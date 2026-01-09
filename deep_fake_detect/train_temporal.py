import argparse
import os
import random
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from deep_fake_detect.DeepFakeDetectModel import DeepFakeDetectModel, TemporalHead
from deep_fake_detect.utils import get_encoder_params
from utils import ConfigParser, print_banner
from data_utils.utils import get_training_reals_and_fakes


class VideoSequenceDataset(Dataset):
    """Builds per-video sequences from cropped frames directories.
    Label per video is 0 (REAL) or 1 (FAKE).
    """

    def __init__(self, mode: str, method: str, max_frames: int = 64, image_size: int = 224):
        assert mode in ["train", "valid"], "Only train/valid supported in this minimal trainer"
        assert method in ["plain_frames", "MRI"], "method must be plain_frames or MRI"
        self.mode = mode
        self.method = method
        self.max_frames = max_frames
        self.cfg = ConfigParser.getInstance()

        if method == "plain_frames":
            root = self.cfg.get_dfdc_crops_train_path() if mode == "train" else self.cfg.get_dfdc_crops_valid_path()
        else:
            root = self.cfg.get_train_mrip2p_png_data_path() if mode == "train" else self.cfg.get_valid_mrip2p_png_data_path()

        self.video_dirs = sorted([d for d in glob(os.path.join(root, "*")) if os.path.isdir(d)])

        # Labels
        if mode == "train":
            reals, fakes = get_training_reals_and_fakes()
            self.real_ids = set([os.path.splitext(x)[0] for x in reals])
            self.fake_ids = set([os.path.splitext(x)[0] for x in fakes])
        else:
            # Expect a CSV mapping of video_id -> label under valid set
            import pandas as pd
            df = pd.read_csv(self.cfg.get_dfdc_valid_label_csv_path(), index_col=0)
            self.real_ids = set([os.path.splitext(x)[0] for x in df[df['label'] == 0].index.values])
            self.fake_ids = set([os.path.splitext(x)[0] for x in df[df['label'] == 1].index.values])

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        vdir = self.video_dirs[idx]
        vid = os.path.basename(vdir)
        label = 1 if vid in self.fake_ids else 0
        frame_paths = sorted(glob(os.path.join(vdir, "*")))
        if len(frame_paths) == 0:
            return None
        if len(frame_paths) > self.max_frames:
            # Uniform sample max_frames indices
            inds = sorted(random.sample(range(len(frame_paths)), self.max_frames))
            frame_paths = [frame_paths[i] for i in inds]
        images = []
        for fp in frame_paths:
            img = torchvision.io.read_image(fp)
            img = img.float() / 255.0
            img = torchvision.transforms.functional.resize(img, [self.transform.transforms[0].size[0], self.transform.transforms[0].size[1]])
            img = torchvision.transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            images.append(img)
        seq = torch.stack(images, dim=0)  # (T, C, H, W)
        return vid, seq, torch.tensor([label], dtype=torch.float32)


def train_temporal(args):
    log_dir = print_banner()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Base CNN encoder
    frame_dim = args.imsize
    encoder_name = args.encoder
    base_model = DeepFakeDetectModel(frame_dim=frame_dim, encoder_name=encoder_name)
    # Load pre-trained per-frame weights
    ckpt = torch.load(args.per_frame_weights, map_location=torch.device('cpu'))
    base_model.load_state_dict(ckpt['model_state_dict'])
    base_model = base_model.to(device)
    base_model.eval()

    embedding_dim, _ = get_encoder_params(encoder_name)
    temporal_head = TemporalHead(embedding_dim=embedding_dim)
    temporal_head = temporal_head.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(temporal_head.parameters(), lr=args.lr)

    # Data
    train_ds = VideoSequenceDataset(mode="train", method=args.method, max_frames=args.max_frames, image_size=frame_dim)
    valid_ds = VideoSequenceDataset(mode="valid", method=args.method, max_frames=args.max_frames, image_size=frame_dim)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=max(0, os.cpu_count() - 2))
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=max(0, os.cpu_count() - 2))

    best_val = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(args.epochs):
        temporal_head.train()
        total_loss = 0.0
        for item in train_loader:
            if item is None:
                continue
            vid, seq, label = item
            seq = seq.squeeze(0).to(device)  # (T, C, H, W)
            label = label.to(device)
            with torch.no_grad():
                feats = base_model.encoder.forward_features(seq)
                feats = base_model.avg_pool(feats).flatten(1)  # (T, D)
            logits = temporal_head(feats.unsqueeze(0))  # (1, 1)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        temporal_head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for item in valid_loader:
                if item is None:
                    continue
                vid, seq, label = item
                seq = seq.squeeze(0).to(device)
                label = label.to(device)
                feats = base_model.encoder.forward_features(seq)
                feats = base_model.avg_pool(feats).flatten(1)
                logits = temporal_head(feats.unsqueeze(0))
                pred = torch.round(torch.sigmoid(logits))
                correct += int((pred == label).sum().item())
                total += 1

        val_acc = (correct / max(1, total)) if total > 0 else 0.0
        print(f"Epoch {epoch+1}/{args.epochs} - TrainLoss={total_loss:.4f} ValAcc={val_acc:.4f}")
        if val_acc >= best_val:
            best_val = val_acc
            torch.save({
                'temporal_state_dict': temporal_head.state_dict(),
                'encoder_name': encoder_name,
                'imsize': frame_dim,
                'method': args.method,
            }, args.out)
            print(f"Saved best temporal head to {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train temporal head over per-frame embeddings')
    parser.add_argument('--method', choices=['plain_frames', 'MRI'], default='plain_frames')
    parser.add_argument('--per_frame_weights', required=True, help='Path to pre-trained per-frame checkpoint (.pth)')
    parser.add_argument('--out', default='assets/weights/temporal_head.pth', help='Output checkpoint path')
    parser.add_argument('--encoder', default='tf_efficientnet_b0_ns')
    parser.add_argument('--imsize', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_frames', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train_temporal(args)



