from __future__ import annotations

import math
from pathlib import Path
import argparse
import os
import warnings
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich import print

from src.config import cfg
from src.data.dataset import VideoFrameDataset
from src.models.generator import SimplePerturbationGenerator
from src.models.discriminator import SmallAttentionDiscriminator


def make_loaders(root: Path, max_videos: int | None = None, frames_per_video: int = 4, fast_no_face: bool = False) -> DataLoader:
    train_root = None
    for name in ["train", "Train"]:
        p = root / name
        if p.exists():
            train_root = p
            break
    if train_root is None:
        raise RuntimeError(f"Train folder not found under {root}")
    ds = VideoFrameDataset(
        train_root,
        frame_stride=cfg.infer.frame_stride,
        frames_per_video=frames_per_video,
        face_size=cfg.infer.face_img_size,
        max_videos=max_videos,
        fast_no_face=fast_no_face,
    )
    # Use single-process loading to avoid pickling Mediapipe objects on Windows
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, pin_memory=False)


def train_one_epoch(
    loader: DataLoader,
    gen: nn.Module,
    disc: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    device: torch.device,
    epoch_index: int | None = None,
) -> Tuple[float, float]:
    gen.train()
    disc.train()
    bce = nn.BCEWithLogitsLoss()
    total_g, total_d, n = 0.0, 0.0, 0
    iterator = tqdm(loader, total=len(loader), desc=f"Epoch {epoch_index or 0}")
    for x, prior, y in iterator:
        x = x.to(device)
        prior = prior.to(device)
        y = y.to(device)

        # Train discriminator
        with torch.no_grad():
            x_adv = gen(x)
        logits_real = disc(x, prior)
        logits_fake = disc(x_adv, prior)
        # Expect real -> 1, fake -> 0
        target_real = torch.ones_like(y)
        target_fake = torch.zeros_like(y)
        loss_d = bce(logits_real, target_real) + bce(logits_fake, target_fake)
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        # Train generator to fool discriminator
        x_adv = gen(x)
        logits_fool = disc(x_adv, prior)
        # Make generated samples be classified as real (1)
        loss_g = bce(logits_fool, torch.ones_like(y))
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

        bs = x.size(0)
        loss_g_val = float(loss_g.item())
        loss_d_val = float(loss_d.item())
        total_g += loss_g_val * bs
        total_d += loss_d_val * bs
        n += bs
        iterator.set_postfix({"loss_g": f"{loss_g_val:.4f}", "loss_d": f"{loss_d_val:.4f}"})
    return total_g / max(n, 1), total_d / max(n, 1)


def save_checkpoint(gen: nn.Module, disc: nn.Module, out_dir: Path, epoch: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "gen": gen.state_dict(), "disc": disc.state_dict()}, out_dir / f"epoch_{epoch:03d}.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=cfg.train.epochs)
    p.add_argument("--batch", type=int, default=cfg.train.batch_size)
    p.add_argument("--lr", type=float, default=cfg.train.lr)
    p.add_argument("--max_videos", type=int, default=0, help="limit number of videos for quick tests")
    p.add_argument("--frames_per_video", type=int, default=4)
    p.add_argument("--fast_no_face", action="store_true", help="skip MediaPipe and use center-crop for speed")
    return p.parse_args()


def main() -> None:
    # Suppress noisy TF/protobuf warnings that make it look stuck
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated", category=UserWarning)
    try:
        from absl import logging as absl_logging  # type: ignore

        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass

    args = parse_args()
    device = torch.device(cfg.infer.device if torch.cuda.is_available() else "cpu")
    print(f"[bold]Training on[/bold] {device}")
    loader = make_loaders(
        cfg.paths.datasets_root,
        max_videos=(args.max_videos if args.max_videos > 0 else None),
        frames_per_video=args.frames_per_video,
        fast_no_face=args.fast_no_face,
    )

    gen = SimplePerturbationGenerator().to(device)
    disc = SmallAttentionDiscriminator().to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr)

    best_score = math.inf
    total_epochs = args.epochs
    for epoch in range(1, total_epochs + 1):
        print(f"[cyan]Epoch {epoch}/{total_epochs}[/cyan] ...")
        loss_g, loss_d = train_one_epoch(loader, gen, disc, opt_g, opt_d, device, epoch_index=epoch)
        print(f"Epoch {epoch}: loss_g={loss_g:.4f} loss_d={loss_d:.4f}")
        save_checkpoint(gen, disc, Path(cfg.paths.models_dir), epoch)
        if loss_g + loss_d < best_score:
            best_score = loss_g + loss_d
            torch.save({"gen": gen.state_dict(), "disc": disc.state_dict()}, Path(cfg.paths.models_dir) / "best.pt")


if __name__ == "__main__":
    main()


