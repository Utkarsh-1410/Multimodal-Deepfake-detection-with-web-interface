import argparse
import os
from tabulate import tabulate
from data_utils.face_detection import *
from deep_fake_detect.utils import *
from deep_fake_detect.DeepFakeDetectModel import *
import torchvision
from data_utils.datasets import *
import warnings
import multiprocessing
import sys
import json

# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)  # Commented out for numpy compatibility


def _smooth_probabilities(probs_np, window):
    if window is None or window <= 1 or probs_np.size == 0:
        return probs_np
    # Simple moving average smoothing
    kernel = np.ones(window) / float(window)
    return np.convolve(probs_np, kernel, mode='same')


def predict_deepfake(input_videofile, df_method, debug=False, verbose=False, temperature=1.0, overwrite_faces=True, overwrite_mri=False, json_out=False, temporal_window=1, return_probs=False, temporal_weights=None):
    num_workers = multiprocessing.cpu_count() - 2
    model_params = dict()
    model_params['batch_size'] = 32
    model_params['imsize'] = 224
    model_params['encoder_name'] = 'tf_efficientnet_b0_ns'

    prob_threshold_fake = 0.5
    fake_fraction = 0.5

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    vid = os.path.basename(input_videofile)[:-4]
    output_path = os.path.join("output", vid)
    plain_faces_data_path = os.path.join(output_path, "plain_frames")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(plain_faces_data_path, exist_ok=True)

    if verbose:
        print(f'Extracting faces from the video')
    # Generate JSON file with location of faces
    extract_landmarks_from_video(input_videofile, output_path, overwrite=overwrite_faces)
    # Crop faces from the video using the JSON file created earlier
    crop_faces_from_video(input_videofile, output_path, plain_faces_data_path, overwrite=overwrite_faces, clean_up=False)

    if df_method == 'plain_frames':
        model_path = 'assets/weights/DeepFake_plain_frames.pth'
        frames_path = plain_faces_data_path
    elif df_method == 'MRI':
        if verbose:
            print(f'Generating MRIs of the faces')
        mri_output = os.path.join(output_path, 'mri')
        predict_mri_using_MRI_GAN(plain_faces_data_path, mri_output, vid, 256, overwrite=overwrite_mri)
        model_path = 'assets/weights/DeepFake_MRI.pth'
        frames_path = mri_output
    elif df_method == 'fusion':
        # Enhanced fusion with weighted ensemble
        from deep_fake_detect.fusion import WeightedEnsembleFusion, smooth_probabilities_temporal
        
        fake_prob_pf, real_prob_pf, pred_pf, probs_pf = predict_deepfake(
            input_videofile, 'plain_frames', debug, verbose, temperature, overwrite_faces, False, json_out, temporal_window, True
        )
        fake_prob_mri, real_prob_mri, pred_mri, probs_mri = predict_deepfake(
            input_videofile, 'MRI', debug, verbose, temperature, False, overwrite_mri, json_out, temporal_window, True
        )
        if probs_pf is None or probs_mri is None or len(probs_pf) == 0 or len(probs_mri) == 0:
            return None, None, None, None if return_probs else (None, None, None)
        
        # Use weighted ensemble fusion (can be configured with weights)
        # Default: MRI gets slightly higher weight (0.55) as it's more specialized
        fusion_weights = {'plain_frames': 0.45, 'MRI': 0.55}
        fusion = WeightedEnsembleFusion(weights=fusion_weights, adaptive=True)
        
        method_probs = {
            'plain_frames': np.array(probs_pf),
            'MRI': np.array(probs_mri)
        }
        
        # Compute confidences for adaptive weighting
        conf_pf = np.abs(np.array(probs_pf) - 0.5) * 2  # Distance from 0.5
        conf_mri = np.abs(np.array(probs_mri) - 0.5) * 2
        method_confidences = {
            'plain_frames': conf_pf,
            'MRI': conf_mri
        }
        
        fused_probs = fusion.fuse_probabilities(method_probs, method_confidences)
        
        # Apply temporal smoothing
        fused_probs = smooth_probabilities_temporal(fused_probs, window_size=temporal_window, method='moving_average')
        
        prob_threshold_fake = 0.5
        fake_frames_high_prob = fused_probs[fused_probs >= prob_threshold_fake]
        real_frames_high_prob = fused_probs[fused_probs < prob_threshold_fake]
        number_fake_frames = len(fake_frames_high_prob)
        number_real_frames = len(real_frames_high_prob)
        total_number_frames = len(fused_probs)
        fake_prob = 0 if number_fake_frames == 0 else round(float(np.mean(fake_frames_high_prob)), 4)
        real_prob = 0 if number_real_frames == 0 else 1 - round(float(np.mean(real_frames_high_prob)), 4)
        pred = pred_strategy(number_fake_frames, number_real_frames, total_number_frames, fake_fraction=0.5)
        if return_probs:
            return fake_prob, real_prob, pred, fused_probs.tolist()
        return fake_prob, real_prob, pred
    elif df_method == 'temporal':
        # Sequence-based inference: extract per-frame embeddings, then temporal head
        if verbose:
            print(f'Running temporal model over frame embeddings')
        model = DeepFakeDetectModel(frame_dim=model_params['imsize'], encoder_name=model_params['encoder_name'])
        check_point_dict = torch.load('assets/weights/DeepFake_plain_frames.pth', map_location=torch.device('cpu'))
        model.load_state_dict(check_point_dict['model_state_dict'])
        model = model.to(device)
        model.eval()

        # Build transforms and sequence dataset
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((model_params['imsize'], model_params['imsize'])),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        data_path = os.path.join(plain_faces_data_path, vid)
        from data_utils.datasets import SequentialFramesDataset
        seq_dataset = SequentialFramesDataset(root=data_path, transforms_=test_transform)
        if len(seq_dataset) == 0:
            if not json_out:
                print('Cannot extract images. Sequence empty')
            return None, None, None
        seq, frame_paths = seq_dataset[0]
        if seq is None or len(frame_paths) == 0:
            if not json_out:
                print('Cannot extract images. Sequence empty')
            return None, None, None
        # seq: (T, C, H, W)
        seq = seq.to(device)
        # Pass frames through encoder to get embeddings
        with torch.no_grad():
            feats = model.encoder.forward_features(seq)           # (T, C, H, W)
            feats = model.avg_pool(feats).flatten(1)              # (T, D)
        # Temporal head
        temporal_head = TemporalHead(embedding_dim=feats.shape[1]).to(device)
        # Load trained temporal weights if provided
        if temporal_weights is not None and os.path.isfile(temporal_weights):
            ckpt_t = torch.load(temporal_weights, map_location=torch.device('cpu'))
            try:
                temporal_head.load_state_dict(ckpt_t['temporal_state_dict'])
            except Exception:
                pass
        temporal_head.eval()
        with torch.no_grad():
            logits = temporal_head(feats.unsqueeze(0))            # (1, 1)
            class_probability_t = torch.sigmoid(logits / max(temperature, 1e-6))
            pred = int(torch.round(class_probability_t).item())
            fake_prob = float(class_probability_t.item())
            real_prob = 1.0 - fake_prob
        return round(fake_prob, 4), round(real_prob, 4), pred
    else:
        raise Exception("Unknown method")

    if verbose:
        print(f'Detecting DeepFakes using method: {df_method}')
    model = DeepFakeDetectModel(frame_dim=model_params['imsize'], encoder_name=model_params['encoder_name'])
    if verbose:
        print(f'Loading model weights {model_path}')
    check_point_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(check_point_dict['model_state_dict'])
    model = model.to(device)
    model.eval()

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((model_params['imsize'], model_params['imsize'])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])

    data_path = os.path.join(frames_path, vid)
    test_dataset = SimpleImageFolder(root=data_path, transforms_=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=model_params['batch_size'],
        num_workers=num_workers,
        pin_memory=use_cuda
    )
    if len(test_loader) == 0:
        if not json_out:
            print('Cannot extract images. Dataloaders empty')
        return None, None, None
    probabilities = []
    all_filenames = []
    all_predicted_labels = []
    with torch.no_grad():
        for batch_id, samples in enumerate(test_loader):
            frames = samples[0].to(device)
            output = model(frames)
            # Temperature scaling for calibrated probabilities
            if temperature <= 0:
                temperature = 1.0
            class_probability_t = torch.sigmoid(output / temperature)
            predicted_t = torch.round(class_probability_t)
            predicted = predicted_t.to('cpu').detach().numpy()
            class_probability = class_probability_t.to('cpu').detach().numpy()
            if len(predicted) > 1:
                all_predicted_labels.extend(predicted.squeeze())
                probabilities.extend(class_probability.squeeze())
                all_filenames.extend(samples[1])
            else:
                all_predicted_labels.append(predicted.squeeze())
                probabilities.append(class_probability.squeeze())
                all_filenames.append(samples[1])

        total_number_frames = len(probabilities)
        probabilities = np.array(probabilities, dtype=float)
        # Temporal smoothing
        probabilities = _smooth_probabilities(probabilities, temporal_window)

        fake_frames_high_prob = probabilities[probabilities >= prob_threshold_fake]
        number_fake_frames = len(fake_frames_high_prob)
        if number_fake_frames == 0:
            fake_prob = 0
        else:
            fake_prob = round(sum(fake_frames_high_prob) / number_fake_frames, 4)

        real_frames_high_prob = probabilities[probabilities < prob_threshold_fake]
        number_real_frames = len(real_frames_high_prob)
        if number_real_frames == 0:
            real_prob = 0
        else:
            real_prob = 1 - round(sum(real_frames_high_prob) / number_real_frames, 4)

        pred = pred_strategy(number_fake_frames, number_real_frames, total_number_frames,
                             fake_fraction=fake_fraction)

        if debug:
            print(f'all {probabilities}')
            print(f'real {real_frames_high_prob}')
            print(f'fake {fake_frames_high_prob}')
            print(
                f"number_fake_frames={number_fake_frames}, number_real_frames={number_real_frames}, total_number_frames={total_number_frames}, fake_fraction={fake_fraction}")
            print(f'fake_prob = {round(fake_prob * 100, 4)}%, real_prob = {round(real_prob * 100, 4)}%  pred={pred}')
        if return_probs:
            return fake_prob, real_prob, pred, probabilities.tolist()
        return fake_prob, real_prob, pred


def individual_test():
    print_line()
    debug = False
    verbose = True
    fake_prob, real_prob, pred = predict_deepfake(
        args.input_videofile,
        args.method,
        debug=debug,
        verbose=verbose,
        temperature=args.temperature,
        overwrite_faces=args.overwrite_faces,
        overwrite_mri=args.overwrite_mri,
        json_out=args.json,
        temporal_window=args.temporal_window,
        temporal_weights=args.temporal_weights
    )
    if pred is None:
        if args.json:
            print(json.dumps({
                'status': 'error',
                'reason': 'no_frames_or_faces',
                'video': args.input_videofile,
                'method': args.method
            }))
        else:
            print_red('Failed to detect DeepFakes')
            return

    label = "REAL" if pred == 0 else "DEEP-FAKE"

    probability = real_prob if pred == 0 else fake_prob
    probability = round(probability * 100, 4)
    if args.json:
        print(json.dumps({
            'status': 'ok',
            'video': args.input_videofile,
            'method': args.method,
            'label': label,
            'probability_percent': float(probability)
        }))
    else:
        print_line()
        if pred == 0:
            print_green(f'The video is {label}, probability={probability}%')
        else:
            print_red(f'The video is {label}, probability={probability}%')
        print_line()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DeepFakes detection App. \n Use demo mode or provide input_videofile and method')
    parser.add_argument('--input_videofile', action='store', help='Input video file')
    parser.add_argument('--method', action='store', choices=['plain_frames', 'MRI', 'fusion', 'temporal'],
                        help='Method type')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for probability calibration (>=0)')
    parser.add_argument('--overwrite_faces', action='store_true', default=False, help='Force re-extracting faces')
    parser.add_argument('--overwrite_mri', action='store_true', default=False, help='Force regenerating MRIs (MRI method)')
    parser.add_argument('--json', action='store_true', help='Output JSON result to stdout')
    parser.add_argument('--temporal_window', type=int, default=1, help='Temporal smoothing window (frames)')
    parser.add_argument('--temporal_weights', type=str, default=None, help='Path to trained temporal head checkpoint')
    args = parser.parse_args()
    if args.input_videofile is not None:
        if args.method is None:
            parser.print_help(sys.stderr)
        else:
            if os.path.isfile(args.input_videofile):
                individual_test()
            else:
                print(f'input file not found ({args.input_videofile})')
    else:
        parser.print_help(sys.stderr)