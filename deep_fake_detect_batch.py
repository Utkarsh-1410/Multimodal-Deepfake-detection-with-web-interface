#!/usr/bin/env python3
"""
Enhanced DeepFake Detection App with Directory Support
Supports both single video files and directories of videos
"""

import argparse
import os
import glob as pyglob
from pathlib import Path
from tabulate import tabulate
from data_utils.face_detection import *
from deep_fake_detect.utils import *
from deep_fake_detect.DeepFakeDetectModel import *
import torchvision
from data_utils.datasets import *
import warnings
import multiprocessing
import sys
import time
from datetime import datetime

# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)  # Commented out for numpy compatibility

# Supported video formats
SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']

def print_line():
    print('-' * 80)

def print_green(text):
    print(f'\033[92m{text}\033[0m')

def print_red(text):
    print(f'\033[91m{text}\033[0m')

def print_yellow(text):
    print(f'\033[93m{text}\033[0m')

def print_blue(text):
    print(f'\033[94m{text}\033[0m')

def get_video_files(input_path):
    """Get all video files from input path (file or directory)"""
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single file
        if input_path.suffix.lower() in SUPPORTED_FORMATS:
            return [str(input_path)]
        else:
            print_red(f"Unsupported file format: {input_path.suffix}")
            return []
    
    elif input_path.is_dir():
        # Directory - find all video files
        video_files = []
        for ext in SUPPORTED_FORMATS:
            pattern = str(input_path / f"**/*{ext}")
            video_files.extend(pyglob.glob(pattern, recursive=True))
        
        # Also check for case variations
        for ext in SUPPORTED_FORMATS:
            pattern = str(input_path / f"**/*{ext.upper()}")
            video_files.extend(pyglob.glob(pattern, recursive=True))
        
        return sorted(video_files)
    
    else:
        print_red(f"Path does not exist: {input_path}")
        return []

def predict_deepfake(input_videofile, df_method, debug=False, verbose=False):
    """Predict if a single video is a DeepFake"""
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
    extract_landmarks_from_video(input_videofile, output_path, overwrite=True)
    # Crop faces from the video using the JSON file created earlier
    crop_faces_from_video(input_videofile, output_path, plain_faces_data_path, overwrite=True, clean_up=False)

    if df_method == 'plain_frames':
        model_path = 'assets/weights/DeepFake_plain_frames.pth'
        frames_path = plain_faces_data_path
    elif df_method == 'MRI':
        if verbose:
            print(f'Generating MRIs of the faces')
        mri_output = os.path.join(output_path, 'mri')
        predict_mri_using_MRI_GAN(plain_faces_data_path, mri_output, vid, 256, overwrite=True)
        model_path = 'assets/weights/DeepFake_MRI.pth'
        frames_path = mri_output
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
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Use SimpleImageFolder from data_utils.datasets to load frame images
    test_dataset = SimpleImageFolder(frames_path, transforms_=test_transform)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=model_params['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    fake_frames_high_prob = []
    real_frames_high_prob = []
    fake_prob_list = []
    real_prob_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if debug:
                print(f'Processing batch {batch_idx + 1}/{len(test_dataloader)}')
            # batch is a tuple: (images, image_names)
            data = batch[0].to(device)
            output = model(data)
            output = torch.nn.functional.softmax(output, dim=1)
            fake_prob = output[:, 1].cpu().numpy()
            real_prob = output[:, 0].cpu().numpy()

            fake_prob_list.extend(fake_prob)
            real_prob_list.extend(real_prob)

            for i in range(len(fake_prob)):
                if fake_prob[i] > prob_threshold_fake:
                    fake_frames_high_prob.append(fake_prob[i])
                else:
                    real_frames_high_prob.append(real_prob[i])

    number_fake_frames = len(fake_frames_high_prob)
    number_real_frames = len(real_frames_high_prob)
    total_number_frames = number_fake_frames + number_real_frames

    if total_number_frames == 0:
        return None, None, None

    fake_prob = number_fake_frames / total_number_frames
    real_prob = number_real_frames / total_number_frames

    if fake_prob > fake_fraction:
        pred = 1  # DeepFake
    else:
        pred = 0  # Real

    if debug:
        print(f'fake {fake_frames_high_prob}')
        print(
            f"number_fake_frames={number_fake_frames}, number_real_frames={number_real_frames}, total_number_frames={total_number_frames}, fake_fraction={fake_fraction}")
        print(f'fake_prob = {round(fake_prob * 100, 4)}%, real_prob = {round(real_prob * 100, 4)}%  pred={pred}')
    return fake_prob, real_prob, pred

def process_single_video(video_path, method, verbose=False):
    """Process a single video and return results"""
    try:
        if verbose:
            print_blue(f"\nProcessing: {os.path.basename(video_path)}")
        
        start_time = time.time()
        fake_prob, real_prob, pred = predict_deepfake(video_path, method, debug=False, verbose=verbose)
        processing_time = time.time() - start_time
        
        if pred is None:
            return {
                'video': os.path.basename(video_path),
                'status': 'FAILED',
                'prediction': 'N/A',
                'confidence': 'N/A',
                'processing_time': f"{processing_time:.2f}s",
                'error': 'No faces detected or processing failed'
            }
        
        label = "DEEP-FAKE" if pred == 1 else "REAL"
        confidence = real_prob if pred == 0 else fake_prob
        confidence = round(confidence * 100, 2)
        
        return {
            'video': os.path.basename(video_path),
            'status': 'SUCCESS',
            'prediction': label,
            'confidence': f"{confidence}%",
            'processing_time': f"{processing_time:.2f}s",
            'error': None
        }
        
    except Exception as e:
        return {
            'video': os.path.basename(video_path),
            'status': 'ERROR',
            'prediction': 'N/A',
            'confidence': 'N/A',
            'processing_time': 'N/A',
            'error': str(e)
        }

def process_videos_batch(input_path, method, verbose=False, save_results=True):
    """Process multiple videos in batch"""
    video_files = get_video_files(input_path)
    
    if not video_files:
        print_red("No video files found in the specified path!")
        return
    
    print_blue(f"\nFound {len(video_files)} video file(s) to process")
    print_line()
    
    results = []
    start_time = time.time()
    
    for i, video_path in enumerate(video_files, 1):
        print_yellow(f"\n[{i}/{len(video_files)}] Processing: {os.path.basename(video_path)}")
        result = process_single_video(video_path, method, verbose)
        results.append(result)
        
        # Print immediate result
        if result['status'] == 'SUCCESS':
            color = print_green if result['prediction'] == 'REAL' else print_red
            color(f"  → {result['prediction']} ({result['confidence']}) - {result['processing_time']}")
        else:
            print_red(f"  → {result['status']}: {result['error']}")
    
    total_time = time.time() - start_time
    
    # Print summary table
    print_line()
    print_blue(f"\nBATCH PROCESSING SUMMARY")
    print_line()
    
    # Prepare table data
    table_data = []
    for result in results:
        table_data.append([
            result['video'],
            result['status'],
            result['prediction'],
            result['confidence'],
            result['processing_time']
        ])
    
    print(tabulate(table_data, 
                   headers=['Video File', 'Status', 'Prediction', 'Confidence', 'Time'],
                   tablefmt='grid'))
    
    # Statistics
    successful = len([r for r in results if r['status'] == 'SUCCESS'])
    failed = len([r for r in results if r['status'] != 'SUCCESS'])
    real_count = len([r for r in results if r['prediction'] == 'REAL'])
    fake_count = len([r for r in results if r['prediction'] == 'DEEP-FAKE'])
    
    print_line()
    print_blue(f"STATISTICS:")
    print(f"  Total videos: {len(video_files)}")
    print(f"  Successfully processed: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Real videos: {real_count}")
    print(f"  DeepFake videos: {fake_count}")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Average time per video: {total_time/len(video_files):.2f}s")
    print_line()
    
    # Save results to CSV if requested
    if save_results:
        save_results_to_csv(results, input_path, method)
    
    return results

def save_results_to_csv(results, input_path, method):
    """Save results to CSV file"""
    import csv
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_name = Path(input_path).name
    csv_filename = f"deepfake_results_{input_name}_{method}_{timestamp}.csv"
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['video', 'status', 'prediction', 'confidence', 'processing_time', 'error', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            result['timestamp'] = datetime.now().isoformat()
            writer.writerow(result)
    
    print_green(f"\nResults saved to: {csv_filename}")

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced DeepFakes Detection App - Supports single videos and directories')
    parser.add_argument('--input', action='store', required=True,
                        help='Input video file or directory containing videos')
    parser.add_argument('--method', action='store', choices=['plain_frames', 'MRI'], required=True,
                        help='Detection method: plain_frames (91%% accuracy) or MRI (74%% accuracy)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to CSV file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print_red(f"Path does not exist: {args.input}")
        sys.exit(1)
    
    print_blue("Enhanced DeepFake Detection App")
    print_blue("=" * 50)
    print(f"Input: {args.input}")
    print(f"Method: {args.method}")
    print(f"Verbose: {args.verbose}")
    print_line()
    
    # Process videos
    results = process_videos_batch(
        args.input, 
        args.method, 
        verbose=args.verbose,
        save_results=not args.no_save
    )
    
    if results:
        print_green("\nBatch processing completed!")
    else:
        print_red("\nBatch processing failed!")

if __name__ == '__main__':
    main()
