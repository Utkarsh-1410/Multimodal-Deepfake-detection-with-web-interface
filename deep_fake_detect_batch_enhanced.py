"""
Enhanced Batch Processing for DeepFake Detection
Supports multiple methods, parallel processing, and comprehensive evaluation
"""

import argparse
import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deep_fake_detect_app import predict_deepfake
from deep_fake_detect.evaluation import DetectionEvaluator, compare_methods


def process_single_video(video_path: str, 
                         method: str,
                         ground_truth: Optional[int] = None,
                         temperature: float = 1.0,
                         temporal_window: int = 5,
                         verbose: bool = False) -> Dict:
    """
    Process a single video file.
    
    Args:
        video_path: Path to video file
        method: Detection method ('plain_frames', 'MRI', 'fusion', 'temporal')
        ground_truth: Optional ground truth label (0=real, 1=fake)
        temperature: Temperature for probability calibration
        temporal_window: Temporal smoothing window
        verbose: Verbose output
    
    Returns:
        Dictionary with results
    """
    try:
        fake_prob, real_prob, pred = predict_deepfake(
            input_videofile=video_path,
            df_method=method,
            debug=False,
            verbose=verbose,
            temperature=temperature,
            overwrite_faces=False,  # Don't overwrite for batch processing
            overwrite_mri=False,
            json_out=False,
            temporal_window=temporal_window
        )
        
        if pred is None:
            return {
                'video': video_path,
                'method': method,
                'status': 'error',
                'error': 'No faces detected'
            }
        
        result = {
            'video': os.path.basename(video_path),
            'video_path': video_path,
            'method': method,
            'status': 'success',
            'prediction': 'REAL' if pred == 0 else 'DEEP-FAKE',
            'fake_probability': float(fake_prob),
            'real_probability': float(real_prob),
            'predicted_label': int(pred)
        }
        
        if ground_truth is not None:
            result['ground_truth'] = ground_truth
            result['ground_truth_label'] = 'REAL' if ground_truth == 0 else 'DEEP-FAKE'
            result['correct'] = int(pred == ground_truth)
        
        return result
    
    except Exception as e:
        return {
            'video': os.path.basename(video_path) if video_path else 'unknown',
            'video_path': video_path,
            'method': method,
            'status': 'error',
            'error': str(e)
        }


def process_videos_batch(video_paths: List[str],
                        method: str,
                        ground_truths: Optional[Dict[str, int]] = None,
                        num_workers: int = 4,
                        temperature: float = 1.0,
                        temporal_window: int = 5,
                        verbose: bool = False) -> List[Dict]:
    """
    Process multiple videos in parallel.
    
    Args:
        video_paths: List of video file paths
        method: Detection method
        ground_truths: Optional dictionary mapping video paths to ground truth labels
        num_workers: Number of parallel workers
        temperature: Temperature for probability calibration
        temporal_window: Temporal smoothing window
        verbose: Verbose output
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Prepare arguments for each video
    tasks = []
    for video_path in video_paths:
        gt = ground_truths.get(video_path) if ground_truths else None
        tasks.append((video_path, method, gt, temperature, temporal_window, verbose))
    
    # Process in parallel
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_video, *task): task[0] 
                      for task in tasks}
            
            with tqdm(total=len(tasks), desc=f"Processing videos ({method})") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
    else:
        # Sequential processing
        for task in tqdm(tasks, desc=f"Processing videos ({method})"):
            result = process_single_video(*task)
            results.append(result)
    
    return results


def load_ground_truth(csv_path: str) -> Dict[str, int]:
    """
    Load ground truth labels from CSV file.
    Expected format: video_filename,label (0=real, 1=fake)
    """
    ground_truths = {}
    if not os.path.exists(csv_path):
        return ground_truths
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = row.get('video', row.get('filename', ''))
            label = int(row.get('label', 0))
            ground_truths[video_name] = label
    
    return ground_truths


def save_results(results: List[Dict], output_path: str, method: str):
    """Save results to JSON and CSV files"""
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_path = output_path.replace('.csv', '.json') if output_path.endswith('.csv') else output_path + '.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV
    csv_path = output_path if output_path.endswith('.csv') else output_path + '.csv'
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    print(f"Results saved to {json_path} and {csv_path}")


def evaluate_results(results: List[Dict], output_dir: str, method: str):
    """Evaluate results and generate metrics"""
    evaluator = DetectionEvaluator()
    
    successful_results = [r for r in results if r.get('status') == 'success']
    if len(successful_results) == 0:
        print("No successful results to evaluate")
        return
    
    # Extract predictions and labels
    predictions = []
    labels = []
    probabilities = []
    
    for result in successful_results:
        if 'ground_truth' in result:
            predictions.append(result['predicted_label'])
            labels.append(result['ground_truth'])
            probabilities.append(result['fake_probability'])
    
    if len(labels) > 0:
        evaluator.update(
            np.array(predictions),
            np.array(labels),
            np.array(probabilities) if probabilities else None
        )
        
        metrics = evaluator.generate_report(output_dir, method_name=method)
        print(f"\nEvaluation Metrics for {method}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='Batch DeepFake Detection')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing videos')
    parser.add_argument('--output', type=str, required=True, help='Output file path (JSON/CSV)')
    parser.add_argument('--method', type=str, choices=['plain_frames', 'MRI', 'fusion', 'temporal'],
                       default='fusion', help='Detection method')
    parser.add_argument('--ground_truth', type=str, help='CSV file with ground truth labels')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for calibration')
    parser.add_argument('--temporal_window', type=int, default=5, help='Temporal smoothing window')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate results if ground truth provided')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(Path(args.input_dir).glob(f'*{ext}'))
        video_paths.extend(Path(args.input_dir).glob(f'*{ext.upper()}'))
    
    video_paths = [str(p) for p in video_paths]
    
    if len(video_paths) == 0:
        print(f"No video files found in {args.input_dir}")
        return
    
    print(f"Found {len(video_paths)} video files")
    
    # Load ground truth if provided
    ground_truths = {}
    if args.ground_truth:
        ground_truths = load_ground_truth(args.ground_truth)
        print(f"Loaded {len(ground_truths)} ground truth labels")
    
    # Process videos
    results = process_videos_batch(
        video_paths=video_paths,
        method=args.method,
        ground_truths=ground_truths,
        num_workers=args.num_workers,
        temperature=args.temperature,
        temporal_window=args.temporal_window,
        verbose=args.verbose
    )
    
    # Save results
    save_results(results, args.output, args.method)
    
    # Evaluate if requested
    if args.evaluate and args.ground_truth:
        output_dir = os.path.dirname(args.output) or '.'
        evaluate_results(results, output_dir, args.method)
    
    # Print summary
    successful = sum(1 for r in results if r.get('status') == 'success')
    errors = len(results) - successful
    print(f"\nSummary: {successful} successful, {errors} errors")


if __name__ == '__main__':
    import numpy as np
    main()

