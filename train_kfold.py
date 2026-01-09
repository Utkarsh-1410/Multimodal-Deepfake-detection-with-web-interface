"""
Train DeepFake Detection Model with K-Fold Cross-Validation
"""

import argparse
import os
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deep_fake_detect.training import train_model
from utils import ConfigParser


def load_fold_config(fold_idx, split_name='train'):
    """Load video list for a specific fold and split"""
    fold_dir = os.path.join('assets', 'kfold_splits', f'fold_{fold_idx + 1}')
    split_file = os.path.join(fold_dir, f'{split_name}_videos.json')
    
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Fold split not found: {split_file}. Run kfold_cv.py first.")
    
    with open(split_file, 'r') as f:
        videos_data = json.load(f)
    
    return videos_data


def create_temporary_csv_for_fold(fold_idx, split_name, output_csv):
    """Create a temporary CSV file for training with fold-specific data"""
    fold_dir = os.path.join('assets', 'kfold_splits', f'fold_{fold_idx + 1}')
    csv_file = os.path.join(fold_dir, f'{split_name}_frame_labels.csv')
    
    if not os.path.exists(csv_file):
        print(f"Frame labels CSV not found: {csv_file}")
        print("Creating it now...")
        from kfold_cv import create_frame_labels_csv_for_fold
        create_frame_labels_csv_for_fold(fold_idx, split_name)
        csv_file = os.path.join(fold_dir, f'{split_name}_frame_labels.csv')
    
    if os.path.exists(csv_file):
        # Copy to temporary location that training script expects
        import shutil
        shutil.copy(csv_file, output_csv)
        print(f"Using fold-specific CSV: {csv_file}")
        return True
    else:
        print(f"Warning: Could not create CSV for fold {fold_idx + 1}, {split_name}")
        return False


def train_fold(fold_idx, method='plain_frames', resume=False):
    """
    Train model for a specific fold
    
    Args:
        fold_idx: Fold index (0-3, will be converted from 1-4 if needed)
        method: 'plain_frames' or 'mri'
        resume: Whether to resume from checkpoint
    """
    if fold_idx > 0 and fold_idx <= 4:
        fold_idx = fold_idx - 1  # Convert to 0-indexed
    
    print("="*60)
    print(f"Training Fold {fold_idx + 1}/4")
    print("="*60)
    
    # Load fold data
    train_videos = load_fold_config(fold_idx, 'train')
    val_videos = load_fold_config(fold_idx, 'val')
    test_videos = load_fold_config(fold_idx, 'test')
    
    print(f"Train videos: {len(train_videos)}")
    print(f"Val videos: {len(val_videos)}")
    print(f"Test videos: {len(test_videos)}")
    
    # Create log directory for this fold
    log_dir = ConfigParser.getInstance().get_log_dir_name()
    fold_log_dir = os.path.join(log_dir, f'fold_{fold_idx + 1}')
    os.makedirs(fold_log_dir, exist_ok=True)
    
    # Save fold info
    fold_info = {
        'fold': fold_idx + 1,
        'train_count': len(train_videos),
        'val_count': len(val_videos),
        'test_count': len(test_videos),
        'method': method
    }
    
    with open(os.path.join(fold_log_dir, 'fold_info.json'), 'w') as f:
        json.dump(fold_info, f, indent=2)
    
    # Create temporary CSV files for this fold
    # Note: This requires modifying the training script to accept custom CSV paths
    # For now, we'll use the standard training which uses config paths
    
    print(f"\nStarting training for Fold {fold_idx + 1}...")
    print(f"Log directory: {fold_log_dir}")
    
    # Train the model
    checkpoint_path = None
    if resume:
        # Find latest checkpoint for this fold
        checkpoint_path = os.path.join(fold_log_dir, 'checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = None
    
    try:
        train_model(log_dir=fold_log_dir, train_resume_checkpoint=checkpoint_path)
        print(f"\n✓ Training completed for Fold {fold_idx + 1}")
    except Exception as e:
        print(f"\n✗ Training failed for Fold {fold_idx + 1}: {e}")
        raise


def train_all_folds(method='plain_frames', resume=False):
    """Train models for all 4 folds"""
    print("="*60)
    print("K-Fold Cross-Validation Training")
    print("="*60)
    print(f"Method: {method}")
    print(f"Folds: 4")
    print("="*60)
    
    results = {}
    
    for fold_idx in range(4):
        try:
            train_fold(fold_idx, method=method, resume=resume)
            results[f'fold_{fold_idx + 1}'] = 'success'
        except Exception as e:
            print(f"Fold {fold_idx + 1} failed: {e}")
            results[f'fold_{fold_idx + 1}'] = f'failed: {e}'
    
    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    for fold, status in results.items():
        print(f"{fold}: {status}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train with K-Fold Cross-Validation')
    parser.add_argument('--fold', type=int, choices=[1, 2, 3, 4], help='Train specific fold (1-4)')
    parser.add_argument('--all', action='store_true', help='Train all 4 folds')
    parser.add_argument('--method', type=str, choices=['plain_frames', 'mri'], default='plain_frames',
                       help='Detection method')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    if args.all:
        train_all_folds(method=args.method, resume=args.resume)
    elif args.fold:
        train_fold(args.fold, method=args.method, resume=args.resume)
    else:
        print("Please specify --fold <1-4> or --all")
        parser.print_help()


if __name__ == '__main__':
    main()

