"""
K-Fold Cross-Validation Utility for Custom Dataset
K=4, Split: 80% Train, 10% Validation, 10% Test
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from glob import glob
from utils import ConfigParser
from collections import defaultdict
import shutil


def get_custom_dataset_videos():
    """Get all videos from custom dataset"""
    real_path = ConfigParser.getInstance().get_custom_dataset_real_path()
    fake_path = ConfigParser.getInstance().get_custom_dataset_fake_path()
    
    real_videos = sorted(glob(os.path.join(real_path, '*')))
    fake_videos = sorted(glob(os.path.join(fake_path, '*')))
    
    # Filter to only video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    real_videos = [v for v in real_videos if os.path.splitext(v)[1].lower() in video_extensions]
    fake_videos = [v for v in fake_videos if os.path.splitext(v)[1].lower() in video_extensions]
    
    return real_videos, fake_videos


def create_kfold_splits(k=4, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Create k-fold cross-validation splits.
    
    Args:
        k: Number of folds (default: 4)
        train_ratio: Training set ratio (default: 0.8)
        val_ratio: Validation set ratio (default: 0.1)
        test_ratio: Test set ratio (default: 0.1)
    
    Returns:
        Dictionary with fold splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    real_videos, fake_videos = get_custom_dataset_videos()
    
    print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos")
    
    # Create labels: 0 for real, 1 for fake
    all_videos = real_videos + fake_videos
    all_labels = [0] * len(real_videos) + [1] * len(fake_videos)
    
    # Use stratified k-fold to maintain class balance
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    folds = {}
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(all_videos, all_labels)):
        # Split train_val into train and validation
        # We want 80% train, 10% val, 10% test
        # So from train_val (90%), we need 80/90 = 88.9% for train, 10/90 = 11.1% for val
        
        train_val_videos = [all_videos[i] for i in train_val_idx]
        train_val_labels = [all_labels[i] for i in train_val_idx]
        test_videos = [all_videos[i] for i in test_idx]
        test_labels = [all_labels[i] for i in test_idx]
        
        # Split train_val into train and validation (maintaining class balance)
        val_ratio_in_train_val = val_ratio / (train_ratio + val_ratio)  # 10% / 90% = 11.11%
        skf_inner = StratifiedKFold(n_splits=int(1/val_ratio_in_train_val), shuffle=True, random_state=42)
        
        # Get one split for validation
        train_idx_inner, val_idx_inner = next(skf_inner.split(train_val_videos, train_val_labels))
        
        train_videos = [train_val_videos[i] for i in train_idx_inner]
        train_labels = [train_val_labels[i] for i in train_idx_inner]
        val_videos = [train_val_videos[i] for i in val_idx_inner]
        val_labels = [train_val_labels[i] for i in val_idx_inner]
        
        folds[fold_idx] = {
            'train': {
                'videos': train_videos,
                'labels': train_labels,
                'count': len(train_videos),
                'real_count': train_labels.count(0),
                'fake_count': train_labels.count(1)
            },
            'val': {
                'videos': val_videos,
                'labels': val_labels,
                'count': len(val_videos),
                'real_count': val_labels.count(0),
                'fake_count': val_labels.count(1)
            },
            'test': {
                'videos': test_videos,
                'labels': test_labels,
                'count': len(test_videos),
                'real_count': test_labels.count(0),
                'fake_count': test_labels.count(1)
            }
        }
        
        print(f"\nFold {fold_idx + 1}/{k}:")
        print(f"  Train: {len(train_videos)} videos ({len(train_videos)/len(all_videos)*100:.1f}%) - Real: {train_labels.count(0)}, Fake: {train_labels.count(1)}")
        print(f"  Val:   {len(val_videos)} videos ({len(val_videos)/len(all_videos)*100:.1f}%) - Real: {val_labels.count(0)}, Fake: {val_labels.count(1)}")
        print(f"  Test:  {len(test_videos)} videos ({len(test_videos)/len(all_videos)*100:.1f}%) - Real: {test_labels.count(0)}, Fake: {test_labels.count(1)}")
    
    return folds


def save_kfold_splits(folds, output_dir='assets/kfold_splits'):
    """Save k-fold splits to JSON files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for fold_idx, fold_data in folds.items():
        fold_dir = os.path.join(output_dir, f'fold_{fold_idx + 1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # Save splits as JSON
        for split_name in ['train', 'val', 'test']:
            split_data = fold_data[split_name]
            output_file = os.path.join(fold_dir, f'{split_name}_videos.json')
            
            # Save as list of video paths with labels
            videos_with_labels = [
                {'path': video, 'label': label, 'class': 'real' if label == 0 else 'fake'}
                for video, label in zip(split_data['videos'], split_data['labels'])
            ]
            
            with open(output_file, 'w') as f:
                json.dump(videos_with_labels, f, indent=2)
            
            print(f"Saved {split_name} split for fold {fold_idx + 1}: {output_file}")
    
    # Save summary
    summary = {
        'k': len(folds),
        'total_videos': sum(len(folds[0][split]['videos']) for split in ['train', 'val', 'test']),
        'folds': {
            f'fold_{i+1}': {
                'train': folds[i]['train']['count'],
                'val': folds[i]['val']['count'],
                'test': folds[i]['test']['count']
            }
            for i in range(len(folds))
        }
    }
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ K-fold splits saved to {output_dir}")
    print(f"✓ Summary saved to {summary_file}")


def create_frame_labels_csv_for_fold(fold_idx, split_name, output_dir='assets/kfold_splits'):
    """
    Create frame labels CSV for a specific fold and split.
    This is needed for the training pipeline.
    """
    fold_dir = os.path.join(output_dir, f'fold_{fold_idx + 1}')
    split_file = os.path.join(fold_dir, f'{split_name}_videos.json')
    
    if not os.path.exists(split_file):
        print(f"Split file not found: {split_file}")
        return None
    
    with open(split_file, 'r') as f:
        videos_data = json.load(f)
    
    # Get landmarks and crop faces directories
    landmarks_path = ConfigParser.getInstance().get_custom_dataset_landmarks_path()
    crops_path = ConfigParser.getInstance().get_custom_dataset_crops_path()
    
    frame_labels = []
    
    for video_data in videos_data:
        video_path = video_data['path']
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        label = video_data['label']
        
        # Check if landmarks exist
        landmark_file = os.path.join(landmarks_path, f"{video_id}.json")
        if not os.path.exists(landmark_file):
            continue
        
        # Get cropped faces directory
        face_dir = os.path.join(crops_path, video_id)
        if not os.path.exists(face_dir):
            continue
        
        # Get all face images
        face_images = glob(os.path.join(face_dir, '*.png'))
        
        for face_img in face_images:
            frame_labels.append({
                'image_path': face_img,
                'label': label,
                'video_id': video_id
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(frame_labels)
    if len(df) > 0:
        csv_file = os.path.join(fold_dir, f'{split_name}_frame_labels.csv')
        df.to_csv(csv_file, index=False)
        print(f"Created {csv_file} with {len(df)} frames")
        return csv_file
    else:
        print(f"No frames found for {split_name} split in fold {fold_idx + 1}")
        return None


def main():
    """Main function to create k-fold splits"""
    print("="*60)
    print("K-Fold Cross-Validation Split Generator")
    print("="*60)
    print(f"K = 4")
    print(f"Split: 80% Train, 10% Validation, 10% Test")
    print("="*60)
    
    # Create k-fold splits
    folds = create_kfold_splits(k=4, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    # Save splits
    save_kfold_splits(folds)
    
    print("\n" + "="*60)
    print("K-Fold splits created successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Complete preprocessing (landmarks, crop faces)")
    print("2. Generate frame labels CSV for each fold:")
    print("   python kfold_cv.py --create_csvs")
    print("3. Train models for each fold:")
    print("   python train_kfold.py --fold 1")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation')
    parser.add_argument('--create_csvs', action='store_true', help='Create frame labels CSV for all folds')
    parser.add_argument('--fold', type=int, help='Create CSV for specific fold (1-4)')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], help='Create CSV for specific split')
    
    args = parser.parse_args()
    
    if args.create_csvs:
        print("Creating frame labels CSV for all folds...")
        for fold_idx in range(4):
            for split_name in ['train', 'val', 'test']:
                create_frame_labels_csv_for_fold(fold_idx, split_name)
    elif args.fold is not None:
        fold_idx = args.fold - 1  # Convert to 0-indexed
        if args.split:
            create_frame_labels_csv_for_fold(fold_idx, args.split)
        else:
            for split_name in ['train', 'val', 'test']:
                create_frame_labels_csv_for_fold(fold_idx, split_name)
    else:
        main()

