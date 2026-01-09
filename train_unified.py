"""
Unified Training Script
Supports training all detection methods (plain_frames, MRI, temporal) with K-fold cross-validation
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deep_fake_detect.training import train_model
from deep_fake_detect.train_temporal import train_temporal
from train_kfold import train_fold, train_all_folds
from utils import ConfigParser, print_banner
from monitoring.logger import StructuredLogger


def train_plain_frames(log_dir: str = None, resume: bool = False):
    """Train plain frames detection model"""
    print_banner("Training Plain Frames Detection Model")
    
    logger = StructuredLogger(log_dir=log_dir or "logs")
    logger.log_event('training_started', {
        'method': 'plain_frames',
        'log_dir': log_dir
    })
    
    try:
        train_model(log_dir=log_dir, train_resume_checkpoint=None if not resume else "checkpoint.pth")
        logger.log_event('training_completed', {'method': 'plain_frames'})
        print("✓ Plain frames training completed")
    except Exception as e:
        logger.log_error('training_error', str(e), {'method': 'plain_frames'})
        raise


def train_mri_based(log_dir: str = None, resume: bool = False):
    """Train MRI-based detection model"""
    print_banner("Training MRI-Based Detection Model")
    
    logger = StructuredLogger(log_dir=log_dir or "logs")
    logger.log_event('training_started', {
        'method': 'MRI',
        'log_dir': log_dir
    })
    
    try:
        # MRI-based training uses same structure but different dataset
        # Update config to use MRI dataset
        train_model(log_dir=log_dir, train_resume_checkpoint=None if not resume else "checkpoint.pth")
        logger.log_event('training_completed', {'method': 'MRI'})
        print("✓ MRI-based training completed")
    except Exception as e:
        logger.log_error('training_error', str(e), {'method': 'MRI'})
        raise


def train_temporal_model(method: str = 'plain_frames',
                        per_frame_weights: str = 'assets/weights/DeepFake_plain_frames.pth',
                        output_path: str = 'assets/weights/temporal_head.pth',
                        epochs: int = 3,
                        max_frames: int = 64,
                        lr: float = 1e-3):
    """Train temporal analysis model"""
    print_banner(f"Training Temporal Model ({method})")
    
    logger = StructuredLogger(log_dir="logs")
    logger.log_event('training_started', {
        'method': 'temporal',
        'base_method': method,
        'per_frame_weights': per_frame_weights
    })
    
    try:
        # Create args object
        class Args:
            pass
        
        args = Args()
        args.method = method
        args.per_frame_weights = per_frame_weights
        args.out = output_path
        args.encoder = 'tf_efficientnet_b0_ns'
        args.imsize = 224
        args.epochs = epochs
        args.max_frames = max_frames
        args.lr = lr
        
        train_temporal(args)
        logger.log_event('training_completed', {'method': 'temporal'})
        print("✓ Temporal model training completed")
    except Exception as e:
        logger.log_error('training_error', str(e), {'method': 'temporal'})
        raise


def train_kfold_unified(method: str = 'plain_frames', 
                        fold: int = None,
                        all_folds: bool = False,
                        resume: bool = False):
    """Train with K-fold cross-validation"""
    print_banner(f"K-Fold Cross-Validation Training ({method})")
    
    logger = StructuredLogger(log_dir="logs")
    logger.log_event('kfold_training_started', {
        'method': method,
        'fold': fold,
        'all_folds': all_folds
    })
    
    try:
        if all_folds:
            train_all_folds(method=method, resume=resume)
        elif fold:
            train_fold(fold - 1, method=method, resume=resume)  # Convert to 0-indexed
        else:
            print("Please specify --fold <1-4> or --all")
            return
        
        logger.log_event('kfold_training_completed', {'method': method})
        print("✓ K-fold training completed")
    except Exception as e:
        logger.log_error('kfold_training_error', str(e), {'method': method})
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Unified Training Script for Multi-Method DeepFake Detection'
    )
    
    parser.add_argument('--method', 
                       choices=['plain_frames', 'MRI', 'temporal', 'kfold'],
                       required=True,
                       help='Training method')
    
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Log directory for training')
    
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    
    # K-fold specific
    parser.add_argument('--fold', type=int, choices=[1, 2, 3, 4],
                       help='Train specific fold (for kfold method)')
    parser.add_argument('--all_folds', action='store_true',
                       help='Train all folds (for kfold method)')
    
    # Temporal specific
    parser.add_argument('--per_frame_weights', type=str,
                       default='assets/weights/DeepFake_plain_frames.pth',
                       help='Path to pre-trained per-frame weights (for temporal)')
    parser.add_argument('--temporal_output', type=str,
                       default='assets/weights/temporal_head.pth',
                       help='Output path for temporal model')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs (for temporal)')
    parser.add_argument('--max_frames', type=int, default=64,
                       help='Maximum frames per video (for temporal)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (for temporal)')
    
    args = parser.parse_args()
    
    if args.method == 'plain_frames':
        train_plain_frames(log_dir=args.log_dir, resume=args.resume)
    
    elif args.method == 'MRI':
        train_mri_based(log_dir=args.log_dir, resume=args.resume)
    
    elif args.method == 'temporal':
        train_temporal_model(
            method='plain_frames',  # Can be extended to support MRI-based temporal
            per_frame_weights=args.per_frame_weights,
            output_path=args.temporal_output,
            epochs=args.epochs,
            max_frames=args.max_frames,
            lr=args.lr
        )
    
    elif args.method == 'kfold':
        if not args.fold and not args.all_folds:
            print("Error: Must specify --fold <1-4> or --all_folds for kfold method")
            parser.print_help()
            return
        
        # Determine base method for kfold
        base_method = 'plain_frames'  # Default, can be extended
        train_kfold_unified(
            method=base_method,
            fold=args.fold,
            all_folds=args.all_folds,
            resume=args.resume
        )
    
    else:
        print(f"Unknown method: {args.method}")
        parser.print_help()


if __name__ == '__main__':
    main()

