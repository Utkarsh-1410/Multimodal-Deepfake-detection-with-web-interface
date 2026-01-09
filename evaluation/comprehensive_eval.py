"""
Comprehensive Evaluation Pipeline
Evaluates all detection methods (plain_frames, MRI, fusion, temporal) with full metrics
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime

from deep_fake_detect.evaluation import DetectionEvaluator, compare_methods
from deep_fake_detect_app import predict_deepfake
from monitoring.logger import StructuredLogger


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator that tests all detection methods and generates reports.
    """
    
    def __init__(self, 
                 output_dir: str = "evaluation_results",
                 logger: Optional[StructuredLogger] = None):
        """
        Args:
            output_dir: Directory to save evaluation results
            logger: Optional structured logger
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        
        self.methods = ['plain_frames', 'MRI', 'fusion', 'temporal']
        self.evaluators = {method: DetectionEvaluator() for method in self.methods}
    
    def evaluate_video(self, 
                      video_path: str,
                      ground_truth: int,
                      methods: Optional[List[str]] = None,
                      temperature: float = 1.0,
                      temporal_window: int = 5) -> Dict[str, Dict]:
        """
        Evaluate a single video with all methods.
        
        Args:
            video_path: Path to video file
            ground_truth: Ground truth label (0=real, 1=fake)
            methods: List of methods to evaluate (None = all)
            temperature: Temperature for probability calibration
            temporal_window: Temporal smoothing window
        
        Returns:
            Dictionary mapping method names to results
        """
        if methods is None:
            methods = self.methods
        
        results = {}
        
        for method in methods:
            try:
                # Run detection
                fake_prob, real_prob, pred = predict_deepfake(
                    input_videofile=video_path,
                    df_method=method,
                    debug=False,
                    verbose=False,
                    temperature=temperature,
                    overwrite_faces=False,  # Don't overwrite if already processed
                    overwrite_mri=False,
                    json_out=False,
                    temporal_window=temporal_window,
                    return_probs=False
                )
                
                if pred is not None:
                    # Update evaluator
                    predictions = np.array([pred])
                    labels = np.array([ground_truth])
                    probabilities = np.array([fake_prob])
                    
                    self.evaluators[method].update(predictions, labels, probabilities)
                    
                    results[method] = {
                        'prediction': int(pred),
                        'fake_probability': float(fake_prob),
                        'real_probability': float(real_prob),
                        'ground_truth': int(ground_truth),
                        'correct': int(pred == ground_truth)
                    }
                else:
                    results[method] = {
                        'error': 'No faces detected'
                    }
            
            except Exception as e:
                results[method] = {
                    'error': str(e)
                }
                
                if self.logger:
                    self.logger.log_error(
                        'evaluation_error',
                        f"Error evaluating {method} on {video_path}",
                        {'error': str(e), 'method': method, 'video': video_path}
                    )
        
        return results
    
    def evaluate_dataset(self,
                        video_list: List[str],
                        ground_truths: Dict[str, int],
                        methods: Optional[List[str]] = None,
                        temperature: float = 1.0,
                        temporal_window: int = 5,
                        batch_size: int = 10) -> Dict[str, Dict]:
        """
        Evaluate a dataset of videos.
        
        Args:
            video_list: List of video file paths
            ground_truths: Dictionary mapping video paths to ground truth labels
            methods: List of methods to evaluate
            temperature: Temperature for probability calibration
            temporal_window: Temporal smoothing window
            batch_size: Batch size for processing
        
        Returns:
            Dictionary with evaluation results for each method
        """
        if methods is None:
            methods = self.methods
        
        # Reset evaluators
        for method in methods:
            self.evaluators[method].reset()
        
        # Process videos
        total = len(video_list)
        processed = 0
        
        for i, video_path in enumerate(video_list):
            if video_path not in ground_truths:
                continue
            
            ground_truth = ground_truths[video_path]
            self.evaluate_video(video_path, ground_truth, methods, temperature, temporal_window)
            
            processed += 1
            if (i + 1) % batch_size == 0:
                print(f"Processed {i + 1}/{total} videos")
        
        # Compute metrics for each method
        all_results = {}
        
        for method in methods:
            metrics = self.evaluators[method].compute_metrics()
            
            # Generate plots
            method_dir = self.output_dir / method
            method_dir.mkdir(exist_ok=True)
            
            self.evaluators[method].generate_report(
                str(method_dir),
                method_name=method
            )
            
            all_results[method] = metrics
        
        # Compare methods
        compare_methods(
            all_results,
            output_path=str(self.output_dir / "method_comparison.png")
        )
        
        # Save summary
        self.save_summary(all_results)
        
        return all_results
    
    def save_summary(self, results: Dict[str, Dict]):
        """Save evaluation summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'methods': results,
            'best_method': max(results.items(), key=lambda x: x[1].get('f1_score', 0))[0] if results else None
        }
        
        summary_file = self.output_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save as CSV for easy viewing
        metrics_df = pd.DataFrame(results).T
        metrics_df.to_csv(self.output_dir / "metrics_comparison.csv")
        
        print(f"Evaluation summary saved to {summary_file}")
    
    def evaluate_kfold_results(self, kfold_dir: str = "logs") -> Dict[str, Dict]:
        """
        Evaluate results from K-fold cross-validation.
        
        Args:
            kfold_dir: Directory containing K-fold training logs
        
        Returns:
            Dictionary with aggregated K-fold results
        """
        kfold_path = Path(kfold_dir)
        if not kfold_path.exists():
            print(f"K-fold directory not found: {kfold_dir}")
            return {}
        
        # Find all fold directories
        fold_dirs = [d for d in kfold_path.iterdir() if d.is_dir() and d.name.startswith('fold_')]
        
        if not fold_dirs:
            print("No fold directories found")
            return {}
        
        fold_results = {}
        
        for fold_dir in sorted(fold_dirs):
            fold_num = fold_dir.name
            fold_results[fold_num] = {}
            
            # Look for evaluation results in each fold
            for method in self.methods:
                method_dir = fold_dir / method
                if method_dir.exists():
                    metrics_file = method_dir / f"{method}_metrics.txt"
                    if metrics_file.exists():
                        # Parse metrics (simplified - would need proper parsing)
                        fold_results[fold_num][method] = "found"
        
        # Aggregate across folds
        aggregated = {}
        for method in self.methods:
            method_metrics = []
            for fold_num, fold_data in fold_results.items():
                if method in fold_data:
                    method_metrics.append(fold_data[method])
            
            if method_metrics:
                aggregated[method] = {
                    'folds_evaluated': len(method_metrics),
                    'status': 'completed' if len(method_metrics) == len(fold_results) else 'partial'
                }
        
        return {
            'fold_results': fold_results,
            'aggregated': aggregated
        }

