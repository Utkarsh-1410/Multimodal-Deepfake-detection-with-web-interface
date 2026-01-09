"""
Comprehensive Evaluation Metrics for DeepFake Detection
Includes accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class DetectionEvaluator:
    """Comprehensive evaluator for deepfake detection models"""
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Classification threshold for binary predictions
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and labels"""
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
    
    def update(self, predictions: np.ndarray, labels: np.ndarray, probabilities: Optional[np.ndarray] = None):
        """
        Update evaluator with new predictions.
        
        Args:
            predictions: Binary predictions (0 or 1)
            labels: True labels (0 or 1)
            probabilities: Prediction probabilities (optional, for ROC/PR curves)
        """
        self.all_predictions.extend(predictions.flatten())
        self.all_labels.extend(labels.flatten())
        if probabilities is not None:
            self.all_probabilities.extend(probabilities.flatten())
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Returns:
            Dictionary of metrics
        """
        if len(self.all_predictions) == 0:
            return {}
        
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        
        # Binary classification metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Additional metrics
        specificity = tn / max(tn + fp, 1)  # True Negative Rate
        sensitivity = recall  # True Positive Rate (same as recall)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        # ROC-AUC and PR-AUC if probabilities available
        if len(self.all_probabilities) > 0 and len(self.all_probabilities) == len(self.all_labels):
            try:
                probs = np.array(self.all_probabilities)
                roc_auc = roc_auc_score(labels, probs)
                pr_auc = average_precision_score(labels, probs)
                metrics['roc_auc'] = float(roc_auc)
                metrics['pr_auc'] = float(pr_auc)
            except Exception as e:
                print(f"Warning: Could not compute AUC metrics: {e}")
        
        return metrics
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None, title: str = "Confusion Matrix"):
        """Plot confusion matrix"""
        if len(self.all_predictions) == 0:
            return
        
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_roc_curve(self, save_path: Optional[str] = None, title: str = "ROC Curve"):
        """Plot ROC curve"""
        if len(self.all_probabilities) == 0:
            return
        
        fpr, tpr, _ = roc_curve(self.all_labels, self.all_probabilities)
        roc_auc = roc_auc_score(self.all_labels, self.all_probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_pr_curve(self, save_path: Optional[str] = None, title: str = "Precision-Recall Curve"):
        """Plot Precision-Recall curve"""
        if len(self.all_probabilities) == 0:
            return
        
        precision, recall, _ = precision_recall_curve(self.all_labels, self.all_probabilities)
        pr_auc = average_precision_score(self.all_labels, self.all_probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def generate_report(self, output_dir: str, method_name: str = "Detection"):
        """
        Generate comprehensive evaluation report with all plots and metrics.
        
        Args:
            output_dir: Directory to save report
            method_name: Name of detection method
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Compute metrics
        metrics = self.compute_metrics()
        
        # Save metrics to file
        metrics_file = output_path / f"{method_name}_metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write(f"Evaluation Report: {method_name}\n")
            f.write("=" * 50 + "\n\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        # Generate plots
        self.plot_confusion_matrix(
            save_path=str(output_path / f"{method_name}_confusion_matrix.png"),
            title=f"Confusion Matrix - {method_name}"
        )
        
        if len(self.all_probabilities) > 0:
            self.plot_roc_curve(
                save_path=str(output_path / f"{method_name}_roc_curve.png"),
                title=f"ROC Curve - {method_name}"
            )
            
            self.plot_pr_curve(
                save_path=str(output_path / f"{method_name}_pr_curve.png"),
                title=f"Precision-Recall Curve - {method_name}"
            )
        
        print(f"Evaluation report saved to {output_dir}")
        return metrics


def compare_methods(method_results: Dict[str, Dict[str, float]], 
                   output_path: Optional[str] = None) -> None:
    """
    Compare multiple detection methods side-by-side.
    
    Args:
        method_results: Dictionary mapping method names to their metrics
        output_path: Optional path to save comparison plot
    """
    methods = list(method_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Extract metric values
    data = {metric: [method_results[method].get(metric, 0) for method in methods] 
            for metric in metrics}
    
    # Create bar plot
    x = np.arange(len(methods))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Score')
    ax.set_title('Method Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

