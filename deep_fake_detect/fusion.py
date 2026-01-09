"""
Enhanced Fusion Methods for Multi-Method DeepFake Detection
Includes weighted ensemble, probability calibration, and temporal smoothing
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict


class WeightedEnsembleFusion:
    """
    Weighted ensemble fusion for combining multiple detection methods.
    Supports adaptive weighting based on method confidence.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None, 
                 adaptive: bool = True, 
                 confidence_threshold: float = 0.7):
        """
        Args:
            weights: Dictionary mapping method names to weights (e.g., {'plain_frames': 0.4, 'MRI': 0.6})
                    If None, uses equal weights
            adaptive: If True, adjusts weights based on method confidence
            confidence_threshold: Threshold for high-confidence predictions
        """
        self.weights = weights if weights is not None else {}
        self.adaptive = adaptive
        self.confidence_threshold = confidence_threshold
    
    def fuse_probabilities(self, 
                           method_probs: Dict[str, np.ndarray],
                           method_confidences: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Fuse probabilities from multiple methods.
        
        Args:
            method_probs: Dictionary mapping method names to probability arrays
            method_confidences: Optional dictionary mapping method names to confidence arrays
        
        Returns:
            Fused probability array
        """
        if len(method_probs) == 0:
            raise ValueError("No method probabilities provided")
        
        method_names = list(method_probs.keys())
        probs_list = [method_probs[name] for name in method_names]
        
        # Normalize lengths
        min_len = min(len(p) for p in probs_list)
        probs_list = [p[:min_len] for p in probs_list]
        
        # Get weights
        if self.adaptive and method_confidences:
            weights = self._compute_adaptive_weights(method_names, method_confidences)
        else:
            weights = self._get_weights(method_names)
        
        # Weighted average
        weighted_sum = np.zeros(min_len)
        weight_sum = 0.0
        
        for i, method_name in enumerate(method_names):
            weight = weights.get(method_name, 1.0 / len(method_names))
            weighted_sum += weight * probs_list[i]
            weight_sum += weight
        
        fused_probs = weighted_sum / max(weight_sum, 1e-6)
        return fused_probs
    
    def _get_weights(self, method_names: List[str]) -> Dict[str, float]:
        """Get weights for methods"""
        if self.weights:
            # Normalize provided weights
            total = sum(self.weights.get(name, 0) for name in method_names)
            if total > 0:
                return {name: self.weights.get(name, 0) / total for name in method_names}
        
        # Equal weights
        weight = 1.0 / len(method_names)
        return {name: weight for name in method_names}
    
    def _compute_adaptive_weights(self, 
                                 method_names: List[str],
                                 method_confidences: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute adaptive weights based on method confidence"""
        weights = {}
        total_confidence = 0.0
        
        for method_name in method_names:
            if method_name in method_confidences:
                # Average confidence for this method
                conf = np.mean(method_confidences[method_name])
                # Higher confidence -> higher weight
                weights[method_name] = conf
                total_confidence += conf
            else:
                weights[method_name] = 0.5  # Default medium confidence
                total_confidence += 0.5
        
        # Normalize
        if total_confidence > 0:
            weights = {name: w / total_confidence for name, w in weights.items()}
        else:
            # Fallback to equal weights
            weight = 1.0 / len(method_names)
            weights = {name: weight for name in method_names}
        
        return weights


class TemperatureScaler(nn.Module):
    """
    Temperature scaling for probability calibration.
    Learns optimal temperature parameter to calibrate model outputs.
    """
    
    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature"""
        return logits / self.temperature
    
    def get_temperature(self) -> float:
        """Get current temperature value"""
        return float(self.temperature.item())


def train_temperature_scaler(model_logits: torch.Tensor, 
                            true_labels: torch.Tensor,
                            lr: float = 0.01,
                            max_iter: int = 50) -> TemperatureScaler:
    """
    Train temperature scaler on validation set.
    
    Args:
        model_logits: Model logits (N, 1) or (N,)
        true_labels: True binary labels (N, 1) or (N,)
        lr: Learning rate
        max_iter: Maximum iterations
    
    Returns:
        Trained TemperatureScaler
    """
    scaler = TemperatureScaler()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)
    
    def closure():
        optimizer.zero_grad()
        scaled_logits = scaler(model_logits)
        loss = criterion(scaled_logits, true_labels)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    return scaler


def smooth_probabilities_temporal(probs: np.ndarray, 
                                 window_size: int = 5,
                                 method: str = 'moving_average') -> np.ndarray:
    """
    Apply temporal smoothing to probability sequence.
    
    Args:
        probs: Probability array (T,)
        window_size: Smoothing window size
        method: Smoothing method ('moving_average', 'gaussian', 'exponential')
    
    Returns:
        Smoothed probability array
    """
    if window_size <= 1 or len(probs) == 0:
        return probs
    
    if method == 'moving_average':
        kernel = np.ones(window_size) / float(window_size)
        return np.convolve(probs, kernel, mode='same')
    
    elif method == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(probs, sigma=window_size / 3.0)
    
    elif method == 'exponential':
        # Exponential moving average
        alpha = 2.0 / (window_size + 1.0)
        smoothed = np.zeros_like(probs)
        smoothed[0] = probs[0]
        for i in range(1, len(probs)):
            smoothed[i] = alpha * probs[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def compute_fusion_metrics(fused_probs: np.ndarray,
                          true_labels: np.ndarray,
                          threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute evaluation metrics for fused predictions.
    
    Args:
        fused_probs: Fused probability array
        true_labels: True binary labels (0 or 1)
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    predictions = (fused_probs >= threshold).astype(int)
    
    tp = np.sum((predictions == 1) & (true_labels == 1))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

