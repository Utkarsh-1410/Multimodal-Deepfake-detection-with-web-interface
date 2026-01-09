"""
Enhanced Probability Calibration using Temperature Scaling
Trains optimal temperature parameter on validation set for better calibrated probabilities
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve


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
    
    def calibrate_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Calibrate probabilities using temperature"""
        scaled_logits = self.forward(logits)
        return torch.sigmoid(scaled_logits)


def train_temperature_scaler(model_logits: torch.Tensor, 
                            true_labels: torch.Tensor,
                            lr: float = 0.01,
                            max_iter: int = 50,
                            device: Optional[torch.device] = None) -> TemperatureScaler:
    """
    Train temperature scaler on validation set.
    
    Args:
        model_logits: Model logits (N, 1) or (N,)
        true_labels: True binary labels (N, 1) or (N,)
        lr: Learning rate
        max_iter: Maximum iterations
        device: Device to use
    
    Returns:
        Trained TemperatureScaler
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure proper shapes
    if model_logits.dim() == 1:
        model_logits = model_logits.unsqueeze(1)
    if true_labels.dim() == 1:
        true_labels = true_labels.unsqueeze(1)
    
    model_logits = model_logits.to(device)
    true_labels = true_labels.to(device).float()
    
    scaler = TemperatureScaler().to(device)
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


class IsotonicCalibrator:
    """
    Isotonic regression for probability calibration.
    Non-parametric method that can handle any monotonic transformation.
    """
    
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
    
    def fit(self, probabilities: np.ndarray, true_labels: np.ndarray):
        """
        Fit isotonic calibrator.
        
        Args:
            probabilities: Predicted probabilities (N,)
            true_labels: True binary labels (N,)
        """
        self.calibrator.fit(probabilities, true_labels)
        self.is_fitted = True
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Transform probabilities using fitted calibrator"""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        return self.calibrator.transform(probabilities)
    
    def fit_transform(self, probabilities: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(probabilities, true_labels)
        return self.transform(probabilities)


def evaluate_calibration(probabilities: np.ndarray, 
                       true_labels: np.ndarray,
                       n_bins: int = 10) -> dict:
    """
    Evaluate calibration quality using ECE (Expected Calibration Error).
    
    Args:
        probabilities: Predicted probabilities (N,)
        true_labels: True binary labels (N,)
        n_bins: Number of bins for calibration curve
    
    Returns:
        Dictionary with calibration metrics
    """
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        true_labels, probabilities, n_bins=n_bins, strategy='uniform'
    )
    
    # Compute ECE (Expected Calibration Error)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    bin_counts = []
    bin_accuracies = []
    bin_confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = true_labels[in_bin].mean()
            avg_confidence_in_bin = probabilities[in_bin].mean()
            
            bin_counts.append(prop_in_bin)
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    # Maximum Calibration Error (MCE)
    mce = max([abs(acc - conf) for acc, conf in zip(bin_accuracies, bin_confidences)]) if bin_accuracies else 0.0
    
    return {
        'ece': float(ece),
        'mce': float(mce),
        'calibration_curve': {
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist()
        },
        'bin_counts': bin_counts,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences
    }


def calibrate_probabilities(model_logits: torch.Tensor,
                           true_labels: torch.Tensor,
                           method: str = 'temperature',
                           validation_logits: Optional[torch.Tensor] = None,
                           validation_labels: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, dict]:
    """
    Calibrate probabilities using specified method.
    
    Args:
        model_logits: Model logits for calibration (N,)
        true_labels: True labels for calibration (N,)
        method: Calibration method ('temperature' or 'isotonic')
        validation_logits: Optional validation logits for temperature scaling
        validation_labels: Optional validation labels for temperature scaling
    
    Returns:
        Tuple of (calibrated_probabilities, calibration_metrics)
    """
    if method == 'temperature':
        # Use validation set if provided, otherwise use training set
        train_logits = validation_logits if validation_logits is not None else model_logits
        train_labels = validation_labels if validation_labels is not None else true_labels
        
        # Train temperature scaler
        scaler = train_temperature_scaler(train_logits, train_labels)
        
        # Calibrate probabilities
        calibrated_probs = scaler.calibrate_probs(model_logits).cpu().numpy().flatten()
        
        # Evaluate calibration
        metrics = evaluate_calibration(calibrated_probs, true_labels.cpu().numpy().flatten())
        metrics['temperature'] = scaler.get_temperature()
        
        return calibrated_probs, metrics
    
    elif method == 'isotonic':
        # Convert to numpy
        probs = torch.sigmoid(model_logits).cpu().numpy().flatten()
        labels = true_labels.cpu().numpy().flatten()
        
        # Fit isotonic calibrator
        calibrator = IsotonicCalibrator()
        calibrated_probs = calibrator.fit_transform(probs, labels)
        
        # Evaluate calibration
        metrics = evaluate_calibration(calibrated_probs, labels)
        
        return calibrated_probs, metrics
    
    else:
        raise ValueError(f"Unknown calibration method: {method}")

