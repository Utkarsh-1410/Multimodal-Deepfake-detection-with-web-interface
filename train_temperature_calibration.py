"""
Train Temperature Scaling for Probability Calibration
Improves probability calibration on validation set
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deep_fake_detect.DeepFakeDetectModel import DeepFakeDetectModel
from deep_fake_detect.fusion import TemperatureScaler, train_temperature_scaler
from deep_fake_detect.datasets import DFDCDatasetSimple
from utils import ConfigParser
import torchvision


def collect_logits_and_labels(model, dataloader, device):
    """Collect model logits and true labels from dataloader"""
    all_logits = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting logits"):
            frames = batch[0].to(device)
            labels = batch[1].to(device)
            
            logits = model(frames)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_logits, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train Temperature Scaling for Calibration')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='assets/weights/temperature_scaler.pth',
                       help='Output path for temperature scaler')
    parser.add_argument('--dataset', type=str, choices=['plain_frames', 'mri'], 
                       default='plain_frames', help='Dataset type')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--max_iter', type=int, default=50, help='Max iterations')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model_params = ConfigParser.getInstance().get_deep_fake_training_params()
    encoder_name = ConfigParser.getInstance().get_default_cnn_encoder_name()
    imsize = 224  # Default
    
    model = DeepFakeDetectModel(frame_dim=imsize, encoder_name=encoder_name)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Prepare validation dataset
    print("Preparing validation dataset...")
    valid_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((imsize, imsize)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
    ])
    
    valid_dataset = DFDCDatasetSimple(
        mode='valid',
        transform=valid_transform,
        data_size=ConfigParser.getInstance().get_valid_sample_size(),
        dataset=args.dataset
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Collect logits and labels
    print("Collecting model predictions...")
    logits, labels = collect_logits_and_labels(model, valid_loader, device)
    
    print(f"Collected {len(logits)} samples")
    print(f"Label distribution: {labels.sum().item()} fake, {len(labels) - labels.sum().item()} real")
    
    # Train temperature scaler
    print("Training temperature scaler...")
    scaler = train_temperature_scaler(
        logits,
        labels,
        lr=args.lr,
        max_iter=args.max_iter
    )
    
    trained_temp = scaler.get_temperature()
    print(f"Trained temperature: {trained_temp:.4f}")
    
    # Save scaler
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'temperature': trained_temp,
        'scaler_state_dict': scaler.state_dict(),
        'model_path': args.model_path,
        'dataset': args.dataset
    }, args.output)
    
    print(f"Temperature scaler saved to {args.output}")
    
    # Evaluate calibration improvement
    print("\nEvaluating calibration...")
    with torch.no_grad():
        # Without temperature scaling
        probs_no_temp = torch.sigmoid(logits).numpy()
        
        # With temperature scaling
        scaled_logits = scaler(logits)
        probs_with_temp = torch.sigmoid(scaled_logits).numpy()
    
    # Compute ECE (Expected Calibration Error) - simplified
    def compute_ece(probs, labels, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    ece_no_temp = compute_ece(probs_no_temp.flatten(), labels.numpy().flatten())
    ece_with_temp = compute_ece(probs_with_temp.flatten(), labels.numpy().flatten())
    
    print(f"ECE without temperature scaling: {ece_no_temp:.4f}")
    print(f"ECE with temperature scaling: {ece_with_temp:.4f}")
    print(f"Improvement: {((ece_no_temp - ece_with_temp) / ece_no_temp * 100):.2f}%")


if __name__ == '__main__':
    main()

