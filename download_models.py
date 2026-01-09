#!/usr/bin/env python3
"""
Script to download pre-trained models for MRI-GAN DeepFake detection
"""

import os
import gdown

def download_models():
    """Download all pre-trained model weights"""
    
    # Create weights directory if it doesn't exist
    os.makedirs('assets/weights', exist_ok=True)
    
    # Model URLs from the repository
    models = {
        'MRI_GAN_weights.chkpt': 'https://drive.google.com/uc?id=1qEfI96SYOWCumzPdQlcZJZvtAW_OXUcH',
        'DeepFake_plain_frames.pth': 'https://drive.google.com/uc?id=1_Pxv6ptxqXKtDJNkodkDmMTD_KRo08za',
        'DeepFake_MRI.pth': 'https://drive.google.com/uc?id=1xKzehNuq1B1th-_-U6OG9v2Q2Odws6VG'
    }
    
    for filename, url in models.items():
        filepath = f'assets/weights/{filename}'
        if os.path.exists(filepath):
            print(f"✓ {filename} already exists, skipping...")
            continue
            
        print(f"Downloading {filename}...")
        try:
            gdown.download(url, filepath, quiet=False)
            print(f"✓ Successfully downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
    
    print("\nModel download completed!")

if __name__ == "__main__":
    download_models()







