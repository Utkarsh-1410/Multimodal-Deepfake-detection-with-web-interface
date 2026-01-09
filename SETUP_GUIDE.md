# Multi-Method DeepFake Detection with Temporal Analysis and Web Interface

## Setup Guide

## 🎉 Setup Complete!

The MRI-GAN DeepFake detection project has been successfully set up on your Windows system. All tests have passed and the environment is ready to use.

## 📁 Project Structure

```
mri_gan_deepfake/
├── assets/
│   └── weights/
│       ├── MRI_GAN_weights.chkpt (654.3 MB)
│       ├── DeepFake_plain_frames.pth (53.0 MB)
│       └── DeepFake_MRI.pth (53.0 MB)
├── data/
│   ├── dfdc/ (DFDC dataset directories)
│   ├── celeb_df_v2/ (Celeb-DF-v2 dataset directories)
│   ├── fdf/ (FDF dataset directories)
│   └── ffhq/ (FFHQ dataset directories)
├── logs/ (Training logs)
├── config_windows.yml (Windows-optimized configuration)
├── requirements_simple.txt (Simplified requirements)
├── download_models.py (Model download script)
├── test_setup.py (Setup verification script)
└── [Original project files...]
```

## 🔧 Environment Details

- **Python Version**: 3.12
- **PyTorch Version**: 2.8.0 (CPU only)
- **CUDA**: Not available (CPU mode)
- **Configuration**: Windows-optimized with reduced batch sizes

## 🚀 Quick Start

### 1. Test the Setup
```bash
python test_setup.py
```

### 2. Download Datasets (Optional)
To use the full functionality, you'll need to download datasets:
- **DFDC Dataset**: https://ai.facebook.com/datasets/dfdc/
- **Celeb-DF-v2**: https://github.com/yuezunli/celeb-deepfakeforensics
- **FFHQ Dataset**: https://github.com/NVlabs/ffhq-dataset
- **FDF Dataset**: https://github.com/hukkelas/FDF

### 3. Run Data Preprocessing
```bash
python data_preprocessing.py --extract_landmarks
python data_preprocessing.py --crop_faces
```

### 4. Train MRI-GAN
```bash
python train_MRI_GAN.py --train_from_scratch
```

### 5. Train DeepFake Detector
```bash
python deep_fake_detect.py --train_from_scratch
```

### 6. Test on Video
```bash
python deep_fake_detect_app.py --input_videofile <path_to_video> --method plain_frames
python deep_fake_detect_app.py --input_videofile <path_to_video> --method MRI
```

## 📋 Available Methods

### Plain Frames Method
- Uses raw video frames for detection
- Higher accuracy (91% on DFDC test set)
- Requires more computational resources

### MRI-based Method
- Uses MRI-GAN generated perceptual maps
- Lower accuracy (74% on DFDC test set)
- More efficient processing

## ⚙️ Configuration

The project uses `config_windows.yml` with Windows-optimized settings:
- **Batch Size**: 32 (reduced for CPU compatibility)
- **Image Size**: 256x256
- **Relative Paths**: All paths are relative to the project directory

## 🔍 Key Features

1. **MRI-GAN**: Generates perceptual dissimilarity maps
2. **DeepFake Detection**: Two detection methods available
3. **Pre-trained Models**: Ready-to-use model weights
4. **Data Processing**: Automated face detection and cropping
5. **Training Pipeline**: Complete training and evaluation scripts

## 📊 Model Performance

- **Plain Frames**: 91% accuracy on DFDC test set
- **MRI-based**: 74% accuracy on DFDC test set
- **MRI-GAN**: Generates perceptual maps highlighting synthetic regions

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Not Available**: The setup uses CPU mode. For GPU acceleration, install CUDA-compatible PyTorch.

2. **Memory Issues**: Reduce batch size in `config_windows.yml` if you encounter memory errors.

3. **Missing Dependencies**: Run `pip install -r requirements_simple.txt` to install missing packages.

4. **Path Issues**: Ensure all dataset paths in `config_windows.yml` are correct.

### Getting Help

- Check the original repository: https://github.com/pratikpv/mri_gan_deepfake
- Read the research paper: https://arxiv.org/abs/2203.00108
- Run `python test_setup.py` to verify your setup

## 📚 Research Citation

If you use this code in your research, please cite:

```bibtex
@misc{2203.00108,
Author = {Pratikkumar Prajapati and Chris Pollett},
Title = {MRI-GAN: A Generalized Approach to Detect DeepFakes using Perceptual Image Assessment},
Year = {2022},
Eprint = {arXiv:2203.00108},
}
```

## 🎯 Next Steps

1. **Download Datasets**: Get the required datasets for full functionality
2. **Experiment**: Try different hyperparameters and configurations
3. **Extend**: Modify the code for your specific use case
4. **Contribute**: Submit improvements back to the community

---

**Setup completed successfully!** 🎉







