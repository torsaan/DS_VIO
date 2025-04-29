# Violence Detection System

A deep learning-based system for detecting violence in video content using multiple state-of-the-art architectures.

## Overview

This repository contains a comprehensive implementation of several advanced deep learning models for violence detection in videos. The system can analyze videos and classify them as containing violent or non-violent content, supporting multiple architectures with a unified training and evaluation pipeline.

## Features

- **Multiple Model Architectures**:
  - 3D CNN (using R3D-18)
  - 2D CNN + LSTM (ResNet50 backbone)
  - Video Transformer (ViT backbone)
  - SlowFast Network
  - Two-Stream Network (RGB + Optical Flow)

- **Complete Pipeline**:
  - Data preprocessing and standardization
  - Flexible data loading with augmentation
  - Training with early stopping and checkpointing
  - Evaluation with comprehensive metrics
  - Model ensemble capabilities
  - Hyperparameter search functionality

- **Performance Optimizations**:
  - Memory-efficient training
  - CUDA optimizations
  - Mixed precision training
  - Gradient checkpointing

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.20.0
opencv-python>=4.5.0
Pillow>=8.0.0
tqdm>=4.62.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
flask>=2.0.0
pathlib>=1.0.0
timm>=0.5.0
```

## Project Structure

```
├── Data/                  # Data directory structure
│   └── VioNonVio/         # Raw videos
│       ├── Violence/      # Videos containing violence
│       └── NonViolence/   # Videos without violence
├── Models/                # Model implementations
│   ├── __init__.py
│   ├── model_3dcnn.py     # 3D CNN model
│   ├── model_2dcnn_lstm.py # 2D CNN + LSTM model
│   ├── model_transformer.py # Video Transformer model
│   ├── model_slowfast.py   # SlowFast Network model
│   └── model_two_stream.py # Two-Stream Network model
├── utils/                 # Utility functions
│   ├── augmentor.py       # Video augmentation
│   ├── dataprep.py        # Data preparation utilities
│   ├── logger.py          # Logging utilities
│   ├── model_utils.py     # Model utility functions
│   ├── precompute_optical_flow.py # Optical flow computation
│   └── video_standardizer.py # Video standardization utilities
├── dataloader.py          # Dataset and dataloader implementations
├── evaluate_best_models.py # Evaluation script for trained models
├── evaluate_ensemble.py   # Ensemble evaluation script
├── evaluations.py         # Evaluation metrics and visualization
├── hyperparameter_search.py # Hyperparameter optimization
├── hyperparameters.py     # Hyperparameter configurations
├── main.py                # Main training script
├── setup_project.py       # Project setup script
└── train.py               # Training utilities
```

## Setup and Usage

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/violence-detection.git
cd violence-detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Standardize videos to consistent resolution, fps, and duration
python setup_project.py --data_dir ./Data/VioNonVio --output_dir ./Data/Processed/standardized
```

Optionally, you can precompute optical flow for the Two-Stream Network:

```bash
python -m utils.precompute_optical_flow --video_dir ./Data/Processed/standardized --output_dir ./Data/Processed/flow
```

### 3. Training

```bash
# Train a single model
python main.py --data_dir ./Data/Processed/standardized --model_types 3d_cnn --batch_size 8

# Train multiple models
python main.py --data_dir ./Data/Processed/standardized --model_types 3d_cnn 2d_cnn_lstm transformer --batch_size 8

# Train with hyperparameter search
python main.py --data_dir ./Data/Processed/standardized --model_types 3d_cnn --hp_search
```

### 4. Evaluation

```bash
# Evaluate best models
python evaluate_best_models.py --data_dir ./Data/Processed/standardized --output_dir ./output

# Evaluate ensemble
python evaluate_ensemble.py --data_dir ./Data/Processed/standardized --output_dir ./output/ensemble
```

## Model Details

### 3D CNN

Uses the R3D-18 architecture with 3D convolutions to directly capture spatial and temporal information.

- Input: Video frames [B, C, T, H, W]
- Temporal modeling: Implicit through 3D convolutions
- Base network: R3D-18 (3D ResNet)

### 2D CNN + LSTM

Uses a pretrained ResNet50 to extract spatial features, then an LSTM for temporal modeling.

- Input: Video frames [B, T, C, H, W]
- Spatial feature extractor: ResNet50
- Temporal modeling: Bidirectional LSTM

### Video Transformer

Uses a Vision Transformer (ViT) backbone with temporal position encoding and transformer encoder.

- Input: Video frames [B, T, C, H, W]
- Spatial feature extractor: Vision Transformer (ViT)
- Temporal modeling: Transformer encoder with positional encoding

### SlowFast Network

Dual-pathway architecture with a slow pathway for spatial semantics and a fast pathway for motion.

- Input: Video frames [B, C, T, H, W]
- Slow pathway: Processes frames at low frame rate for spatial features
- Fast pathway: Processes frames at high frame rate for temporal features
- Fusion types: early, middle, or late fusion

### Two-Stream Network

Separate networks for RGB frames and optical flow, fused for the final prediction.

- Input: RGB frames [B, C, T, H, W] and optical flow [B, 2, T-1, H, W]
- Spatial stream: Processes RGB frames for appearance information
- Temporal stream: Processes optical flow for motion information
- Fusion: late (score fusion) or conv (feature fusion)

## Performance

Our best performing models achieve the following accuracy on the test set:

| Model            | Accuracy (%) | ROC AUC | PR AUC |
|------------------|:------------:|:-------:|:------:|
| 3D CNN           | 96.67        | 0.9896  | 0.9903 |
| 2D CNN + LSTM    | 95.14        | 0.9875  | 0.9874 |
| Video Transformer| 96.93        | 0.9907  | 0.9908 |
| SlowFast Network | 72.38        | 0.8329  | 0.8472 |
| Two-Stream Network| 71.10       | 0.8210  | 0.8370 |
| Ensemble         | 97.44        | 0.9925  | 0.9930 |

## Acknowledgements

This project uses the following libraries and resources:
- PyTorch and TorchVision
- timm (PyTorch Image Models)
- scikit-learn for evaluation metrics

