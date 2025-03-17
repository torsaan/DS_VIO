#!/usr/bin/env python3
# Testing/test_training.py
"""
Test script to verify the training pipeline works correctly.
Tests a single training epoch with a small dataset.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from pathlib import Path
import json

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.dataprep import prepare_violence_nonviolence_data
from dataloader import get_dataloaders
from train import train_model, train_epoch, validate, clear_cuda_memory
import hyperparameters as hp
from Models.model_3dcnn import Model3DCNN
from Models.model_simplecnn import SimpleCNN

def setup_device(gpu_id):
    """Set up computation device (CPU or GPU)"""
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        
        # Print GPU memory info
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9  # GB
        allocated_memory = torch.cuda.memory_allocated(device) / 1e9  # GB
        reserved_memory = torch.cuda.memory_reserved(device) / 1e9  # GB
        
        print(f"GPU Memory: Total {total_memory:.2f} GB, Allocated {allocated_memory:.2f} GB, Reserved {reserved_memory:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device

def initialize_model(model_type, device, use_pose=False):
    """Initialize model based on model type"""
    # Import the configuration function
    from hyperparameters import get_model_config
    
    # Get model configuration with any overrides
    config = get_model_config(model_type, use_pose=use_pose)
    
    # Initialize the appropriate model
    if model_type == '3d_cnn':
        from Models.model_3dcnn import Model3DCNN
        model = Model3DCNN(**config).to(device)
    elif model_type == 'simple_cnn':
        from Models.model_simplecnn import SimpleCNN
        model = SimpleCNN(**config).to(device)
    elif model_type == 'temporal_3d_cnn':
        from Models.model_Temporal3DCNN import Temporal3DCNN
        model = Temporal3DCNN(**config).to(device)
    elif model_type == '2d_cnn_lstm':
        from Models.model_2dcnn_lstm import Model2DCNNLSTM
        model = Model2DCNNLSTM(**config).to(device)
    else:
        raise ValueError(f"Unsupported model type for this test: {model_type}")
    
    return model

def test_training_epoch(model, train_loader, device, grad_clip=None):
    """Test a single training epoch"""
    print("\n" + "="*50)
    print("Testing Training Epoch")
    print("="*50)
    
    # Set up optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Time the epoch
    start_time = time.time()
    
    # Train for one epoch
    epoch_loss, epoch_acc = train_epoch(
        model, train_loader, optimizer, criterion, device, grad_clip=grad_clip
    )
    
    duration = time.time() - start_time
    
    print(f"Epoch completed in {duration:.2f} seconds")
    print(f"Training Loss: {epoch_loss:.4f}")
    print(f"Training Accuracy: {epoch_acc:.2f}%")
    
    # Check if loss and accuracy are reasonable
    if np.isnan(epoch_loss):
        print("Error: Training loss is NaN")
        return False
    
    return True

def test_validation(model, val_loader, device):
    """Test validation routine"""
    print("\n" + "="*50)
    print("Testing Validation")
    print("="*50)
    
    # Set up criterion
    criterion = nn.CrossEntropyLoss()
    
    # Time the validation
    start_time = time.time()
    
    # Run validation
    metrics, all_preds, all_targets, all_probs = validate(
        model, val_loader, criterion, device
    )
    
    duration = time.time() - start_time
    
    print(f"Validation completed in {duration:.2f} seconds")
    print(f"Validation Loss: {metrics['val_loss']:.4f}")
    print(f"Validation Accuracy: {metrics['val_acc']:.2f}%")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    
    # Check if loss and accuracy are reasonable
    if np.isnan(metrics['val_loss']):
        print("Error: Validation loss is NaN")
        return False
    
    return True

def test_full_training(model_type, data_dir, output_dir, device, epochs=2, batch_size=4, use_pose=False):
    """Test full training pipeline with a small dataset"""
    print("\n" + "="*50)
    print(f"Testing Full Training Pipeline with {model_type}")
    print("="*50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a small subset of data
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        prepare_violence_nonviolence_data(data_dir)
    
    # Use only a small subset for testing
    train_subset_paths = train_paths[:10]
    train_subset_labels = train_labels[:10]
    val_subset_paths = val_paths[:5]
    val_subset_labels = val_labels[:5]
    
    # Create dataloaders
    pose_dir = None  # Set to an actual path if testing with pose data
    train_loader, val_loader, _ = get_dataloaders(
        train_subset_paths, train_subset_labels,
        val_subset_paths, val_subset_labels,
        val_subset_paths, val_subset_labels,  # Reuse val for test
        pose_dir=pose_dir,
        batch_size=batch_size,
        num_workers=2,
        model_type=model_type
    )
    
    # Initialize model
    model = initialize_model(model_type, device, use_pose)
    
    # Train model
    try:
        trained_model = train_model(
            model_name=model_type,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            device=device,
            output_dir=output_dir,
            patience=3  # Low patience for testing
        )
        
        # Check if model was returned
        if trained_model is None:
            print("Error: train_model did not return a model")
            return False
        
        # Check if training log was created
        log_path = os.path.join(output_dir, model_type, "training_log.csv")
        if not os.path.exists(log_path):
            print(f"Warning: Training log not found at {log_path}")
        else:
            print(f"Training log found at {log_path}")
        
        # Check if model checkpoint was created
        checkpoint_path = os.path.join(output_dir, model_type, f"last_{model_type}_model.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Model checkpoint not found at {checkpoint_path}")
        else:
            print(f"Model checkpoint found at {checkpoint_path}")
        
        return True
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_hyperparameter_loading():
    """Test hyperparameter loading functionality"""
    print("\n" + "="*50)
    print("Testing Hyperparameter Loading")
    print("="*50)
    
    try:
        # Test getting model config
        from hyperparameters import get_model_config, get_training_config, get_optimizer
        
        # Test model configs
        model_types = ['3d_cnn', '2d_cnn_lstm', 'transformer', 'i3d', 
                       'simple_cnn', 'temporal_3d_cnn', 'slowfast', 
                       'r2plus1d', 'two_stream', 'cnn_lstm']
        
        for model_type in model_types:
            # Get default config
            config = get_model_config(model_type)
            print(f"{model_type} config: {config}")
            
            # Get config with overrides
            config_override = get_model_config(model_type, num_classes=3, dropout_prob=0.7)
            print(f"{model_type} config with overrides: {config_override}")
            
            assert config_override['num_classes'] == 3, "Override didn't work for num_classes"
            if 'dropout_prob' in config_override:
                assert config_override['dropout_prob'] == 0.7, "Override didn't work for dropout_prob"
            
            # Get training config
            train_config = get_training_config(model_type)
            print(f"{model_type} training config: {train_config}")
        
        # Test optimizer with a dummy model
        dummy_model = nn.Sequential(nn.Linear(10, 10))
        
        # Test different optimizers
        adam = get_optimizer(dummy_model, optimizer_name='adam')
        sgd = get_optimizer(dummy_model, optimizer_name='sgd')
        adamw = get_optimizer(dummy_model, optimizer_name='adamw')
        
        print(f"Adam optimizer: {type(adam).__name__}")
        print(f"SGD optimizer: {type(sgd).__name__}")
        print(f"AdamW optimizer: {type(adamw).__name__}")
        
        # Test with specific learning rate
        custom_lr = get_optimizer(dummy_model, lr=0.0123)
        print(f"Custom LR: {custom_lr.param_groups[0]['lr']}")
        assert custom_lr.param_groups[0]['lr'] == 0.0123, "Custom learning rate wasn't applied"
        
        print("All hyperparameter loading tests passed")
        return True
    except Exception as e:
        print(f"Error testing hyperparameters: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test training pipeline")
    parser.add_argument("--data_dir", type=str, default="./Data/Processed/standardized",
                      help="Directory containing standardized videos")
    parser.add_argument("--output_dir", type=str, default="./Testing/test_output",
                      help="Directory to save test output")
    parser.add_argument("--model_type", type=str, default="simple_cnn",
                      choices=['3d_cnn', 'simple_cnn', 'temporal_3d_cnn', '2d_cnn_lstm'],
                      help="Model type to test")
    parser.add_argument("--epochs", type=int, default=2,
                      help="Number of epochs for full training test")
    parser.add_argument("--batch_size", type=int, default=2,
                      help="Batch size for training")
    parser.add_argument("--gpu", type=int, default=0,
                      help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--use_pose", action="store_true",
                      help="Test with pose data")
    parser.add_argument("--skip_hyperparams", action="store_true",
                      help="Skip testing hyperparameter loading")
    parser.add_argument("--skip_training", action="store_true",
                      help="Skip testing full training pipeline")
    args = parser.parse_args()
    
    # Set up device
    device = setup_device(args.gpu)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test hyperparameter loading
    hyperparams_ok = True
    if not args.skip_hyperparams:
        hyperparams_ok = test_hyperparameter_loading()
    
    # Test training pipeline
    training_ok = True
    if not args.skip_training:
        training_ok = test_full_training(
            args.model_type, 
            args.data_dir, 
            args.output_dir, 
            device, 
            args.epochs,
            args.batch_size,
            args.use_pose
        )
    
    # Print summary
    print("\n" + "="*50)
    print("Training Pipeline Tests Results")
    print("="*50)
    
    if not args.skip_hyperparams:
        print(f"Hyperparameter Loading: {'✓ PASSED' if hyperparams_ok else '✗ FAILED'}")
        
    if not args.skip_training:
        print(f"Training Pipeline: {'✓ PASSED' if training_ok else '✗ FAILED'}")
    
    # Return success only if all tests pass
    if hyperparams_ok and training_ok:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())