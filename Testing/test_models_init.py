#!/usr/bin/env python3
# Testing/test_models_init.py
"""
Test script to verify model initialization and basic operations.
Tests each model's initialization, forward pass, and backward pass individually.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.model_utils import create_fake_batch, print_tensor_shape, print_model_summary

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

def clear_cuda_memory():
    """Clear CUDA memory between tests"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

def get_model_params(model_type):
    """Get hyperparameters for model initialization"""
    base_params = {
        'num_classes': 2
    }
    
    if model_type == '3d_cnn':
        return base_params
    elif model_type == '2d_cnn_lstm':
        return {**base_params, 'lstm_hidden_size': 256, 'lstm_num_layers': 1}
    elif model_type == 'transformer':
        return {**base_params, 'embed_dim': 256, 'num_heads': 4, 'num_layers': 2}
    elif model_type == 'i3d':
        return base_params
    elif model_type == 'slowfast':
        return {**base_params, 'alpha': 4, 'beta': 1/8}
    elif model_type == 'r2plus1d':
        return base_params
    elif model_type == 'two_stream':
        return {**base_params, 'spatial_weight': 1.0, 'temporal_weight': 1.5, 'fusion': 'late'}
    elif model_type == 'simple_cnn':
        return base_params
    elif model_type == 'temporal_3d_cnn':
        return base_params
    elif model_type == 'cnn_lstm':
        return {**base_params, 'lstm_hidden_size': 256, 'num_layers': 1}
    else:
        return base_params

def initialize_model(model_type, device, verbose=False):
    """Initialize model with basic hyperparameters"""
    try:
        # Get hyperparameters for this model type
        params = get_model_params(model_type)
        
        # Initialize model based on type
        if model_type == '3d_cnn':
            from Models.model_3dcnn import Model3DCNN
            model = Model3DCNN(**params)
        elif model_type == '2d_cnn_lstm':
            from Models.model_2dcnn_lstm import Model2DCNNLSTM
            model = Model2DCNNLSTM(**params)
        elif model_type == 'transformer':
            from Models.model_transformer import VideoTransformer
            model = VideoTransformer(**params)
        elif model_type == 'i3d':
            from Models.model_i3d import TransferLearningI3D
            model = TransferLearningI3D(**params)
        elif model_type == 'slowfast':
            from Models.model_slowfast import SlowFastNetwork
            model = SlowFastNetwork(**params)
        elif model_type == 'r2plus1d':
            from Models.model_r2plus1d import R2Plus1DNet
            model = R2Plus1DNet(**params)
        elif model_type == 'two_stream':
            from Models.model_two_stream import TwoStreamNetwork
            model = TwoStreamNetwork(**params)
        elif model_type == 'simple_cnn':
            from Models.model_simplecnn import SimpleCNN
            model = SimpleCNN(**params)
        elif model_type == 'temporal_3d_cnn':
            from Models.model_Temporal3DCNN import Temporal3DCNN
            model = Temporal3DCNN(**params)
        elif model_type == 'cnn_lstm':
            from Models.violence_cnn_lstm import ViolenceCNNLSTM
            model = ViolenceCNNLSTM(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Move to device
        model = model.to(device)
        
        # Print model summary if verbose
        if verbose:
            print_model_summary(model)
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    except Exception as e:
        print(f"Error initializing {model_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_forward_pass(model, model_type, device, batch_size=2, verbose=False):
    """Test forward pass with dummy data"""
    try:
        # Create dummy batch
        inputs, labels = create_fake_batch(
            batch_size=batch_size, 
            model_type=model_type
        )
        
        # Move data to device
        if isinstance(inputs, tuple):
            inputs = tuple(x.to(device) for x in inputs)
        else:
            inputs = inputs.to(device)
        labels = labels.to(device)
        
        if verbose:
            print(f"Input shape: {inputs.shape if not isinstance(inputs, tuple) else [x.shape for x in inputs]}")
        
        # Switch to eval mode
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
        
        if verbose:
            print(f"Output shape: {outputs.shape}")
        
        # Check output shape
        assert outputs.shape[0] == batch_size, f"Expected batch size {batch_size}, got {outputs.shape[0]}"
        assert outputs.shape[1] == 2, f"Expected 2 output classes, got {outputs.shape[1]}"
        
        return True
    except Exception as e:
        print(f"Error in forward pass: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_pass(model, model_type, device, batch_size=2, verbose=False):
    """Test backward pass with dummy data"""
    try:
        # Create dummy batch
        inputs, labels = create_fake_batch(
            batch_size=batch_size, 
            model_type=model_type
        )
        
        # Move data to device
        if isinstance(inputs, tuple):
            inputs = tuple(x.to(device) for x in inputs)
        else:
            inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Switch to train mode
        model.train()
        
        # Set up optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        if verbose:
            print(f"Loss: {loss.item():.4f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that gradients were computed
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_grad = True
                    break
        
        assert has_grad, "No gradients were computed during backward pass"
        
        return True
    except Exception as e:
        print(f"Error in backward pass: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_checkpointing(model, model_type, tmp_dir, device):
    """Test saving and loading model checkpoints"""
    try:
        # Save model
        checkpoint_path = os.path.join(tmp_dir, f"{model_type}_test.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        
        # Initialize a new model instance
        new_model = initialize_model(model_type, device)
        
        # Load the saved weights
        new_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Model loaded from {checkpoint_path}")
        
        # Test forward pass with loaded model
        result = test_forward_pass(new_model, model_type, device)
        
        # Clean up
        os.remove(checkpoint_path)
        
        return result
    except Exception as e:
        print(f"Error testing checkpointing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model(model_type, device, batch_size=2, verbose=False, tmp_dir="/tmp"):
    """Run all tests for a single model"""
    print(f"\n{'='*20} Testing {model_type} {'='*20}")
    
    # Initialize model
    print(f"Initializing {model_type}...")
    model = initialize_model(model_type, device, verbose)
    if model is None:
        print(f"{model_type} initialization FAILED!")
        return False
    
    print(f"Model initialized successfully.")
    
    # Test forward pass
    print(f"Testing forward pass...")
    forward_ok = test_forward_pass(model, model_type, device, batch_size, verbose)
    if not forward_ok:
        print(f"{model_type} forward pass FAILED!")
        return False
    
    print(f"Forward pass successful.")
    
    # Test backward pass
    print(f"Testing backward pass...")
    backward_ok = test_backward_pass(model, model_type, device, batch_size, verbose)
    if not backward_ok:
        print(f"{model_type} backward pass FAILED!")
        return False
    
    print(f"Backward pass successful.")
    
    # Test checkpointing
    print(f"Testing checkpointing...")
    checkpoint_ok = test_model_checkpointing(model, model_type, tmp_dir, device)
    if not checkpoint_ok:
        print(f"{model_type} checkpointing FAILED!")
        return False
    
    print(f"Checkpointing successful.")
    print(f"{model_type} test PASSED!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Test model initializations")
    parser.add_argument("--model_types", nargs="+", 
                    default=['3d_cnn', '2d_cnn_lstm', 'transformer', 'i3d', 
                             'simple_cnn', 'temporal_3d_cnn', 'slowfast', 
                             'r2plus1d', 'two_stream', 'cnn_lstm'],
                    help="Model types to test")
    parser.add_argument("--batch_size", type=int, default=2, help="Mini-batch size")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--tmp_dir", type=str, default="/tmp", help="Temporary directory for checkpoint testing")
    args = parser.parse_args()
    
    # Set up device
    device = setup_device(args.gpu)
    
    # Ensure temp directory exists
    os.makedirs(args.tmp_dir, exist_ok=True)
    
    # Results storage
    results = {}
    
    # Test each model type
    for model_type in args.model_types:
        # Test the model
        results[model_type] = test_model(
            model_type, device, args.batch_size, 
            verbose=args.verbose, tmp_dir=args.tmp_dir
        )
        
        # Clear memory after testing each model
        clear_cuda_memory()
    
    # Print summary
    print("\n" + "="*20 + " Test Summary " + "="*20)
    print(f"{'Model Type':<15} {'Status':<10}")
    print("-" * 25)
    
    for model_type, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{model_type:<15} {status:<10}")
    
    # Count failures
    failures = sum(1 for passed in results.values() if not passed)
    if failures > 0:
        print(f"\n{failures} models failed the test.")
        return 1
    else:
        print("\nAll models passed the test!")
        return 0

if __name__ == "__main__":
    sys.exit(main())