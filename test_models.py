#!/usr/bin/env python3
# test_models_fixed.py
"""
Updated script to test all video-based violence detection models.
Includes fixes for tensor shape handling and proper model initialization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import sys
import os
from utils.model_utils import create_fake_batch, print_tensor_shape, print_model_summary

def parse_args():
    parser = argparse.ArgumentParser(description="Test all violence detection models")
    parser.add_argument("--model_types", nargs="+", 
                    default=['3d_cnn', '2d_cnn_lstm', 'transformer', 'i3d', 
                             'simple_cnn', 'temporal_3d_cnn', 'slowfast', 
                             'r2plus1d', 'two_stream'],
                    help="Model types to test")
    parser.add_argument("--batch_size", type=int, default=2, help="Mini-batch size")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--use_pose", action="store_true", help="Test with pose data")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    return parser.parse_args()

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

def get_hyperparameters(model_type, use_pose=False):
    """Get hyperparameters for a specific model type"""
    if model_type == '3d_cnn':
        return {
            'num_classes': 2,
            'use_pose': use_pose
        }
    elif model_type == '2d_cnn_lstm':
        return {
            'num_classes': 2,
            'hidden_size': 512,  
            'use_pose': use_pose
        }
    elif model_type == 'transformer':
        return {
            'num_classes': 2,
            'use_pose': use_pose
        }
    elif model_type == 'i3d':
        return {
            'num_classes': 2,
            'use_pose': use_pose
        }
    elif model_type == 'slowfast':
        return {
            'num_classes': 2,
            'pretrained': True,
            'alpha': 8,
            'beta': 1/8,
            'dropout_prob': 0.5
        }
    elif model_type == 'r2plus1d':
        return {
            'num_classes': 2,
            'pretrained': True,
            'dropout_prob': 0.5
        }
    elif model_type == 'two_stream':
        return {
            'num_classes': 2,
            'spatial_weight': 1.0,
            'temporal_weight': 1.5,
            'pretrained': True,
            'spatial_backbone': 'r3d_18',
            'dropout_prob': 0.5,
            'fusion': 'late'
        }
    elif model_type == 'simple_cnn':
        return {
            'num_classes': 2
        }
    elif model_type == 'temporal_3d_cnn':
        return {
            'num_classes': 2
        }
    else:
        return {'num_classes': 2}
    
def initialize_model(model_type, device, use_pose=False, hyperparams=None):
    """Initialize model based on model type with optional hyperparameters"""
    if hyperparams is None:
        hyperparams = {}
    
    # Ensure num_classes is set
    hyperparams['num_classes'] = hyperparams.get('num_classes', 2)
    
    # Set use_pose parameter if model supports it
    if model_type in ['2d_cnn_lstm', 'transformer', 'i3d']:
        hyperparams['use_pose'] = use_pose
    
    try:
        if model_type == '3d_cnn':
            from Models.model_3dcnn import Model3DCNN
            model = Model3DCNN(**hyperparams).to(device)
            
        elif model_type == '2d_cnn_lstm':
            from Models.model_2dcnn_lstm import Model2DCNNLSTM
            model = Model2DCNNLSTM(**hyperparams).to(device)
            
        elif model_type == 'transformer':
            from Models.model_transformer import VideoTransformer
            model = VideoTransformer(**hyperparams).to(device)
            
        elif model_type == 'i3d':
            from Models.model_i3d import TransferLearningI3D
            model = TransferLearningI3D(**hyperparams).to(device)
            
        elif model_type == 'simple_cnn':
            from Models.model_simplecnn import SimpleCNN
            model = SimpleCNN(**hyperparams).to(device)
            
        elif model_type == 'temporal_3d_cnn':
            from Models.model_Temporal3DCNN import Temporal3DCNN
            model = Temporal3DCNN(**hyperparams).to(device)
            
        elif model_type == 'slowfast':
            from Models.model_slowfast import SlowFastNetwork
            model = SlowFastNetwork(**hyperparams).to(device)
            
        elif model_type == 'r2plus1d':
            from Models.model_r2plus1d import R2Plus1DNet
            model = R2Plus1DNet(**hyperparams).to(device)
            
        elif model_type == 'two_stream':
            from Models.model_two_stream import TwoStreamNetwork
            model = TwoStreamNetwork(**hyperparams).to(device)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    except Exception as e:
        print(f"Error initializing {model_type}: {str(e)}")
        raise

def test_model(model_type, device, batch_size=2, use_pose=False, verbose=False):
    """
    Test if a model can be initialized and forward/backward passes work
    
    Args:
        model_type: Type of model to test
        device: Device to use
        batch_size: Batch size for testing
        use_pose: Whether to include pose data
        verbose: Whether to print detailed information
        
    Returns:
        True if test passed, False otherwise
    """
    print(f"\n{'='*20} Testing {model_type} {'='*20}")
    
    try:
        # Get hyperparameters
        hyperparams = get_hyperparameters(model_type, use_pose)
        
        # Initialize model
        print(f"Initializing {model_type}...")
        model = initialize_model(model_type, device, use_pose, hyperparams)
        print(f"Model initialized successfully.")
        
        if verbose:
            print_model_summary(model)
        
        # Create a dummy batch
        print(f"Creating dummy batch...")
        inputs, labels = create_fake_batch(
            batch_size=batch_size, 
            model_type=model_type, 
            use_pose=use_pose
        )
        
        # Move data to device
        if isinstance(inputs, tuple):
            inputs = tuple(x.to(device) for x in inputs)
        else:
            inputs = inputs.to(device)
        labels = labels.to(device)
        
        if verbose:
            print_tensor_shape(inputs, "Input")
        
        # Set up loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Forward pass
        print(f"Testing forward pass...")
        outputs = model(inputs)
        print(f"Forward pass successful. Output shape: {outputs.shape}")
        
        # Compute loss
        loss = criterion(outputs, labels)
        print(f"Loss calculation successful. Loss: {loss.item():.4f}")
        
        # Backward pass
        print(f"Testing backward pass...")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Backward pass successful.")
        
        print(f"{model_type} test PASSED!")
        return True
        
    except Exception as e:
        print(f"Error testing {model_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{model_type} test FAILED!")
        return False

def clear_cuda_memory():
    """Clear CUDA memory to prevent OOM errors between model tests"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up device
    device = setup_device(args.gpu)
    
    # Results storage
    results = {}
    
    # Test each model type
    for model_type in args.model_types:
        # Test the model
        results[model_type] = test_model(
            model_type, device, args.batch_size, 
            use_pose=args.use_pose, verbose=args.verbose
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