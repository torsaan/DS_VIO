#!/usr/bin/env python3
# Testing/test_cuda_memory.py
"""
Test script to verify CUDA memory management.
Tests memory allocation, usage tracking, and proper cleanup.
"""

import os
import sys
import torch
import gc
import time
import argparse
from pathlib import Path
import numpy as np

def print_memory_stats():
    """Print current CUDA memory statistics"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.cuda.current_device()
    
    total = torch.cuda.get_device_properties(device).total_memory / (1024**2)  # MB
    allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB
    reserved = torch.cuda.memory_reserved(device) / (1024**2)  # MB
    free = total - allocated
    
    print(f"  Total: {total:.2f} MB, Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Free: {free:.2f} MB")

def clear_cuda_memory():
    """Clear CUDA memory"""
    if torch.cuda.is_available():
        # Empty cache
        torch.cuda.empty_cache()
        
        # Run garbage collector
        gc.collect()
        
        print("CUDA memory cleared.")

def test_memory_allocation(size_mb=100, device="cuda"):
    """Test allocating and freeing memory"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return False
    
    print(f"Allocating {size_mb} MB tensor on {device}")
    
    # Calculate number of elements needed for the requested size
    # Each float32 element is 4 bytes
    n_elements = int(size_mb * (1024**2) / 4)
    
    try:
        # Print memory before allocation
        print("Before allocation:")
        print_memory_stats()
        
        # Allocate tensor
        x = torch.ones(n_elements, dtype=torch.float32, device=device)
        
        # Print memory after allocation
        print("After allocation:")
        print_memory_stats()
        
        # Free the tensor
        del x
        
        # Print memory after deletion but before clearing cache
        print("After deletion (before clearing cache):")
        print_memory_stats()
        
        # Clear cache
        clear_cuda_memory()
        
        # Print memory after clearing cache
        print("After clearing cache:")
        print_memory_stats()
        
        return True
    except Exception as e:
        print(f"Error during memory allocation test: {str(e)}")
        return False

def test_memory_leak(n_iterations=10, tensor_size_mb=10, device="cuda"):
    """Test for memory leaks in a loop"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return False
    
    print(f"Testing for memory leaks with {n_iterations} iterations")
    print(f"Creating and deleting {tensor_size_mb} MB tensors")
    
    # Elements for tensor size
    n_elements = int(tensor_size_mb * (1024**2) / 4)
    
    try:
        # Print initial memory state
        print("Initial memory state:")
        print_memory_stats()
        
        # First record baseline memory usage
        baseline_allocated = torch.cuda.memory_allocated(device) / (1024**2)
        
        # Run iterations
        memory_usage = []
        for i in range(n_iterations):
            # Create tensor
            x = torch.ones(n_elements, dtype=torch.float32, device=device)
            
            # Do some operations
            y = x * 2
            z = y + x
            
            # Record memory
            current_allocated = torch.cuda.memory_allocated(device) / (1024**2)
            memory_usage.append(current_allocated)
            
            # Delete tensors
            del x, y, z
            
            # Clear cache every other iteration
            if i % 2 == 0:
                torch.cuda.empty_cache()
                
            print(f"Iteration {i+1}/{n_iterations}: {current_allocated:.2f} MB allocated")
        
        # Final memory check after all iterations
        print("Final memory state before cleanup:")
        print_memory_stats()
        
        # Clear memory fully
        clear_cuda_memory()
        
        # Check memory after cleanup
        print("Final memory state after cleanup:")
        print_memory_stats()
        
        # Final allocated memory
        final_allocated = torch.cuda.memory_allocated(device) / (1024**2)
        
        # Check for potential leaks
        memory_diff = final_allocated - baseline_allocated
        if memory_diff > 1.0:  # If more than 1MB difference
            print(f"WARNING: Potential memory leak detected. {memory_diff:.2f} MB more allocated after test.")
            return False
        else:
            print(f"No memory leak detected. Memory difference: {memory_diff:.2f} MB")
            return True
            
    except Exception as e:
        print(f"Error during memory leak test: {str(e)}")
        return False

def test_model_memory_usage(model_type, batch_size=2, device="cuda"):
    """Test memory usage of a specific model"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return False
    
    # Add parent directory to path to import modules
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    try:
        from utils.model_utils import create_fake_batch
        
        print(f"Testing memory usage for {model_type} model")
        
        # Print initial memory
        print("Initial memory state:")
        print_memory_stats()
        
        # Import the model
        if model_type == '3d_cnn':
            from Models.model_3dcnn import Model3DCNN
            model = Model3DCNN(num_classes=2).to(device)
        elif model_type == '2d_cnn_lstm':
            from Models.model_2dcnn_lstm import Model2DCNNLSTM
            model = Model2DCNNLSTM(num_classes=2).to(device)
        elif model_type == 'transformer':
            from Models.model_transformer import VideoTransformer
            model = VideoTransformer(num_classes=2).to(device)
        elif model_type == 'i3d':
            from Models.model_i3d import TransferLearningI3D
            model = TransferLearningI3D(num_classes=2).to(device)
        elif model_type == 'simple_cnn':
            from Models.model_simplecnn import SimpleCNN
            model = SimpleCNN(num_classes=2).to(device)
        elif model_type == 'temporal_3d_cnn':
            from Models.model_Temporal3DCNN import Temporal3DCNN
            model = Temporal3DCNN(num_classes=2).to(device)
        elif model_type == 'slowfast':
            from Models.model_slowfast import SlowFastNetwork
            model = SlowFastNetwork(num_classes=2).to(device)
        elif model_type == 'r2plus1d':
            from Models.model_r2plus1d import R2Plus1DNet
            model = R2Plus1DNet(num_classes=2).to(device)
        elif model_type == 'two_stream':
            from Models.model_two_stream import TwoStreamNetwork
            model = TwoStreamNetwork(num_classes=2).to(device)
        elif model_type == 'cnn_lstm':
            from Models.violence_cnn_lstm import ViolenceCNNLSTM
            model = ViolenceCNNLSTM(num_classes=2).to(device)
        else:
            print(f"Unknown model type: {model_type}")
            return False
        
        # Print memory after model creation
        print("Memory after loading model:")
        print_memory_stats()
        
        # Create input data
        inputs, labels = create_fake_batch(batch_size=batch_size, model_type=model_type)
        
        # Move to device
        if isinstance(inputs, tuple):
            inputs = tuple(x.to(device) for x in inputs)
        else:
            inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Print memory after creating inputs
        print("Memory after loading inputs:")
        print_memory_stats()
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        
        # Print memory after forward pass
        print("Memory after forward pass:")
        print_memory_stats()
        
        # Backward pass with gradient calculation
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Forward pass in training mode
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Print memory after backward pass
        print("Memory after backward pass:")
        print_memory_stats()
        
        # Optimizer step
        optimizer.step()
        
        # Print memory after optimizer step
        print("Memory after optimizer step:")
        print_memory_stats()
        
        # Free resources
        del model, inputs, labels, outputs, optimizer, criterion
        
        # Clear memory
        clear_cuda_memory()
        
        # Print final memory state
        print("Final memory state:")
        print_memory_stats()
        
        return True
    except Exception as e:
        print(f"Error testing model memory usage: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_models(model_types, device="cuda"):
    """Test loading multiple models in sequence to check for memory leaks"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return False
    
    print(f"Testing sequential loading of {len(model_types)} models")
    
    # Print initial memory
    print("Initial memory state:")
    print_memory_stats()
    
    # Track results
    results = []
    
    for model_type in model_types:
        print(f"\nTesting {model_type}...")
        
        # Clear memory before each model
        clear_cuda_memory()
        
        # Test this model
        result = test_model_memory_usage(model_type, device=device)
        results.append((model_type, result))
    
    # Print summary
    print("\nSequential Model Loading Results:")
    print("-" * 40)
    for model_type, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{model_type:<20} {status}")
    
    # Final memory check
    print("\nFinal memory state after all models:")
    print_memory_stats()
    
    # All tests passed?
    return all(result for _, result in results)

def main():
    parser = argparse.ArgumentParser(description="Test CUDA memory management")
    parser.add_argument("--allocation_size", type=int, default=100,
                      help="Size in MB for the memory allocation test")
    parser.add_argument("--leak_iterations", type=int, default=10,
                      help="Number of iterations for the memory leak test")
    parser.add_argument("--test_model", type=str, default=None,
                      choices=['3d_cnn', '2d_cnn_lstm', 'transformer', 'i3d', 
                               'simple_cnn', 'temporal_3d_cnn', 'slowfast', 
                               'r2plus1d', 'two_stream', 'cnn_lstm', 'all'],
                      help="Test memory usage with a specific model")
    parser.add_argument("--batch_size", type=int, default=2,
                      help="Batch size for model memory tests")
    parser.add_argument("--skip_basic", action="store_true",
                      help="Skip basic memory tests")
    parser.add_argument("--gpu", type=int, default=0,
                      help="GPU device index to use")
    args = parser.parse_args()
    
    # Set device
    if not torch.cuda.is_available():
        print("CUDA is not available on this system. Tests will be skipped.")
        return 1
    
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Using device: {torch.cuda.get_device_name(device)}")
    
    # Run basic memory tests
    if not args.skip_basic:
        print("\n" + "="*50)
        print("Testing memory allocation and deallocation")
        print("="*50)
        allocation_ok = test_memory_allocation(args.allocation_size, device)
        
        print("\n" + "="*50)
        print("Testing for memory leaks")
        print("="*50)
        leak_ok = test_memory_leak(args.leak_iterations, device=device)
    else:
        allocation_ok = True
        leak_ok = True
    
    # Test model memory usage
    model_ok = True
    if args.test_model:
        print("\n" + "="*50)
        if args.test_model == 'all':
            print("Testing memory usage with all models")
            print("="*50)
            model_types = ['3d_cnn', '2d_cnn_lstm', 'transformer', 'i3d', 
                          'simple_cnn', 'temporal_3d_cnn', 'slowfast', 
                          'r2plus1d', 'two_stream', 'cnn_lstm']
            model_ok = test_multiple_models(model_types, device=device)
        else:
            print(f"Testing memory usage with {args.test_model} model")
            print("="*50)
            model_ok = test_model_memory_usage(args.test_model, args.batch_size, device=device)
    
    # Print summary
    print("\n" + "="*50)
    print("CUDA Memory Tests Summary")
    print("="*50)
    
    if not args.skip_basic:
        print(f"Memory Allocation: {'✓ PASSED' if allocation_ok else '✗ FAILED'}")
        print(f"Memory Leak Test: {'✓ PASSED' if leak_ok else '✗ FAILED'}")
    
    if args.test_model:
        print(f"Model Memory Test: {'✓ PASSED' if model_ok else '✗ FAILED'}")
    
    # Return code based on test results
    if allocation_ok and leak_ok and model_ok:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())