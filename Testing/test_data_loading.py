#!/usr/bin/env python3
# Testing/test_data_loading.py
"""
Test script to verify the data loading pipeline works correctly.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import time


# Add parent directory to path to import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from dataloader import get_dataloaders, EnhancedViolenceDataset
from utils.dataprep import prepare_violence_nonviolence_data, load_data_splits

def test_data_splits(data_dir=None, splits_path=None):
    """Test data split functionality"""
    print("\n" + "="*50)
    print("Testing Data Splits")
    print("="*50)
    
    if splits_path and os.path.exists(splits_path):
        print(f"Loading data splits from: {splits_path}")
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
            load_data_splits(splits_path)
    elif data_dir and os.path.exists(data_dir):
        print(f"Preparing data splits from: {data_dir}")
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
            prepare_violence_nonviolence_data(data_dir)
    else:
        print("Error: Neither valid splits_path nor data_dir provided")
        return False
    
    # Print splits statistics
    print(f"Train set: {len(train_paths)} videos, Violence: {sum(train_labels)}, Non-violence: {len(train_labels) - sum(train_labels)}")
    print(f"Val set: {len(val_paths)} videos, Violence: {sum(val_labels)}, Non-violence: {len(val_labels) - sum(val_labels)}")
    print(f"Test set: {len(test_paths)} videos, Violence: {sum(test_labels)}, Non-violence: {len(test_labels) - sum(test_labels)}")
    
    # Check that files actually exist
    missing_files = []
    for path in train_paths + val_paths + test_paths:
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        print(f"Warning: {len(missing_files)} files do not exist")
        for path in missing_files[:5]:  # Print first 5 missing files
            print(f"  - {path}")
        if len(missing_files) > 5:
            print(f"  - ... and {len(missing_files) - 5} more")
        return False
    else:
        print("All video files exist on disk")
        return True

def test_dataset(data_dir=None, splits_path=None, pose_dir=None, model_type='3d_cnn'):
    """Test dataset class"""
    print("\n" + "="*50)
    print(f"Testing Dataset for {model_type}")
    print("="*50)
    
    if splits_path and os.path.exists(splits_path):
        print(f"Loading data splits from: {splits_path}")
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
            load_data_splits(splits_path)
    elif data_dir and os.path.exists(data_dir):
        print(f"Preparing data splits from: {data_dir}")
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
            prepare_violence_nonviolence_data(data_dir)
    else:
        print("Error: Neither valid splits_path nor data_dir provided")
        return False
    
    # Use first few samples for testing
    test_train_paths = train_paths[:5]
    test_train_labels = train_labels[:5]
    
    # Create a dataset instance
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    start_time = time.time()
    
    dataset = EnhancedViolenceDataset(
        test_train_paths, 
        test_train_labels, 
        pose_dir=pose_dir,
        transform=transform, 
        num_frames=16, 
        target_fps=15,
        model_type=model_type,
        training=True
    )
    
    print(f"Dataset created in {time.time() - start_time:.2f} seconds")
    print(f"Dataset length: {len(dataset)}")
    
    # Test __getitem__
    print("\nTesting dataset __getitem__:")
    
    item_start_time = time.time()
    try:
        sample = dataset[0]
        print(f"Sample loaded in {time.time() - item_start_time:.2f} seconds")
        
        if isinstance(sample, (list, tuple)) and len(sample) == 3:
            frames, pose, label = sample
            print(f"Sample contains frames, pose, and label")
            print(f"Frames shape: {frames.shape}")
            print(f"Pose shape: {pose.shape}")
            print(f"Label: {label.item()}")
        elif isinstance(sample, (list, tuple)) and len(sample) == 2:
            frames, label = sample
            print(f"Sample contains only frames and label")
            print(f"Frames shape: {frames.shape}")
            print(f"Label: {label.item()}")
        else:
            print(f"Unexpected sample format: {type(sample)}")
            return False
        
        return True
    except Exception as e:
        print(f"Error loading sample: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader(data_dir=None, splits_path=None, pose_dir=None, model_type='3d_cnn', batch_size=4):
    """Test dataloader functionality"""
    print("\n" + "="*50)
    print(f"Testing DataLoader for {model_type}")
    print("="*50)
    
    if splits_path and os.path.exists(splits_path):
        print(f"Loading data splits from: {splits_path}")
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
            load_data_splits(splits_path)
    elif data_dir and os.path.exists(data_dir):
        print(f"Preparing data splits from: {data_dir}")
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
            prepare_violence_nonviolence_data(data_dir)
    else:
        print("Error: Neither valid splits_path nor data_dir provided")
        return False
    
    # Use first few samples for testing
    test_train_paths = train_paths[:10]
    test_train_labels = train_labels[:10]
    test_val_paths = val_paths[:5]
    test_val_labels = val_labels[:5]
    
    print("Creating test dataloaders")
    start_time = time.time()
    
    train_loader, val_loader, _ = get_dataloaders(
        test_train_paths, test_train_labels,
        test_val_paths, test_val_labels,
        test_val_paths, test_val_labels,  # Reuse validation for test
        pose_dir=pose_dir,
        batch_size=batch_size,
        num_workers=2,
        model_type=model_type
    )
    
    print(f"Dataloaders created in {time.time() - start_time:.2f} seconds")
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    
    # Test iterating through dataloader
    print("\nTesting dataloader iteration:")
    
    batch_start_time = time.time()
    try:
        batch = next(iter(train_loader))
        print(f"Batch loaded in {time.time() - batch_start_time:.2f} seconds")
        
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            frames, pose, labels = batch
            print(f"Batch contains frames, pose, and labels")
            print(f"Frames shape: {frames.shape}")
            print(f"Pose shape: {pose.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Labels: {labels}")
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            frames, labels = batch
            print(f"Batch contains only frames and labels")
            print(f"Frames shape: {frames.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Labels: {labels}")
        else:
            print(f"Unexpected batch format: {type(batch)}")
            return False
        
        # Test data augmentation by comparing multiple batches
        if model_type != 'two_stream':  # Two-stream has flow calculation, not compatible with this test
            print("\nTesting data augmentation:")
            
            # Multiple iterations to see different augmentations
            for i in range(3):
                batch_i = next(iter(train_loader))
                
                if isinstance(batch_i, (list, tuple)) and len(batch_i) >= 2:
                    frames_i = batch_i[0]
                    if i == 0:
                        # Save first batch frames
                        first_batch = frames_i.clone()
                    else:
                        # Compare with first batch
                        diff = torch.abs(frames_i - first_batch).mean().item()
                        print(f"Batch {i+1} difference from first batch: {diff:.6f}")
                        if diff < 0.001:
                            print("Warning: Batches are nearly identical. Augmentation may not be working.")
                        else:
                            print(f"Batches differ as expected (augmentation working)")
        
        return True
    except Exception as e:
        print(f"Error loading batch: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test data loading pipeline")
    parser.add_argument("--data_dir", type=str, default="./Data/Processed/standardized",
                      help="Directory containing standardized videos")
    parser.add_argument("--splits_path", type=str, default="./data_splits.json",
                      help="Path to data splits JSON file")
    parser.add_argument("--pose_dir", type=str, default=None,
                      help="Directory containing pose keypoints (optional)")
    parser.add_argument("--model_type", type=str, default="3d_cnn",
                      choices=["3d_cnn", "2d_cnn_lstm", "transformer", "i3d", 
                               "simple_cnn", "temporal_3d_cnn", "slowfast", 
                               "r2plus1d", "two_stream", "cnn_lstm"],
                      help="Model type to test data loading for")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for testing dataloader")
    args = parser.parse_args()
    
    # Ensure data is available
    if not os.path.exists(args.data_dir) and not os.path.exists(args.splits_path):
        print(f"Error: Neither data directory {args.data_dir} nor splits file {args.splits_path} exists")
        return 1
    
    # Run tests
    data_splits_ok = test_data_splits(args.data_dir, args.splits_path)
    dataset_ok = test_dataset(args.data_dir, args.splits_path, args.pose_dir, args.model_type)
    dataloader_ok = test_dataloader(args.data_dir, args.splits_path, args.pose_dir, args.model_type, args.batch_size)
    
    # Report results
    print("\n" + "="*50)
    print("Data Loading Tests Results")
    print("="*50)
    print(f"Data Splits: {'✓ PASSED' if data_splits_ok else '✗ FAILED'}")
    print(f"Dataset Class: {'✓ PASSED' if dataset_ok else '✗ FAILED'}")
    print(f"DataLoader: {'✓ PASSED' if dataloader_ok else '✗ FAILED'}")
    
    # Return success only if all tests pass
    return 0 if all([data_splits_ok, dataset_ok, dataloader_ok]) else 1

if __name__ == "__main__":
    sys.exit(main())