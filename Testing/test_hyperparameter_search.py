#!/usr/bin/env python3
# Testing/test_hyperparameter_search.py
"""
Test script to verify the hyperparameter search functionality.
Tests grid search implementation on a small subset of data.
"""

import os
import sys
import torch
import json
import numpy as np
import argparse
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

CUDA_LAUNCH_BLOCKING=1 
# Add parent directory to path to import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.dataprep import prepare_violence_nonviolence_data
from hyperparameter_search import grid_search, get_best_hyperparameters
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

def mini_grid_search(data_dir, output_dir, device, model_class=SimpleCNN):
    """
    Test grid search functionality with a simple model and limited param grid
    """
    print("\n" + "="*50)
    print(f"Testing Grid Search with {model_class.__name__}")
    print("="*50)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a small subset of data
    train_paths, train_labels, val_paths, val_labels, _, _ = \
        prepare_violence_nonviolence_data(data_dir)
    
    # Use only 10 samples for quick testing
    train_subset_paths = train_paths[:10]
    train_subset_labels = train_labels[:10]
    val_subset_paths = val_paths[:5]
    val_subset_labels = val_labels[:5]
    
    # Define a simple parameter grid
    param_grid = {
        'num_classes': [2],  # Fixed for binary classification
        'dropout_prob': [0.3, 0.5]  # Only test two values for speed
    }
    
    # Base parameters
    base_params = {
        'num_classes': 2
    }
    
    # Run grid search
    start_time = time.time()
    results = grid_search(
        model_class,
        train_subset_paths, train_subset_labels,
        val_subset_paths, val_subset_labels,
        param_grid,
        base_params,
        device=device,
        output_dir=os.path.join(output_dir, f"{model_class.__name__}_grid_search"),
        num_epochs=2  # Just 2 epochs for quick testing
    )
    
    elapsed_time = time.time() - start_time
    print(f"Grid search completed in {elapsed_time:.2f} seconds")
    
    # Check if results are valid
    if results and 'best_params' in results and 'best_auc' in results:
        print(f"Best parameters: {results['best_params']}")
        print(f"Best AUC: {results['best_auc']:.4f}")
        
        # Check if best_auc is valid
        if np.isnan(results['best_auc']):
            print("Error: Best AUC is NaN")
            return False
        
        # Save results
        results_path = os.path.join(output_dir, f"{model_class.__name__}_grid_search_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy values to Python types
            results_copy = results.copy()
            results_copy['best_auc'] = float(results_copy['best_auc'])
            
            if 'results' in results_copy:
                for res in results_copy['results']:
                    if 'metrics' in res and 'roc_auc' in res['metrics']:
                        res['metrics']['roc_auc'] = float(res['metrics']['roc_auc'])
            
            json.dump(results_copy, f, indent=2)
        
        print(f"Results saved to {results_path}")
        
        # Plot results if there are multiple
        if 'results' in results and len(results['results']) > 1:
            # Extract dropout values and AUCs
            dropout_values = [res['params']['dropout_prob'] for res in results['results']]
            auc_values = [res['metrics']['roc_auc'] for res in results['results']]
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(dropout_values, auc_values, 'bo-')
            plt.xlabel('Dropout Probability')
            plt.ylabel('AUC-ROC')
            plt.title(f'Grid Search Results for {model_class.__name__}')
            plt.grid(True)
            
            # Save plot
            plot_path = os.path.join(output_dir, f"{model_class.__name__}_grid_search_plot.png")
            plt.savefig(plot_path)
            plt.close()
            
            print(f"Plot saved to {plot_path}")
        
        return True
    else:
        print("Error: Invalid grid search results")
        return False

def test_best_hyperparameters(data_dir, output_dir, device):
    """
    Test the get_best_hyperparameters function with a simple model
    """
    print("\n" + "="*50)
    print("Testing get_best_hyperparameters Function")
    print("="*50)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a small subset of data
    train_paths, train_labels, val_paths, val_labels, _, _ = \
        prepare_violence_nonviolence_data(data_dir)
    
    # Use only 10 samples for quick testing
    train_subset_paths = train_paths[:10]
    train_subset_labels = train_labels[:10]
    val_subset_paths = val_paths[:5]
    val_subset_labels = val_labels[:5]
    
    try:
        # Import the model class
        from Models.model_simplecnn import SimpleCNN
        
        # Run get_best_hyperparameters (which internally uses grid_search)
        results = get_best_hyperparameters(
            SimpleCNN,
            train_subset_paths, train_subset_labels,
            val_subset_paths, val_subset_labels,
            output_dir=os.path.join(output_dir, "best_hyperparam_search")
        )
        
        # Check results
        if results and 'best_params' in results:
            print(f"Best parameters found: {results['best_params']}")
            print(f"Best AUC: {results['best_auc']:.4f}")
            return True
        else:
            print("Error: Invalid results from get_best_hyperparameters")
            return False
    except Exception as e:
        print(f"Error testing get_best_hyperparameters: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_sequential_search(data_dir, output_dir, device):
    """
    Test the sequential hyperparameter search functionality
    """
    print("\n" + "="*50)
    print("Testing Sequential Hyperparameter Search")
    print("="*50)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a small subset of data
    train_paths, train_labels, val_paths, val_labels, _, _ = \
        prepare_violence_nonviolence_data(data_dir)
    
    # Use only 10 samples for quick testing
    train_subset_paths = train_paths[:10]
    train_subset_labels = train_labels[:10]
    val_subset_paths = val_paths[:5]
    val_subset_labels = val_labels[:5]
    
    try:
        # Import function and model classes
        from hyperparameter_search import sequential_hyperparameter_search
        from Models.model_simplecnn import SimpleCNN
        from Models.model_3dcnn import Model3DCNN
        
        # Define model classes to test
        model_classes = {
            'simple_cnn': SimpleCNN,
            '3d_cnn': Model3DCNN
        }
        
        # Define parameter grids for each model
        param_grids = {
            'simple_cnn': {
                'num_classes': [2],
                'dropout_prob': [0.3, 0.5]
            },
            '3d_cnn': {
                'num_classes': [2],
                'dropout_prob': [0.3, 0.5]
            }
        }
        
        # Define base parameters
        base_params = {
            'simple_cnn': {'num_classes': 2},
            '3d_cnn': {'num_classes': 2}
        }
        
        # Run sequential search
        results = sequential_hyperparameter_search(
            model_classes,
            train_subset_paths, train_subset_labels,
            val_subset_paths, val_subset_labels,
            param_grids,
            base_params,
            output_dir=os.path.join(output_dir, "sequential_search"),
            device=device,
            num_epochs=2  # Just 2 epochs for quick testing
        )
        
        # Check results
        if results:
            print("Sequential search results:")
            for model_name, result in results.items():
                print(f"  {model_name}: Best AUC = {result['best_auc']:.4f}")
                print(f"  {model_name}: Best params = {result['best_params']}")
            
            # Check for JSON output
            json_path = os.path.join(output_dir, "sequential_search", "all_results.json")
            if os.path.exists(json_path):
                print(f"Results JSON exists at {json_path}")
            else:
                print(f"Warning: Results JSON not found at {json_path}")
            
            # Check for summary text
            summary_path = os.path.join(output_dir, "sequential_search", "search_summary.txt")
            if os.path.exists(summary_path):
                print(f"Summary text exists at {summary_path}")
            else:
                print(f"Warning: Summary text not found at {summary_path}")
            
            return True
        else:
            print("Error: Invalid results from sequential_hyperparameter_search")
            return False
    except Exception as e:
        print(f"Error testing sequential search: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test hyperparameter search functionality")
    parser.add_argument("--data_dir", type=str, default="./Data/Processed/standardized",
                      help="Directory containing standardized videos")
    parser.add_argument("--output_dir", type=str, default="./Testing/hyperparam_test_output",
                      help="Directory to save test output")
    parser.add_argument("--gpu", type=int, default=0,
                      help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--skip_grid_search", action="store_true",
                      help="Skip grid search test")
    parser.add_argument("--skip_best_hyperparams", action="store_true",
                      help="Skip best hyperparameters test")
    parser.add_argument("--skip_sequential", action="store_true",
                      help="Skip sequential search test")
    args = parser.parse_args()
    
    # Set up device
    device = setup_device(args.gpu)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run tests
    grid_search_ok = True
    if not args.skip_grid_search:
        try:
            grid_search_ok = mini_grid_search(args.data_dir, args.output_dir, device)
        except Exception as e:
            print(f"Error in grid search test: {str(e)}")
            import traceback
            traceback.print_exc()
            grid_search_ok = False
    
    best_hyperparams_ok = True
    if not args.skip_best_hyperparams:
        try:
            best_hyperparams_ok = test_best_hyperparameters(args.data_dir, args.output_dir, device)
        except Exception as e:
            print(f"Error in best hyperparameters test: {str(e)}")
            import traceback
            traceback.print_exc()
            best_hyperparams_ok = False
    
    sequential_ok = True
    if not args.skip_sequential:
        try:
            sequential_ok = test_sequential_search(args.data_dir, args.output_dir, device)
        except Exception as e:
            print(f"Error in sequential search test: {str(e)}")
            import traceback
            traceback.print_exc()
            sequential_ok = False
    
    # Print summary
    print("\n" + "="*50)
    print("Hyperparameter Search Tests Results")
    print("="*50)
    
    if not args.skip_grid_search:
        print(f"Grid Search: {'✓ PASSED' if grid_search_ok else '✗ FAILED'}")
    
    if not args.skip_best_hyperparams:
        print(f"Best Hyperparameters: {'✓ PASSED' if best_hyperparams_ok else '✗ FAILED'}")
    
    if not args.skip_sequential:
        print(f"Sequential Search: {'✓ PASSED' if sequential_ok else '✗ FAILED'}")
    
    # Return success only if all tests pass
    if grid_search_ok and best_hyperparams_ok and sequential_ok:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())