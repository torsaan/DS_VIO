# hyperparameter_search.py
import os
import itertools
import json
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from train import train_model, clear_cuda_memory
from dataloader import get_dataloaders
from evaluations import evaluate_model
import matplotlib.pyplot as plt
import argparse
from utils.dataprep import prepare_violence_nonviolence_data
from dataloader import get_dataloaders


def grid_search(model_class, train_paths, train_labels, val_paths, val_labels,
               param_grid, base_params=None, device=torch.device("cuda"),
               output_dir="./hyperparam_search", num_epochs=10,
               batch_size=8, num_workers=4, model_type='3d_cnn'):
    """
    Perform grid search for hyperparameter optimization
    
    Args:
        model_class: Model class to instantiate
        train_paths, val_paths: Lists of video paths for training and validation
        train_labels, val_labels: Lists of labels for training and validation
        param_grid: Dictionary mapping parameter names to lists of values to try
        base_params: Dictionary of base parameters for model instantiation
        device: Device to use for training
        output_dir: Directory to save results
        num_epochs: Number of epochs to train each model
        batch_size: Batch size
        num_workers: Number of worker processes for DataLoader
        model_type: Type of model ('3d_cnn', '2d_cnn_lstm', etc.)
        
    Returns:
        Dictionary with best parameters and results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create base params if None
    if base_params is None:
        base_params = {}
    
    # Get all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    # Initialize results storage
    results = []
    best_auc = 0.0
    best_params = None
    
    # Set up criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    # Set up CSV logger with timestamp
    from utils.logger import CSVLogger
    from datetime import datetime
    
    # Define CSV header fields
    fieldnames = ['timestamp', 'combination_id'] + param_names + ['val_loss', 'val_acc', 'roc_auc', 'pr_auc']
    
    # Initialize logger
    logger = CSVLogger(
        os.path.join(output_dir, 'grid_search_log.csv'),
        fieldnames=fieldnames
    )
    
    # Loop through all parameter combinations
    for i, combination in enumerate(tqdm(param_combinations, desc="Hyperparameter Search")):
        # Clear CUDA memory
        clear_cuda_memory()
        
        # Create parameter dictionary for this combination
        params = base_params.copy()
        for name, value in zip(param_names, combination):
            params[name] = value
        
        # Create model name
        model_name = f"model_{i}"
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Log this combination
        param_str = ", ".join([f"{name}={value}" for name, value in zip(param_names, combination)])
        print(f"\nTrying combination {i+1}/{len(param_combinations)}: {param_str}")
        
        # Create model with current parameters
        model = model_class(**params).to(device)
        
        # Create dataloaders with current batch size
        train_loader, val_loader, _ = get_dataloaders(
            train_paths, train_labels, 
            val_paths, val_labels,
            val_paths, val_labels,  # Use validation set as test set
            batch_size=batch_size,
            num_workers=num_workers,
            model_type=model_type
        )
        
        # Train model
        trained_model = train_model(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            device=device,
            output_dir=output_dir,
            patience=3  # Use shorter patience for hyperparameter search
        )
        
        # Evaluate model on validation set
        val_loss, val_acc, _, _, all_probs, metrics_dict = evaluate_model(
            trained_model, val_loader, criterion, device
        )
        
        # Extract evaluation metrics
        roc_auc = metrics_dict.get('roc_auc', 0)
        pr_auc = metrics_dict.get('pr_auc', 0)
        
        # Save metrics
        combination_result = {
            'params': {name: value for name, value in zip(param_names, combination)},
            'metrics': metrics_dict
        }
        results.append(combination_result)
        
        # Log results to CSV
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'combination_id': i
        }
        
        # Add parameter values
        for name, value in zip(param_names, combination):
            log_entry[name] = value
            
        # Add metrics
        log_entry['val_loss'] = val_loss
        log_entry['val_acc'] = val_acc
        log_entry['roc_auc'] = roc_auc
        log_entry['pr_auc'] = pr_auc
        
        # Write to CSV log
        logger.log(log_entry)
        
        # Check if this is the best model
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_params = combination_result['params']
            print(f"New best parameters found with AUC {best_auc:.4f}")
        
        # Save current results
        with open(os.path.join(output_dir, 'grid_search_results.json'), 'w') as f:
            json.dump({
                'results': results,
                'best_params': best_params,
                'best_auc': best_auc
            }, f, indent=4)
        
        # Clear memory
        del model, trained_model
        clear_cuda_memory()
    
    # Plot results
    plot_grid_search_results(results, param_names, output_dir)
    
    return {
        'results': results,
        'best_params': best_params,
        'best_auc': best_auc
    }

def plot_grid_search_results(results, param_names, output_dir):
    """Plot grid search results for different parameters"""
    plt.figure(figsize=(15, 10))
    
    for param_name in param_names:
        # Group results by this parameter
        param_values = []
        auc_values = []
        
        for result in results:
            param_values.append(result['params'][param_name])
            auc_values.append(result['metrics']['roc_auc'])
        
        # Create scatter plot
        plt.scatter(param_values, auc_values, label=param_name)
        
        # Try to fit a line if there are enough unique values
        unique_values = sorted(set(param_values))
        if len(unique_values) > 2:
            # Calculate mean AUC for each parameter value
            mean_aucs = []
            for value in unique_values:
                indices = [i for i, v in enumerate(param_values) if v == value]
                mean_auc = np.mean([auc_values[i] for i in indices])
                mean_aucs.append(mean_auc)
            
            # Plot line connecting mean values
            plt.plot(unique_values, mean_aucs, 'o-', label=f"{param_name}_mean")
    
    plt.xlabel('Parameter Value')
    plt.ylabel('Validation AUC-ROC')
    plt.title('Grid Search Results')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'grid_search_results.png'))
    plt.close()

def get_best_hyperparameters(model_class, train_paths, train_labels, val_paths, val_labels, 
                             output_dir="./hyperparam_search"):
    """Example function showing how to use grid search for specific models"""
    
    # Define parameter grid based on model type
    if model_class.__name__ == 'Model3DCNN':
        param_grid = {
            'num_classes': [2],  # Fixed for binary classification
            'dropout_prob': [0.3, 0.5, 0.7],
            'learning_rate': [1e-4, 5e-4, 1e-3],
            'weight_decay': [1e-4, 1e-5]
        }
        
        base_params = {
            'num_classes': 2,
            'pretrained': True
        }
        
        model_type = '3d_cnn'
        
    elif model_class.__name__ == 'Model2DCNNLSTM':
        param_grid = {
            'num_classes': [2],  # Fixed for binary classification
            'lstm_hidden_size': [256, 512],
            'lstm_num_layers': [1, 2],
            'dropout_prob': [0.3, 0.5],
            'learning_rate': [1e-4, 5e-4],
            'bidirectional': [True, False]
        }
        
        base_params = {
            'num_classes': 2,
            'pretrained': True
        }
        
        model_type = '2d_cnn_lstm'
        
    elif model_class.__name__ == 'VideoTransformer':
        param_grid = {
            'num_classes': [2],  # Fixed for binary classification
            'embed_dim': [256, 512],
            'num_heads': [4, 8],
            'num_layers': [2, 4],
            'dropout': [0.1, 0.3],
            'learning_rate': [1e-4, 5e-4]
        }
        
        base_params = {
            'num_classes': 2,
            'use_pose': False
        }
        
        model_type = 'transformer'
    
    elif model_class.__name__ == 'SlowFastNetwork':
        param_grid = {
            'num_classes': [2],  # Fixed for binary classification
            'alpha': [4, 8],  # Speed ratio options
            'beta': [1/8, 1/4],  # Channel ratio options
            'dropout_prob': [0.3, 0.5],
            'fusion_places': [['res2', 'res3', 'res4', 'res5'], ['res3', 'res4', 'res5']]
        }
        
        base_params = {
            'num_classes': 2,
            'pretrained': True
        }
        
        model_type = 'slowfast'
        
    elif model_class.__name__ == 'R2Plus1DNet':
        param_grid = {
            'num_classes': [2],  # Fixed for binary classification
            'dropout_prob': [0.3, 0.5, 0.7],
            'frozen_layers': [None, ['stem'], ['stem', 'layer1']]
        }
        
        base_params = {
            'num_classes': 2,
            'pretrained': True
        }
        
        model_type = 'r2plus1d'
        
    elif model_class.__name__ == 'TwoStreamNetwork':
        param_grid = {
            'num_classes': [2],  # Fixed for binary classification
            'spatial_weight': [0.8, 1.0, 1.2],
            'temporal_weight': [1.2, 1.5, 1.8],
            'dropout_prob': [0.3, 0.5],
            'fusion': ['late', 'conv']
        }
        
        base_params = {
            'num_classes': 2,
            'pretrained': True,
            'spatial_backbone': 'r3d_18'
        }
        
        model_type = 'two_stream'
        
    else:
        raise ValueError(f"Unknown model class: {model_class.__name__}")
    
    # Run grid search
    return grid_search(
        model_class, 
        train_paths, train_labels, 
        val_paths, val_labels,
        param_grid, 
        base_params,
        output_dir=output_dir,
        model_type=model_type
    )

def sequential_hyperparameter_search(model_classes, train_paths, train_labels, val_paths, val_labels,
                                        param_grids, base_params=None, output_dir="./hyperparam_search",
                                        device=torch.device("cuda"), num_epochs=10):
        """
        Run hyperparameter search for multiple models sequentially
        
        Args:
            model_classes: Dictionary mapping model names to model classes
            train_paths, val_paths: Lists of video paths for training and validation
            train_labels, val_labels: Lists of labels for training and validation
            param_grids: Dictionary mapping model names to parameter grids
            base_params: Dictionary mapping model names to base parameters
            output_dir: Directory to save results
            device: Device to use for training
            num_epochs: Number of epochs to train each model
            
        Returns:
            Dictionary with best parameters for each model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        from datetime import datetime
        
        # Initialize results summary
        all_results = {}
        
        # Log start of hyperparameter search
        with open(os.path.join(output_dir, 'search_summary.txt'), 'w') as f:
            f.write(f"Sequential Hyperparameter Search started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models to test: {', '.join(model_classes.keys())}\n\n")
        
        # Create base_params if None
        if base_params is None:
            base_params = {model_name: {} for model_name in model_classes.keys()}
        
        # Run search for each model
        for model_name, model_class in model_classes.items():
            print(f"\n{'='*80}")
            print(f"Starting hyperparameter search for {model_name}")
            print(f"{'='*80}")
            
            # Get parameter grid and base parameters for this model
            param_grid = param_grids.get(model_name, {})
            base_param = base_params.get(model_name, {})
            
            if not param_grid:
                print(f"No parameter grid defined for {model_name}, skipping...")
                continue
            
            # Create model-specific output directory
            model_output_dir = os.path.join(output_dir, model_name)
            
            # Determine model type for dataloader configuration
            if "3d_cnn" in model_name.lower():
                model_type = "3d_cnn"
            elif "lstm" in model_name.lower():
                model_type = "2d_cnn_lstm"
            elif "transformer" in model_name.lower():
                model_type = "transformer"
            else:
                model_type = "3d_cnn"  # Default
            
            # Run grid search for this model
            try:
                start_time = datetime.now()
                print(f"Starting search at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                results = grid_search(
                    model_class=model_class,
                    train_paths=train_paths,
                    train_labels=train_labels,
                    val_paths=val_paths,
                    val_labels=val_labels,
                    param_grid=param_grid,
                    base_params=base_param,
                    device=device,
                    output_dir=model_output_dir,
                    num_epochs=num_epochs,
                    model_type=model_type
                )
                
                end_time = datetime.now()
                duration = end_time - start_time
                
                # Store results
                all_results[model_name] = {
                    'best_params': results['best_params'],
                    'best_auc': results['best_auc'],
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': str(duration)
                }
                
                # Log results
                with open(os.path.join(output_dir, 'search_summary.txt'), 'a') as f:
                    f.write(f"Results for {model_name}:\n")
                    f.write(f"  Duration: {duration}\n")
                    f.write(f"  Best AUC: {results['best_auc']:.4f}\n")
                    f.write(f"  Best parameters: {results['best_params']}\n\n")
                
            except Exception as e:
                print(f"Error during hyperparameter search for {model_name}: {e}")
                # Log error
                with open(os.path.join(output_dir, 'search_summary.txt'), 'a') as f:
                    f.write(f"Error during hyperparameter search for {model_name}: {str(e)}\n\n")
        
        # Save all results to a single JSON file
        with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Log completion
        with open(os.path.join(output_dir, 'search_summary.txt'), 'a') as f:
            f.write(f"\nSequential Hyperparameter Search completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Find best overall model
            if all_results:
                best_model = max(all_results.items(), key=lambda x: x[1]['best_auc'])
                f.write(f"\nBest overall model: {best_model[0]} with AUC {best_model[1]['best_auc']:.4f}\n")
        
        return all_results
    
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run sequential hyperparameter search for multiple models")
    parser.add_argument("--data_dir", type=str, default="./Data/Processed/standardized",
                      help="Directory containing videos")
    parser.add_argument("--output_dir", type=str, default="./hyperparam_search",
                      help="Directory to save search results")
    parser.add_argument("--gpu", type=int, default=0,
                      help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--num_epochs", type=int, default=10,
                      help="Number of epochs for each trial")
    parser.add_argument("--model_types", nargs="+", 
                      default=['3d_cnn', '2d_cnn_lstm', 'transformer'],
                      help="Model types to search")
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    
    # Prepare data
    train_paths, train_labels, val_paths, val_labels, _, _ = prepare_violence_nonviolence_data(args.data_dir)
    
    # Use subset of data for faster search
    subset_size = min(len(train_paths) // 4, 100)  # Use at most 100 samples
    train_paths_subset = train_paths[:subset_size]
    train_labels_subset = train_labels[:subset_size]
    
    # Define model classes
    from Models.model_3dcnn import Model3DCNN
    from Models.model_2dcnn_lstm import Model2DCNNLSTM
    from Models.model_transformer import VideoTransformer
    from Models.model_i3d import TransferLearningI3D
    from Models.model_slowfast import SlowFastNetwork
    from Models.model_r2plus1d import R2Plus1DNet
    
    model_classes = {
        '3d_cnn': Model3DCNN,
        '2d_cnn_lstm': Model2DCNNLSTM,
        'transformer': VideoTransformer,
        'i3d': TransferLearningI3D,
        'slowfast': SlowFastNetwork,
        'r2plus1d': R2Plus1DNet
    }
    
    # Filter model classes based on args.model_types
    selected_model_classes = {k: v for k, v in model_classes.items() if k in args.model_types}
    
    # Define parameter grids for each model
    param_grids = {
        '3d_cnn': {
            'dropout_prob': [0.3, 0.5, 0.7],
            'use_pose': [False, True]
        },
        '2d_cnn_lstm': {
            'lstm_hidden_size': [256, 512],
            'lstm_num_layers': [1, 2],
            'dropout_prob': [0.3, 0.5],
            'use_pose': [False, True]
        },
        'transformer': {
            'embed_dim': [256, 512],
            'num_heads': [4, 8],
            'num_layers': [2, 4],
            'dropout': [0.1, 0.3]
        },
        'i3d': {
            'dropout_prob': [0.3, 0.5, 0.7],
            'use_pose': [False, True]
        },
        'slowfast': {
            'alpha': [4, 8],
            'beta': [1/8, 1/4],
            'dropout_prob': [0.3, 0.5]
        },
        'r2plus1d': {
            'dropout_prob': [0.3, 0.5, 0.7],
            'frozen_layers': [None, ['stem'], ['stem', 'layer1']]
        }
    }
    
    # Run sequential search
    results = sequential_hyperparameter_search(
        model_classes=selected_model_classes,
        train_paths=train_paths_subset,
        train_labels=train_labels_subset,
        val_paths=val_paths,
        val_labels=val_labels,
        param_grids={k: v for k, v in param_grids.items() if k in args.model_types},
        output_dir=args.output_dir,
        device=device,
        num_epochs=args.num_epochs
    )
    
    # Print summary
    print("\nHyperparameter search complete!")
    print(f"Results saved to {args.output_dir}")
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Best AUC: {result['best_auc']:.4f}")
        print(f"  Best parameters: {result['best_params']}")
        print(f"  Duration: {result['duration']}")

if __name__ == "__main__":
    main()