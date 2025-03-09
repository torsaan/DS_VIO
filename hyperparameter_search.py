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
        _, _, _, _, _, metrics_dict = evaluate_model(
            trained_model, val_loader, criterion, device
        )
        
        # Save metrics
        combination_result = {
            'params': {name: value for name, value in zip(param_names, combination)},
            'metrics': metrics_dict
        }
        results.append(combination_result)
        
        # Check if this is the best model
        val_auc = metrics_dict['roc_auc']
        if val_auc > best_auc:
            best_auc = val_auc
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

# Example usage in main.py:
"""
from hyperparameter_search import get_best_hyperparameters
from Models.model_3dcnn import Model3DCNN

# Get best hyperparameters for 3D CNN model
best_params = get_best_hyperparameters(
    Model3DCNN,
    train_paths, train_labels,
    val_paths, val_labels,
    output_dir="./output/hyperparam_search_3dcnn"
)

# Use best parameters to initialize the final model
model = Model3DCNN(**best_params['best_params']).to(device)

# Train with the best parameters
trained_model = train_model(
    model_name="3d_cnn_best",
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=num_epochs,
    device=device,
    output_dir=output_dir
)
"""