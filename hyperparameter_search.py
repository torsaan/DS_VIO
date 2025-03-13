# hyperparameter_search.py
import os
import itertools
import json
import torch
import torch.optim as optim
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
    
    # Always ensure pose is disabled
    base_params['use_pose'] = False
    
    # Extract training specific parameters from param_grid
    train_param_names = ['learning_rate', 'weight_decay']
    model_param_names = [name for name in param_grid.keys() if name not in train_param_names]
    
    # Separate grids
    train_param_grid = {name: param_grid[name] for name in train_param_names if name in param_grid}
    model_param_grid = {name: param_grid[name] for name in model_param_names}
    
    # Get all parameter combinations
    model_param_names = list(model_param_grid.keys())
    model_param_values = list(model_param_grid.values())
    model_combinations = list(itertools.product(*model_param_values))
    
    train_param_names = list(train_param_grid.keys())
    train_param_values = list(train_param_grid.values())
    train_combinations = list(itertools.product(*train_param_values))
    
    # Combine all parameters for complete combinations
    all_combinations = []
    for model_combo in model_combinations:
        for train_combo in train_combinations:
            all_combinations.append((model_combo, train_combo))
    
    # Initialize results storage
    results = []
    best_auc = 0.0
    best_model_params = None
    best_train_params = None
    
    # Set up criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    # Loop through all parameter combinations
    for i, (model_combo, train_combo) in enumerate(tqdm(all_combinations, desc="Hyperparameter Search")):
        # Clear CUDA memory
        clear_cuda_memory()
        
        # Create parameter dictionaries for this combination
        model_params = base_params.copy()
        for name, value in zip(model_param_names, model_combo):
            model_params[name] = value
        
        train_params = {}
        for name, value in zip(train_param_names, train_combo):
            train_params[name] = value
        
        # Create model name
        model_name = f"model_{i}"
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Log this combination
        model_str = ", ".join([f"{name}={value}" for name, value in zip(model_param_names, model_combo)])
        train_str = ", ".join([f"{name}={value}" for name, value in zip(train_param_names, train_combo)])
        print(f"\nTrying combination {i+1}/{len(all_combinations)}:")
        print(f"  Model params: {model_str}")
        print(f"  Train params: {train_str}")
        
        # Create model with current parameters
        model = model_class(**model_params).to(device)
        
        # Create optimizer with current parameters
        optimizer = None
        if 'learning_rate' in train_params:
            lr = train_params['learning_rate']
            weight_decay = train_params.get('weight_decay', 0)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
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
            patience=3,  # Use shorter patience for hyperparameter search
            optimizer=optimizer  # Pass custom optimizer if created
        )
        
        # Evaluate model on validation set
        val_loss, val_acc, all_preds, all_targets, all_probs = evaluate_model(
            trained_model, val_loader, criterion, device
        )
        
        # Calculate metrics
        try:
            roc_auc = roc_auc_score(all_targets, all_probs[:, 1])
        except:
            roc_auc = 0.0
            
        metrics_dict = {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'roc_auc': roc_auc
        }
        
        # Save metrics
        combination_result = {
            'model_params': {name: value for name, value in zip(model_param_names, model_combo)},
            'train_params': {name: value for name, value in zip(train_param_names, train_combo)},
            'metrics': metrics_dict
        }
        results.append(combination_result)
        
        # Check if this is the best model
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model_params = {name: value for name, value in zip(model_param_names, model_combo)}
            best_train_params = {name: value for name, value in zip(train_param_names, train_combo)}
            print(f"New best parameters found with AUC {best_auc:.4f}")
        
        # Save current results
        with open(os.path.join(output_dir, 'grid_search_results.json'), 'w') as f:
            json.dump({
                'results': results,
                'best_model_params': best_model_params,
                'best_train_params': best_train_params,
                'best_auc': best_auc
            }, f, indent=4)
        
        # Clear memory
        del model, trained_model
        clear_cuda_memory()
    
    # Plot results
    plot_grid_search_results(results, model_param_names + train_param_names, output_dir)
    
    # Combine best parameters
    best_params = {}
    if best_model_params:
        best_params.update(best_model_params)
    if best_train_params:
        best_params.update(best_train_params)
    
    return {
        'results': results,
        'best_params': best_params,
        'best_model_params': best_model_params, 
        'best_train_params': best_train_params,
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
            # Check if parameter is in model_params or train_params
            if param_name in result.get('model_params', {}):
                param_values.append(result['model_params'][param_name])
            elif param_name in result.get('train_params', {}):
                param_values.append(result['train_params'][param_name])
            else:
                continue
                
            auc_values.append(result['metrics']['roc_auc'])
        
        # Skip if no values found
        if not param_values:
            continue
            
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
            'learning_rate': [1e-4, 5e-4]
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
            'num_classes': 2
        }
        
        model_type = 'transformer'
    
    elif model_class.__name__ == 'SlowFastNetwork':
        param_grid = {
            'num_classes': [2],  # Fixed for binary classification
            'alpha': [4, 8],  # Speed ratio options
            'beta': [1/8, 1/4],  # Channel ratio options
            'dropout_prob': [0.3, 0.5],
            'learning_rate': [1e-4, 5e-4]
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
            'learning_rate': [1e-4, 5e-4, 1e-3],
            'weight_decay': [1e-4, 1e-5]
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
            'learning_rate': [1e-4, 5e-4],
            'fusion': ['late', 'conv']
        }
        
        base_params = {
            'num_classes': 2,
            'pretrained': True,
            'spatial_backbone': 'r3d_18'
        }
        
        model_type = 'two_stream'
        
    elif model_class.__name__ == 'TransferLearningI3D':
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
        
        model_type = 'i3d'
        
    elif model_class.__name__ == 'ModelHybrid':
        param_grid = {
            'num_classes': [2],  # Fixed for binary classification
            'learning_rate': [1e-4, 5e-4, 1e-3],
            'weight_decay': [1e-4, 1e-5]
        }
        
        base_params = {
            'num_classes': 2
        }
        
        model_type = 'hybrid'
        
    elif model_class.__name__ == 'ModelEDTNN':
        param_grid = {
            'num_classes': [2],  # Fixed for binary classification
            'knot_type': ['trefoil', 'figure-eight'],
            'node_density': [32, 64, 128],
            'features_per_node': [8, 16, 32],
            'collapse_method': ['entropy', 'energy', 'tension'],
            'learning_rate': [1e-4, 5e-4],
            'weight_decay': [1e-5]
        }
        
        base_params = {
            'num_classes': 2,
            'pretrained': True
        }
        
        model_type = 'edtnn'
    
    else:
        raise ValueError(f"Unknown model class: {model_class.__name__}")
    
    # Run grid search
    results = grid_search(
        model_class, 
        train_paths, train_labels, 
        val_paths, val_labels,
        param_grid, 
        base_params,
        output_dir=output_dir,
        model_type=model_type
    )
    
    # Also save the best model with best parameters for ready use
    if results['best_params']:
        print("\nTraining final model with best parameters...")
        
        # Create model with best parameters
        model_params = base_params.copy()
        for param, value in results['best_model_params'].items():
            model_params[param] = value
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        final_model = model_class(**model_params).to(device)
        
        # Create optimizer with best parameters
        lr = results['best_train_params'].get('learning_rate', 0.0001)
        weight_decay = results['best_train_params'].get('weight_decay', 0)
        optimizer = optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Get full dataloaders for final training
        train_loader, val_loader, _ = get_dataloaders(
            train_paths, train_labels, 
            val_paths, val_labels,
            val_paths, val_labels,
            batch_size=8,
            num_workers=4,
            model_type=model_type
        )
        
        # Train with best parameters for more epochs
        best_model = train_model(
            model_name=f"best_{model_class.__name__}",
            model=final_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=20,  # More epochs for final model
            device=device,
            output_dir=output_dir,
            optimizer=optimizer
        )
        
        # Save final model
        torch.save(best_model.state_dict(), os.path.join(output_dir, f"final_best_{model_class.__name__}.pth"))
        
        # Save best parameters to a separate file for easy loading
        with open(os.path.join(output_dir, f"best_params_{model_class.__name__}.json"), 'w') as f:
            json.dump({
                'model_params': results['best_model_params'],
                'train_params': results['best_train_params']
            }, f, indent=4)
    
    return results