# hyperparameter_search.py
import os
import itertools
import json
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from train import train_model, clear_cuda_memory
from dataloader import get_dataloaders
from evaluations import evaluate_model
import matplotlib.pyplot as plt
import argparse
from utils.dataprep import prepare_violence_nonviolence_data
from dataloader import get_dataloaders
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
from train import EarlyStopping
import argparse as args

from Models.model_simplecnn import SimpleCNN
from Models.model_2dcnn_lstm import Model2DCNNLSTM
from Models.model_3dcnn import Model3DCNN
from Models.model_transformer import VideoTransformer
from Models.model_slowfast import SlowFastNetwork
from Models.model_r2plus1d import R2Plus1DNet
from Models.violence_cnn_lstm import ViolenceCNNLSTM
from Models.model_two_stream import TwoStreamNetwork
from Models.model_Temporal3DCNN import Temporal3DCNN
from Models.model_i3d import TransferLearningI3D    
from Models.model_hybrid import ModelHybrid


import warnings

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated")
warnings.filterwarnings("ignore", message="The verbose parameter is deprecated")
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    else:
        return obj

def grid_search(model_class, train_paths, train_labels, val_paths, val_labels,
               param_grid, base_params=None, device=torch.device("cuda"),
               output_dir="./hyperparam_search", num_epochs=6,
               batch_size=2, num_workers=4, model_type='3d_cnn'):
    """
    Perform grid search for hyperparameter optimization with memory optimizations
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
    
    # Define model parameters vs optimizer parameters
    optimizer_param_names = ['learning_rate', 'weight_decay', 'momentum', 'beta1', 'beta2']
    
    # Loop through all parameter combinations
    for i, combination in enumerate(tqdm(param_combinations, desc="Hyperparameter Search")):
        # Clear CUDA memory thoroughly
        clear_cuda_memory()
        
        # Create parameter dictionary for this combination
        params = base_params.copy()
        
        # Separate model params from optimizer params
        model_params = {}
        optimizer_params = {}
        
        for name, value in zip(param_names, combination):
            if name in optimizer_param_names:
                optimizer_params[name] = value
            else:
                model_params[name] = value
                
        # Update params with model params
        params.update(model_params)
        
        # Create model name
        model_name = f"model_{i}"
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Log this combination
        param_str = ", ".join([f"{name}={value}" for name, value in zip(param_names, combination)])
        print(f"\nTrying combination {i+1}/{len(param_combinations)}: {param_str}")
        
        try:
            # Create model with current parameters
            model = model_class(**params)
            
            # Enable model specific memory optimizations
            if model_type == 'transformer':
                # For transformer models, enable gradient checkpointing
                if hasattr(model, 'backbone') and hasattr(model.backbone, 'transformer'):
                    model.backbone.transformer.gradient_checkpointing_enable()
                elif hasattr(model, 'temporal_encoder'):
                    # Enable gradient checkpointing for your custom transformer if it exists
                    for layer in model.temporal_encoder.layers:
                        layer.use_checkpoint = True
            
            # Move model to device after optimizations
            model = model.to(device)
            
            # Create dataloaders with current batch size but smaller prefetch factor
            # to reduce memory overhead and dynamic batch sizes for different models
            adjusted_batch_size = batch_size
            
            # Adjust batch size based on model type to prevent OOM
            if model_type == '3d_cnn':
                adjusted_batch_size = max(1, batch_size // 2)  # Halve batch size for 3D CNN
            elif model_type == 'transformer':
                adjusted_batch_size = max(1, batch_size // 2)  # Halve batch size for transformers
            
            train_loader, val_loader, _ = get_dataloaders(
                train_paths, train_labels, 
                val_paths, val_labels,
                val_paths, val_labels,  # Use validation set as test set
                batch_size=adjusted_batch_size,
                num_workers=num_workers,
                prefetch_factor=2,  # Reduce prefetching to save memory
                pin_memory=True,
                model_type=model_type
            )
            
            # Train model with mixed precision and memory optimizations
            trained_model = train_model_with_optimization(
                model_name=model_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                device=device,
                output_dir=output_dir,
                patience=3,  # Use shorter patience for hyperparameter search
                use_amp=True,  # Enable automatic mixed precision
                **optimizer_params  # Pass optimizer params here
            )
            
            # Evaluate model on validation set
            metrics_dict, all_preds, all_targets, all_probs = evaluate_model_with_optimization(model, val_loader, criterion, device, use_amp=True)
            val_loss = metrics_dict.get('val_loss', 0)
            val_acc = metrics_dict.get('val_acc', 0)
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
            results_to_save = convert_ndarray_to_list({
            'results': results,
            'best_params': best_params,
            'best_auc': float(best_auc)
        })

            with open(os.path.join(output_dir, 'grid_search_results.json'), 'w') as f:
                json.dump(results_to_save, f, indent=4)
                
        except Exception as e:
            print(f"Error during hyperparameter search for {model_type}: {str(e)}")
            # Log error in CSV
            log_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'combination_id': i
            }
            for name, value in zip(param_names, combination):
                log_entry[name] = value
            log_entry['val_loss'] = None
            log_entry['val_acc'] = None
            log_entry['roc_auc'] = None
            log_entry['pr_auc'] = None
            logger.log(log_entry)
            
        finally:
            # Always clear memory
            if 'model' in locals():
                del model
            if 'trained_model' in locals():
                del trained_model
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
    if not results:
        print("No valid results to plot")
        return
        
    plt.figure(figsize=(15, 10))
    
    for param_name in param_names:
        # Group results by this parameter
        param_values = []
        auc_values = []
        
        for result in results:
            # Check if this result has valid metrics
            if 'metrics' not in result or 'roc_auc' not in result['metrics']:
                continue
                
            param_values.append(result['params'][param_name])
            auc_values.append(result['metrics']['roc_auc'])
        
        # Only plot if we have data points
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
                             output_dir="./hyperparam_search", num_epochs=6, batch_size=2):
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
            'lstm_num_layers': [2, 3],
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
            'num_classes': 2,
            
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
        
    elif model_class.__name__ == 'ViolenceCNNLSTM':
        param_grid = {
            'num_classes': [2],  # Fixed for binary classification
            'lstm_hidden_size': [256, 512],
            'num_layers': [1, 2, 3],
            'dropout': [0.3, 0.5, 0.7],
            'activation': ['relu', 'gelu'],
            'learning_rate': [1e-4, 5e-4]
        }
        
        base_params = {
            'num_classes': 2
        }
    
        model_type = 'cnn_lstm'
        
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
        
    # Add SimpleCNN support
    elif model_class.__name__ == 'SimpleCNN':
        param_grid = {
            'num_classes': [2],  # Fixed for binary classification
            'dropout_prob': [0.3, 0.5, 0.7]
        }
        
        base_params = {
            'num_classes': 2
            
        }
        
        model_type = 'simple_cnn'
        
    # Add Temporal3DCNN support
    elif model_class.__name__ == 'Temporal3DCNN':
        param_grid = {
            'num_classes': [2],  # Fixed for binary classification
            'learning_rate': [1e-4, 5e-4, 1e-3]
        }
        
        base_params = {
            'num_classes': 2
        }
        
        model_type = 'temporal_3d_cnn'
        
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
        model_type=model_type,
        num_epochs=num_epochs,  # Add these parameters
        batch_size=batch_size
    )

def sequential_hyperparameter_search(model_classes, train_paths, train_labels, val_paths, val_labels,
                                    param_grids, base_params=None, output_dir="./hyperparam_search",
                                    device=torch.device("cuda"), num_epochs=6, batch_size=2):
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
                batch_size=batch_size,  # Add this line
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


def train_model_with_optimization(model_name, model, train_loader, val_loader, num_epochs=6, 
                                  device=torch.device("cuda"), output_dir="./output", 
                                  patience=7, resume_from=None, grad_clip=None, use_amp=True, **kwargs):
    """
    Train a model with memory optimizations including mixed precision training.
    Assumes that batches are in the format (frames, targets) with no pose data.
    """
    from hyperparameters import get_optimizer, get_training_config
    import os
    import torch.cuda.amp as amp
    from train import EarlyStopping  # Ensure EarlyStopping is imported

    # Create model directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr = kwargs.get('learning_rate', 0.0001)
    optimizer_name = kwargs.get('optimizer', 'adam')
    
    optimizer = get_optimizer(
        model, 
        model_type=model_name,
        optimizer_name=optimizer_name,
        lr=lr,
        **kwargs
    )
    
    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, mode='max')
    
    # Set up gradient scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    best_auc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            # Expect batch to be (frames, targets)
            frames, targets = batch
            frames, targets = frames.to(device), targets.to(device)
            inputs = frames
            
            optimizer.zero_grad()
            
            with amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss /= total
        train_acc = 100. * correct / total
        
        # Evaluate on validation set
        metrics_dict, _, _, all_probs = evaluate_model_with_optimization(
            model, val_loader, criterion, device, use_amp=use_amp
        )
        val_loss = metrics_dict.get('val_loss', 0)
        val_acc = metrics_dict.get('val_acc', 0)
        roc_auc = metrics_dict.get('roc_auc', 0)
        pr_auc = metrics_dict.get('pr_auc', 0)
                
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}")
        
        if early_stopping(roc_auc):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
        
        if roc_auc > best_auc:
            best_auc = roc_auc
            checkpoint_path = os.path.join(model_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model with AUC-ROC: {best_auc:.4f}")
    
    best_model_path = os.path.join(model_dir, f"{model_name}_best.pth")
    if os.path.exists(best_model_path):
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        except Exception as e:
            print(f"Warning: Could not load best model: {e}")
        return model


def evaluate_model_with_optimization(model, data_loader, criterion, device, use_amp=True):
    import torch.cuda.amp as amp
    import numpy as np
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = np.array([], dtype=np.int64)
    all_targets = np.array([], dtype=np.int64)
    
    # Determine num_classes from the final classifier layer rather than an arbitrary module
    try:
        num_classes = model.classifier[-1].out_features
    except Exception as e:
        num_classes = 2  # Fallback if not available
    
    all_probs = np.array([]).reshape(0, num_classes)
    
    with torch.no_grad():
        for batch in data_loader:
            # Expect batch to be (frames, targets)
            frames, targets = batch
            frames, targets = frames.to(device), targets.to(device)
            inputs = frames
            
            with amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            val_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            batch_preds = predicted.cpu().numpy()
            batch_targets = targets.cpu().numpy()
            batch_probs = probs.cpu().numpy()
            
            all_preds = np.concatenate([all_preds, batch_preds])
            all_targets = np.concatenate([all_targets, batch_targets])
            all_probs = np.concatenate([all_probs, batch_probs])
            
            # Optionally clear CUDA cache if needed
            if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.9 * torch.cuda.get_device_properties(device).total_memory:
                torch.cuda.empty_cache()
    
    val_loss /= total
    val_acc = 100. * correct / total
    metrics_dict = {
        'val_loss': val_loss,
        'val_acc': val_acc,
        'roc_auc': 0.0, 
        'pr_auc': 0.0,
        'fpr': [],
        'tpr': [],
        'precision': [],
        'recall': []
    }
    
    if len(all_targets) > 0 and len(np.unique(all_targets)) > 1:
        try:
            # For binary classification, use probability of class 1
            positive_probs = all_probs[:, 1] if all_probs.shape[1] >= 2 else all_probs[:, 0]
            fpr, tpr, _ = roc_curve(all_targets, positive_probs)
            roc_auc_value = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(all_targets, positive_probs)
            pr_auc_value = average_precision_score(all_targets, positive_probs)
            metrics_dict.update({
                'roc_auc': roc_auc_value,
                'pr_auc': pr_auc_value,
                'fpr': fpr,
                'tpr': tpr,
                'precision': precision,
                'recall': recall
            })
        except Exception as e:
            print(f"Warning: Could not calculate AUC metrics: {e}")
    else:
        print("Warning: Not enough unique classes to calculate AUC metrics")
    
    return metrics_dict, all_preds, all_targets, all_probs

def clear_cuda_memory():
    """Clear CUDA memory between runs with better error handling"""
    if not torch.cuda.is_available():
        return
        
    try:
        # Get initial memory stats
        initial_allocated = torch.cuda.memory_allocated() / 1e9
        initial_reserved = torch.cuda.memory_reserved() / 1e9
        
        # Delete any existing tensors
        import gc
        gc.collect()
        
        # Reset CUDA device completely (more aggressive)
        torch.cuda.empty_cache()
        # For extreme cases, can consider:
        if torch.cuda.memory_allocated() > 0:  # If memory still allocated
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete
        
        # Print memory savings
        final_allocated = torch.cuda.memory_allocated() / 1e9
        final_reserved = torch.cuda.memory_reserved() / 1e9
        
        print(f"GPU Memory: {initial_allocated:.2f} GB → {final_allocated:.2f} GB allocated, "
              f"{initial_reserved:.2f} GB → {final_reserved:.2f} GB reserved")
    except RuntimeError as e:
        print(f"Warning: Error clearing CUDA memory: {e}")
        print("Continuing with available memory...")

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
    parser.add_argument("--batch_size", type=int, default=2,
                      help="Batch size for training")
    parser.add_argument("--model_types", nargs="+", 
                      default=['2d_cnn_lstm', '3d_cnn', 'transformer'],
                      help="Model types to search")
    parser.add_argument("--use_mixed_precision", action="store_true",
                      help="Use mixed precision training to save memory")
    args = parser.parse_args()
    
    # Configure memory settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    
    # Prepare data
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = prepare_violence_nonviolence_data(args.data_dir)
    
    # Define model classes dictionary
    model_classes = {
        'simple_cnn': SimpleCNN,
        '2d_cnn_lstm': Model2DCNNLSTM,
        '3d_cnn': Model3DCNN,
        'transformer': VideoTransformer,
        'slowfast': SlowFastNetwork,
        'r2plus1d': R2Plus1DNet,
        'cnn_lstm': ViolenceCNNLSTM,
        'two_stream': TwoStreamNetwork,
        'temporal_3d_cnn': Temporal3DCNN,
        'i3d': TransferLearningI3D,
        'hybrid': ModelHybrid
    }
    
    # Filter model classes based on args.model_types
    selected_model_classes = {k: v for k, v in model_classes.items() if k in args.model_types}
    
    # Create parameter grids with memory-optimized values - model specific parameters only!
    param_grids = {
        'simple_cnn': {
            'dropout_prob': [0.3, 0.5]
        },
        '3d_cnn': {
            'dropout_prob': [0.3, 0.5]
        },
        '2d_cnn_lstm': {
            'lstm_hidden_size': [256],
            'lstm_num_layers': [1],
            'dropout_prob': [0.3, 0.5]
        },
        'transformer': {
            'embed_dim': [128],
            'num_heads': [2],
            'num_layers': [1],
            'dropout': [0.1, 0.3],
        
        },
        'cnn_lstm': {
            'lstm_hidden_size': [256],
            'num_layers': [1],
            'dropout': [0.3, 0.5]
        },
        'slowfast': {
            'alpha': [4],
            'beta': [1/8],
            'dropout_prob': [0.3]
        },
        'r2plus1d': {
            'dropout_prob': [0.3, 0.5]
        },
        'two_stream': {
            'spatial_weight': [0.8],
            'temporal_weight': [1.2],
            'dropout_prob': [0.3]
        },
        'temporal_3d_cnn': {
            # Minimal params for testing
        },
        'i3d': {
            'dropout_prob': [0.3]
        },
        'hybrid': {
            'dropout': [0.3]
        }
    }
    
    # Create base params - common parameters for initialization
    base_params = {
        'simple_cnn': {'num_classes': 2},
        '3d_cnn': {'num_classes': 2, 'pretrained': True},
        '2d_cnn_lstm': {'num_classes': 2, 'pretrained': True},
        'transformer': {'num_classes': 2}, 
        'cnn_lstm': {'num_classes': 2},
        'slowfast': {'num_classes': 2, 'pretrained': True},
        'r2plus1d': {'num_classes': 2, 'pretrained': True},
        'two_stream': {'num_classes': 2, 'pretrained': True},
        'temporal_3d_cnn': {'num_classes': 2},
        'i3d': {'num_classes': 2},
        'hybrid': {'num_classes': 2}
    }
    
    # Run sequential search
    results = sequential_hyperparameter_search(
        model_classes=selected_model_classes,
        train_paths=train_paths[:100],  # Use subset for faster search
        train_labels=train_labels[:100],
        val_paths=val_paths[:50],
        val_labels=val_labels[:50],
        param_grids={k: v for k, v in param_grids.items() if k in args.model_types},
        base_params={k: v for k, v in base_params.items() if k in args.model_types},
        output_dir=args.output_dir,
        device=device,
        num_epochs=args.num_epochs
    )
    
    print("\nHyperparameter search complete!")
    print(f"Results saved to {args.output_dir}")

# Call the main function if script is executed directly
if __name__ == "__main__":
    main()