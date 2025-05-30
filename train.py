# trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from utils.logger import CSVLogger
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import gc
import time
import json
import torch
import gc
import time

class EarlyStopping:
    """Early stopping to stop training when validation performance doesn't improve."""
    def __init__(self, patience=7, min_delta=0, verbose=True, mode='min', baseline=None):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in monitored value to qualify as improvement
            verbose: If True, prints message when early stopping is triggered
            mode: 'min' or 'max' based on whether we want to minimize or maximize the monitored value
            baseline: Baseline value for the monitored metric. Training will stop if the model doesn't show
                      improvement over this baseline
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf if mode == 'min' else -np.inf
        
    def __call__(self, val_loss):
        score = -val_loss if self.mode == 'min' else val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose and self.best_score is not None:
                improvement = score - self.best_score
                print(f'Validation metric improved ({self.best_score:.5f} --> {score:.5f}), improvement: {improvement:.5f}')
            self.best_score = score
            self.counter = 0
            
        return self.early_stop

def clear_cuda_memory():
    """Clear CUDA cache to prevent memory issues between model training"""

    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        # Print memory stats before clearing
        allocated_before = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        reserved_before = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
        
        # Empty cache
        torch.cuda.empty_cache()
        
        # Run garbage collector
        gc.collect()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        # Try IPC collect if available (new in PyTorch 1.9+)
        try:
            torch.cuda.ipc_collect()
        except (AttributeError, RuntimeError):
            pass
        
        # Wait a moment to ensure memory is freed
        time.sleep(1)
        
        # Print memory stats after clearing
        allocated_after = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        reserved_after = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
        
        print(f"GPU Memory: {allocated_before:.3f}GB → {allocated_after:.3f}GB allocated, "
              f"{reserved_before:.3f}GB → {reserved_after:.3f}GB reserved")

def train_epoch(model, data_loader, optimizer, criterion, device, scheduler=None, grad_clip=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        # Handle both 2-tuple (frames, targets) and 3-tuple (frames, flow, targets)
        if len(batch) == 3:
            frames, flow, targets = batch
            frames, flow, targets = frames.to(device), flow.to(device), targets.to(device)
            inputs = (frames, flow)
        else:
            frames, targets = batch
            frames, targets = frames.to(device), targets.to(device)
            inputs = frames
        
        optimizer.zero_grad()
        
        # Use mixed precision for faster training and memory efficiency
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        loss.backward()
        
        # Apply gradient clipping for stability
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        optimizer.step()
        
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        running_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct / total,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device, model_type='3d_cnn'):
    """
    Validate model on validation set
    """
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    total = 0
    correct = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            # Handle different batch structures based on model_type
            if model_type == 'two_stream':
                frames, flow, targets = batch
                frames, flow, targets = frames.to(device), flow.to(device), targets.to(device)
                inputs = (frames, flow)
            else:
                frames, targets = batch
                frames, targets = frames.to(device), targets.to(device)
                inputs = frames
                
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Calculate metrics
            val_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Store predictions and targets for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Store probabilities for ROC/PR curves
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    
    # Calculate metrics
    val_accuracy = 100.0 * correct / total
    val_loss = val_loss / total
    
    metrics = {
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    }
    
    return metrics, all_preds, all_targets, all_probs

def plot_roc_curve(fpr, tpr, roc_auc, epoch, output_dir, model_name):
    """Plot and save ROC curve"""
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (Epoch {epoch})')
    plt.legend(loc="lower right")
    
    # Save plot
    curves_dir = os.path.join(output_dir, model_name, 'curves')
    os.makedirs(curves_dir, exist_ok=True)
    plt.savefig(os.path.join(curves_dir, f'roc_curve_epoch_{epoch}.png'))
    plt.close()

def plot_pr_curve(precision, recall, pr_auc, epoch, output_dir, model_name):
    """Plot and save Precision-Recall curve"""
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (Epoch {epoch})')
    plt.legend(loc="lower left")
    
    # Save plot
    curves_dir = os.path.join(output_dir, model_name, 'curves')
    os.makedirs(curves_dir, exist_ok=True)
    plt.savefig(os.path.join(curves_dir, f'pr_curve_epoch_{epoch}.png'))
    plt.close()

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, model_path, is_best=False):
    """Save model checkpoint for resuming training later"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, model_path)
    
    # If this is the best model, save a copy
    if is_best:
        best_path = model_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)

def load_checkpoint(model, optimizer, scheduler, model_path, device):
    """Load a checkpoint to resume training"""
    if not os.path.exists(model_path):
        print(f"Checkpoint {model_path} not found. Starting from scratch.")
        return model, optimizer, scheduler, 0, {}
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    metrics = checkpoint.get('metrics', {})
    
    print(f"Resumed from epoch {checkpoint['epoch']}")
    return model, optimizer, scheduler, start_epoch, metrics

# In train_model function in train.py
def train_model(model_name, model, train_loader, val_loader, num_epochs=None, 
                device=torch.device("cuda"), output_dir="./output", 
                patience=7, resume_from=None, grad_clip=None, **kwargs):
    """
    Train a model and save checkpoints with early stopping and AUC-ROC metrics
    
    Args:
        model_name: Name of the model (used for saving)
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs (default: from hyperparameters)
        device: Device to use for training
        output_dir: Directory to save model checkpoints and logs
        patience: Number of epochs with no improvement after which training will be stopped
        resume_from: Path to checkpoint to resume training from (None for starting from scratch)
        grad_clip: Value for gradient clipping (None to disable)
        **kwargs: Additional parameters for optimizer configuration
        
    Returns:
        Trained model
    """
    # Import hyperparameters
    from hyperparameters import get_optimizer, get_training_config, MODEL_CONFIGS, NUM_EPOCHS
    
    # Use default num_epochs if not specified
    if num_epochs is None:
        num_epochs = NUM_EPOCHS
    
    # Create model directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Clear CUDA memory before starting
    clear_cuda_memory()
    
    # Set up criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Get training config for this model type
    try:
        training_config = get_training_config(model_name)
        optimizer_name = training_config.get('optimizer', 'adam')
        default_lr = training_config.get('lr', 0.0001)
    except ValueError:
        # Fallback if model not found in training config
        optimizer_name = 'adam'
        default_lr = 0.0001
    
    # Override with kwargs if provided
    optimizer_name = kwargs.get('optimizer', optimizer_name)
    lr = kwargs.get('lr', default_lr)
    
    # Create optimizer using the helper function with named parameters
    optimizer = get_optimizer(
        model, 
        model_type=model_name,  # Pass model_name as model_type
        optimizer_name=optimizer_name,
        lr=lr,
        **kwargs
    )
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # Initialize or load from checkpoint
    start_epoch = 0
    if resume_from is not None:
        checkpoint_path = resume_from
        model, optimizer, scheduler, start_epoch, saved_metrics = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device
        )
        
        # After loading the optimizer from checkpoint:
        if 'learning_rate' in kwargs:
            print(f"Updating optimizer learning rate to {kwargs['learning_rate']}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = kwargs['learning_rate']
    
    # Set up logger
    logger = CSVLogger(
        os.path.join(model_dir, 'training_log.csv'),
        fieldnames=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'roc_auc', 'pr_auc', 'lr']
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, mode='max')
    
    # Track best model
    best_auc = 0.0
    best_model_path = os.path.join(model_dir, f"best_{model_name}_model.pth")
    last_model_path = os.path.join(model_dir, f"last_{model_name}_model.pth")
    
    # Train for specified number of epochs
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip=grad_clip
        )
        
        # Validate
        val_metrics, all_preds, all_targets, all_probs = validate(
            model, val_loader, criterion, device, model_type=model_name
        )
        
        # Extract validation metrics
        val_loss = val_metrics['val_loss']
        val_acc = val_metrics['val_acc']
        roc_auc = val_metrics['roc_auc']
        pr_auc = val_metrics['pr_auc']
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}")
        
        logger.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Plot ROC and PR curves
        if epoch % 5 == 0 or epoch == num_epochs - 1:  # Every 5 epochs or last epoch
            plot_roc_curve(
                val_metrics['fpr'], val_metrics['tpr'], roc_auc, 
                epoch + 1, output_dir, model_name
            )
            plot_pr_curve(
                val_metrics['precision'], val_metrics['recall'], pr_auc, 
                epoch + 1, output_dir, model_name
            )
        
        # Save best model based on AUC-ROC
        is_best = roc_auc > best_auc
        if is_best:
            best_auc = roc_auc
            print(f"New best model with AUC-ROC: {best_auc:.4f}")
        
        # Save checkpoint with all training state
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_metrics, 
            last_model_path, is_best
        )
        
        # Check for early stopping based on AUC-ROC
        if early_stopping(roc_auc):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(model_dir, f"{model_name}_epoch{epoch+1}.pth")
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics, 
                checkpoint_path
            )
        
        # Clear CUDA cache after each epoch to prevent memory fragmentation
        clear_cuda_memory()
    
    # Load best model weights
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model with AUC-ROC: {best_auc:.4f}")
    
    return model
def clear_cuda_memory():
    """Clear CUDA cache to prevent memory issues between model training"""
    import torch
    import gc
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run garbage collector
    gc.collect()
    
    
def load_checkpoint(model, optimizer=None, scheduler=None, model_path=None, device=None):
    """Load a checkpoint to resume training"""
    if not model_path or not os.path.exists(model_path):
        print(f"Checkpoint {model_path} not found. Starting from scratch.")
        return model, optimizer, scheduler, 0, {}
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    metrics = checkpoint.get('metrics', {})
    
    print(f"Resumed from epoch {checkpoint.get('epoch', 0)}")
    return model, optimizer, scheduler, start_epoch, metrics