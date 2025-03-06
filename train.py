# trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from utils.logger import CSVLogger

def train_epoch(model, data_loader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        # Handle different input types (with or without pose data)
        if isinstance(batch, list) and len(batch) == 3:  # Video + Pose + Label
            frames, pose, targets = batch
            frames, pose, targets = frames.to(device), pose.to(device), targets.to(device)
            inputs = (frames, pose)
        elif isinstance(batch, list) and len(batch) == 2:  # Video + Label
            frames, targets = batch
            frames, targets = frames.to(device), targets.to(device)
            inputs = frames
        else:
            raise ValueError("Unexpected batch format")
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct / total
        })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, data_loader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            # Handle different input types (with or without pose data)
            if isinstance(batch, list) and len(batch) == 3:  # Video + Pose + Label
                frames, pose, targets = batch
                frames, pose, targets = frames.to(device), pose.to(device), targets.to(device)
                inputs = (frames, pose)
            elif isinstance(batch, list) and len(batch) == 2:  # Video + Label
                frames, targets = batch
                frames, targets = frames.to(device), targets.to(device)
                inputs = frames
            else:
                raise ValueError("Unexpected batch format")
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Update statistics
            running_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate validation metrics
    val_loss = running_loss / total
    val_acc = 100. * correct / total
    
    return val_loss, val_acc, all_preds, all_targets

def train_model(model_name, model, train_loader, val_loader, num_epochs=30, 
                device=torch.device("cuda"), output_dir="./output"):
    """
    Train a model and save checkpoints
    
    Args:
        model_name: Name of the model (used for saving)
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        device: Device to use for training
        output_dir: Directory to save model checkpoints and logs
        
    Returns:
        Trained model
    """
    # Create model directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Set up logger
    logger = CSVLogger(
        os.path.join(model_dir, 'training_log.csv'),
        fieldnames=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
    )
    
    # Track best model
    best_val_loss = float('inf')
    best_model_path = os.path.join(model_dir, f"best_{model_name}_model.pth")
    
    # Train for specified number of epochs
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        logger.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(model_dir, f"{model_name}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
    
    # Load best model weights
    model.load_state_dict(torch.load(best_model_path))
    
    return model