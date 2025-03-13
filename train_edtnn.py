# train_edtnn.py
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.dataprep import prepare_violence_nonviolence_data
from dataloader import get_dataloaders
from train import clear_cuda_memory
from evaluations import evaluate_model, generate_metrics_report, plot_confusion_matrix
from Models.model_edtnn import ModelEDTNN, ResonanceLoss

def train_edtnn(model, train_loader, val_loader, device, num_epochs=30, 
              learning_rate=0.0001, resonance_weight=0.1, output_dir="./output/edtnn",
              patience=7):
    """
    Train the ED-TNN model for violence detection.
    
    Args:
        model: The ED-TNN model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cpu or cuda)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        resonance_weight: Weight for resonance loss component
        output_dir: Directory to save model checkpoints and results
        patience: Number of epochs for early stopping
        
    Returns:
        Trained model and training history
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up loss function with resonance component
    criterion = ResonanceLoss(
        model.topology,
        base_criterion=nn.CrossEntropyLoss(),
        resonance_weight=resonance_weight
    )
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    # Training loop
    best_val_acc = 0.0
    no_improve_count = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Handle different input types (with or without pose data)
            if model.use_pose and len(batch) == 3:  # Video + Pose + Label
                frames, pose, targets = batch
                frames, pose, targets = frames.to(device), pose.to(device), targets.to(device)
                inputs = (frames, pose)
            else:  # Video + Label
                frames, targets = batch
                frames, targets = frames.to(device), targets.to(device)
                inputs = frames
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets, model.entangled_layer)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / total
        epoch_train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Handle different input types (with or without pose data)
                if model.use_pose and len(batch) == 3:  # Video + Pose + Label
                    frames, pose, targets = batch
                    frames, pose, targets = frames.to(device), pose.to(device), targets.to(device)
                    inputs = (frames, pose)
                else:  # Video + Label
                    frames, targets = batch
                    frames, targets = frames.to(device), targets.to(device)
                    inputs = frames
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = criterion(outputs, targets, model.entangled_layer)
                
                # Calculate softmax probabilities
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                # Update statistics
                val_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store predictions and targets for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())



                # Calculate validation metrics
        epoch_val_loss = val_loss / total
        epoch_val_acc = 100. * correct / total
        
        # Calculate ROC AUC score if possible
        try:
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(all_targets, np.array(all_probs)[:, 1])
        except:
            val_auc = 0.0
        
        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)
        
        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_auc'].append(val_auc)
        
        # Print epoch summary
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        print(f"Val AUC: {val_auc:.4f}")
        
        # Save checkpoint if this is the best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            checkpoint_path = os.path.join(output_dir, "edtnn_best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': best_val_acc,
                'topology_type': model.topology.knot_type,
                'node_density': model.topology.node_density
            }, checkpoint_path)
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Save topology visualization at specific epochs
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            fig = model.topology.visualize_topology()
            fig.savefig(os.path.join(output_dir, f"topology_epoch_{epoch+1}.png"))
            plt.close(fig)
            
            # Visualize node tensions
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            node_positions = np.array(model.topology.nodes)
            tensions = model.entangled_layer.knot_tension.detach().cpu().numpy()
            scatter = ax.scatter(
                node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
                c=tensions, cmap='viridis', s=60
            )
            ax.set_title(f"Node Tensions at Epoch {epoch+1}")
            fig.colorbar(scatter, ax=ax, label="Tension")
            fig.savefig(os.path.join(output_dir, f"tensions_epoch_{epoch+1}.png"))
            plt.close(fig)
        
        # Check for early stopping
        if no_improve_count >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
        
        # Clear CUDA cache
        clear_cuda_memory()
    
    # Load the best model for return
    checkpoint = torch.load(os.path.join(output_dir, "edtnn_best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    # Plot AUC
    plt.subplot(1, 3, 3)
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('Validation AUC-ROC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()
    
    return model, history


def evaluate_edtnn(model, test_loader, device, output_dir="./output/edtnn"):
    """
    Evaluate the ED-TNN model on test data.
    
    Args:
        model: The trained ED-TNN model
        test_loader: DataLoader for test data
        device: Device to use for evaluation
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Use existing evaluation function from your codebase
    result = evaluate_model(model, test_loader, criterion, device)
    
    # Extract results
    if len(result) == 5:
        test_loss, test_acc, all_preds, all_targets, all_probs = result
        try:
            from sklearn.metrics import roc_auc_score
            roc_auc = roc_auc_score(all_targets, all_probs[:, 1])
        except Exception as e:
            print(f"Error computing ROC AUC: {e}")
            roc_auc = 0.0
    else:
        test_loss, test_acc, all_preds, all_targets, all_probs, metrics_dict = result
        roc_auc = metrics_dict.get('roc_auc', 0.0)
    
    # Calculate metrics
    report, cm = generate_metrics_report(
        all_preds, all_targets,
        output_path=os.path.join(output_dir, 'edtnn_metrics.json')
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm,
        output_path=os.path.join(output_dir, 'edtnn_confusion_matrix.png')
    )
    
    # Print summary
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs,
        'report': report,
        'confusion_matrix': cm,
        'roc_auc': roc_auc
    }


def main():
    """Main function to train and evaluate ED-TNN model"""
    parser = argparse.ArgumentParser(description="Train ED-TNN for violence detection")
    parser.add_argument("--data_dir", type=str, default="./Data/Processed/standardized",
                       help="Directory containing videos")
    parser.add_argument("--pose_dir", type=str, default=None,
                       help="Directory containing pose keypoints (optional)")
    parser.add_argument("--output_dir", type=str, default="./output/edtnn",
                       help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                       help="Learning rate for optimizer")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--resonance_weight", type=float, default=0.1,
                       help="Weight for resonance loss component")
    parser.add_argument("--patience", type=int, default=7,
                       help="Patience for early stopping")
    parser.add_argument("--knot_type", type=str, default="trefoil",
                       choices=["trefoil", "figure-eight"],
                       help="Type of knot topology to use")
    parser.add_argument("--node_density", type=int, default=64,
                       help="Number of nodes in the topology")
    parser.add_argument("--features_per_node", type=int, default=16,
                       help="Number of features per node")
    parser.add_argument("--collapse_method", type=str, default="entropy",
                       choices=["entropy", "energy", "tension"],
                       help="Method for collapse layer")
    parser.add_argument("--use_pose", action="store_true",
                       help="Use pose data in addition to video")
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        prepare_violence_nonviolence_data(args.data_dir)
    
    # Create dataloaders
    pose_dir = args.pose_dir if args.use_pose else None
    train_loader, val_loader, test_loader = get_dataloaders(
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
        pose_dir=pose_dir,
        batch_size=args.batch_size,
        model_type='3d_cnn'  # Using 3D CNN format for the frames
    )
    
    # Initialize ED-TNN model
    print(f"Initializing ED-TNN model with {args.knot_type} knot topology...")
    model = ModelEDTNN(
        num_classes=2,
        knot_type=args.knot_type,
        node_density=args.node_density,
        features_per_node=args.features_per_node,
        collapse_method=args.collapse_method,
        use_pose=args.use_pose
    ).to(device)
    
    # Display model structure
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Visualize initial topology
    fig = model.topology.visualize_topology()
    fig.savefig(os.path.join(args.output_dir, "initial_topology.png"))
    plt.close(fig)
    
    # Train the model
    print("\nTraining ED-TNN model...")
    trained_model, history = train_edtnn(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        resonance_weight=args.resonance_weight,
        output_dir=args.output_dir,
        patience=args.patience
    )
    
    # Evaluate the model
    print("\nEvaluating ED-TNN model...")
    evaluation_results = evaluate_edtnn(
        trained_model,
        test_loader,
        device,
        output_dir=args.output_dir
    )
    
    # Print final results
    print("\nTraining completed!")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"Best validation AUC: {max(history['val_auc']):.4f}")
    print(f"Test accuracy: {evaluation_results['test_accuracy']:.2f}%")
    print(f"Test AUC: {evaluation_results['roc_auc']:.4f}")
    
    # Save results summary
    summary = {
        'test_accuracy': float(evaluation_results['test_accuracy']),
        'test_auc': float(evaluation_results['roc_auc']),
        'best_val_accuracy': float(max(history['val_acc'])),
        'best_val_auc': float(max(history['val_auc'])),
        'model_parameters': {
            'knot_type': args.knot_type,
            'node_density': args.node_density,
            'features_per_node': args.features_per_node,
            'collapse_method': args.collapse_method,
            'use_pose': args.use_pose
        }
    }
    
    import json
    with open(os.path.join(args.output_dir, "results_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()# train_edtnn.py
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.dataprep import prepare_violence_nonviolence_data
from dataloader import get_dataloaders
from train import clear_cuda_memory
from evaluations import evaluate_model, generate_metrics_report, plot_confusion_matrix
from Models.model_edtnn import ModelEDTNN, ResonanceLoss

def train_edtnn(model, train_loader, val_loader, device, num_epochs=30, 
              learning_rate=0.0001, resonance_weight=0.1, output_dir="./output/edtnn",
              patience=7):
    """
    Train the ED-TNN model for violence detection.
    
    Args:
        model: The ED-TNN model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cpu or cuda)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        resonance_weight: Weight for resonance loss component
        output_dir: Directory to save model checkpoints and results
        patience: Number of epochs for early stopping
        
    Returns:
        Trained model and training history
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up loss function with resonance component
    criterion = ResonanceLoss(
        model.topology,
        base_criterion=nn.CrossEntropyLoss(),
        resonance_weight=resonance_weight
    )
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    # Training loop
    best_val_acc = 0.0
    no_improve_count = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Handle different input types (with or without pose data)
            if model.use_pose and len(batch) == 3:  # Video + Pose + Label
                frames, pose, targets = batch
                frames, pose, targets = frames.to(device), pose.to(device), targets.to(device)
                inputs = (frames, pose)
            else:  # Video + Label
                frames, targets = batch
                frames, targets = frames.to(device), targets.to(device)
                inputs = frames
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets, model.entangled_layer)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / total
        epoch_train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Handle different input types (with or without pose data)
                if model.use_pose and len(batch) == 3:  # Video + Pose + Label
                    frames, pose, targets = batch
                    frames, pose, targets = frames.to(device), pose.to(device), targets.to(device)
                    inputs = (frames, pose)
                else:  # Video + Label
                    frames, targets = batch
                    frames, targets = frames.to(device), targets.to(device)
                    inputs = frames
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = criterion(outputs, targets, model.entangled_layer)
                
                # Calculate softmax probabilities
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                # Update statistics
                val_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store predictions and targets for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())