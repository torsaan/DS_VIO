# evaluator.py
import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate a single model on test dataset - Optimized version
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to use
        
    Returns:
        test_loss, test_acc, all_preds, all_targets, all_probs
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # Pre-allocate memory for predictions and targets if dataset size is known
    num_samples = len(test_loader.dataset)
    all_preds = torch.zeros(num_samples, dtype=torch.long)
    all_targets = torch.zeros(num_samples, dtype=torch.long)
    all_probs = torch.zeros(num_samples, 2)  # Assuming binary classification
    
    start_idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Get frames and labels
            frames, targets = batch
            batch_size = targets.size(0)
            frames = frames.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(frames)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Calculate softmax probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Update statistics
            test_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, 1)
            total += batch_size
            correct += (predicted == targets).sum().item()
            
            # Store predictions and targets in pre-allocated tensors
            end_idx = start_idx + batch_size
            all_preds[start_idx:end_idx] = predicted.cpu()
            all_targets[start_idx:end_idx] = targets.cpu()
            all_probs[start_idx:end_idx] = probs.cpu()
            
            start_idx = end_idx
    
    # Calculate test metrics
    test_loss = test_loss / total
    test_acc = 100. * correct / total
    
    # Convert to numpy arrays only once at the end
    return test_loss, test_acc, all_preds.numpy(), all_targets.numpy(), all_probs.numpy()

def generate_metrics_report(all_preds, all_targets, output_path=None):
    """Generate classification metrics report and optionally save to file"""
    # Calculate metrics once to avoid redundant computation
    target_names = ["NonViolence", "Violence"]
    
    # Calculate confusion matrix first (more efficient)
    cm = confusion_matrix(all_targets, all_preds)
    
    # Generate classification report once
    report_str = classification_report(
        all_targets, all_preds,
        target_names=target_names
    )
    
    # Get dict version for JSON output
    report_dict = classification_report(
        all_targets, all_preds,
        target_names=target_names,
        output_dict=True
    )
    
    # Print reports
    print("\nClassification Report:")
    print(report_str)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save to file if output path provided
    if output_path:
        # Use a more efficient approach for file writing
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=4)
        except (IOError, OSError) as e:
            print(f"Warning: Could not save metrics to {output_path}: {e}")
    
    return report_dict, cm

def plot_confusion_matrix(cm, output_path=None):
    """Plot confusion matrix and optionally save to file"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=["NonViolence", "Violence"],
        yticklabels=["NonViolence", "Violence"]
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    # Save to file if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def evaluate_and_compare_models(models_dict, test_loaders, device, output_dir="./output"):
    """
    Evaluate and compare multiple models
    
    Args:
        models_dict: Dictionary of models {model_name: model}
        test_loaders: Dictionary of test loaders {model_name: loader}
        device: Device to use
        output_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation results
    """
    # Create output directory once at the beginning
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize shared objects
    criterion = torch.nn.CrossEntropyLoss()
    results = {}
    
    # Track model performance for easy comparison
    model_accuracies = []
    
    for model_name, model in models_dict.items():
        print(f"\n{'='*20} Evaluating {model_name} {'='*20}")
        
        # Get test loader and ensure model is in eval mode
        test_loader = test_loaders[model_name]
        model.eval()
        
        # Evaluate model
        test_loss, test_acc, all_preds, all_targets, all_probs = evaluate_model(
            model, test_loader, criterion, device
        )
        
        # Create model output directory
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate metrics report
        report, cm = generate_metrics_report(
            all_preds, all_targets,
            output_path=os.path.join(model_dir, 'metrics.json')
        )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            cm,
            output_path=os.path.join(model_dir, 'confusion_matrix.png')
        )
        
        # Store results
        model_accuracies.append((model_name, test_acc))
        results[model_name] = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs,
            'report': report,
            'confusion_matrix': cm.tolist()
        }
        
        # Display immediate results
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Print model comparison table
    model_accuracies.sort(key=lambda x: x[1], reverse=True)
    print("\n" + "="*20 + " Model Comparison " + "="*20)
    print("{:<15} {:<10}".format("Model", "Accuracy"))
    print("-" * 25)
    
    for model_name, acc in model_accuracies:
        print("{:<15} {:<10.2f}%".format(model_name, acc))
    
    # Find best model (already sorted)
    best_model_name, best_acc = model_accuracies[0]
    print(f"\nBest model: {best_model_name} with accuracy {best_acc:.2f}%")
    
    # Save results summary - avoid redundant key lookup and list comprehension
    summary = {}
    for model_name, model_results in results.items():
        summary[model_name] = {k: v for k, v in model_results.items() 
                              if k not in ['predictions', 'targets', 'probabilities']}
    
    # Write json with error handling
    try:
        with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
    except IOError as e:
        print(f"Warning: Failed to write evaluation summary: {e}")
    
    return results

def ensemble_predictions(models_dict, test_loaders, device, output_dir="./output"):
    """
    Create an ensemble of models using majority voting
    
    Args:
        models_dict: Dictionary of models {model_name: model}
        test_loaders: Dictionary of test loaders {model_name: loader}
        device: Device to use
        output_dir: Directory to save results
        
    Returns:
        Dictionary with ensemble results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_model_preds = []
    all_model_probs = []
    targets = None
    
    # Collect predictions from all models
    for model_name, model in models_dict.items():
        print(f"Getting predictions from {model_name}...")
        
        model.eval()
        test_loader = test_loaders[model_name]
        
        # Pre-allocate memory for predictions and probabilities
        num_samples = len(test_loader.dataset)
        model_preds = np.zeros(num_samples, dtype=np.int64)
        model_probs = np.zeros((num_samples, 2), dtype=np.float32)  # Assuming binary classification
        
        # Pre-allocate targets array if not yet done
        if targets is None:
            targets = np.zeros(num_samples, dtype=np.int64)
        
        start_idx = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Ensemble - {model_name}"):
                # Get frames and labels
                frames, batch_targets = batch
                batch_size = batch_targets.size(0)
                frames = frames.to(device, non_blocking=True)
                
                # Store targets (only need to do this once)
                if model_name == list(models_dict.keys())[0]:
                    end_idx = start_idx + batch_size
                    targets[start_idx:end_idx] = batch_targets.numpy()
                
                # Forward pass
                outputs = model(frames)
                
                # Get predictions and probabilities
                batch_probs = torch.nn.functional.softmax(outputs, dim=1)
                _, batch_preds = torch.max(outputs, 1)
                
                # Store in pre-allocated arrays
                end_idx = start_idx + batch_size
                model_preds[start_idx:end_idx] = batch_preds.cpu().numpy()
                model_probs[start_idx:end_idx] = batch_probs.cpu().numpy()
                
                start_idx = end_idx
        
        all_model_preds.append(model_preds)
        all_model_probs.append(model_probs)
    
    # Convert to numpy arrays - already arrays, just stack them
    all_model_preds = np.stack(all_model_preds)
    all_model_probs = np.stack(all_model_probs)
    
    # Majority voting - use optimized approach
    ensemble_preds = np.zeros(num_samples, dtype=np.int64)
    for i in range(num_samples):
        ensemble_preds[i] = np.bincount(all_model_preds[:, i]).argmax()
    
    # Average probabilities
    ensemble_probs = np.mean(all_model_probs, axis=0)
    
    # Calculate accuracy
    correct = (ensemble_preds == targets).sum()
    ensemble_acc = 100. * correct / num_samples
    
    # Generate report
    try:
        report, cm = generate_metrics_report(
            ensemble_preds, targets,
            output_path=os.path.join(output_dir, 'ensemble_metrics.json')
        )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            cm,
            output_path=os.path.join(output_dir, 'ensemble_confusion_matrix.png')
        )
    except Exception as e:
        print(f"Warning: Error generating ensemble metrics: {e}")
        report, cm = {}, np.zeros((2, 2))
    
    # Store results
    ensemble_results = {
        'accuracy': float(ensemble_acc),  # Convert numpy types to native Python types for JSON serialization
        'predictions': ensemble_preds.tolist(),
        'targets': targets.tolist(),
        'probabilities': ensemble_probs.tolist(),
        'report': report,
        'confusion_matrix': cm.tolist()
    }
    
    print(f"Ensemble Accuracy: {ensemble_acc:.2f}%")
    
    return ensemble_results