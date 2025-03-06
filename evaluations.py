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
    Evaluate a single model on test dataset
    
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
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
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
            
            # Calculate softmax probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Update statistics
            test_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate test metrics
    test_loss = test_loss / total
    test_acc = 100. * correct / total
    
    return test_loss, test_acc, np.array(all_preds), np.array(all_targets), np.array(all_probs)

def generate_metrics_report(all_preds, all_targets, output_path=None):
    """Generate classification metrics report and optionally save to file"""
    # Calculate metrics
    report = classification_report(
        all_targets, all_preds, 
        target_names=["NonViolence", "Violence"],
        output_dict=True
    )
    
    # Print report
    print("\nClassification Report:")
    print(classification_report(
        all_targets, all_preds, 
        target_names=["NonViolence", "Violence"]
    ))
    
    # Print confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save to file if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
    
    return report, cm

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
    criterion = torch.nn.CrossEntropyLoss()
    results = {}
    
    for model_name, model in models_dict.items():
        print(f"\n{'='*20} Evaluating {model_name} {'='*20}")
        
        # Make sure model is in evaluation mode
        model.eval()
        
        # Get test loader
        test_loader = test_loaders[model_name]
        
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
        results[model_name] = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs,
            'report': report,
            'confusion_matrix': cm.tolist()
        }
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Compare all models
    print("\n" + "="*20 + " Model Comparison " + "="*20)
    print("{:<15} {:<10}".format("Model", "Accuracy"))
    print("-" * 25)
    
    for model_name, result in results.items():
        print("{:<15} {:<10.2f}%".format(model_name, result['test_accuracy']))
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"\nBest model: {best_model[0]} with accuracy {best_model[1]['test_accuracy']:.2f}%")
    
    # Save results summary
    with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
        # Filter out numpy arrays from results
        summary = {
            model_name: {
                k: v for k, v in model_results.items() 
                if k not in ['predictions', 'targets', 'probabilities']
            }
            for model_name, model_results in results.items()
        }
        json.dump(summary, f, indent=4)
    
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
    all_model_preds = []
    all_model_probs = []
    targets = None
    
    # Collect predictions from all models
    for model_name, model in models_dict.items():
        print(f"Getting predictions from {model_name}...")
        
        model.eval()
        test_loader = test_loaders[model_name]
        preds = []
        probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Ensemble - {model_name}"):
                # Handle different input types (with or without pose data)
                if isinstance(batch, list) and len(batch) == 3:  # Video + Pose + Label
                    frames, pose, batch_targets = batch
                    frames, pose = frames.to(device), pose.to(device)
                    inputs = (frames, pose)
                elif isinstance(batch, list) and len(batch) == 2:  # Video + Label
                    frames, batch_targets = batch
                    frames = frames.to(device)
                    inputs = frames
                else:
                    raise ValueError("Unexpected batch format")
                
                # Store targets from first model only
                if targets is None:
                    targets = batch_targets.numpy()
                elif len(batch_targets) > 0:
                    targets = np.concatenate([targets, batch_targets.numpy()])
                
                # Forward pass
                outputs = model(inputs)
                
                # Get predictions and probabilities
                batch_probs = torch.nn.functional.softmax(outputs, dim=1)
                _, batch_preds = outputs.max(1)
                
                preds.extend(batch_preds.cpu().numpy())
                probs.extend(batch_probs.cpu().numpy())
        
        all_model_preds.append(preds)
        all_model_probs.append(probs)
    
    # Convert to numpy arrays
    all_model_preds = np.array(all_model_preds)
    all_model_probs = np.array(all_model_probs)
    
    # Majority voting
    ensemble_preds = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), 
        axis=0, 
        arr=all_model_preds
    )
    
    # Average probabilities
    ensemble_probs = np.mean(all_model_probs, axis=0)
    
    # Calculate accuracy
    ensemble_acc = 100. * (ensemble_preds == targets).mean()
    
    # Generate report
    report, cm = generate_metrics_report(
        ensemble_preds, targets,
        output_path=os.path.join(output_dir, 'ensemble_metrics.json')
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm,
        output_path=os.path.join(output_dir, 'ensemble_confusion_matrix.png')
    )
    
    # Store results
    ensemble_results = {
        'accuracy': ensemble_acc,
        'predictions': ensemble_preds,
        'targets': targets,
        'probabilities': ensemble_probs,
        'report': report,
        'confusion_matrix': cm.tolist()
    }
    
    return ensemble_results