# evaluator.py
import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gc

def clear_cuda_memory():
    """Explicitly clear CUDA memory to prevent allocation issues"""
    torch.cuda.empty_cache()
    gc.collect()

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate a single model on test dataset with enhanced metrics
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to use
        
    Returns:
        test_loss, test_acc, all_preds, all_targets, all_probs, metrics_dict
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
            frames, targets = batch
            frames, targets = frames.to(device), targets.to(device)
            inputs = frames
            
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
    
    # Calculate ROC and PR curves
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # For binary classification, use the probability of the positive class
    positive_probs = all_probs[:, 1] if all_probs.shape[1] == 2 else all_probs[:, 0]
    
    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(all_targets, positive_probs)
    roc_auc = auc(fpr, tpr)
    
    # Calculate PR curve
    precision, recall, pr_thresholds = precision_recall_curve(all_targets, positive_probs)
    pr_auc = average_precision_score(all_targets, positive_probs)
    
    metrics_dict = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'precision': precision.tolist(),
        'recall': recall.tolist()
    }
    
    return test_loss, test_acc, all_preds, all_targets, all_probs, metrics_dict

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

def plot_roc_curve(fpr, tpr, roc_auc, output_path=None, title='Receiver Operating Characteristic'):
    """Plot ROC curve and optionally save to file"""
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    # Save to file if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pr_curve(precision, recall, pr_auc, output_path=None, title='Precision-Recall Curve'):
    """Plot Precision-Recall curve and optionally save to file"""
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="upper right")
    
    # Save to file if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def evaluate_and_compare_models(models_dict, test_loaders, device, output_dir="./output"):
    """
    Evaluate and compare multiple models with enhanced metrics
    
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
        test_loss, test_acc, all_preds, all_targets, all_probs, metrics_dict = evaluate_model(
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
        
        # Plot ROC curve
        plot_roc_curve(
            metrics_dict['fpr'], metrics_dict['tpr'], metrics_dict['roc_auc'],
            output_path=os.path.join(model_dir, 'roc_curve.png'),
            title=f'{model_name} - ROC Curve'
        )
        
        # Plot PR curve
        plot_pr_curve(
            metrics_dict['precision'], metrics_dict['recall'], metrics_dict['pr_auc'],
            output_path=os.path.join(model_dir, 'pr_curve.png'),
            title=f'{model_name} - Precision-Recall Curve'
        )
        
        # Store results
        results[model_name] = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs,
            'report': report,
            'confusion_matrix': cm.tolist(),
            'roc_auc': metrics_dict['roc_auc'],
            'pr_auc': metrics_dict['pr_auc']
        }
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"ROC AUC: {metrics_dict['roc_auc']:.4f}")
        print(f"PR AUC: {metrics_dict['pr_auc']:.4f}")
        
        # Clear memory after each model evaluation
        clear_cuda_memory()
    
    # Compare all models
    print("\n" + "="*20 + " Model Comparison " + "="*20)
    print("{:<15} {:<10} {:<10} {:<10}".format("Model", "Accuracy", "ROC AUC", "PR AUC"))
    print("-" * 45)
    
    for model_name, result in results.items():
        print("{:<15} {:<10.2f}% {:<10.4f} {:<10.4f}".format(
            model_name, 
            result['test_accuracy'], 
            result['roc_auc'], 
            result['pr_auc']
        ))
    
    # Find best model based on ROC AUC
    best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
    print(f"\nBest model (by ROC AUC): {best_model[0]} with AUC {best_model[1]['roc_auc']:.4f}")
    
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

def ensemble_predictions(models, test_loaders, device, output_dir=None, weights=None):
    """
    Evaluate models as an ensemble with optional weighting
    
    Args:
        models: Dict of model_name -> model
        test_loaders: Dict of model_name -> dataloader
        device: Computation device
        output_dir: Directory to save results
        weights: Dict of model_name -> weight (optional)
    """
    all_targets = []
    all_model_probs = {}
    
    # Process each model separately
    for model_name, model in models.items():
        test_loader = test_loaders[model_name]
        model.eval()
        model_probs = []
        
        print(f"Getting predictions from {model_name}...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Ensemble - {model_name}"):
                # Handle different batch structures based on model type
                if model_name == 'two_stream':
                    # For two_stream model, batch contains RGB frames, optical flow, and labels
                    rgb_frames, optical_flow, targets = batch
                    inputs = (rgb_frames.to(device), optical_flow.to(device))
                else:
                    # For other models, batch contains frames and labels
                    frames, targets = batch
                    inputs = frames.to(device)
                
                # Store targets only once (from the first model)
                if model_name == list(models.keys())[0]:
                    all_targets.append(targets.cpu().numpy())
                
                # Get model predictions
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                model_probs.append(probs.cpu().numpy())
        
        # Concatenate all batches
        all_model_probs[model_name] = np.concatenate(model_probs, axis=0)
        
        # Clear GPU memory after each model
        torch.cuda.empty_cache()
    
    # Combine all targets
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Define weights based on performance if not provided
    if weights is None:
        # Option 1: Equal weighting (current approach)
        weights = {model_name: 1.0 for model_name in models.keys()}
        
        # Option 2: Auto-compute weights based on validation accuracy
        # This requires having validation accuracy for each model
        # weights = {model_name: model_val_accuracies[model_name] for model_name in models.keys()}
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    normalized_weights = {k: v/total_weight for k, v in weights.items()}
    
    # Apply weighted averaging
    ensemble_probs = np.zeros_like(all_model_probs[list(models.keys())[0]])
    for model_name in models.keys():
        ensemble_probs += all_model_probs[model_name] * normalized_weights[model_name]
    
    # Calculate metrics
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    accuracy = 100 * np.mean(ensemble_preds == all_targets)
    
    # Calculate ROC and PR curves
    if ensemble_probs.shape[1] == 2:  # Binary classification
        roc_auc = roc_auc_score(all_targets, ensemble_probs[:, 1])
        precision, recall, _ = precision_recall_curve(all_targets, ensemble_probs[:, 1])
        pr_auc = auc(recall, precision)
    else:
        # For multi-class, use one-vs-rest approach
        roc_auc = roc_auc_score(all_targets, ensemble_probs, multi_class='ovr')
        pr_auc = 0.0  # Not easily calculated for multi-class
    
    # Save results if output_dir provided
    if output_dir:
        ensemble_dir = os.path.join(output_dir, 'ensemble')
        os.makedirs(ensemble_dir, exist_ok=True)
        
        # Rest of the function remains the same...
        
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'predictions': ensemble_preds,
        'probabilities': ensemble_probs,
        'targets': all_targets
    }