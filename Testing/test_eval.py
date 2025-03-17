#!/usr/bin/env python3
# Testing/test_evaluation.py
"""
Test script to verify the model evaluation functionality.
Tests metrics calculation, plotting, and comparison features.
"""

import os
import sys
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from evaluations import generate_metrics_report, plot_confusion_matrix, plot_roc_curve, plot_pr_curve
from evaluations import evaluate_model, evaluate_and_compare_models, ensemble_predictions, clear_cuda_memory

def setup_device(gpu_id):
    """Set up computation device (CPU or GPU)"""
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device

def test_metrics_calculation():
    """Test metrics calculation functions with synthetic data"""
    print("\n" + "="*50)
    print("Testing Metrics Calculation")
    print("="*50)
    
    try:
        # Create synthetic predictions and targets
        np.random.seed(42)  # For reproducibility
        all_preds = np.random.randint(0, 2, 100)  # Binary predictions
        all_targets = np.random.randint(0, 2, 100)  # Binary targets
        
        # Make some predictions match targets for reasonable accuracy
        match_ratio = 0.7
        match_indices = np.random.choice(100, int(100 * match_ratio), replace=False)
        all_preds[match_indices] = all_targets[match_indices]
        
        # Generate synthetic probabilities
        all_probs = np.random.random(100)
        # Adjust probabilities to correlate with targets
        all_probs = np.where(all_targets == 1, 0.5 + all_probs * 0.5, all_probs * 0.5)
        
        print("Created synthetic data:")
        print(f"  Predictions array shape: {all_preds.shape}")
        print(f"  Targets array shape: {all_targets.shape}")
        print(f"  Probabilities array shape: {all_probs.shape}")
        
        # Test generate_metrics_report
        print("\nTesting generate_metrics_report...")
        report, cm = generate_metrics_report(all_preds, all_targets)
        
        print("Classification report generated:")
        print(f"  Report type: {type(report)}")
        print(f"  Confusion matrix shape: {cm.shape}")
        
        # Check report structure
        expected_keys = ['accuracy', 'macro avg', 'weighted avg', '0', '1']
        for key in expected_keys:
            if key not in report:
                print(f"  Error: Expected key '{key}' missing from report")
                return False
        
        # Test ROC curve calculation
        print("\nTesting ROC curve calculation...")
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        roc_auc = auc(fpr, tpr)
        
        print(f"  ROC AUC: {roc_auc:.4f}")
        
        # Test PR curve calculation
        print("\nTesting PR curve calculation...")
        precision, recall, _ = precision_recall_curve(all_targets, all_probs)
        pr_auc = average_precision_score(all_targets, all_probs)
        
        print(f"  PR AUC: {pr_auc:.4f}")
        
        return True
    except Exception as e:
        print(f"Error in metrics calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_plotting_functions(output_dir):
    """Test plotting functions with synthetic data"""
    print("\n" + "="*50)
    print("Testing Plotting Functions")
    print("="*50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create synthetic data for plotting
        np.random.seed(42)  # For reproducibility
        
        # Confusion matrix
        print("\nTesting plot_confusion_matrix...")
        cm = np.array([[70, 30], [20, 80]])  # Synthetic confusion matrix
        
        cm_path = os.path.join(output_dir, "test_confusion_matrix.png")
        plot_confusion_matrix(cm, output_path=cm_path)
        
        if os.path.exists(cm_path):
            print(f"  Confusion matrix plot saved to {cm_path}")
        else:
            print(f"  Error: Confusion matrix plot not created at {cm_path}")
            return False
        
        # ROC curve
        print("\nTesting plot_roc_curve...")
        # Generate synthetic ROC curve data
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)  # Simple curved shape
        roc_auc = auc(fpr, tpr)
        
        roc_path = os.path.join(output_dir, "test_roc_curve.png")
        plot_roc_curve(fpr, tpr, roc_auc, output_path=roc_path)
        
        if os.path.exists(roc_path):
            print(f"  ROC curve plot saved to {roc_path}")
        else:
            print(f"  Error: ROC curve plot not created at {roc_path}")
            return False
        
        # PR curve
        print("\nTesting plot_pr_curve...")
        # Generate synthetic PR curve data
        recall = np.linspace(0, 1, 100)
        precision = 1 - recall/2  # Simple decreasing line
        pr_auc = auc(recall, precision)
        
        pr_path = os.path.join(output_dir, "test_pr_curve.png")
        plot_pr_curve(precision, recall, pr_auc, output_path=pr_path)
        
        if os.path.exists(pr_path):
            print(f"  PR curve plot saved to {pr_path}")
        else:
            print(f"  Error: PR curve plot not created at {pr_path}")
            return False
        
        return True
    except Exception as e:
        print(f"Error in plotting functions: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_dummy_model(num_classes=2):
    """Create a simple dummy model for testing"""
    class DummyModel(torch.nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.fc = torch.nn.Linear(10, num_classes)
        
        def forward(self, x):
            # Handle different input types
            if isinstance(x, tuple):
                # If tuple, use the first element
                x = x[0]
            
            # For any input shape, return a tensor of shape [batch_size, num_classes]
            batch_size = x.shape[0]
            return self.fc(torch.ones(batch_size, 10, device=x.device))
    
    return DummyModel(num_classes)

def create_dummy_dataloader(model_type='3d_cnn', batch_size=4, num_batches=3, use_pose=False):
    """Create a dummy dataloader for testing"""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dummy data
    if model_type in ['3d_cnn', 'i3d', 'slowfast', 'r2plus1d']:
        # For 3D CNN: [batch_size, channels, frames, height, width]
        inputs = torch.randn(batch_size * num_batches, 3, 16, 224, 224)
    else:
        # For other models: [batch_size, frames, channels, height, width]
        inputs = torch.randn(batch_size * num_batches, 16, 3, 224, 224)
    
    # Create pose data if needed
    if use_pose:
        pose = torch.randn(batch_size * num_batches, 16, 66)  # [batch_size, frames, keypoints]
    
    # Create random labels with balanced classes
    labels = torch.cat([
        torch.zeros(batch_size * num_batches // 2, dtype=torch.long),
        torch.ones(batch_size * num_batches - batch_size * num_batches // 2, dtype=torch.long)
    ])
    
    # Create dataset
    if use_pose:
        dataset = TensorDataset(inputs, pose, labels)
    else:
        dataset = TensorDataset(inputs, labels)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def test_evaluate_model(device, output_dir):
    """Test the evaluate_model function"""
    print("\n" + "="*50)
    print("Testing evaluate_model Function")
    print("="*50)
    
    try:
        # Create a dummy model
        model = create_dummy_model().to(device)
        
        # Create a dummy dataloader
        test_loader = create_dummy_dataloader(model_type='3d_cnn', batch_size=4, num_batches=3)
        
        # Set up criterion
        criterion = torch.nn.CrossEntropyLoss()
        
        # Run evaluation
        print("Running evaluate_model...")
        test_loss, test_acc, all_preds, all_targets, all_probs, metrics_dict = evaluate_model(
            model, test_loader, criterion, device
        )
        
        # Check results
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  ROC AUC: {metrics_dict['roc_auc']:.4f}")
        print(f"  PR AUC: {metrics_dict['pr_auc']:.4f}")
        
        # Check shapes
        print(f"  Predictions shape: {all_preds.shape}")
        print(f"  Targets shape: {all_targets.shape}")
        print(f"  Probabilities shape: {all_probs.shape}")
        
        # Verify metrics_dict structure
        expected_keys = ['test_loss', 'test_accuracy', 'roc_auc', 'pr_auc', 'fpr', 'tpr', 'precision', 'recall']
        for key in expected_keys:
            if key not in metrics_dict:
                print(f"  Error: Expected key '{key}' missing from metrics_dict")
                return False
        
        return True
    except Exception as e:
        print(f"Error in evaluate_model test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_compare_models(device, output_dir):
    """Test the evaluate_and_compare_models function"""
    print("\n" + "="*50)
    print("Testing evaluate_and_compare_models Function")
    print("="*50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create dummy models
        models = {
            'model1': create_dummy_model().to(device),
            'model2': create_dummy_model().to(device)
        }
        
        # Create dummy dataloaders
        test_loaders = {
            'model1': create_dummy_dataloader(model_type='3d_cnn', batch_size=4, num_batches=3),
            'model2': create_dummy_dataloader(model_type='3d_cnn', batch_size=4, num_batches=3)
        }
        
        # Run comparison
        print("Running evaluate_and_compare_models...")
        results = evaluate_and_compare_models(
            models, test_loaders, device, output_dir=output_dir
        )
        
        # Check results
        if not results:
            print("  Error: evaluate_and_compare_models returned None")
            return False
        
        print("Evaluation results:")
        for model_name, result in results.items():
            print(f"  {model_name}: Accuracy = {result['test_accuracy']:.2f}%, ROC AUC = {result['roc_auc']:.4f}")
        
        # Check for output files
        expected_files = [
            os.path.join(output_dir, 'evaluation_summary.json'),
            os.path.join(output_dir, 'model1', 'metrics.json'),
            os.path.join(output_dir, 'model1', 'confusion_matrix.png'),
            os.path.join(output_dir, 'model1', 'roc_curve.png'),
            os.path.join(output_dir, 'model1', 'pr_curve.png'),
            os.path.join(output_dir, 'model2', 'metrics.json'),
            os.path.join(output_dir, 'model2', 'confusion_matrix.png'),
            os.path.join(output_dir, 'model2', 'roc_curve.png'),
            os.path.join(output_dir, 'model2', 'pr_curve.png')
        ]
        
        missing_files = [f for f in expected_files if not os.path.exists(f)]
        if missing_files:
            print("  Warning: Some expected output files are missing:")
            for f in missing_files:
                print(f"    - {f}")
        else:
            print("  All expected output files were created")
        
        return True
    except Exception as e:
        print(f"Error in evaluate_and_compare_models test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble_predictions(device, output_dir):
    """Test the ensemble_predictions function"""
    print("\n" + "="*50)
    print("Testing ensemble_predictions Function")
    print("="*50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create dummy models
        models = {
            'model1': create_dummy_model().to(device),
            'model2': create_dummy_model().to(device),
            'model3': create_dummy_model().to(device)
        }
        
        # Create dummy dataloaders (use the same for all models in this test)
        dataloader = create_dummy_dataloader(model_type='3d_cnn', batch_size=4, num_batches=3)
        test_loaders = {
            'model1': dataloader,
            'model2': dataloader,
            'model3': dataloader
        }
        
        # Run ensemble predictions
        print("Running ensemble_predictions...")
        ensemble_results = ensemble_predictions(
            models, test_loaders, device, output_dir=output_dir
        )
        
        # Check results
        if not ensemble_results:
            print("  Error: ensemble_predictions returned None")
            return False
        
        print(f"Ensemble accuracy: {ensemble_results['accuracy']:.2f}%")
        print(f"Ensemble ROC AUC: {ensemble_results['roc_auc']:.4f}")
        print(f"Ensemble PR AUC: {ensemble_results['pr_auc']:.4f}")
        
        # Check array shapes
        print(f"  Predictions shape: {ensemble_results['predictions'].shape}")
        print(f"  Targets shape: {ensemble_results['targets'].shape}")
        print(f"  Probabilities shape: {ensemble_results['probabilities'].shape}")
        
        # Check for output files
        expected_files = [
            os.path.join(output_dir, 'ensemble', 'ensemble_metrics.json'),
            os.path.join(output_dir, 'ensemble', 'ensemble_confusion_matrix.png'),
            os.path.join(output_dir, 'ensemble', 'ensemble_roc_curve.png'),
            os.path.join(output_dir, 'ensemble', 'ensemble_pr_curve.png')
        ]
        
        missing_files = [f for f in expected_files if not os.path.exists(f)]
        if missing_files:
            print("  Warning: Some expected output files are missing:")
            for f in missing_files:
                print(f"    - {f}")
        else:
            print("  All expected output files were created")
        
        return True
    except Exception as e:
        print(f"Error in ensemble_predictions test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test model evaluation functionality")
    parser.add_argument("--output_dir", type=str, default="./Testing/evaluation_test_output",
                      help="Directory to save test output")
    parser.add_argument("--gpu", type=int, default=-1,
                      help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--skip_metrics", action="store_true",
                      help="Skip metrics calculation tests")
    parser.add_argument("--skip_plotting", action="store_true",
                      help="Skip plotting function tests")
    parser.add_argument("--skip_evaluate", action="store_true",
                      help="Skip evaluate_model test")
    parser.add_argument("--skip_compare", action="store_true",
                      help="Skip compare_models test")
    parser.add_argument("--skip_ensemble", action="store_true",
                      help="Skip ensemble_predictions test")
    args = parser.parse_args()
    
    # Set up device
    device = setup_device(args.gpu)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run tests
    metrics_ok = True
    plotting_ok = True
    evaluate_ok = True
    compare_ok = True
    ensemble_ok = True
    
    if not args.skip_metrics:
        metrics_ok = test_metrics_calculation()
    
    if not args.skip_plotting:
        plotting_ok = test_plotting_functions(args.output_dir)
    
    if not args.skip_evaluate:
        evaluate_ok = test_evaluate_model(device, args.output_dir)
    
    if not args.skip_compare:
        compare_ok = test_compare_models(device, args.output_dir)
    
    if not args.skip_ensemble:
        ensemble_ok = test_ensemble_predictions(device, args.output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("Evaluation Tests Summary")
    print("="*50)
    
    if not args.skip_metrics:
        print(f"Metrics Calculation: {'✓ PASSED' if metrics_ok else '✗ FAILED'}")
    
    if not args.skip_plotting:
        print(f"Plotting Functions: {'✓ PASSED' if plotting_ok else '✗ FAILED'}")
    
    if not args.skip_evaluate:
        print(f"evaluate_model: {'✓ PASSED' if evaluate_ok else '✗ FAILED'}")
    
    if not args.skip_compare:
        print(f"evaluate_and_compare_models: {'✓ PASSED' if compare_ok else '✗ FAILED'}")
    
    if not args.skip_ensemble:
        print(f"ensemble_predictions: {'✓ PASSED' if ensemble_ok else '✗ FAILED'}")
    
    # Return success only if all tests pass
    if metrics_ok and plotting_ok and evaluate_ok and compare_ok and ensemble_ok:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())