import os
import argparse
import torch
import gc
import torch.cuda.amp as amp
from utils.dataprep import prepare_violence_nonviolence_data
from dataloader import get_dataloaders
from train import clear_cuda_memory, load_checkpoint
from evaluations import ensemble_predictions
import json
import sys
from main import initialize_model, get_hyperparameters

# Define model configs based on your hyperparameters module
MODEL_CONFIGS = {
    '3d_cnn': {'num_classes': 2, 'dropout_prob': 0.5, 'pretrained': True},
    'slowfast': {'num_classes': 2, 'alpha': 8, 'beta': 0.125, 'dropout_prob': 0.5, 'pretrained': True},
    'two_stream': {'num_classes': 2, 'spatial_weight': 1.0, 'temporal_weight': 1.5, 'dropout_prob': 0.5, 
                  'fusion': 'late', 'spatial_backbone': 'r3d_18', 'pretrained': True},
    '2d_cnn_lstm': {'num_classes': 2, 'lstm_hidden_size': 512, 'lstm_num_layers': 3, 'dropout_prob': 0.5, 'pretrained': True},
    'transformer': {'num_classes': 2, 'num_heads': 8, 'num_layers': 6, 'embed_dim': 512, 'dropout': 0.1}
}

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Evaluation for Violence Detection")
    parser.add_argument("--model_types", nargs="+", 
                    default=['two_stream', 'slowfast', '3d_cnn', '2d_cnn_lstm', 'transformer'],
                    help="Models to include in ensemble")
    parser.add_argument("--data_dir", type=str, default="./Data/Processed/standardized", 
                        help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="./output/ensemble_evaluation", 
                        help="Directory for evaluation outputs")
    parser.add_argument("--trained_models_dir", type=str, default="./output",
                        help="Base directory containing trained models")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Workers for data loading")
    parser.add_argument("--gpu", type=int, default=0, 
                        help="GPU ID (-1 for CPU)")
    
    return parser.parse_args()

def setup_device(gpu_id):
    """Set up computation device (CPU or GPU)"""
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def load_trained_model(model_type, model_dir, device):
    """Load a trained model from checkpoint with robust error handling"""
    # Get model configuration
    if model_type in MODEL_CONFIGS:
        model_params = MODEL_CONFIGS[model_type]
    else:
        model_params = get_hyperparameters(model_type)
    
    # Initialize the model with the correct parameters
    try:
        model = initialize_model(model_type, device, **model_params)
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Trying with default parameters...")
        model = initialize_model(model_type, device)
    
    # Find the checkpoint file
    checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    best_checkpoints = [f for f in checkpoint_files if 'best' in f]
    checkpoint_path = None
    
    if best_checkpoints:
        checkpoint_path = os.path.join(model_dir, best_checkpoints[0])
    else:
        print(f"No best checkpoint found for {model_type}, looking for any checkpoint...")
        if checkpoint_files:
            checkpoint_path = os.path.join(model_dir, checkpoint_files[0])
    
    if not checkpoint_path:
        print(f"No checkpoint found for {model_type}")
        return None
    
    # Load checkpoint
    print(f"Loading {model_type} from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try different state dict formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load with strict=False to ignore mismatched parameters
    try:
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model state dict with strict=False")
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def main():
    """Main function for ensemble evaluation"""
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = setup_device(args.gpu)
    
    print("Preparing data...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = prepare_violence_nonviolence_data(args.data_dir)
    
    models = {}
    test_loaders = {}
    
    for model_type in args.model_types:
        print(f"\n{'='*20} Loading {model_type} {'='*20}")
        
        # Load data loaders for this model type
        _, _, test_loader = get_dataloaders(
            train_paths, train_labels, 
            val_paths, val_labels, 
            test_paths, test_labels,
            batch_size=args.batch_size if model_type != 'two_stream' else args.batch_size // 2,
            num_workers=args.num_workers,
            target_fps=15,
            num_frames=16,
            model_type=model_type,
            pin_memory=False,  # Change from True to False
            flow_dir=None
        )
        
        # Load the trained model
        model_dir = os.path.join(args.trained_models_dir, model_type)
        if not os.path.exists(model_dir):
            print(f"Model directory not found for {model_type}, skipping...")
            continue
            
        model = load_trained_model(model_type, model_dir, device)
        if model is None:
            continue
            
        models[model_type] = model
        test_loaders[model_type] = test_loader
        
        # Clear memory after loading each model
        clear_cuda_memory()
    
    if len(models) < 2:
        print("Need at least 2 models for ensemble evaluation.")
        return
        
    print("Clearing GPU memory before ensemble evaluation...")
    clear_cuda_memory()
    torch.cuda.empty_cache()
    
    for model_name, model in models.items():
        model.eval()  # Ensure model is in evaluation mode
    
    print("\n\n" + "="*20 + " ENSEMBLE EVALUATION " + "="*20)
    
    # Define weights based on your known model performances
    weights = {
        '3d_cnn': 2.4,       # Strong performer with 96.68% accuracy
        'transformer': 2.5,   # Best performer with 96.93% accuracy
        '2d_cnn_lstm': 2.3,   # Very good performer with 95.14% accuracy and high AUC
        'slowfast': 0.7,      # Significantly lower performance at 72.38% accuracy
        'two_stream': 0.6     # Lowest performance at 71.10% accuracy
    }

    # Pass weights to ensemble_predictions
    ensemble_results = ensemble_predictions(models, test_loaders, device, args.output_dir, weights=weights)
    
    print(f"Ensemble accuracy: {ensemble_results['accuracy']:.2f}%")
    print(f"Ensemble ROC AUC: {ensemble_results['roc_auc']:.4f}")
    print(f"Ensemble PR AUC: {ensemble_results['pr_auc']:.4f}")
    
    # Create the ensemble directory
    ensemble_dir = os.path.join(args.output_dir, 'ensemble')
    os.makedirs(ensemble_dir, exist_ok=True)

    # Save ensemble metrics as JSON
    metrics_path = os.path.join(ensemble_dir, 'ensemble_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'accuracy': float(ensemble_results['accuracy']),
            'roc_auc': float(ensemble_results['roc_auc']),
            'pr_auc': float(ensemble_results['pr_auc'])
        }, f, indent=4)
    print(f"Saved ensemble metrics to {metrics_path}")

    # Generate and save confusion matrix
    from evaluations import generate_metrics_report, plot_confusion_matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(ensemble_results['targets'], ensemble_results['predictions'])
    cm_path = os.path.join(ensemble_dir, 'ensemble_confusion_matrix.png')
    plot_confusion_matrix(cm, output_path=cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    # Generate and save ROC curve
    from evaluations import plot_roc_curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(ensemble_results['targets'], ensemble_results['probabilities'][:, 1])
    roc_path = os.path.join(ensemble_dir, 'ensemble_roc_curve.png')
    plot_roc_curve(fpr, tpr, ensemble_results['roc_auc'], output_path=roc_path, 
                   title='Ensemble Model - ROC Curve')
    print(f"Saved ROC curve to {roc_path}")

    # Generate and save PR curve
    from evaluations import plot_pr_curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(ensemble_results['targets'], 
                                                 ensemble_results['probabilities'][:, 1])
    pr_path = os.path.join(ensemble_dir, 'ensemble_pr_curve.png')
    plot_pr_curve(precision, recall, ensemble_results['pr_auc'], output_path=pr_path,
                  title='Ensemble Model - Precision-Recall Curve')
    print(f"Saved PR curve to {pr_path}")
    
    print("\nEnsemble evaluation completed!")
    clear_cuda_memory()

if __name__ == "__main__":
    main()