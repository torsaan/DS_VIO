import os
import torch
import argparse
import json
from evaluations import evaluate_and_compare_models, ensemble_predictions
from dataloader import get_dataloaders
from utils.dataprep import prepare_violence_nonviolence_data
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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Violence Detection Model Evaluation")
    parser.add_argument("--data_dir", type=str, default="./Data/Processed/standardized",
                      help="Directory containing violence/non-violence videos")
    parser.add_argument("--output_dir", type=str, default="./output",
                      help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for evaluation")
    parser.add_argument("--gpu", type=int, default=0,
                      help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of worker processes for data loading")
    parser.add_argument("--model_types", nargs="+", 
                      default=['3d_cnn'],
                      help="Model types to evaluate")
    return parser.parse_args()

def initialize_model(model_type, hyperparams, device):
    """Initialize model based on type with hyperparameters"""
    if model_type == '3d_cnn':
        model = Model3DCNN(**hyperparams).to(device)
    elif model_type == '2d_cnn_lstm':
        model = Model2DCNNLSTM(**hyperparams).to(device)
    elif model_type == 'transformer':
        model = VideoTransformer(**hyperparams).to(device)
    elif model_type == 'slowfast':
        model = SlowFastNetwork(**hyperparams).to(device)
    elif model_type == 'r2plus1d':
        model = R2Plus1DNet(**hyperparams).to(device)
    elif model_type == 'cnn_lstm':
        model = ViolenceCNNLSTM(**hyperparams).to(device)
    elif model_type == 'two_stream':
        model = TwoStreamNetwork(**hyperparams).to(device)
    elif model_type == 'simple_cnn':
        model = SimpleCNN(**hyperparams).to(device)
    elif model_type == 'temporal_3d_cnn':
        model = Temporal3DCNN(**hyperparams).to(device)
    elif model_type == 'i3d':
        model = TransferLearningI3D(**hyperparams).to(device)
    elif model_type == 'hybrid':
        model = ModelHybrid(**hyperparams).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model

def load_best_model(model_type, output_dir, device):
    """Load best model from checkpoints with architecture matching"""
    model_dir = os.path.join(output_dir, model_type)
    
    # Look for best model checkpoint - try different filename patterns
    possible_filenames = [
        f"last_{model_type}_model_best.pth",  # Your specific format
        f"{model_type}_epoch*.pth",           # Epoch-specific format (using glob)
        f"best_{model_type}_model.pth",       # Common format 1
        f"{model_type}_best.pth",             # Common format 2
        f"last_{model_type}_model.pth",       # Latest checkpoint
        f"{model_type}_model.pth"             # Generic format
    ]
    
    best_model_path = None
    
    # Check for exact filename matches first
    for pattern in possible_filenames:
        if '*' not in pattern:  # Skip glob patterns in first pass
            path = os.path.join(model_dir, pattern)
            if os.path.exists(path):
                best_model_path = path
                print(f"Found model checkpoint: {pattern}")
                break
    
    # If not found, try glob patterns
    if best_model_path is None:
        import glob
        for pattern in possible_filenames:
            if '*' in pattern:  # Only check glob patterns
                matches = glob.glob(os.path.join(model_dir, pattern))
                if matches:
                    # Sort by modification time, newest first
                    matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    best_model_path = matches[0]
                    print(f"Found model checkpoint using pattern: {os.path.basename(best_model_path)}")
                    break
    
    # If still not found, try to find any checkpoint
    if best_model_path is None:
        all_checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if all_checkpoints:
            # Sort by modification time, newest first
            all_checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
            best_model_path = os.path.join(model_dir, all_checkpoints[0])
            print(f"Using most recent checkpoint: {all_checkpoints[0]}")
        else:
            raise FileNotFoundError(f"No checkpoint found for {model_type}")
    
    # Load checkpoint to extract architecture hyperparameters
    print(f"Loading checkpoint: {os.path.basename(best_model_path)}")
    checkpoint = torch.load(best_model_path, map_location=device)
    
    # Try to extract architecture info from the checkpoint
    architecture_params = None
    
    # Option 1: Check if the checkpoint has a 'hyperparams' key
    if isinstance(checkpoint, dict) and 'hyperparams' in checkpoint:
        architecture_params = checkpoint['hyperparams']
        print("Found hyperparameters in checkpoint")
    
    # Option 2: Load from hyperparams file
    if architecture_params is None:
        hyperparams_file = os.path.join(model_dir, 'hyperparams.json')
        if os.path.exists(hyperparams_file):
            with open(hyperparams_file, 'r') as f:
                architecture_params = json.load(f)
            print(f"Loaded hyperparameters from file")
    
    # Option 3: Try to infer parameters from the model state_dict
    if architecture_params is None and model_type == '2d_cnn_lstm':
        # For 2D CNN-LSTM, we need to determine LSTM parameters from the state dict
        print("Inferring LSTM hyperparameters from state_dict")
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        
        # Get number of LSTM layers
        lstm_layers = 0
        while f'lstm.weight_ih_l{lstm_layers}' in state_dict:
            lstm_layers += 1
        
        # Get LSTM hidden size (half of bias size since bidirectional)
        if 'lstm.bias_ih_l0' in state_dict:
            lstm_hidden_size = state_dict['lstm.bias_ih_l0'].shape[0] // 2
        else:
            lstm_hidden_size = 256  # default fallback
        
        # Check if bidirectional
        bidirectional = 'lstm.weight_ih_l0_reverse' in state_dict
        
        architecture_params = {
            'num_classes': 2,
            'lstm_hidden_size': lstm_hidden_size,
            'lstm_num_layers': lstm_layers,
            'dropout_prob': 0.5,  # reasonable default
            'pretrained': True,
            'bidirectional': bidirectional
        }
        print(f"Inferred LSTM params: layers={lstm_layers}, hidden_size={lstm_hidden_size}, bidirectional={bidirectional}")
    
    # If we still don't have params, use defaults
    if architecture_params is None:
        print(f"No hyperparameters found. Using defaults.")
        architecture_params = {'num_classes': 2}
        if model_type == '3d_cnn':
            architecture_params.update({'dropout_prob': 0.5, 'pretrained': True})
        elif model_type == '2d_cnn_lstm':
            architecture_params.update({
                'lstm_hidden_size': 256, 
                'lstm_num_layers': 2, 
                'dropout_prob': 0.5, 
                'pretrained': True,
                'bidirectional': True
            })
        elif model_type == 'transformer':
            architecture_params.update({
                'embed_dim': 512, 
                'num_heads': 8, 
                'num_layers': 4, 
                'dropout': 0.1
            })
    
    # Optionally filter optimizer parameters
    optimizer_params = ['learning_rate', 'weight_decay', 'momentum', 'beta1', 'beta2']
    model_params = {k: v for k, v in architecture_params.items() if k not in optimizer_params}
    
    print(f"Initializing {model_type} with parameters: {model_params}")
    
    # Initialize model with hyperparameters
    model = initialize_model(model_type, model_params, device)
    
    # Load model weights
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from state_dict")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from full checkpoint")
    except Exception as e:
        print(f"Error loading model weights: {str(e)}")
        raise
    
    print(f"Successfully loaded model for {model_type}")
    return model

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"Total GPU memory: {total_memory:.2f} GB")
    
    # Prepare data
    print("Preparing data...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        prepare_violence_nonviolence_data(args.data_dir)
    
    print(f"Total test videos: {len(test_paths)}")
    
    # Load models and create test loaders
    models = {}
    test_loaders = {}
    
    for model_type in args.model_types:
        try:
            print(f"\n{'='*20} Loading {model_type} {'='*20}")
            
            # Load best model
            model = load_best_model(model_type, args.output_dir, device)
            models[model_type] = model
            
            # Create test loader
            _, _, test_loader = get_dataloaders(
                train_paths, train_labels,
                val_paths, val_labels,
                test_paths, test_labels,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                model_type=model_type,
                pin_memory=True
            )
            test_loaders[model_type] = test_loader
            
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
    
    if not models:
        print("No models were successfully loaded. Exiting.")
        return
    
    # Evaluate models
    print("\n{'='*20} Evaluating models {'='*20}")
    results = evaluate_and_compare_models(models, test_loaders, device, args.output_dir)
    
    # Create ensemble if more than one model
    if len(models) > 1:
        print("\n{'='*20} Creating ensemble {'='*20}")
        ensemble_results = ensemble_predictions(models, test_loaders, device, args.output_dir)
        print(f"Ensemble accuracy: {ensemble_results['accuracy']:.2f}%")
        print(f"Ensemble ROC AUC: {ensemble_results['roc_auc']:.4f}")
        print(f"Ensemble PR AUC: {ensemble_results['pr_auc']:.4f}")
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()