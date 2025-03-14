#main.py
import os
import argparse
import torch
import gc
from utils.dataprep import prepare_violence_nonviolence_data
from dataloader import get_dataloaders
from train import train_model, clear_cuda_memory, load_checkpoint
from evaluations import evaluate_and_compare_models, ensemble_predictions
import hyperparameters as hp
from Models.model_3dcnn import Model3DCNN
from Models.model_2dcnn_lstm import Model2DCNNLSTM
from Models.model_transformer import VideoTransformer
from Models.model_i3d import TransferLearningI3D
from Models.model_two_stream import TwoStreamNetwork
from Models.model_slowfast import SlowFastNetwork
from Models.model_r2plus1d import R2Plus1DNet
from Models.model_simplecnn import SimpleCNN
from Models.model_hybrid import ModelHybrid
from Models.model_Temporal3DCNN import Temporal3DCNN
from Models.violence_cnn_lstm import ViolenceCNNLSTM



def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Violence Detection Training")
    parser.add_argument("--data_dir", type=str, default="./Data/Processed/standardized", 
                        help="Directory containing the violence detection dataset")
    parser.add_argument("--pose_dir", type=str, default="./Data/pose_keypoints", 
                        help="Directory containing pose keypoints")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save model outputs")
    parser.add_argument("--batch_size", type=int, default=hp.BATCH_SIZE, 
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=hp.NUM_EPOCHS, 
                        help="Number of training epochs")
    parser.add_argument("--use_pose", action="store_true", 
                        help="Use pose features in addition to video frames")
    parser.add_argument("--gpu", type=int, default=0, 
                        help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for data loading")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint if available")
    parser.add_argument("--early_stopping", type=int, default=7,
                        help="Patience for early stopping (epochs)")
    parser.add_argument("--hp_search", action="store_true",
                        help="Perform hyperparameter search before training")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping value (use 0 to disable)")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Use pin_memory in DataLoader for faster GPU transfers")
    parser.add_argument("--model_types", nargs="+", 
                    default=['3d_cnn', '2d_cnn_lstm', 'transformer', 'i3d', 
                             'simple_cnn', 'temporal_3d_cnn', 'slowfast', 
                             'r2plus1d', 'two_stream'],
                    help="Model types to train")
    return parser.parse_args()

def setup_device(gpu_id):
    """Set up computation device (CPU or GPU) with memory cleanup"""
    # Clear CUDA memory before setting up
    torch.cuda.empty_cache()
    gc.collect()
    
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        
        # Print GPU memory info
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9  # GB
        allocated_memory = torch.cuda.memory_allocated(device) / 1e9  # GB
        reserved_memory = torch.cuda.memory_reserved(device) / 1e9  # GB
        
        print(f"GPU Memory: Total {total_memory:.2f} GB, Allocated {allocated_memory:.2f} GB, Reserved {reserved_memory:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device

def initialize_model(model_type, device, use_pose=False, **overrides):
    """
    Initialize model based on model type with hyperparameters from central configuration
    
    Args:
        model_type: Type of model to initialize
        device: Device to use (CPU or GPU)
        use_pose: Whether to use pose data
        **overrides: Any hyperparameters to override defaults
        
    Returns:
        Initialized PyTorch model on the specified device
    """
    # Import the configuration function
    from hyperparameters import get_model_config
    
    # Get model configuration with any overrides
    config = get_model_config(model_type, **overrides)
    
    # Add use_pose only for models that support it
    if use_pose and model_type in ['3d_cnn', '2d_cnn_lstm', 'transformer', 'i3d']:
        config['use_pose'] = use_pose
    elif 'use_pose' in config and model_type not in ['3d_cnn', '2d_cnn_lstm', 'transformer', 'i3d']:
        # Remove use_pose from models that don't support it
        config.pop('use_pose')
    
    # Initialize the appropriate model with the config
    if model_type == '3d_cnn':
        from Models.model_3dcnn import Model3DCNN
        model = Model3DCNN(**config).to(device)
        
    elif model_type == '2d_cnn_lstm':
        from Models.model_2dcnn_lstm import Model2DCNNLSTM
        model = Model2DCNNLSTM(**config).to(device)
        
    elif model_type == 'transformer':
        from Models.model_transformer import VideoTransformer
        model = VideoTransformer(**config).to(device)
        
    elif model_type == 'i3d':
        from Models.model_i3d import TransferLearningI3D
        model = TransferLearningI3D(**config).to(device)
        
    elif model_type == 'simple_cnn':
        from Models.model_simplecnn import SimpleCNN
        model = SimpleCNN(**config).to(device)
        
    elif model_type == 'temporal_3d_cnn':
        from Models.model_Temporal3DCNN import Temporal3DCNN
        model = Temporal3DCNN(**config).to(device)
        
    # New models
    elif model_type == 'slowfast':
        from Models.model_slowfast import SlowFastNetwork
        model = SlowFastNetwork(**config).to(device)
        
    elif model_type == 'r2plus1d':
        from Models.model_r2plus1d import R2Plus1DNet
        model = R2Plus1DNet(**config).to(device)
        
    elif model_type == 'two_stream':
        from Models.model_two_stream import TwoStreamNetwork
        model = TwoStreamNetwork(**config).to(device)
    
    elif model_type == 'cnn_lstm':
        model_params = get_hyperparameters(model_type, use_pose)
        from Models.violence_cnn_lstm import ViolenceCNNLSTM
        model = ViolenceCNNLSTM(**config).to(device)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Print model configuration if verbose logging is enabled
    print(f"Initialized {model_type} with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return model

def get_hyperparameters(model_type, use_pose=False):
    """Get hyperparameters for a specific model type"""
    if model_type == '3d_cnn':
        return {
            'num_classes': 2,
            'dropout_prob': 0.5,
            'pretrained': True
        }
    elif model_type == '2d_cnn_lstm':
        return {
            'num_classes': 2,
            'lstm_hidden_size': 512,
            'lstm_num_layers': 2,
            'dropout_prob': 0.5,
            'use_pose': use_pose,
            'pretrained': True
        }
    elif model_type == 'transformer':
        return {
            'num_classes': 2,
            'embed_dim': 512,
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.1,
            'use_pose': use_pose
        }
    elif model_type == 'i3d':
        return {
            'num_classes': 2,
            'dropout_prob': 0.5,
            'use_pose': use_pose,
            'pretrained': True
        }
    # Add parameters for the new models
    elif model_type == 'slowfast':
        return {
            'num_classes': 2,
            'pretrained': True,
            'alpha': 8,  # Speed ratio between Fast and Slow pathways
            'beta': 1/8,  # Channel ratio between Fast and Slow pathways
            'dropout_prob': 0.5
        }
    elif model_type == 'r2plus1d':
        return {
            'num_classes': 2,
            'pretrained': True,
            'dropout_prob': 0.5,
            'frozen_layers': None  # Set to ['stem', 'layer1'] to freeze early layers
        }
    elif model_type == 'two_stream':
        return {
            'num_classes': 2,
            'spatial_weight': 1.0,  # Weight for spatial stream predictions
            'temporal_weight': 1.5,  # Weight for temporal stream (usually higher)
            'pretrained': True,
            'spatial_backbone': 'r3d_18',
            'dropout_prob': 0.5,
            'fusion': 'late'  # 'late' or 'conv'
        }
    elif model_type == 'simple_cnn':
        return {
            'num_classes': 2
        }
    elif model_type == 'temporal_3d_cnn':
        return {
            'num_classes': 2
        }
    elif model_type == 'cnn_lstm':
        return {
            'num_classes': 2,
            'lstm_hidden_size': 512,
            'num_layers': 2,
            'dropout': 0.5,
            'activation': 'relu'
        }
    else:
        return {'num_classes': 2}
    
    
    
    
    
def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = setup_device(args.gpu)
    
    # Prepare data
    print("Preparing data...")
    (train_paths, train_labels, 
     val_paths, val_labels, 
     test_paths, test_labels) = prepare_violence_nonviolence_data(args.data_dir)
    
    # Create dataloaders for each model type
    models = {}
    test_loaders = {}
    
    # Set grad_clip to None if 0
    grad_clip = args.grad_clip if args.grad_clip > 0 else None
    
    for model_type in args.model_types:
        print(f"\n{'='*20} Setting up {model_type} {'='*20}")
        
        # Only use pose_dir if use_pose flag is True; otherwise, set to None.
        current_pose_dir = args.pose_dir if args.use_pose else None

        # Get data loaders for this model type using current_pose_dir
        train_loader, val_loader, test_loader = get_dataloaders(
            train_paths, train_labels, 
            val_paths, val_labels, 
            test_paths, test_labels,
            current_pose_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            model_type=model_type,
            pin_memory=args.pin_memory
        )
        
        # Hyperparameter search if requested
        hyperparams = None
        if args.hp_search:
            from hyperparameter_search import get_best_hyperparameters
            
            print(f"Performing hyperparameter search for {model_type}...")
            model_class = None
            
            if model_type == '3d_cnn':
                model_class = Model3DCNN
            elif model_type == '2d_cnn_lstm':
                model_class = Model2DCNNLSTM
            elif model_type == 'transformer':
                model_class = VideoTransformer
            elif model_type == 'i3d':
                model_class = TransferLearningI3D
            
            if model_class is not None:
                search_results = get_best_hyperparameters(
                    model_class,
                    train_paths[:len(train_paths)//4],  # Use subset for speed
                    train_labels[:len(train_labels)//4],
                    val_paths,
                    val_labels,
                    output_dir=os.path.join(args.output_dir, f"hp_search_{model_type}")
                )
                
                hyperparams = search_results['best_params']
                print(f"Best hyperparameters: {hyperparams}")
        else:
            # Use default hyperparameters
            hyperparams = get_hyperparameters(model_type, args.use_pose)
        
        # Initialize model
        # First, create a copy of hyperparams
            model_params = hyperparams.copy()

            # Remove use_pose from the hyperparams if it exists
            if 'use_pose' in model_params:
                model_params.pop('use_pose')

            # Initialize model with correct arguments
            model = initialize_model(model_type, device, args.use_pose, **model_params)
        
        # Setup checkpoint path for resuming
        checkpoint_path = None
        if args.resume:
            checkpoint_dir = os.path.join(args.output_dir, model_type)
            if os.path.exists(checkpoint_dir):
                # Look for checkpoint files
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                
                # Prioritize 'best' checkpoint
                best_checkpoints = [f for f in checkpoint_files if 'best' in f]
                last_checkpoints = [f for f in checkpoint_files if 'last' in f]
                
                if best_checkpoints:
                    checkpoint_path = os.path.join(checkpoint_dir, best_checkpoints[0])
                elif last_checkpoints:
                    checkpoint_path = os.path.join(checkpoint_dir, last_checkpoints[0])
                elif checkpoint_files:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
        
        # Train model
        print(f"\n{'='*20} Training {model_type} {'='*20}")
        trained_model = train_model(
            model_name=model_type,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            device=device,
            output_dir=args.output_dir,
            patience=args.early_stopping,
            resume_from=checkpoint_path,
            grad_clip=grad_clip
        )
        
        models[model_type] = trained_model
        test_loaders[model_type] = test_loader
        
        # Clear memory after training each model
        clear_cuda_memory()
    
    # Evaluate all models
    print("\n\n" + "="*20 + " EVALUATION " + "="*20)
    results = evaluate_and_compare_models(models, test_loaders, device, args.output_dir)
    
    # Create ensemble if multiple models are trained
    if len(models) > 1:
        print("\n\n" + "="*20 + " ENSEMBLE RESULTS " + "="*20)
        ensemble_results = ensemble_predictions(models, test_loaders, device, args.output_dir)
        
        # Log ensemble results
        print(f"Ensemble accuracy: {ensemble_results['accuracy']:.2f}%")
        print(f"Ensemble ROC AUC: {ensemble_results['roc_auc']:.4f}")
        print(f"Ensemble PR AUC: {ensemble_results['pr_auc']:.4f}")
    
    print("\nTraining and evaluation completed!")
    
    # Final memory cleanup
    clear_cuda_memory()

if __name__ == "__main__":
    main()