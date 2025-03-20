# main.py
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
import json

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Violence Detection Training")
    parser.add_argument("--data_dir", type=str, default="./Data/Processed/standardized", 
                        help="Directory containing the violence detection dataset")
    # Removed --pose_dir argument
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save model outputs")
    parser.add_argument("--batch_size", type=int, default=hp.BATCH_SIZE, 
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=hp.NUM_EPOCHS, 
                        help="Number of training epochs")
    # Removed --use_pose argument
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
    torch.cuda.empty_cache()
    gc.collect()
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        allocated_memory = torch.cuda.memory_allocated(device) / 1e9
        reserved_memory = torch.cuda.memory_reserved(device) / 1e9
        print(f"GPU Memory: Total {total_memory:.2f} GB, Allocated {allocated_memory:.2f} GB, Reserved {reserved_memory:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def initialize_model(model_type, device, **overrides):
    """
    Initialize model based on model type with hyperparameters from central configuration.
    Pose data is no longer used.
    """
    from hyperparameters import get_model_config
    config = get_model_config(model_type, **overrides)
    # Remove any use_pose parameter if present
    config.pop('use_pose', None)
    
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
        from Models.model_2dcnn_lstm import ViolenceCNNLSTM
        model = ViolenceCNNLSTM(**config).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Initialized {model_type} with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    return model

def get_hyperparameters(model_type):
    """Get hyperparameters for a specific model type (pose data removed)."""
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
            'pretrained': True
        }
    elif model_type == 'transformer':
        return {
            'num_classes': 2,
            'embed_dim': 512,
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.1
        }
    elif model_type == 'i3d':
        return {
            'num_classes': 2,
            'dropout_prob': 0.5,
            'pretrained': True
        }
    elif model_type == 'slowfast':
        return {
            'num_classes': 2,
            'pretrained': True,
            'alpha': 8,
            'beta': 1/8,
            'dropout_prob': 0.5,
            'fusion_type': 'late'
        }
    elif model_type == 'r2plus1d':
        return {
            'num_classes': 2,
            'pretrained': True,
            'dropout_prob': 0.5,
            'frozen_layers': None
        }
    elif model_type == 'two_stream':
        return {
            'num_classes': 2,
            'spatial_weight': 1.0,
            'temporal_weight': 1.5,
            'pretrained': True,
            'spatial_backbone': 'r3d_18',
            'dropout_prob': 0.5,
            'fusion': 'late'
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
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = setup_device(args.gpu)
    
    print("Preparing data...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = prepare_violence_nonviolence_data(args.data_dir)
    
    models = {}
    test_loaders = {}
    grad_clip = args.grad_clip if args.grad_clip > 0 else None
    
    for model_type in args.model_types:
        print(f"\n{'='*20} Setting up {model_type} {'='*20}")
        
        # Since pose data is no longer used, we set current_pose_dir to None
        current_pose_dir = None
        
        # Get dataloaders (do not pass a pose_dir)
        train_loader, val_loader, test_loader = get_dataloaders(
            train_paths, train_labels, 
            val_paths, val_labels, 
            test_paths, test_labels,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_fps=10,
            num_frames=16,
            model_type=model_type,
            pin_memory=args.pin_memory
        )
        
        # Hyperparameter search if requested
        hyperparams = None
        if args.hp_search:
            from hyperparameter_search import get_best_hyperparameters
            print(f"Performing hyperparameter search for {model_type}...")
            
            # Define model_class for all supported model types
            model_class = None
            if model_type == '3d_cnn':
                from Models.model_3dcnn import Model3DCNN
                model_class = Model3DCNN
            elif model_type == '2d_cnn_lstm':
                from Models.model_2dcnn_lstm import Model2DCNNLSTM
                model_class = Model2DCNNLSTM
            elif model_type == 'transformer':
                from Models.model_transformer import VideoTransformer
                model_class = VideoTransformer
            elif model_type == 'i3d':
                from Models.model_i3d import TransferLearningI3D
                model_class = TransferLearningI3D
            elif model_type == 'simple_cnn':
                from Models.model_simplecnn import SimpleCNN
                model_class = SimpleCNN
            elif model_type == 'temporal_3d_cnn':
                from Models.model_Temporal3DCNN import Temporal3DCNN
                model_class = Temporal3DCNN
            elif model_type == 'slowfast':
                from Models.model_slowfast import SlowFastNetwork
                model_class = SlowFastNetwork
            elif model_type == 'r2plus1d':
                from Models.model_r2plus1d import R2Plus1DNet
                model_class = R2Plus1DNet
            elif model_type == 'two_stream':
                from Models.model_two_stream import TwoStreamNetwork
                model_class = TwoStreamNetwork
            elif model_type == 'cnn_lstm':
                from Models.violence_cnn_lstm import ViolenceCNNLSTM
                model_class = ViolenceCNNLSTM
            elif model_type == 'hybrid':
                from Models.model_hybrid import ModelHybrid
                model_class = ModelHybrid
            
            if model_class is not None:
                search_results = get_best_hyperparameters(
                model_class,
                train_paths[:len(train_paths)//4],
                train_labels[:len(train_labels)//4],
                val_paths,
                val_labels,
                output_dir=os.path.join(args.output_dir, f"hp_search_{model_type}"),
                num_epochs=args.num_epochs,
                batch_size=args.batch_size
            )
                hyperparams = search_results['best_params']
                print(f"Best hyperparameters: {hyperparams}")
                
                # Save hyperparams to a file for resume
                model_dir = os.path.join(args.output_dir, model_type)
                os.makedirs(model_dir, exist_ok=True)
                with open(os.path.join(model_dir, 'best_hyperparams.json'), 'w') as f:
                    json.dump(hyperparams, f)
            else:
                hyperparams = get_hyperparameters(model_type)
        elif args.resume:
            # Try to load hyperparams from a saved file
            model_dir = os.path.join(args.output_dir, model_type)
            hyperparams_file = os.path.join(model_dir, 'best_hyperparams.json')
            if os.path.exists(hyperparams_file):
                try:
                    with open(hyperparams_file, 'r') as f:
                        hyperparams = json.load(f)
                    print(f"Loaded hyperparameters for {model_type} from saved file.")
                except Exception as e:
                    print(f"Error loading hyperparameters: {e}")
                    hyperparams = get_hyperparameters(model_type)
            else:
                print(f"No saved hyperparameters found for {model_type}, using defaults.")
                hyperparams = get_hyperparameters(model_type)
        else:
            # Normal training mode, use defaults
            hyperparams = get_hyperparameters(model_type)
            # Add this section for handling resume mode
            if args.resume:
                # Check if we can load hyperparams from a saved file
                checkpoint_dir = os.path.join(args.output_dir, model_type)
                if os.path.exists(os.path.join(checkpoint_dir, 'best_hyperparams.json')):
                    with open(os.path.join(checkpoint_dir, 'best_hyperparams.json'), 'r') as f:
                        hyperparams = json.load(f)
                    print(f"Loaded hyperparameters for {model_type} from checkpoint.")
                elif hyperparams is None:
                    # If no saved hyperparams and no search done, use defaults
                    hyperparams = get_hyperparameters(model_type)
                    print(f"Using default hyperparameters for {model_type}.")

            print("Aggressively clearing GPU memory before model initialization...")
        # Clear all cache
        clear_cuda_memory()
        # Force garbage collection
        import gc
        gc.collect()
        # Delete any other large objects that might be in memory
        if 'model_class' in locals():
            del model_class
        # Reset CUDA statistics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            # Force sync to ensure all CUDA operations are completed
            torch.cuda.synchronize()
        # Wait a moment to ensure memory is freed
        import time
        time.sleep(2)
        # Check memory now
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"GPU Memory after clearing: {allocated:.3f} GB allocated, {reserved:.3f} GB reserved")

        # Add this code to filter out optimizer parameters:
        optimizer_param_names = ['learning_rate', 'weight_decay', 'momentum', 'beta1', 'beta2']
        model_params = {k: v for k, v in hyperparams.items() if k not in optimizer_param_names}
        optimizer_params = {k: v for k, v in hyperparams.items() if k in optimizer_param_names}

        # Remove any 'use_pose' key if present
        model_params.pop('use_pose', None)

        # Initialize model (with ONLY model parameters)
        model = initialize_model(model_type, device, **model_params)
        
        # Check for resume checkpoint if needed
        checkpoint_path = None
        if args.resume:
            checkpoint_dir = os.path.join(args.output_dir, model_type)
            if os.path.exists(checkpoint_dir):
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                best_checkpoints = [f for f in checkpoint_files if 'best' in f]
                last_checkpoints = [f for f in checkpoint_files if 'last' in f]
                if best_checkpoints:
                    checkpoint_path = os.path.join(checkpoint_dir, best_checkpoints[0])
                elif last_checkpoints:
                    checkpoint_path = os.path.join(checkpoint_dir, last_checkpoints[0])
                elif checkpoint_files:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
        
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
            grad_clip=grad_clip,
            **optimizer_params  # Pass the optimizer params here
        )
        
        models[model_type] = trained_model
        test_loaders[model_type] = test_loader
        
        clear_cuda_memory()
    
    print("\n\n" + "="*20 + " EVALUATION " + "="*20)
    results = evaluate_and_compare_models(models, test_loaders, device, args.output_dir)
    
    if len(models) > 1:
        print("\n\n" + "="*20 + " ENSEMBLE RESULTS " + "="*20)
        ensemble_results = ensemble_predictions(models, test_loaders, device, args.output_dir)
        print(f"Ensemble accuracy: {ensemble_results['accuracy']:.2f}%")
        print(f"Ensemble ROC AUC: {ensemble_results['roc_auc']:.4f}")
        print(f"Ensemble PR AUC: {ensemble_results['pr_auc']:.4f}")
    
    print("\nTraining and evaluation completed!")
    clear_cuda_memory()

if __name__ == "__main__":
    main()
