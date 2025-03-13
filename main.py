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
from Models.dl_models import ViolenceLSTM

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Violence Detection Training")
    parser.add_argument("--data_dir", type=str, default="./Data/Processed/standardized", 
                        help="Directory containing the violence detection dataset")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save model outputs")
    parser.add_argument("--batch_size", type=int, default=hp.BATCH_SIZE, 
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=hp.NUM_EPOCHS, 
                        help="Number of training epochs")
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
    parser.add_argument("--hp_search_epochs", type=int, default=10,
                        help="Number of epochs for each hyperparameter combination")
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

def initialize_model(model_type, device, hyperparams=None):
    """Initialize model based on model type with optional hyperparameters"""
    if hyperparams is None:
        hyperparams = {}
    
    # Set default values for required parameters
    required_params = {
        'num_classes': 2,
        'use_pose': False
    }
    
    # Merge required params with provided hyperparams, ensuring no pose data
    model_params = {**required_params, **hyperparams}
    model_params['use_pose'] = False  # Always set to False
    
    if model_type == '3d_cnn':
        model = Model3DCNN(**model_params).to(device)
        
    elif model_type == '2d_cnn_lstm':
        model = Model2DCNNLSTM(**model_params).to(device)
        
    elif model_type == 'transformer':
        model = VideoTransformer(**model_params).to(device)
        
    elif model_type == 'i3d':
        model = TransferLearningI3D(**model_params).to(device)
        
    elif model_type == 'simple_cnn':
        model = SimpleCNN(**model_params).to(device)
        
    elif model_type == 'temporal_3d_cnn':
        model = Temporal3DCNN(**model_params).to(device)
        
    elif model_type == 'slowfast':
        model = SlowFastNetwork(**model_params).to(device)
        
    elif model_type == 'r2plus1d':
        model = R2Plus1DNet(**model_params).to(device)
        
    elif model_type == 'two_stream':
        model = TwoStreamNetwork(**model_params).to(device)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def get_hyperparameters(model_type):
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
    # Parameters for the new models
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
        
        # Get data loaders for this model type
        train_loader, val_loader, test_loader = get_dataloaders(
            train_paths, train_labels, 
            val_paths, val_labels, 
            test_paths, test_labels,
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
            elif model_type == 'slowfast':
                model_class = SlowFastNetwork
            elif model_type == 'r2plus1d':
                model_class = R2Plus1DNet
            elif model_type == 'two_stream':
                model_class = TwoStreamNetwork
            elif model_type == 'simple_cnn':
                model_class = SimpleCNN
            elif model_type == 'temporal_3d_cnn':
                model_class = Temporal3DCNN
            
            if model_class is not None:
                # Limit search data for faster execution
                subset_size = len(train_paths) // 4  # Use 25% of data for search
                search_train_paths = train_paths[:subset_size]
                search_train_labels = train_labels[:subset_size]
                
                hp_output_dir = os.path.join(args.output_dir, f"hp_search_{model_type}")
                
                search_results = get_best_hyperparameters(
                    model_class,
                    search_train_paths,
                    search_train_labels,
                    val_paths,
                    val_labels,
                    output_dir=hp_output_dir
                )
                
                # Load best parameters for model creation
                model_hyperparams = {}
                if 'best_params' in search_results:
                    model_hyperparams = search_results['best_params']
                elif 'best_model_params' in search_results:
                    model_hyperparams = search_results['best_model_params']
                
                # Create optimizer parameters if available
                optimizer_hyperparams = {}
                if 'best_train_params' in search_results:
                    optimizer_hyperparams = search_results['best_train_params']
                
                hyperparams = model_hyperparams
                print(f"Best hyperparameters: {hyperparams}")
                
                # Check if a final trained model is available to load directly
                final_model_path = os.path.join(hp_output_dir, f"final_best_{model_class.__name__}.pth")
                if os.path.exists(final_model_path):
                    print(f"Loading pre-trained best model from hyperparameter search")
                    model = initialize_model(model_type, device, hyperparams)
                    model.load_state_dict(torch.load(final_model_path, map_location=device))
                    models[model_type] = model
                    test_loaders[model_type] = test_loader
                    
                    # Skip to next model type
                    continue
        else:
            # Use default hyperparameters
            hyperparams = get_hyperparameters(model_type)
        
        # Initialize model
        model = initialize_model(model_type, device, hyperparams)
        
        # Create optimizer with best parameters if available
        optimizer = None
        if args.hp_search and 'optimizer_hyperparams' in locals() and optimizer_hyperparams:
            import torch.optim as optim
            lr = optimizer_hyperparams.get('learning_rate', 0.0001)
            weight_decay = optimizer_hyperparams.get('weight_decay', 1e-5)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
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
            grad_clip=grad_clip,
            optimizer=optimizer
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
        
        if 'auc' in ensemble_results:
            print(f"Ensemble AUC: {ensemble_results['auc']:.4f}")
    
    print("\nTraining and evaluation completed!")
    
    # Final memory cleanup
    clear_cuda_memory()

if __name__ == "__main__":
    main()