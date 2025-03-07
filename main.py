#!/usr/bin/env python3
# main.py - Violence Detection main script

import os
import argparse
import torch
from utils.dataprep import prepare_violence_nonviolence_data
from dataloader import get_dataloaders
from train import train_model
from evaluations import evaluate_and_compare_models, ensemble_predictions
import hyperparameters as hp
from Models.model_3dcnn import Model3DCNN
from Models.model_2dcnn_lstm import Model2DCNNLSTM
from Models.model_transformer import VideoTransformer
from Models.model_i3d import TransferLearningI3D





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
    parser.add_argument("--model_types", nargs="+", 
                        default=['3d_cnn', '2d_cnn_lstm', 'transformer', 'i3d'],
                        help="Model types to train")
    parser.add_argument("--use_pose", action="store_true", 
                        help="Use pose features in addition to video frames")
    parser.add_argument("--gpu", type=int, default=0, 
                        help="GPU ID to use (-1 for CPU)")
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

def initialize_model(model_type, device, use_pose=False):
    """Initialize model based on model type"""
    if model_type == '3d_cnn':
        model = Model3DCNN().to(device)
    elif model_type == '2d_cnn_lstm':
        model = Model2DCNNLSTM(use_pose=use_pose).to(device)
    elif model_type == 'transformer':
        model = VideoTransformer(use_pose=use_pose).to(device)
    elif model_type == 'i3d':
        model = TransferLearningI3D(use_pose=use_pose).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

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
            num_workers=4,
            model_type=model_type
        )
        
        # Initialize model (for simple_cnn, no pose is used)
        model = initialize_model(model_type, device, args.use_pose)
        
        # Train model
        print(f"\n{'='*20} Training {model_type} {'='*20}")
        trained_model = train_model(
            model_name=model_type,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            device=device,
            output_dir=args.output_dir
        )
        
        models[model_type] = trained_model
        test_loaders[model_type] = test_loader
    
    # Evaluate all models
    print("\n\n" + "="*20 + " EVALUATION " + "="*20)
    results = evaluate_and_compare_models(models, test_loaders, device, args.output_dir)
    
    # Create ensemble if multiple models are trained
    if len(models) > 1:
        print("\n\n" + "="*20 + " ENSEMBLE RESULTS " + "="*20)
        ensemble_results = ensemble_predictions(models, test_loaders, device)
        
        # Log ensemble results
        print(f"Ensemble accuracy: {ensemble_results['accuracy']:.2f}%")
    
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main()
    
    
    #########JEG ELSKER JØDER ########