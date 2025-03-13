# simplified_hyperparameter_search.py
import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

from utils.dataprep import prepare_violence_nonviolence_data
from dataloader import get_dataloaders
from train import clear_cuda_memory
from Models.model_edtnn import ModelEDTNN, ResonanceLoss
from hybrid_edtnn_ensemble import HybridEDTNNEnsemble, FeatureExtractor

def train_simplified_edtnn(model, train_loader, val_loader, device, num_epochs=10, 
                         learning_rate=0.0001, resonance_weight=0.1, output_dir="./output"):
    """
    Simplified training function for ED-TNN to avoid dependency issues.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up loss function
    criterion = ResonanceLoss(
        model.topology,
        base_criterion=torch.nn.CrossEntropyLoss(),
        resonance_weight=resonance_weight
    )
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Initialize training history
    best_val_acc = 0.0
    patience = 5
    counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Handle different input types
            if model.use_pose and len(batch) == 3:  # Video + Pose + Label
                frames, pose, targets = batch
                frames, pose, targets = frames.to(device), pose.to(device), targets.to(device)
                inputs = (frames, pose)
            else:  # Video + Label
                frames, targets = batch
                frames, targets = frames.to(device), targets.to(device)
                inputs = frames
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets, model.entangled_layer)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / total
        epoch_train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Handle different input types
                if model.use_pose and len(batch) == 3:  # Video + Pose + Label
                    frames, pose, targets = batch
                    frames, pose, targets = frames.to(device), pose.to(device), targets.to(device)
                    inputs = (frames, pose)
                else:  # Video + Label
                    frames, targets = batch
                    frames, targets = frames.to(device), targets.to(device)
                    inputs = frames
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = criterion(outputs, targets, model.entangled_layer)
                
                # Update statistics
                val_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate epoch metrics
        epoch_val_loss = val_loss / total
        epoch_val_acc = 100. * correct / total
        
        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)
        
        # Print epoch summary
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        # Check for early stopping
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # Save best model
            checkpoint_path = os.path.join(output_dir, "best_model.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc
            }, checkpoint_path)
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break
        
        # Clear CUDA cache
        clear_cuda_memory()
    
    # Load best model
    checkpoint = torch.load(os.path.join(output_dir, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, best_val_acc

def run_simplified_search(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
                        edtnn_params, rf_params, if_params, ensemble_params,
                        batch_size=8, num_workers=4, device=torch.device("cuda"),
                        output_dir="./simplified_search", pose_dir=None, max_edtnn_epochs=10):
    """
    Run simplified hyperparameter search by training and evaluating one configuration.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
        pose_dir=pose_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        model_type='3d_cnn'
    )
    
    # Initialize and train ED-TNN model
    print("Initializing ED-TNN model...")
    edtnn_model = ModelEDTNN(
        num_classes=edtnn_params.get('num_classes', 2),
        knot_type=edtnn_params.get('knot_type', 'trefoil'),
        node_density=edtnn_params.get('node_density', 64),
        features_per_node=edtnn_params.get('features_per_node', 16),
        collapse_method=edtnn_params.get('collapse_method', 'entropy'),
        use_pose=edtnn_params.get('use_pose', pose_dir is not None)
    ).to(device)
    
    print("Training ED-TNN model...")
    edtnn_model, best_val_acc = train_simplified_edtnn(
        model=edtnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=max_edtnn_epochs,
        learning_rate=edtnn_params.get('learning_rate', 0.0001),
        resonance_weight=edtnn_params.get('resonance_weight', 0.1),
        output_dir=os.path.join(output_dir, "edtnn")
    )
    
    # Initialize hybrid ensemble
    print("\nInitializing Hybrid Ensemble...")
    hybrid_ensemble = HybridEDTNNEnsemble(
        edtnn_model=edtnn_model,
        device=device,
        output_dir=output_dir
    )
    
    # Configure ensemble with parameters
    hybrid_ensemble.edtnn_weight = ensemble_params.get('edtnn_weight', 0.6)
    hybrid_ensemble.rf_weight = ensemble_params.get('rf_weight', 0.4)
    
    # Configure Random Forest
    hybrid_ensemble.random_forest = RandomForestClassifier(
        n_estimators=rf_params.get('n_estimators', 100),
        max_depth=rf_params.get('max_depth', None),
        min_samples_split=rf_params.get('min_samples_split', 2),
        min_samples_leaf=rf_params.get('min_samples_leaf', 1),
        max_features=rf_params.get('max_features', 'sqrt'),
        bootstrap=rf_params.get('bootstrap', True),
        random_state=42,
        n_jobs=-1
    )
    
    # Configure Isolation Forest
    hybrid_ensemble.isolation_forest = IsolationForest(
        n_estimators=if_params.get('n_estimators', 100),
        contamination=if_params.get('contamination', 0.1),
        max_samples=if_params.get('max_samples', 'auto'),
        max_features=if_params.get('max_features', 1.0),
        random_state=42,
        n_jobs=-1
    )
    
    # Train hybrid ensemble
    print("Training Hybrid Ensemble...")
    val_results = hybrid_ensemble.train(train_loader, val_loader)
    
    # Evaluate on test set
    print("\nEvaluating Hybrid Ensemble...")
    test_results = hybrid_ensemble.evaluate(test_loader)
    
    # Save the ensemble
    hybrid_ensemble.save()
    
    # Save results
    results = {
        'edtnn_params': edtnn_params,
        'rf_params': rf_params,
        'if_params': if_params,
        'ensemble_params': ensemble_params,
        'edtnn_val_acc': best_val_acc,
        'validation': {
            'accuracy': float(val_results['accuracy']),
            'auc': float(val_results['auc'])
        },
        'test': {
            'accuracy': float(test_results['accuracy']),
            'auc': float(test_results['auc']),
            'edtnn_accuracy': float(test_results['edtnn_accuracy']),
            'rf_accuracy': float(test_results['rf_accuracy'])
        }
    }
    
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to", os.path.join(output_dir, "results.json"))
    print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
    print(f"Validation AUC: {val_results['auc']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test AUC: {test_results['auc']:.4f}")
    print(f"ED-TNN Accuracy: {test_results['edtnn_accuracy']:.4f}")
    print(f"Random Forest Accuracy: {test_results['rf_accuracy']:.4f}")
    
    return results

def incremental_search(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
                    device=torch.device("cuda"), output_dir="./incremental_search",
                    pose_dir=None, batch_size=8, num_workers=4, max_edtnn_epochs=10):
    """
    Run incremental hyperparameter search instead of a full grid search.
    This approach starts with a base configuration and then iteratively improves each component.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Base configuration
    base_config = {
        'edtnn': {
            'knot_type': 'trefoil',
            'node_density': 64,
            'features_per_node': 16,
            'learning_rate': 0.0001,
            'resonance_weight': 0.1,
            'use_pose': pose_dir is not None,
            'num_classes': 2,
            'collapse_method': 'entropy'
        },
        'rf': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True
        },
        'if': {
            'n_estimators': 100,
            'contamination': 0.1,
            'max_samples': 'auto',
            'max_features': 1.0
        },
        'ensemble': {
            'edtnn_weight': 0.6,
            'rf_weight': 0.4
        }
    }
    
    # Step 1: Start with the base configuration
    print("STEP 1: Testing base configuration")
    base_results = run_simplified_search(
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
        base_config['edtnn'], base_config['rf'], base_config['if'], base_config['ensemble'],
        batch_size=batch_size, num_workers=num_workers, device=device,
        output_dir=os.path.join(output_dir, "01_base"), pose_dir=pose_dir,
        max_edtnn_epochs=max_edtnn_epochs
    )
    
    best_val_auc = base_results['validation']['auc']
    best_config = base_config.copy()
    
    # Step 2: Optimize ED-TNN parameters
    print("\nSTEP 2: Optimizing ED-TNN parameters")
    
    # Test different knot types
    knot_types = ['trefoil', 'figure-eight']
    for knot_type in knot_types:
        if knot_type == best_config['edtnn']['knot_type']:
            continue  # Skip the one we already tested
        
        print(f"\nTesting knot_type: {knot_type}")
        
        # Create a new config with the modified parameter
        edtnn_params = best_config['edtnn'].copy()
        edtnn_params['knot_type'] = knot_type
        
        # Run with this configuration
        results = run_simplified_search(
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
            edtnn_params, best_config['rf'], best_config['if'], best_config['ensemble'],
            batch_size=batch_size, num_workers=num_workers, device=device,
            output_dir=os.path.join(output_dir, f"02_knot_{knot_type}"), pose_dir=pose_dir,
            max_edtnn_epochs=max_edtnn_epochs
        )
        
        # Update if better
        if results['validation']['auc'] > best_val_auc:
            best_val_auc = results['validation']['auc']
            best_config['edtnn'] = edtnn_params
            print(f"New best configuration found! Validation AUC: {best_val_auc:.4f}")
    
    # Test different node densities
    node_densities = [32, 64, 128]
    for node_density in node_densities:
        if node_density == best_config['edtnn']['node_density']:
            continue  # Skip the one we already tested
        
        print(f"\nTesting node_density: {node_density}")
        
        # Create a new config with the modified parameter
        edtnn_params = best_config['edtnn'].copy()
        edtnn_params['node_density'] = node_density
        
        # Run with this configuration
        results = run_simplified_search(
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
            edtnn_params, best_config['rf'], best_config['if'], best_config['ensemble'],
            batch_size=batch_size, num_workers=num_workers, device=device,
            output_dir=os.path.join(output_dir, f"03_density_{node_density}"), pose_dir=pose_dir,
            max_edtnn_epochs=max_edtnn_epochs
        )
        
        # Update if better
        if results['validation']['auc'] > best_val_auc:
            best_val_auc = results['validation']['auc']
            best_config['edtnn'] = edtnn_params
            print(f"New best configuration found! Validation AUC: {best_val_auc:.4f}")
    
    # Step 3: Optimize Random Forest parameters
    print("\nSTEP 3: Optimizing Random Forest parameters")
    
    # Test different n_estimators
    n_estimators_values = [50, 100, 200]
    for n_estimators in n_estimators_values:
        if n_estimators == best_config['rf']['n_estimators']:
            continue  # Skip the one we already tested
        
        print(f"\nTesting n_estimators: {n_estimators}")
        
        # Create a new config with the modified parameter
        rf_params = best_config['rf'].copy()
        rf_params['n_estimators'] = n_estimators
        
        # Run with this configuration
        results = run_simplified_search(
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
            best_config['edtnn'], rf_params, best_config['if'], best_config['ensemble'],
            batch_size=batch_size, num_workers=num_workers, device=device,
            output_dir=os.path.join(output_dir, f"04_rf_est_{n_estimators}"), pose_dir=pose_dir,
            max_edtnn_epochs=max_edtnn_epochs
        )
        
        # Update if better
        if results['validation']['auc'] > best_val_auc:
            best_val_auc = results['validation']['auc']
            best_config['rf'] = rf_params
            print(f"New best configuration found! Validation AUC: {best_val_auc:.4f}")
    
    # Step 4: Optimize ensemble weights
    print("\nSTEP 4: Optimizing ensemble weights")
    
    # Test different weight combinations
    weight_pairs = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (0.9, 0.1)]
    for edtnn_weight, rf_weight in weight_pairs:
        if edtnn_weight == best_config['ensemble']['edtnn_weight'] and rf_weight == best_config['ensemble']['rf_weight']:
            continue  # Skip the one we already tested
        
        print(f"\nTesting weights: edtnn={edtnn_weight}, rf={rf_weight}")
        
        # Create a new config with the modified parameter
        ensemble_params = best_config['ensemble'].copy()
        ensemble_params['edtnn_weight'] = edtnn_weight
        ensemble_params['rf_weight'] = rf_weight
        
        # Run with this configuration
        results = run_simplified_search(
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
            best_config['edtnn'], best_config['rf'], best_config['if'], ensemble_params,
            batch_size=batch_size, num_workers=num_workers, device=device,
            output_dir=os.path.join(output_dir, f"05_weights_{edtnn_weight}_{rf_weight}"), pose_dir=pose_dir,
            max_edtnn_epochs=max_edtnn_epochs
        )
        
        # Update if better
        if results['validation']['auc'] > best_val_auc:
            best_val_auc = results['validation']['auc']
            best_config['ensemble'] = ensemble_params
            print(f"New best configuration found! Validation AUC: {best_val_auc:.4f}")
    
    # Final: Run with the best configuration
    print("\nFINAL: Running with best configuration")
    final_results = run_simplified_search(
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
        best_config['edtnn'], best_config['rf'], best_config['if'], best_config['ensemble'],
        batch_size=batch_size, num_workers=num_workers, device=device,
        output_dir=os.path.join(output_dir, "06_final_best"), pose_dir=pose_dir,
        max_edtnn_epochs=max_edtnn_epochs*2  # Double epochs for final training
    )
    
    # Save best configuration
    with open(os.path.join(output_dir, "best_config.json"), 'w') as f:
        json.dump(best_config, f, indent=4)
    
    print("\nIncremental search completed!")
    print(f"Best configuration saved to {os.path.join(output_dir, 'best_config.json')}")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Final test AUC: {final_results['test']['auc']:.4f}")
    
    return best_config, final_results

def main():
    """Main function to run simplified hyperparameter search"""
    parser = argparse.ArgumentParser(description="Simplified Hyperparameter Search for Hybrid ED-TNN Ensemble")
    parser.add_argument("--data_dir", type=str, default="./Data/Processed/standardized",
                      help="Directory containing videos")
    parser.add_argument("--pose_dir", type=str, default=None,
                      help="Directory containing pose keypoints (optional)")
    parser.add_argument("--output_dir", type=str, default="./incremental_search",
                      help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for training")
    parser.add_argument("--gpu", type=int, default=0,
                      help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of worker processes for DataLoader")
    parser.add_argument("--max_edtnn_epochs", type=int, default=10,
                      help="Maximum number of epochs for ED-TNN training")
    parser.add_argument("--subset_size", type=int, default=100,
                      help="Number of samples to use for training subset")
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    print("Preparing data...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        prepare_violence_nonviolence_data(args.data_dir)
    
    # Subsample training data for efficiency
    if args.subset_size < len(train_paths):
        np.random.seed(42)
        indices = np.random.choice(len(train_paths), args.subset_size, replace=False)
        train_paths_subset = [train_paths[i] for i in indices]
        train_labels_subset = [train_labels[i] for i in indices]
    else:
        train_paths_subset = train_paths
        train_labels_subset = train_labels
    
    # Run incremental hyperparameter search
    best_config, final_results = incremental_search(
        train_paths_subset, train_labels_subset,
        val_paths, val_labels,
        test_paths, test_labels,
        device=device,
        output_dir=args.output_dir,
        pose_dir=args.pose_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_edtnn_epochs=args.max_edtnn_epochs
    )
    
    # Final summary
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"Best ED-TNN Configuration:")
    for k, v in best_config['edtnn'].items():
        print(f"  {k}: {v}")
    print(f"\nBest Random Forest Configuration:")
    for k, v in best_config['rf'].items():
        print(f"  {k}: {v}")
    print(f"\nBest Isolation Forest Configuration:")
    for k, v in best_config['if'].items():
        print(f"  {k}: {v}")
    print(f"\nBest Ensemble Configuration:")
    for k, v in best_config['ensemble'].items():
        print(f"  {k}: {v}")
    print("\nPerformance:")
    print(f"  Final Validation Accuracy: {final_results['validation']['accuracy']:.4f}")
    print(f"  Final Validation AUC: {final_results['validation']['auc']:.4f}")
    print(f"  Final Test Accuracy: {final_results['test']['accuracy']:.4f}")
    print(f"  Final Test AUC: {final_results['test']['auc']:.4f}")
    print(f"  ED-TNN Test Accuracy: {final_results['test']['edtnn_accuracy']:.4f}")
    print(f"  Random Forest Test Accuracy: {final_results['test']['rf_accuracy']:.4f}")


if __name__ == "__main__":
    main()