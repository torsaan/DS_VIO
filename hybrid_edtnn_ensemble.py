# hybrid_edtnn_ensemble.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib

from utils.dataprep import prepare_violence_nonviolence_data
from dataloader import get_dataloaders
from train import clear_cuda_memory
from evaluations import plot_confusion_matrix
from Models.model_edtnn import ModelEDTNN, ResonanceLoss


class FeatureExtractor:
    """
    Helper class to extract features from intermediate layers of the ED-TNN model.
    """
    def __init__(self, model, layer_name='propagator'):
        """
        Initialize the feature extractor.
        
        Args:
            model: The ED-TNN model
            layer_name: Name of the layer to extract features from
        """
        self.model = model
        self.layer_name = layer_name
        self.features = None
        
        # Register forward hook on the specified layer
        if layer_name == 'propagator':
            self.hook = model.propagator.register_forward_hook(self.hook_fn)
        elif layer_name == 'entangled_layer':
            self.hook = model.entangled_layer.register_forward_hook(self.hook_fn)
        else:
            raise ValueError(f"Unsupported layer name: {layer_name}")
    
    def hook_fn(self, module, input, output):
        """Hook function to capture layer output"""
        self.features = output
    
    def extract_features(self, loader, device):
        """
        Extract features for all samples in the loader.
        
        Args:
            loader: DataLoader with the dataset
            device: Device to run the model on
            
        Returns:
            Extracted features and labels
        """
        self.model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Extracting {self.layer_name} features"):
                # Handle different input types (with or without pose data)
                if self.model.use_pose and len(batch) == 3:  # Video + Pose + Label
                    frames, pose, targets = batch
                    frames, pose = frames.to(device), pose.to(device)
                    inputs = (frames, pose)
                else:  # Video + Label
                    frames, targets = batch
                    frames = frames.to(device)
                    inputs = frames
                
                # Forward pass to trigger the hook
                self.model(inputs)
                
                # Get the features from the hook
                batch_features = self.features.cpu().numpy()
                
                # Store features and labels
                all_features.append(batch_features)
                all_labels.extend(targets.numpy())
        
        # Concatenate all batches
        features = np.vstack([f.reshape(f.shape[0], -1) for f in all_features])
        labels = np.array(all_labels)
        
        return features, labels
    
    def remove_hook(self):
        """Remove the hook when done to free resources"""
        self.hook.remove()


class HybridEDTNNEnsemble:
    """
    A hybrid ensemble that combines ED-TNN with classical ML models.
    """
    def __init__(self, edtnn_model, device, output_dir="./output/hybrid_edtnn"):
        """
        Initialize the hybrid ensemble.
        
        Args:
            edtnn_model: Trained ED-TNN model
            device: Device to run the model on
            output_dir: Directory to save results
        """
        self.edtnn_model = edtnn_model
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize classical ML components
        self.feature_extractor = FeatureExtractor(edtnn_model, layer_name='propagator')
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.random_forest = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        
        # Weights for ensemble averaging
        self.edtnn_weight = 0.6
        self.rf_weight = 0.4
    
    def train(self, train_loader, val_loader):
        """
        Train the hybrid ensemble.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            
        Returns:
            Validation metrics
        """
        # Step 1: Extract features from ED-TNN
        print("Extracting features from ED-TNN...")
        train_features, train_labels = self.feature_extractor.extract_features(train_loader, self.device)
        val_features, val_labels = self.feature_extractor.extract_features(val_loader, self.device)
        
        # Step 2: Scale the features
        print("Scaling features...")
        train_features_scaled = self.scaler.fit_transform(train_features)
        val_features_scaled = self.scaler.transform(val_features)
        
        # Step 3: Train Isolation Forest for anomaly detection
        print("Training Isolation Forest...")
        self.isolation_forest.fit(train_features_scaled)
        
        # Compute anomaly scores
        train_anomaly_scores = -self.isolation_forest.score_samples(train_features_scaled)
        val_anomaly_scores = -self.isolation_forest.score_samples(val_features_scaled)
        
        # Add anomaly scores as an extra feature
        train_features_with_anomaly = np.column_stack([train_features_scaled, train_anomaly_scores])
        val_features_with_anomaly = np.column_stack([val_features_scaled, val_anomaly_scores])
        
        # Step 4: Train Random Forest with the combined features
        print("Training Random Forest classifier...")
        self.random_forest.fit(train_features_with_anomaly, train_labels)
        
        # Step 5: Evaluate on validation set
        # Get Random Forest predictions
        rf_probs = self.random_forest.predict_proba(val_features_with_anomaly)
        
        # Get ED-TNN predictions
        edtnn_probs = self._get_edtnn_probabilities(val_loader)
        
        # Combine predictions
        ensemble_probs = self.edtnn_weight * edtnn_probs + self.rf_weight * rf_probs
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(val_labels, ensemble_preds)
        try:
            auc = roc_auc_score(val_labels, ensemble_probs[:, 1])
        except:
            auc = 0.0
        
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation AUC: {auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'predictions': ensemble_preds,
            'probabilities': ensemble_probs,
            'labels': val_labels
        }
    
    def evaluate(self, test_loader):
        """
        Evaluate the hybrid ensemble on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Test metrics
        """
        # Extract features from ED-TNN
        test_features, test_labels = self.feature_extractor.extract_features(test_loader, self.device)
        
        # Scale the features
        test_features_scaled = self.scaler.transform(test_features)
        
        # Compute anomaly scores
        test_anomaly_scores = -self.isolation_forest.score_samples(test_features_scaled)
        
        # Add anomaly scores as an extra feature
        test_features_with_anomaly = np.column_stack([test_features_scaled, test_anomaly_scores])
        
        # Get Random Forest predictions
        rf_probs = self.random_forest.predict_proba(test_features_with_anomaly)
        
        # Get ED-TNN predictions
        edtnn_probs = self._get_edtnn_probabilities(test_loader)
        
        # Combine predictions
        ensemble_probs = self.edtnn_weight * edtnn_probs + self.rf_weight * rf_probs
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, ensemble_preds)
        try:
            auc = roc_auc_score(test_labels, ensemble_probs[:, 1])
        except:
            auc = 0.0
        
        # Generate classification report
        report = classification_report(test_labels, ensemble_preds, 
                                     target_names=["NonViolence", "Violence"],
                                     output_dict=True)
        
        # Generate confusion matrix
        cm = confusion_matrix(test_labels, ensemble_preds)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            cm,
            output_path=os.path.join(self.output_dir, 'hybrid_confusion_matrix.png')
        )
        
        # Calculate metrics for ED-TNN and Random Forest individually
        rf_preds = np.argmax(rf_probs, axis=1)
        edtnn_preds = np.argmax(edtnn_probs, axis=1)
        
        rf_accuracy = accuracy_score(test_labels, rf_preds)
        edtnn_accuracy = accuracy_score(test_labels, edtnn_preds)
        
        print(f"ED-TNN Accuracy: {edtnn_accuracy:.4f}")
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        print(f"Hybrid Ensemble Accuracy: {accuracy:.4f}")
        print(f"Hybrid Ensemble AUC: {auc:.4f}")
        
        # Save feature importance plot
        self._plot_feature_importance()
        
        # Save anomaly score visualization
        self._plot_anomaly_scores(test_features_scaled, test_anomaly_scores, test_labels)
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'predictions': ensemble_preds,
            'probabilities': ensemble_probs,
            'labels': test_labels,
            'report': report,
            'confusion_matrix': cm,
            'edtnn_accuracy': edtnn_accuracy,
            'rf_accuracy': rf_accuracy
        }
    
    def _get_edtnn_probabilities(self, loader):
        """Get probability predictions from the ED-TNN model"""
        self.edtnn_model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Getting ED-TNN predictions"):
                # Handle different input types (with or without pose data)
                if self.edtnn_model.use_pose and len(batch) == 3:  # Video + Pose + Label
                    frames, pose, _ = batch
                    frames, pose = frames.to(self.device), pose.to(self.device)
                    inputs = (frames, pose)
                else:  # Video + Label
                    frames, _ = batch
                    frames = frames.to(self.device)
                    inputs = frames
                
                # Forward pass
                outputs = self.edtnn_model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                all_probs.extend(probs)
        
        return np.array(all_probs)
    
    def _plot_feature_importance(self):
        """Plot feature importance from Random Forest"""
        importances = self.random_forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot top 20 features or all if less than 20
        n_features = min(20, len(importances))
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(n_features), importances[indices[:n_features]], align="center")
        plt.xticks(range(n_features), indices[:n_features])
        plt.xlim([-1, n_features])
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "feature_importance.png"))
        plt.close()
    
    def _plot_anomaly_scores(self, features, anomaly_scores, labels):
        """Plot anomaly scores and their correlation with violence class"""
        # Use t-SNE to reduce dimensionality for visualization
        from sklearn.manifold import TSNE
        
        # Sample if too many points
        max_points = 1000
        if len(features) > max_points:
            indices = np.random.choice(len(features), max_points, replace=False)
            sampled_features = features[indices]
            sampled_scores = anomaly_scores[indices]
            sampled_labels = labels[indices]
        else:
            sampled_features = features
            sampled_scores = anomaly_scores
            sampled_labels = labels
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(sampled_features)
        
        # Plot
        plt.figure(figsize=(15, 10))
        
        # First subplot: t-SNE colored by class
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=sampled_labels, 
                             s=50, alpha=0.7, cmap='coolwarm')
        plt.colorbar(scatter, label='Class (0=NonViolence, 1=Violence)')
        plt.title('t-SNE Visualization of Features (Colored by Class)')
        
        # Second subplot: t-SNE colored by anomaly score
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=sampled_scores, 
                             s=50, alpha=0.7, cmap='YlOrRd')
        plt.colorbar(scatter, label='Anomaly Score')
        plt.title('t-SNE Visualization of Features (Colored by Anomaly Score)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "anomaly_visualization.png"))
        plt.close()
        
        # Plot histogram of anomaly scores by class
        plt.figure(figsize=(12, 6))
        
        violence_scores = anomaly_scores[labels == 1]
        nonviolence_scores = anomaly_scores[labels == 0]
        
        plt.hist(nonviolence_scores, bins=50, alpha=0.5, label='NonViolence')
        plt.hist(violence_scores, bins=50, alpha=0.5, label='Violence')
        
        plt.xlabel('Anomaly Score')
        plt.ylabel('Count')
        plt.title('Distribution of Anomaly Scores by Class')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "anomaly_distribution.png"))
        plt.close()
    
    def save(self):
        """Save all components of the hybrid ensemble"""
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.output_dir, "scaler.joblib"))
        
        # Save Isolation Forest
        joblib.dump(self.isolation_forest, os.path.join(self.output_dir, "isolation_forest.joblib"))
        
        # Save Random Forest
        joblib.dump(self.random_forest, os.path.join(self.output_dir, "random_forest.joblib"))
        
        # Save configuration
        import json
        config = {
            'edtnn_weight': self.edtnn_weight,
            'rf_weight': self.rf_weight,
            'topology_type': self.edtnn_model.topology.knot_type,
            'node_density': self.edtnn_model.topology.node_density,
            'features_per_node': self.edtnn_model.features_per_node
        }
        
        with open(os.path.join(self.output_dir, "ensemble_config.json"), 'w') as f:
            json.dump(config, f, indent=4)
    
    def load(self):
        """Load all components of the hybrid ensemble"""
        # Load scaler
        self.scaler = joblib.load(os.path.join(self.output_dir, "scaler.joblib"))
        
        # Load Isolation Forest
        self.isolation_forest = joblib.load(os.path.join(self.output_dir, "isolation_forest.joblib"))
        
        # Load Random Forest
        self.random_forest = joblib.load(os.path.join(self.output_dir, "random_forest.joblib"))
        
        # Load configuration
        import json
        with open(os.path.join(self.output_dir, "ensemble_config.json"), 'r') as f:
            config = json.load(f)
        
        self.edtnn_weight = config['edtnn_weight']
        self.rf_weight = config['rf_weight']