#!/usr/bin/env python
"""
Hybrid Ensemble for ED-TNN Violence Detection
This script uses the ED-TNN as a feature extractor, extracts intermediate
topological features using a forward hook on the propagator layer, computes 
an anomaly score with IsolationForest, trains a RandomForest classifier, 
and ensembles the deep network’s predictions with the classical branch.
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score

# Import the ED-TNN model from your edtnn_violence_detection.py
from edtnn_violence_detection import EDTNN_ViolenceDetection  # :contentReference[oaicite:0]{index=0}
# Import the data preparation functions
from utils.dataprep import prepare_violence_nonviolence_data
from dataloader import get_dataloaders

# Global variable to store intermediate features extracted from the propagator layer
extracted_features = []

def hook_function(module, input, output):
    """
    Forward hook to capture the output of the propagator layer.
    The output is expected to be of shape [batch_size, num_nodes, feature_dim].
    We flatten it to [batch_size, num_nodes * feature_dim].
    """
    flattened = output.view(output.size(0), -1).detach().cpu().numpy()
    extracted_features.append(flattened)

def extract_intermediate_features(model, dataloader, device):
    """
    Pass the dataset through the model and extract intermediate features using the hook.
    
    Returns:
        features_array: NumPy array of extracted features.
        labels_array: Corresponding labels.
    """
    global extracted_features
    extracted_features = []  # Reset before extraction
    hook_handle = model.propagator.register_forward_hook(hook_function)
    
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Handle batch: supports both (video, label) and (video, pose, label)
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    video, pose, labels = batch
                    inputs = (video.to(device), pose.to(device))
                else:
                    video, labels = batch
                    inputs = video.to(device)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
            all_labels.append(labels.cpu().numpy())
            _ = model(inputs)  # forward pass triggers the hook

    hook_handle.remove()
    features_array = np.concatenate(extracted_features, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    return features_array, labels_array

def train_classical_ensemble(features, labels):
    """
    Train an IsolationForest to compute anomaly scores and then
    a RandomForest classifier using the features augmented by the anomaly score.
    
    Returns:
        clf: Trained RandomForestClassifier.
        iso_forest: Trained IsolationForest.
    """
    # Train IsolationForest on the features
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(features)
    # Compute anomaly (outlier) scores
    anomaly_scores = iso_forest.decision_function(features).reshape(-1, 1)
    # Append anomaly scores as an extra feature
    combined_features = np.hstack([features, anomaly_scores])
    # Train RandomForest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(combined_features, labels)
    return clf, iso_forest

def ensemble_predictions(model, classical_clf, iso_forest, dataloader, device, weight_deep=0.5, weight_classical=0.5):
    """
    Generate ensemble predictions by combining the ED-TNN softmax probabilities
    with the probabilities from the classical classifier.
    
    Returns:
        ensemble_preds: Final predicted labels.
        all_labels_array: Ground truth labels.
    """
    global extracted_features
    extracted_features = []  # Clear any previous features
    deep_probs_list = []
    all_labels = []
    
    # Register hook for test-time feature extraction
    hook_handle = model.propagator.register_forward_hook(hook_function)
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    video, pose, labels = batch
                    inputs = (video.to(device), pose.to(device))
                else:
                    video, labels = batch
                    inputs = video.to(device)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
            outputs = model(inputs)
            deep_probs = torch.softmax(outputs, dim=1).cpu().numpy()
            deep_probs_list.append(deep_probs)
            all_labels.append(labels.cpu().numpy())
    hook_handle.remove()
    
    # Get classical features from the hook and flatten them
    features_array = np.concatenate(extracted_features, axis=0)
    anomaly_scores = iso_forest.decision_function(features_array).reshape(-1, 1)
    combined_features = np.hstack([features_array, anomaly_scores])
    classical_probs = classical_clf.predict_proba(combined_features)
    
    deep_probs_all = np.concatenate(deep_probs_list, axis=0)
    all_labels_array = np.concatenate(all_labels, axis=0)
    
    # Combine probabilities via weighted average
    ensemble_probs = weight_deep * deep_probs_all + weight_classical * classical_probs
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    return ensemble_preds, all_labels_array

def main():
    parser = argparse.ArgumentParser(description="Hybrid Ensemble for ED-TNN Violence Detection")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for feature extraction")
    parser.add_argument("--data_dir", type=str, default="./Data/Processed/standardized", help="Directory for standardized dataset")
    parser.add_argument("--pose_dir", type=str, default=None, help="Directory for pose keypoints (optional)")
    parser.add_argument("--use_pose", action="store_true", help="Use pose data along with video frames")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use (-1 for CPU)")
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the ED-TNN model
    model = EDTNN_ViolenceDetection(num_classes=2, 
                                    knot_type='trefoil', 
                                    node_density=64, 
                                    features_per_node=16, 
                                    collapse_method='entropy', 
                                    use_pose=args.use_pose)
    model.to(device)
    
    # Optionally, load pre-trained weights here:
    # model.load_state_dict(torch.load("edtnn_pretrained.pth", map_location=device))
    
    # Prepare data: first obtain file lists from your data directory
    print("Preparing data...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        prepare_violence_nonviolence_data(args.data_dir)
    
    # Now, get dataloaders using the file lists (not a 'data_dir' keyword)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
        pose_dir=args.pose_dir,
        batch_size=args.batch_size,
        num_workers=4,
        model_type='3d_cnn'  # Adjust as needed; ED-TNN expects 3D CNN–formatted inputs
    )
    
    # --- Phase 1: Feature Extraction & Classical Model Training ---
    print("Extracting intermediate features from the training set...")
    train_features, train_labels_arr = extract_intermediate_features(model, train_loader, device)
    print(f"Extracted training features shape: {train_features.shape}")
    
    print("Training classical ensemble model...")
    classical_clf, iso_forest = train_classical_ensemble(train_features, train_labels_arr)
    print("Classical model trained.")
    
    # --- Phase 2: Ensemble Evaluation on Test Set ---
    print("Generating ensemble predictions on the test set...")
    ensemble_preds, test_labels_array = ensemble_predictions(model, classical_clf, iso_forest, test_loader, device)
    ensemble_acc = accuracy_score(test_labels_array, ensemble_preds)
    print(f"Ensemble Accuracy: {ensemble_acc * 100:.2f}%")
    
if __name__ == "__main__":
    main()
