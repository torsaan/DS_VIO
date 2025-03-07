# ml_train.py
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns

from utils.dataprep import prepare_violence_nonviolence_data
from ml_dataloader import preprocess_and_extract_features, load_features_from_directory
from Models.ml_base import registry
from Models.ml_forest_models import RandomForestModel, GradientBoostingModel, XGBoostModel
from Models.ml_linear_models import SVMModel, LinearSVMModel, LogisticRegressionModel
from Models.ml_feature_models import PoseBasedModel, MotionBasedModel, VisualBasedModel
from Models.ml_ensemble_models import EnsembleViolenceClassifier, FeatureFusionModel

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train ML models for violence detection")
    parser.add_argument("--data_dir", type=str, default="./Data/Processed/standardized", 
                        help="Directory containing the violence detection dataset")
    parser.add_argument("--feature_dir", type=str, default="./features", 
                        help="Directory to save/load extracted features")
    parser.add_argument("--output_dir", type=str, default="./output/ml_models", 
                        help="Directory to save model outputs")
    parser.add_argument("--extract_features", action="store_true", 
                        help="Extract features before training (otherwise load from feature_dir)")
    parser.add_argument("--sample_rate", type=int, default=10, 
                        help="Number of frames to sample per video for feature extraction")
    parser.add_argument("--model_types", nargs="+", 
                        default=['random_forest', 'svm', 'xgboost', 'pose_model', 'motion_model'],
                        help="ML model types to train")
    parser.add_argument("--ensemble", action="store_true", 
                        help="Train ensemble model from best individual models")
    parser.add_argument("--fusion", action="store_true", 
                        help="Train feature fusion model")
    return parser.parse_args()

def plot_confusion_matrix(cm, classes, output_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_feature_importance(importances, output_path=None):
    """Plot feature importance"""
    # Convert importances to DataFrame for plotting
    import pandas as pd
    
    # Extract top 20 features
    top_n = 20
    if len(importances) > top_n:
        importances = importances[:top_n]
    
    # Create DataFrame
    df = pd.DataFrame({
        "Feature": [f[0] for f in importances],
        "Importance": [f[1] for f in importances]
    })
    
    # Sort by importance
    df = df.sort_values("Importance", ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Importance", y="Feature", data=df)
    plt.title(f"Top {top_n} Feature Importances")
    
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def prepare_feature_datasets(features, test_size=0.2, val_size=0.25, random_state=42):
    """
    Prepare feature datasets for training, validation, and testing
    
    Args:
        features: List of feature dictionaries
        test_size: Fraction of data to use for testing
        val_size: Fraction of remaining data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Train, validation, and test feature lists, plus feature type dictionaries
    """
    # Filter features with labels
    labeled_features = [f for f in features if f.get("label") is not None]
    
    # Extract labels
    labels = [f["label"] for f in labeled_features]
    
    # Split into train+val and test
    train_val_features, test_features, train_val_labels, test_labels = train_test_split(
        labeled_features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Split train+val into train and validation
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_val_features, train_val_labels, 
        test_size=val_size, random_state=random_state, stratify=train_val_labels
    )
    
    print(f"Training set: {len(train_features)} samples")
    print(f"Validation set: {len(val_features)} samples")
    print(f"Test set: {len(test_features)} samples")
    
    # Prepare dictionaries for each feature type
    train_dict = prepare_feature_dict(train_features)
    val_dict = prepare_feature_dict(val_features)
    test_dict = prepare_feature_dict(test_features)
    
    return train_features, val_features, test_features, train_dict, val_dict, test_dict

def prepare_feature_dict(features):
    """
    Prepare dictionary mapping feature types to arrays
    
    Args:
        features: List of feature dictionaries
        
    Returns:
        Dictionary mapping feature types to (X, y) tuples
    """
    # Define feature extractors
    extractors = {
        "pose": extract_pose_features,
        "optical_flow": extract_flow_features,
        "mhi": extract_mhi_features,
        "histograms": extract_histogram_features
    }
    
    # Initialize results
    result = {}
    
    # Process each feature type
    for feature_type, extractor in extractors.items():
        X = []
        y = []
        
        for feature_data in features:
            # Skip entries without labels
            if feature_data.get("label") is None:
                continue
            
            # Extract features for this type
            if feature_type == "pose" and "pose_keypoints" in feature_data:
                features = extractor(feature_data["pose_keypoints"])
                if len(features) > 0:
                    X.append(features)
                    y.append(feature_data["label"])
            elif feature_type == "optical_flow" and "optical_flow" in feature_data:
                features = extractor(feature_data["optical_flow"])
                if len(features) > 0:
                    X.append(features)
                    y.append(feature_data["label"])
            elif feature_type == "mhi" and "mhi_features" in feature_data:
                features = extractor(feature_data["mhi_features"])
                if len(features) > 0:
                    X.append(features)
                    y.append(feature_data["label"])
            elif feature_type == "histograms" and "histograms" in feature_data:
                features = extractor(feature_data["histograms"])
                if len(features) > 0:
                    X.append(features)
                    y.append(feature_data["label"])
        
        # Convert to numpy arrays
        if X and y:
            X = np.array(X)
            y = np.array(y)
            result[feature_type] = (X, y)
        else:
            print(f"Warning: No data for feature type {feature_type}")
    
    return result

def extract_pose_features(pose_keypoints):
    """Extract statistical features from pose keypoints"""
    if not pose_keypoints:
        return []
    
    # Convert to numpy array
    keypoints = np.array(pose_keypoints)
    
    # Skip if empty
    if keypoints.size == 0:
        return []
    
    # Calculate statistics over frames
    mean_keypoints = np.mean(keypoints, axis=0)
    std_keypoints = np.std(keypoints, axis=0)
    max_keypoints = np.max(keypoints, axis=0)
    
    # Calculate velocity (difference between consecutive frames)
    if keypoints.shape[0] > 1:
        velocity = np.diff(keypoints, axis=0)
        mean_velocity = np.mean(np.abs(velocity), axis=0)
        max_velocity = np.max(np.abs(velocity), axis=0)
    else:
        mean_velocity = np.zeros_like(mean_keypoints)
        max_velocity = np.zeros_like(mean_keypoints)
    
    # Combine features
    features = np.concatenate([
        mean_keypoints, std_keypoints, max_keypoints,
        mean_velocity, max_velocity
    ])
    
    return features

def extract_flow_features(optical_flow):
    """Extract statistical features from optical flow"""
    if not optical_flow:
        return []
    
    # Convert to numpy array
    flow = np.array(optical_flow)
    
    # Skip if empty
    if flow.size == 0:
        return []
    
    # Calculate statistics
    mean_flow = np.mean(flow, axis=0)
    std_flow = np.std(flow, axis=0)
    max_flow = np.max(flow, axis=0)
    
    # Combine features
    features = np.concatenate([mean_flow, std_flow, max_flow])
    
    return features

def extract_mhi_features(mhi_features):
    """Pass through MHI features"""
    if not mhi_features:
        return []
    
    return np.array(mhi_features)

def extract_histogram_features(histograms):
    """Extract statistical features from histograms"""
    if not histograms:
        return []
    
    # Convert to numpy array
    hists = np.array(histograms)
    
    # Skip if empty
    if hists.size == 0:
        return []
    
    # Calculate statistics
    mean_hist = np.mean(hists, axis=0)
    std_hist = np.std(hists, axis=0)
    
    # Combine features
    features = np.concatenate([mean_hist, std_hist])
    
    return features

def train_models(model_types, train_dict, val_dict, output_dir):
    """
    Train ML models
    
    Args:
        model_types: List of model types to train
        train_dict: Dictionary mapping feature types to training data
        val_dict: Dictionary mapping feature types to validation data
        output_dir: Directory to save model outputs
        
    Returns:
        Dictionary mapping model names to trained models
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results
    trained_models = {}
    evaluation_results = {}
    
    # Train models for each feature type
    for feature_type, (X_train, y_train) in train_dict.items():
        print(f"\n{'-'*40}")
        print(f"Training models for feature type: {feature_type}")
        print(f"Training data shape: {X_train.shape}, Labels: {np.bincount(y_train)}")
        
        # Create feature name list
        if feature_type == "pose":
            feature_names = [f"pose_{i}" for i in range(X_train.shape[1])]
        elif feature_type == "optical_flow":
            feature_names = [f"flow_{i}" for i in range(X_train.shape[1])]
        elif feature_type == "mhi":
            feature_names = [f"mhi_{i}" for i in range(X_train.shape[1])]
        elif feature_type == "histograms":
            feature_names = [f"hist_{i}" for i in range(X_train.shape[1])]
        else:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Train each model type
        for model_type in model_types:
            if model_type in ["fusion", "ensemble"]:
                # Handle these separately
                continue
            
            print(f"\nTraining {model_type} model...")
            
            # Create model
            if model_type == "random_forest":
                model = RandomForestModel(select_k_best=min(100, X_train.shape[1]//2))
            elif model_type == "gradient_boosting":
                model = GradientBoostingModel()
            elif model_type == "xgboost":
                model = XGBoostModel()
            elif model_type == "svm":
                model = SVMModel(kernel='rbf', C=10.0)
            elif model_type == "linear_svm":
                model = LinearSVMModel()
            elif model_type == "logistic_regression":
                model = LogisticRegressionModel()
            elif model_type == "pose_model":
                if feature_type != "pose":
                    continue  # Skip for other feature types
                model = PoseBasedModel()
            elif model_type == "motion_model":
                if feature_type not in ["optical_flow", "mhi"]:
                    continue  # Skip for other feature types
                model = MotionBasedModel()
            elif model_type == "visual_model":
                if feature_type != "histograms":
                    continue  # Skip for other feature types
                model = VisualBasedModel()
            else:
                model = registry.create(model_type)
            
            # Train model
            model.train(X_train, y_train, feature_names)
            
            # Get feature importances if available
            importances = model.get_feature_importances(feature_names)
            
            # Plot feature importance
            if importances:
                importance_path = os.path.join(output_dir, f"{feature_type}_{model_type}_importance.png")
                plot_feature_importance(importances, importance_path)
            
            # Evaluate on validation set
            if feature_type in val_dict:
                X_val, y_val = val_dict[feature_type]
                eval_results = model.evaluate(X_val, y_val)
                
                # Print results
                print(f"Validation accuracy: {eval_results['accuracy']:.4f}")
                print(f"Validation precision: {eval_results['precision']:.4f}")
                print(f"Validation recall: {eval_results['recall']:.4f}")
                print(f"Validation F1 score: {eval_results['f1_score']:.4f}")
                
                # Plot confusion matrix
                cm_path = os.path.join(output_dir, f"{feature_type}_{model_type}_cm.png")
                plot_confusion_matrix(
                    eval_results["confusion_matrix"],
                    ["Non-Violence", "Violence"],
                    cm_path
                )
                
                # Store evaluation results
                model_key = f"{feature_type}_{model_type}"
                evaluation_results[model_key] = eval_results
            
            # Save model
            model_key = f"{feature_type}_{model_type}"
            model_dir = os.path.join(output_dir, feature_type)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{model_key}.joblib")
            model.save(model_path)
            
            # Store model
            trained_models[model_key] = model
    
    # Save evaluation results
    eval_path = os.path.join(output_dir, "evaluation_results.json")
    with open(eval_path, 'w') as f:
        # Filter out non-serializable elements
        results_to_save = {}
        for key, value in evaluation_results.items():
            results_to_save[key] = {
                "accuracy": value["accuracy"],
                "precision": value["precision"],
                "recall": value["recall"],
                "f1_score": value["f1_score"],
                "confusion_matrix": value["confusion_matrix"].tolist()
            }
        json.dump(results_to_save, f, indent=2)
    
    return trained_models, evaluation_results

def train_ensemble(trained_models, evaluation_results, train_dict, val_dict, output_dir):
    """
    Train ensemble model from best individual models
    
    Args:
        trained_models: Dictionary mapping model keys to trained models
        evaluation_results: Dictionary mapping model keys to evaluation results
        train_dict: Dictionary mapping feature types to training data
        val_dict: Dictionary mapping feature types to validation data
        output_dir: Directory to save model outputs
        
    Returns:
        Trained ensemble model
    """
    print(f"\n{'-'*40}")
    print("Training ensemble model...")
    
    # Select best model for each feature type based on F1 score
    best_models = {}
    for feature_type in train_dict.keys():
        best_f1 = -1
        best_key = None
        
        for model_key, results in evaluation_results.items():
            if model_key.startswith(feature_type) and results["f1_score"] > best_f1:
                best_f1 = results["f1_score"]
                best_key = model_key
        
        if best_key is not None:
            best_models[feature_type] = {
                "key": best_key,
                "model": trained_models[best_key],
                "f1_score": best_f1
            }
    
    # Print selected models
    print("Selected models for ensemble:")
    for feature_type, model_info in best_models.items():
        print(f"  {feature_type}: {model_info['key']} (F1: {model_info['f1_score']:.4f})")
    
    # Create ensemble model
    ensemble = EnsembleViolenceClassifier(
        models=[
            {
                "model": model_info["model"],
                "feature_type": feature_type,
                "weight": model_info["f1_score"]  # Weight by F1 score
            }
            for feature_type, model_info in best_models.items()
        ],
        voting="soft"
    )
    
    # Train ensemble model
    # Note: Individual models are already trained, so we just need to set the trained flag
    ensemble.trained = True
    
    # Evaluate on validation set
    if val_dict:
        eval_results = ensemble.evaluate(val_dict, [val_dict[k][1] for k in val_dict.keys() if k in best_models][0])
        
        # Print results
        print(f"Ensemble validation accuracy: {eval_results['accuracy']:.4f}")
        print(f"Ensemble validation precision: {eval_results['precision']:.4f}")
        print(f"Ensemble validation recall: {eval_results['recall']:.4f}")
        print(f"Ensemble validation F1 score: {eval_results['f1_score']:.4f}")
        
        # Plot confusion matrix
        cm_path = os.path.join(output_dir, "ensemble_cm.png")
        plot_confusion_matrix(
            eval_results["confusion_matrix"],
            ["Non-Violence", "Violence"],
            cm_path
        )
    
    # Save ensemble model
    ensemble_dir = os.path.join(output_dir, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    ensemble.save(ensemble_dir)
    
    return ensemble

def train_fusion_model(train_dict, val_dict, output_dir, model_type="random_forest"):
    """
    Train feature fusion model
    
    Args:
        train_dict: Dictionary mapping feature types to training data
        val_dict: Dictionary mapping feature types to validation data
        output_dir: Directory to save model outputs
        model_type: Base model type for fusion
        
    Returns:
        Trained fusion model
    """
    print(f"\n{'-'*40}")
    print(f"Training feature fusion model with {model_type}...")
    
    # Create fusion model
    fusion_model = FeatureFusionModel(base_model_id=model_type)
    
    # Add feature types
    for feature_type in train_dict.keys():
        if feature_type == "pose":
            fusion_model.add_feature_type(feature_type, extract_pose_features)
        elif feature_type == "optical_flow":
            fusion_model.add_feature_type(feature_type, extract_flow_features)
        elif feature_type == "mhi":
            fusion_model.add_feature_type(feature_type, extract_mhi_features)
        elif feature_type == "histograms":
            fusion_model.add_feature_type(feature_type, extract_histogram_features)
    
    # Prepare fused training data
    X_train_fused = {}
    y_train = None
    for feature_type, (X, y) in train_dict.items():
        X_train_fused[feature_type] = X
        if y_train is None:
            y_train = y
    
    # Train fusion model
    fusion_model.fit(X_train_fused, y_train)
    
    # Evaluate on validation set
    if val_dict:
        X_val_fused = {}
        y_val = None
        for feature_type, (X, y) in val_dict.items():
            X_val_fused[feature_type] = X
            if y_val is None:
                y_val = y
        
        eval_results = fusion_model.evaluate(X_val_fused, y_val)
        
        # Print results
        print(f"Fusion validation accuracy: {eval_results['accuracy']:.4f}")
        print(f"Fusion validation precision: {eval_results['precision']:.4f}")
        print(f"Fusion validation recall: {eval_results['recall']:.4f}")
        print(f"Fusion validation F1 score: {eval_results['f1_score']:.4f}")
        
        # Plot confusion matrix
        cm_path = os.path.join(output_dir, "fusion_cm.png")
        plot_confusion_matrix(
            eval_results["confusion_matrix"],
            ["Non-Violence", "Violence"],
            cm_path
        )
    
    # Save fusion model
    fusion_dir = os.path.join(output_dir, "fusion")
    os.makedirs(fusion_dir, exist_ok=True)
    fusion_model.save(os.path.join(fusion_dir, "fusion_model.joblib"))
    
    return fusion_model

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = prepare_violence_nonviolence_data(args.data_dir)
    
    # Extract or load features
    if args.extract_features:
        print("Extracting features...")
        # Create feature extractor
        os.makedirs(args.feature_dir, exist_ok=True)
        
        # Extract features for each set
        print("Extracting training features...")
        train_features = preprocess_and_extract_features(
            train_paths, train_labels, args.feature_dir, args.sample_rate
        )
        
        print("Extracting validation features...")
        val_features = preprocess_and_extract_features(
            val_paths, val_labels, args.feature_dir, args.sample_rate
        )
        
        print("Extracting test features...")
        test_features = preprocess_and_extract_features(
            test_paths, test_labels, args.feature_dir, args.sample_rate
        )
    else:
        print("Loading features from directory...")
        all_features = load_features_from_directory(args.feature_dir)
        
        # Split into train, val, test
        train_features, val_features, test_features, train_dict, val_dict, test_dict = prepare_feature_datasets(all_features)
    
    # Prepare feature dictionaries if not already done
    if 'train_dict' not in locals():
        train_dict = prepare_feature_dict(train_features)
        val_dict = prepare_feature_dict(val_features)
        test_dict = prepare_feature_dict(test_features)
    
    # Train models
    print("Training ML models...")
    trained_models, evaluation_results = train_models(
        args.model_types, train_dict, val_dict, args.output_dir
    )
    
    # Train ensemble model if requested
    if args.ensemble:
        ensemble_model = train_ensemble(
            trained_models, evaluation_results, train_dict, val_dict, args.output_dir
        )
    
    # Train fusion model if requested
    if args.fusion:
        fusion_model = train_fusion_model(
            train_dict, val_dict, args.output_dir
        )
    
    # Evaluate on test set
    print(f"\n{'-'*40}")
    print("Evaluating on test set...")
    
    # Find best model based on validation F1 score
    best_model_key = max(evaluation_results.items(), key=lambda x: x[1]["f1_score"])[0]
    best_model = trained_models[best_model_key]
    best_feature_type = best_model_key.split('_')[0]
    
    print(f"Best model: {best_model_key} (F1: {evaluation_results[best_model_key]['f1_score']:.4f})")
    
    # Evaluate on test set
    if best_feature_type in test_dict:
        X_test, y_test = test_dict[best_feature_type]
        test_results = best_model.evaluate(X_test, y_test)
        
        # Print results
        print(f"Test accuracy: {test_results['accuracy']:.4f}")
        print(f"Test precision: {test_results['precision']:.4f}")
        print(f"Test recall: {test_results['recall']:.4f}")
        print(f"Test F1 score: {test_results['f1_score']:.4f}")
        
        # Plot confusion matrix
        cm_path = os.path.join(args.output_dir, "best_model_test_cm.png")
        plot_confusion_matrix(
            test_results["confusion_matrix"],
            ["Non-Violence", "Violence"],
            cm_path
        )
    
    # Evaluate ensemble on test set if available
    if args.ensemble and 'ensemble_model' in locals():
        print("\nEvaluating ensemble model on test set...")
        test_results = ensemble_model.evaluate(test_dict, [test_dict[k][1] for k in test_dict.keys() if k in best_models][0])
        
        # Print results
        print(f"Ensemble test accuracy: {test_results['accuracy']:.4f}")
        print(f"Ensemble test precision: {test_results['precision']:.4f}")
        print(f"Ensemble test recall: {test_results['recall']:.4f}")
        print(f"Ensemble test F1 score: {test_results['f1_score']:.4f}")
        
        # Plot confusion matrix
        cm_path = os.path.join(args.output_dir, "ensemble_test_cm.png")
        plot_confusion_matrix(
            test_results["confusion_matrix"],
            ["Non-Violence", "Violence"],
            cm_path
        )
    
    # Evaluate fusion model on test set if available
    if args.fusion and 'fusion_model' in locals():
        print("\nEvaluating fusion model on test set...")
        X_test_fused = {}
        y_test = None
        for feature_type, (X, y) in test_dict.items():
            X_test_fused[feature_type] = X
            if y_test is None:
                y_test = y
        
        test_results = fusion_model.evaluate(X_test_fused, y_test)
        
        # Print results
        print(f"Fusion test accuracy: {test_results['accuracy']:.4f}")
        print(f"Fusion test precision: {test_results['precision']:.4f}")
        print(f"Fusion test recall: {test_results['recall']:.4f}")
        print(f"Fusion test F1 score: {test_results['f1_score']:.4f}")
        
        # Plot confusion matrix
        cm_path = os.path.join(args.output_dir, "fusion_test_cm.png")
        plot_confusion_matrix(
            test_results["confusion_matrix"],
            ["Non-Violence", "Violence"],
            cm_path
        )
    
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main()