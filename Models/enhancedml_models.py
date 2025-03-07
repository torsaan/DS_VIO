# Models/enhanced_ml_models.py
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

class MLModelFactory:
    """Factory for creating and managing ML models for violence detection"""
    
    def __init__(self, model_dir="./saved_models/ml"):
        """
        Initialize ML model factory
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Available model configurations
        self.model_configs = {
            # Random Forest models
            "rf_basic": {
                "name": "Random Forest (Basic)",
                "model": RandomForestClassifier(n_estimators=100, random_state=42),
                "feature_selector": None
            },
            "rf_advanced": {
                "name": "Random Forest (Advanced)",
                "model": RandomForestClassifier(
                    n_estimators=200, 
                    max_depth=None, 
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42
                ),
                "feature_selector": SelectKBest(f_classif, k=100)
            },
            
            # SVM models
            "svm_linear": {
                "name": "SVM (Linear)",
                "model": SVC(kernel='linear', C=1.0, probability=True, random_state=42),
                "feature_selector": None
            },
            "svm_rbf": {
                "name": "SVM (RBF)",
                "model": SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42),
                "feature_selector": SelectKBest(f_classif, k=50)
            },
            
            # Gradient Boosting
            "gb": {
                "name": "Gradient Boosting",
                "model": GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                ),
                "feature_selector": None
            },
            
            # XGBoost
            "xgb": {
                "name": "XGBoost",
                "model": XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                ),
                "feature_selector": None
            }
        }
    
    def create_model(self, model_type):
        """
        Create a model pipeline with feature selection and scaling
        
        Args:
            model_type: Type of model to create (must be in model_configs)
            
        Returns:
            scikit-learn Pipeline for the specified model
        """
        if model_type not in self.model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        config = self.model_configs[model_type]
        
        # Build pipeline components
        steps = [("scaler", StandardScaler())]
        
        # Add feature selector if specified
        if config["feature_selector"] is not None:
            steps.append(("feature_selector", config["feature_selector"]))
        
        # Add the model
        steps.append(("model", config["model"]))
        
        # Create and return the pipeline
        return Pipeline(steps)
    
    def train_model(self, model_type, X_train, y_train, feature_names=None):
        """
        Train a model on the provided data
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training labels
            feature_names: Names of features (optional)
            
        Returns:
            Trained model pipeline
        """
        # Create model pipeline
        pipeline = self.create_model(model_type)
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Get feature importances if available and feature names are provided
        if feature_names is not None:
            importances = self._get_feature_importances(pipeline, feature_names)
            print(f"Top 10 features for {model_type}:")
            for feature, importance in importances[:10]:
                print(f"  {feature}: {importance:.4f}")
        
        return pipeline
    
    def _get_feature_importances(self, pipeline, feature_names):
        """
        Extract feature importances from a trained pipeline
        
        Args:
            pipeline: Trained model pipeline
            feature_names: Names of features
            
        Returns:
            List of (feature_name, importance) tuples, sorted by importance
        """
        # Get the feature selector if present
        feature_selector = None
        for name, step in pipeline.named_steps.items():
            if name == "feature_selector" and step is not None:
                feature_selector = step
                break
        
        # Get the model
        model = pipeline.named_steps["model"]
        
        # If we have a feature selector, get the selected features
        selected_indices = np.arange(len(feature_names))
        if feature_selector is not None and hasattr(feature_selector, "get_support"):
            selected_indices = feature_selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
        else:
            selected_features = feature_names
        
        # Get feature importances based on model type
        importances = []
        
        if hasattr(model, "feature_importances_"):
            # For tree-based models
            importances = [(selected_features[i], imp) for i, imp in enumerate(model.feature_importances_)]
        elif hasattr(model, "coef_"):
            # For linear models
            if len(model.coef_.shape) == 1:
                # Binary classification
                importances = [(selected_features[i], abs(imp)) for i, imp in enumerate(model.coef_)]
            else:
                # Multi-class classification
                importances = [(selected_features[i], np.mean(abs(model.coef_[:, i]))) for i in range(len(selected_features))]
        
        # Sort by importance (descending)
        importances.sort(key=lambda x: x[1], reverse=True)
        
        return importances
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate a trained model on test data
        
        Args:
            model: Trained model pipeline
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Return evaluation results
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix,
            "predictions": y_pred,
            "probabilities": y_proba
        }
    
    def save_model(self, model, model_type, save_dir=None):
        """
        Save a trained model to disk
        
        Args:
            model: Trained model pipeline
            model_type: Type of model
            save_dir: Directory to save the model (defaults to self.model_dir)
            
        Returns:
            Path to the saved model
        """
        if save_dir is None:
            save_dir = self.model_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, f"{model_type}.joblib")
        joblib.dump(model, model_path)
        
        print(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_type, load_dir=None):
        """
        Load a trained model from disk
        
        Args:
            model_type: Type of model to load
            load_dir: Directory to load the model from (defaults to self.model_dir)
            
        Returns:
            Loaded model pipeline
        """
        if load_dir is None:
            load_dir = self.model_dir
        
        model_path = os.path.join(load_dir, f"{model_type}.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        return model


class ViolenceClassifier:
    """
    Violence classifier using multiple ML models and feature types
    """
    
    def __init__(self, model_types=None, feature_types=None):
        """
        Initialize classifier
        
        Args:
            model_types: List of model types to use
            feature_types: List of feature types to use
        """
        self.model_factory = MLModelFactory()
        
        # Default model types
        self.model_types = model_types or ["rf_basic", "svm_rbf", "xgb"]
        
        # Default feature types
        self.feature_types = feature_types or [
            "pose", "optical_flow", "mhi", "histograms"
        ]
        
        # Models for each feature type
        self.models = {}
        
        # Feature extractors
        self.feature_extractors = {}
    
    def _prepare_feature_extractors(self):
        """Prepare feature extraction functions for each feature type"""
        
        # Pose features
        def extract_pose_features(pose_keypoints):
            """Extract statistical features from pose keypoints"""
            if not pose_keypoints:
                return []
            
            # Convert to numpy array
            keypoints = np.array(pose_keypoints)
            
            # Calculate statistics over frames
            mean_keypoints = np.mean(keypoints, axis=0)
            std_keypoints = np.std(keypoints, axis=0)
            max_keypoints = np.max(keypoints, axis=0)
            
            # Calculate velocity (difference between consecutive frames)
            velocity = np.diff(keypoints, axis=0)
            mean_velocity = np.mean(np.abs(velocity), axis=0)
            max_velocity = np.max(np.abs(velocity), axis=0)
            
            # Combine features
            features = np.concatenate([
                mean_keypoints, std_keypoints, max_keypoints,
                mean_velocity, max_velocity
            ])
            
            return features
        
        # Optical flow features
        def extract_flow_features(optical_flow):
            """Extract statistical features from optical flow"""
            if not optical_flow:
                return []
            
            # Convert to numpy array
            flow = np.array(optical_flow)
            
            # Calculate statistics
            mean_flow = np.mean(flow, axis=0)
            std_flow = np.std(flow, axis=0)
            max_flow = np.max(flow, axis=0)
            
            # Combine features
            features = np.concatenate([mean_flow, std_flow, max_flow])
            
            return features
        
        # MHI features (already processed)
        def extract_mhi_features(mhi_features):
            """Pass through MHI features"""
            return np.array(mhi_features)
        
        # Histogram features
        def extract_histogram_features(histograms):
            """Extract statistical features from histograms"""
            if not histograms:
                return []
            
            # Convert to numpy array
            hists = np.array(histograms)
            
            # Calculate statistics
            mean_hist = np.mean(hists, axis=0)
            std_hist = np.std(hists, axis=0)
            
            # Combine features
            features = np.concatenate([mean_hist, std_hist])
            
            return features
            
        # Register extractors
        self.feature_extractors = {
            "pose": extract_pose_features,
            "optical_flow": extract_flow_features,
            "mhi": extract_mhi_features,
            "histograms": extract_histogram_features
        }
    
    def extract_features(self, feature_data):
        """
        Extract features from feature data
        
        Args:
            feature_data: Dictionary containing different feature types
            
        Returns:
            Dictionary mapping feature types to extracted features
        """
        extracted = {}
        
        # Process each feature type
        for feature_type in self.feature_types:
            if feature_type == "pose" and "pose_keypoints" in feature_data:
                extracted[feature_type] = self.feature_extractors["pose"](feature_data["pose_keypoints"])
            elif feature_type == "optical_flow" and "optical_flow" in feature_data:
                extracted[feature_type] = self.feature_extractors["optical_flow"](feature_data["optical_flow"])
            elif feature_type == "mhi" and "mhi_features" in feature_data:
                extracted[feature_type] = self.feature_extractors["mhi"](feature_data["mhi_features"])
            elif feature_type == "histograms" and "histograms" in feature_data:
                extracted[feature_type] = self.feature_extractors["histograms"](feature_data["histograms"])
        
        return extracted
    
    def prepare_dataset(self, feature_dataset):
        """
        Prepare dataset for training/testing
        
        Args:
            feature_dataset: List of feature dictionaries
            
        Returns:
            Dictionary mapping feature types to (X, y) tuples
        """
        dataset = {}
        
        # Process each feature type
        for feature_type in self.feature_types:
            X = []
            y = []
            
            for feature_data in feature_dataset:
                # Skip entries without labels
                if feature_data.get("label") is None:
                    continue
                
                # Extract features for this type
                extracted = self.extract_features(feature_data)
                
                if feature_type in extracted and len(extracted[feature_type]) > 0:
                    X.append(extracted[feature_type])
                    y.append(feature_data["label"])
            
            # Convert to numpy arrays
            if X and y:
                X = np.array(X)
                y = np.array(y)
                dataset[feature_type] = (X, y)
            else:
                print(f"Warning: No data for feature type {feature_type}")
        
        return dataset
    
    def train(self, feature_dataset):
        """
        Train models on feature dataset
        
        Args:
            feature_dataset: List of feature dictionaries
            
        Returns:
            Dictionary of trained models
        """
        # Prepare feature extractors
        self._prepare_feature_extractors()
        
        # Prepare dataset
        dataset = self.prepare_dataset(feature_dataset)
        
        # Train models for each feature type
        for feature_type, (X, y) in dataset.items():
            print(f"\nTraining models for feature type: {feature_type}")
            print(f"Dataset shape: {X.shape}, Labels: {np.bincount(y)}")
            
            # Generate feature names
            if feature_type == "pose":
                feature_names = [f"pose_{i}" for i in range(X.shape[1])]
            elif feature_type == "optical_flow":
                feature_names = [f"flow_{i}" for i in range(X.shape[1])]
            elif feature_type == "mhi":
                feature_names = [f"mhi_{i}" for i in range(X.shape[1])]
            elif feature_type == "histograms":
                feature_names = [f"hist_{i}" for i in range(X.shape[1])]
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Train each model type
            for model_type in self.model_types:
                print(f"Training {model_type} model...")
                model = self.model_factory.train_model(
                    model_type, X, y, feature_names
                )
                
                # Save model
                model_key = f"{feature_type}_{model_type}"
                self.models[model_key] = model
                
                # Save to disk
                self.model_factory.save_model(
                    model, model_key,
                    save_dir=os.path.join(self.model_factory.model_dir, feature_type)
                )
        
        return self.models
    
    def evaluate(self, feature_dataset):
        """
        Evaluate models on feature dataset
        
        Args:
            feature_dataset: List of feature dictionaries
            
        Returns:
            Dictionary of evaluation results
        """
        # Prepare feature extractors
        self._prepare_feature_extractors()
        
        # Prepare dataset
        dataset = self.prepare_dataset(feature_dataset)
        
        # Evaluate models for each feature type
        results = {}
        
        for feature_type, (X, y) in dataset.items():
            print(f"\nEvaluating models for feature type: {feature_type}")
            
            # Evaluate each model type
            for model_type in self.model_types:
                model_key = f"{feature_type}_{model_type}"
                
                if model_key in self.models:
                    print(f"Evaluating {model_key} model...")
                    model = self.models[model_key]
                    
                    # Evaluate model
                    eval_result = self.model_factory.evaluate_model(model, X, y)
                    results[model_key] = eval_result
                    
                    # Print results
                    print(f"  Accuracy: {eval_result['accuracy']:.4f}")
                    print(f"  Precision: {eval_result['precision']:.4f}")
                    print(f"  Recall: {eval_result['recall']:.4f}")
                    print(f"  F1 Score: {eval_result['f1_score']:.4f}")
                else:
                    print(f"Warning: No trained model for {model_key}")
        
        return results
    
    def load_models(self, model_dir=None):
        """
        Load trained models from disk
        
        Args:
            model_dir: Directory to load models from
            
        Returns:
            Dictionary of loaded models
        """
        if model_dir is None:
            model_dir = self.model_factory.model_dir
        
        # Load models for each feature type
        for feature_type in self.feature_types:
            feature_dir = os.path.join(model_dir, feature_type)
            
            if not os.path.exists(feature_dir):
                print(f"Warning: No models found for feature type {feature_type}")
                continue
            
            # Load each model type
            for model_type in self.model_types:
                model_key = f"{feature_type}_{model_type}"
                model_path = os.path.join(feature_dir, f"{model_key}.joblib")
                
                if os.path.exists(model_path):
                    print(f"Loading model: {model_key}")
                    self.models[model_key] = joblib.load(model_path)
                else:
                    print(f"Warning: Model file not found: {model_path}")
        
        return self.models
    
    def predict(self, feature_data):
        """
        Make predictions using trained models
        
        Args:
            feature_data: Dictionary containing different feature types
            
        Returns:
            Dictionary mapping model keys to predictions
        """
        # Extract features
        extracted = self.extract_features(feature_data)
        
        # Make predictions with each model
        predictions = {}
        
        for feature_type in self.feature_types:
            if feature_type in extracted and len(extracted[feature_type]) > 0:
                # Reshape for single sample prediction
                X = extracted[feature_type].reshape(1, -1)
                
                # Predict with each model type
                for model_type in self.model_types:
                    model_key = f"{feature_type}_{model_type}"
                    
                    if model_key in self.models:
                        model = self.models[model_key]
                        
                        # Make prediction
                        pred = model.predict(X)[0]
                        prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
                        
                        predictions[model_key] = {
                            "prediction": int(pred),
                            "probability": float(prob) if prob is not None else None
                        }
        
        return predictions
    
    def ensemble_predict(self, feature_data, weighted=True):
        """
        Make ensemble prediction from all models
        
        Args:
            feature_data: Dictionary containing different feature types
            weighted: Whether to use weighted voting (by probability)
            
        Returns:
            Dictionary with ensemble prediction and individual results
        """
        # Get predictions from all models
        all_preds = self.predict(feature_data)
        
        if not all_preds:
            return {"prediction": None, "probability": None, "individual": {}}
        
        # Collect votes
        votes = []
        probabilities = []
        
        for model_key, pred_info in all_preds.items():
            votes.append(pred_info["prediction"])
            
            if weighted and pred_info["probability"] is not None:
                # For binary classification, convert 0 to probability of class 0
                prob = pred_info["probability"] if pred_info["prediction"] == 1 else 1 - pred_info["probability"]
                probabilities.append(prob)
            else:
                probabilities.append(1.0)  # Equal weight
        
        # Calculate weighted prediction
        if weighted and probabilities:
            # Normalize weights
            weights = np.array(probabilities) / sum(probabilities)
            
            # Weighted vote for class 1 (violence)
            class_1_vote = sum([v * w for v, w in zip(votes, weights)])
            ensemble_pred = 1 if class_1_vote >= 0.5 else 0
            ensemble_prob = class_1_vote
        else:
            # Simple majority vote
            ensemble_pred = 1 if sum(votes) > len(votes) / 2 else 0
            ensemble_prob = sum(votes) / len(votes)
        
        # Return ensemble prediction and individual model predictions
        return {
            "prediction": ensemble_pred,
            "probability": ensemble_prob,
            "individual": all_preds
        }