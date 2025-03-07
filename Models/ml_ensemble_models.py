# Models/ml_ensemble_models.py
import numpy as np
import os
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from .ml_base import registry

class EnsembleViolenceClassifier(BaseEstimator, ClassifierMixin):
    """Ensemble classifier that combines multiple models for violence detection"""
    
    def __init__(self, models=None, voting='soft'):
        """
        Initialize ensemble classifier
        
        Args:
            models: List of models to use in the ensemble
            voting: Voting strategy ('hard' or 'soft')
        """
        self.models = models or []
        self.voting = voting
        self.trained = False
    
    def fit(self, X_dict, y):
        """
        Train all models in the ensemble
        
        Args:
            X_dict: Dictionary mapping feature types to feature arrays
            y: Target labels
            
        Returns:
            Self (trained model)
        """
        # Train each model with its corresponding features
        for model_info in self.models:
            model = model_info['model']
            feature_type = model_info['feature_type']
            
            if feature_type in X_dict:
                print(f"Training {model.model_name} with {feature_type} features...")
                model.train(X_dict[feature_type], y)
            else:
                print(f"Warning: Missing {feature_type} features for {model.model_name}")
        
        self.trained = True
        return self
    
    def predict(self, X_dict):
        """
        Make predictions using the ensemble
        
        Args:
            X_dict: Dictionary mapping feature types to feature arrays
            
        Returns:
            Predicted labels
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit first.")
        
        # Get predictions from each model
        all_preds = []
        all_weights = []
        
        for model_info in self.models:
            model = model_info['model']
            feature_type = model_info['feature_type']
            weight = model_info.get('weight', 1.0)
            
            if feature_type in X_dict and hasattr(model, 'predict'):
                # Get predictions
                preds = model.predict(X_dict[feature_type])
                all_preds.append(preds)
                all_weights.append(weight)
        
        if not all_preds:
            raise ValueError("No predictions available")
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_weights = np.array(all_weights) / sum(all_weights)  # Normalize weights
        
        # Make ensemble prediction
        if self.voting == 'soft' and hasattr(self, 'predict_proba'):
            # Weighted probability-based voting
            ensemble_proba = self.predict_proba(X_dict)
            return (ensemble_proba[:, 1] >= 0.5).astype(int)
        else:
            # Weighted majority voting
            weighted_votes = np.zeros(all_preds[0].shape)
            for preds, weight in zip(all_preds, all_weights):
                weighted_votes += preds * weight
            
            return (weighted_votes >= 0.5).astype(int)
    
    def predict_proba(self, X_dict):
        """
        Get probability estimates
        
        Args:
            X_dict: Dictionary mapping feature types to feature arrays
            
        Returns:
            Probability estimates
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit first.")
        
        # Get probabilities from each model
        all_probas = []
        all_weights = []
        
        for model_info in self.models:
            model = model_info['model']
            feature_type = model_info['feature_type']
            weight = model_info.get('weight', 1.0)
            
            if feature_type in X_dict and hasattr(model, 'predict_proba'):
                # Get probabilities
                probas = model.predict_proba(X_dict[feature_type])
                all_probas.append(probas)
                all_weights.append(weight)
        
        if not all_probas:
            raise ValueError("No probability estimates available")
        
        # Convert to numpy arrays
        all_weights = np.array(all_weights) / sum(all_weights)  # Normalize weights
        
        # Calculate weighted average probabilities
        ensemble_proba = np.zeros(all_probas[0].shape)
        for probas, weight in zip(all_probas, all_weights):
            ensemble_proba += probas * weight
        
        return ensemble_proba
    
    def evaluate(self, X_dict, y):
        """
        Evaluate the ensemble model
        
        Args:
            X_dict: Dictionary mapping feature types to feature arrays
            y: Target labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit first.")
        
        # Make predictions
        y_pred = self.predict(X_dict)
        y_proba = self.predict_proba(X_dict)[:, 1] if hasattr(self, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Evaluate individual models
        model_results = {}
        for model_info in self.models:
            model = model_info['model']
            feature_type = model_info['feature_type']
            
            if feature_type in X_dict:
                model_results[model.model_name] = model.evaluate(X_dict[feature_type], y)
        
        # Return evaluation results
        return {
            "model_name": "EnsembleViolenceClassifier",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix,
            "predictions": y_pred,
            "probabilities": y_proba,
            "model_results": model_results
        }
    
    def save(self, model_dir):
        """
        Save the ensemble model
        
        Args:
            model_dir: Directory to save the model
            
        Returns:
            Dictionary with saved model paths
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit first.")
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Save ensemble configuration
        config = {
            "voting": self.voting,
            "models": []
        }
        
        # Save each model
        for i, model_info in enumerate(self.models):
            model = model_info['model']
            feature_type = model_info['feature_type']
            weight = model_info.get('weight', 1.0)
            
            # Save individual model
            model_path = os.path.join(model_dir, f"model_{i}_{feature_type}.joblib")
            model.save(model_path)
            
            # Add to configuration
            config["models"].append({
                "model_path": model_path,
                "model_name": model.model_name,
                "feature_type": feature_type,
                "weight": weight
            })
        
        # Save configuration
        config_path = os.path.join(model_dir, "ensemble_config.joblib")
        joblib.dump(config, config_path)
        
        return {
            "config_path": config_path,
            "model_dir": model_dir
        }
    
    @classmethod
    def load(cls, config_path):
        """
        Load an ensemble model from configuration
        
        Args:
            config_path: Path to the ensemble configuration
            
        Returns:
            Loaded ensemble model
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration
        config = joblib.load(config_path)
        
        # Create ensemble model
        ensemble = cls(voting=config["voting"])
        ensemble.models = []
        
        # Load each model
        for model_info in config["models"]:
            model_path = model_info["model_path"]
            feature_type = model_info["feature_type"]
            weight = model_info.get("weight", 1.0)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model
            model = joblib.load(model_path)
            
            # Add to ensemble
            ensemble.models.append({
                "model": model,
                "feature_type": feature_type,
                "weight": weight
            })
        
        ensemble.trained = True
        return ensemble


class FeatureFusionModel(BaseEstimator, ClassifierMixin):
    """Model that fuses multiple feature types for violence detection"""
    
    def __init__(self, base_model_id="random_forest"):
        """
        Initialize feature fusion model
        
        Args:
            base_model_id: Base model ID to use
        """
        self.base_model_id = base_model_id
        self.base_model = registry.create(base_model_id)
        self.feature_types = []
        self.trained = False
    
    def add_feature_type(self, feature_type, extractor=None):
        """
        Add a feature type to the fusion model
        
        Args:
            feature_type: Type of feature
            extractor: Feature extractor function
            
        Returns:
            Self
        """
        self.feature_types.append({
            "type": feature_type,
            "extractor": extractor
        })
        
        return self
    
    def extract_and_fuse_features(self, X_dict):
        """
        Extract and fuse features from multiple types
        
        Args:
            X_dict: Dictionary mapping feature types to feature arrays
            
        Returns:
            Fused features
        """
        extracted_features = []
        
        for feature_info in self.feature_types:
            feature_type = feature_info["type"]
            extractor = feature_info["extractor"]
            
            if feature_type in X_dict:
                if extractor is not None:
                    # Extract features using provided extractor
                    features = extractor(X_dict[feature_type])
                else:
                    # Use features directly
                    features = X_dict[feature_type]
                
                extracted_features.append(features)
        
        # Concatenate all features
        if extracted_features:
            return np.concatenate(extracted_features, axis=1 if len(extracted_features[0].shape) > 1 else 0)
        else:
            return np.array([])
    
    def fit(self, X_dict, y):
        """
        Train the fusion model
        
        Args:
            X_dict: Dictionary mapping feature types to feature arrays
            y: Target labels
            
        Returns:
            Self (trained model)
        """
        # Extract and fuse features
        X_fused = self.extract_and_fuse_features(X_dict)
        
        if X_fused.size == 0:
            raise ValueError("No features available for fusion")
        
        # Train base model
        self.base_model.train(X_fused, y)
        self.trained = True
        
        return self
    
    def predict(self, X_dict):
        """
        Make predictions using the fusion model
        
        Args:
            X_dict: Dictionary mapping feature types to feature arrays
            
        Returns:
            Predicted labels
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit first.")
        
        # Extract and fuse features
        X_fused = self.extract_and_fuse_features(X_dict)
        
        if X_fused.size == 0:
            raise ValueError("No features available for fusion")
        
        # Make predictions
        return self.base_model.predict(X_fused)
    
    def predict_proba(self, X_dict):
        """
        Get probability estimates
        
        Args:
            X_dict: Dictionary mapping feature types to feature arrays
            
        Returns:
            Probability estimates
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit first.")
        
        # Extract and fuse features
        X_fused = self.extract_and_fuse_features(X_dict)
        
        if X_fused.size == 0:
            raise ValueError("No features available for fusion")
        
        # Get probabilities
        return self.base_model.predict_proba(X_fused)
    
    def evaluate(self, X_dict, y):
        """
        Evaluate the fusion model
        
        Args:
            X_dict: Dictionary mapping feature types to feature arrays
            y: Target labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit first.")
        
        # Extract and fuse features
        X_fused = self.extract_and_fuse_features(X_dict)
        
        if X_fused.size == 0:
            raise ValueError("No features available for fusion")
        
        # Evaluate model
        return self.base_model.evaluate(X_fused, y)
    
    def save(self, filepath):
        """
        Save the fusion model
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Path to the saved model
        """
        if not self.trained:
            raise ValueError("Model not trained. Call fit first.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        joblib.dump(self, filepath)
        
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """
        Load a fusion model from disk
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        return joblib.load(filepath)


# Register models
registry.register(EnsembleViolenceClassifier, "ensemble")
registry.register(FeatureFusionModel, "fusion")