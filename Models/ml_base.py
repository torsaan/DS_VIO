# Models/ml_base.py
import os
import numpy as np
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class BaseMLModel:
    """Base class for ML models with common functionality"""
    
    def __init__(self, feature_selector=None, model_name="BaseModel"):
        """
        Initialize base ML model
        
        Args:
            feature_selector: Feature selector instance or None
            model_name: Name of the model
        """
        self.model_name = model_name
        self.feature_selector = feature_selector
        self.pipeline = None
        self.trained = False
    
    def create_pipeline(self, estimator):
        """
        Create a scikit-learn pipeline with the estimator
        
        Args:
            estimator: Scikit-learn estimator
            
        Returns:
            Pipeline instance
        """
        # Build pipeline components
        steps = [("scaler", StandardScaler())]
        
        # Add feature selector if specified
        if self.feature_selector is not None:
            steps.append(("feature_selector", self.feature_selector))
        
        # Add the model
        steps.append(("model", estimator))
        
        # Create and return the pipeline
        return Pipeline(steps)
    
    def train(self, X_train, y_train, feature_names=None):
        """
        Train the model on the provided data
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Names of features (optional)
            
        Returns:
            Self (trained model)
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call create_pipeline first.")
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        self.trained = True
        
        # Get feature importances if available
        if feature_names is not None:
            self.get_feature_importances(feature_names)
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train first.")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Features to predict
            
        Returns:
            Probability estimates
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train first.")
        
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)
        else:
            # For models that don't support probability estimates
            preds = self.pipeline.predict(X)
            # Convert to "probabilities" (0 or 1)
            proba = np.zeros((X.shape[0], 2))
            proba[np.arange(X.shape[0]), preds] = 1
            return proba
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train first.")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1] if hasattr(self, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Return evaluation results
        return {
            "model_name": self.model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix,
            "predictions": y_pred,
            "probabilities": y_proba
        }
    
    def get_feature_importances(self, feature_names):
        """
        Get feature importances from the trained model
        
        Args:
            feature_names: List of feature names
            
        Returns:
            List of (feature_name, importance) tuples, sorted by importance
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train first.")
        
        # Get the feature selector if present
        feature_selector = None
        for name, step in self.pipeline.named_steps.items():
            if name == "feature_selector" and step is not None:
                feature_selector = step
                break
        
        # Get the model
        model = self.pipeline.named_steps["model"]
        
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
    
    def save(self, filepath):
        """
        Save the model to disk
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Path to the saved model
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train first.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        joblib.dump(self, filepath)
        
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from disk
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        return joblib.load(filepath)


class ModelRegistry:
    """Registry for managing ML models"""
    
    def __init__(self):
        """Initialize model registry"""
        self.models = {}
    
    def register(self, model_class, model_id=None):
        """
        Register a model class
        
        Args:
            model_class: Class to register
            model_id: ID to use for the model (defaults to class name)
            
        Returns:
            model_id
        """
        if model_id is None:
            model_id = model_class.__name__
        
        self.models[model_id] = model_class
        return model_id
    
    def get(self, model_id):
        """
        Get a model class by ID
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model class
        """
        if model_id not in self.models:
            raise ValueError(f"Unknown model ID: {model_id}")
        
        return self.models[model_id]
    
    def create(self, model_id, **kwargs):
        """
        Create a model instance
        
        Args:
            model_id: ID of the model
            **kwargs: Arguments to pass to the model constructor
            
        Returns:
            Model instance
        """
        model_class = self.get(model_id)
        return model_class(**kwargs)
    
    def list_models(self):
        """
        List all registered models
        
        Returns:
            List of model IDs
        """
        return list(self.models.keys())


# Create a global model registry
registry = ModelRegistry()