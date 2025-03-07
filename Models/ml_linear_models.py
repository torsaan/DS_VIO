# Models/ml_linear_models.py
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from .ml_base import BaseMLModel, registry

class SVMModel(BaseMLModel):
    """Support Vector Machine classifier for violence detection"""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', probability=True,
                random_state=42, select_k_best=None):
        """
        Initialize SVM model
        
        Args:
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
            probability: Whether to enable probability estimates
            random_state: Random state for reproducibility
            select_k_best: Number of features to select (None for no selection)
        """
        # Create feature selector if specified
        feature_selector = None
        if select_k_best is not None:
            feature_selector = SelectKBest(f_classif, k=select_k_best)
        
        super().__init__(feature_selector, model_name=f"SVM-{kernel}")
        
        # Create classifier
        self.classifier = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=random_state
        )
        
        # Create pipeline
        self.pipeline = self.create_pipeline(self.classifier)


class LinearSVMModel(BaseMLModel):
    """Linear SVM classifier for violence detection (faster than SVC with linear kernel)"""
    
    def __init__(self, C=1.0, dual=True, random_state=42, select_k_best=None):
        """
        Initialize Linear SVM model
        
        Args:
            C: Regularization parameter
            dual: Solve the dual or primal optimization problem
            random_state: Random state for reproducibility
            select_k_best: Number of features to select (None for no selection)
        """
        # Create feature selector if specified
        feature_selector = None
        if select_k_best is not None:
            feature_selector = SelectKBest(f_classif, k=select_k_best)
        
        super().__init__(feature_selector, model_name="LinearSVM")
        
        # Create classifier
        self.classifier = LinearSVC(
            C=C,
            dual=dual,
            random_state=random_state
        )
        
        # Create pipeline
        self.pipeline = self.create_pipeline(self.classifier)
    
    def predict_proba(self, X):
        """
        Get probability estimates for LinearSVC (which doesn't support predict_proba natively)
        
        Args:
            X: Features to predict
            
        Returns:
            Probability estimates
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train first.")
        
        # Get decision function scores
        decision_values = self.pipeline.decision_function(X)
        
        # Convert to probabilities using sigmoid function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        probs = sigmoid(decision_values)
        
        # Return probabilities for both classes
        return np.vstack([1 - probs, probs]).T


class LogisticRegressionModel(BaseMLModel):
    """Logistic Regression classifier for violence detection"""
    
    def __init__(self, C=1.0, penalty='l2', solver='lbfgs', max_iter=1000,
                random_state=42, select_k_best=None):
        """
        Initialize Logistic Regression model
        
        Args:
            C: Regularization parameter
            penalty: Penalty type ('l1', 'l2', 'elasticnet', 'none')
            solver: Algorithm for optimization
            max_iter: Maximum number of iterations
            random_state: Random state for reproducibility
            select_k_best: Number of features to select (None for no selection)
        """
        # Create feature selector if specified
        feature_selector = None
        if select_k_best is not None:
            feature_selector = SelectKBest(f_classif, k=select_k_best)
        
        super().__init__(feature_selector, model_name="LogisticRegression")
        
        # Create classifier
        self.classifier = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state
        )
        
        # Create pipeline
        self.pipeline = self.create_pipeline(self.classifier)
    
    def get_coefficients(self, feature_names):
        """
        Get model coefficients with feature names
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to coefficients
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train first.")
        
        # Get the raw classifier
        classifier = self.pipeline.named_steps["model"]
        
        # Get the feature selector if present
        selected_indices = np.arange(len(feature_names))
        if "feature_selector" in self.pipeline.named_steps:
            feature_selector = self.pipeline.named_steps["feature_selector"]
            if feature_selector is not None and hasattr(feature_selector, "get_support"):
                selected_indices = feature_selector.get_support(indices=True)
        
        # Get selected feature names
        selected_features = [feature_names[i] for i in selected_indices]
        
        # Get coefficients
        coefficients = classifier.coef_[0]
        
        # Create dictionary mapping feature names to coefficients
        coef_dict = {selected_features[i]: coef for i, coef in enumerate(coefficients)}
        
        # Sort by absolute coefficient value
        sorted_coefficients = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return sorted_coefficients


# Register models
registry.register(SVMModel, "svm")
registry.register(LinearSVMModel, "linear_svm")
registry.register(LogisticRegressionModel, "logistic_regression")