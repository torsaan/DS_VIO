# Models/ml_feature_models.py
import numpy as np
import os
from sklearn.feature_selection import SelectKBest, f_classif
from .ml_base import BaseMLModel, registry

class PoseBasedModel(BaseMLModel):
    """Model specialized for pose keypoint features"""
    
    def __init__(self, base_model_id="random_forest", select_k_best=50):
        """
        Initialize pose-based model
        
        Args:
            base_model_id: Base model ID to use
            select_k_best: Number of features to select
        """
        # Create feature selector
        feature_selector = SelectKBest(f_classif, k=select_k_best)
        
        super().__init__(feature_selector, model_name=f"Pose-{base_model_id}")
        
        # Create base model
        self.base_model = registry.create(base_model_id)
        
        # Use the base model's pipeline
        self.pipeline = self.base_model.pipeline
    
    def extract_features(self, pose_keypoints):
        """
        Extract features from pose keypoints
        
        Args:
            pose_keypoints: List of pose keypoints across frames
            
        Returns:
            Extracted features
        """
        # Convert to numpy array
        keypoints = np.array(pose_keypoints)
        
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
            # Handle single frame case
            mean_velocity = np.zeros_like(mean_keypoints)
            max_velocity = np.zeros_like(mean_keypoints)
        
        # Calculate acceleration (difference of velocity)
        if keypoints.shape[0] > 2:
            acceleration = np.diff(velocity, axis=0)
            mean_acceleration = np.mean(np.abs(acceleration), axis=0)
            max_acceleration = np.max(np.abs(acceleration), axis=0)
        else:
            # Handle cases with too few frames
            mean_acceleration = np.zeros_like(mean_keypoints)
            max_acceleration = np.zeros_like(mean_keypoints)
        
        # Combine features
        features = np.concatenate([
            mean_keypoints, std_keypoints, max_keypoints,
            mean_velocity, max_velocity,
            mean_acceleration, max_acceleration
        ])
        
        return features


class MotionBasedModel(BaseMLModel):
    """Model specialized for motion features (optical flow and MHI)"""
    
    def __init__(self, base_model_id="xgboost", select_k_best=30):
        """
        Initialize motion-based model
        
        Args:
            base_model_id: Base model ID to use
            select_k_best: Number of features to select
        """
        # Create feature selector
        feature_selector = SelectKBest(f_classif, k=select_k_best)
        
        super().__init__(feature_selector, model_name=f"Motion-{base_model_id}")
        
        # Create base model
        self.base_model = registry.create(base_model_id)
        
        # Use the base model's pipeline
        self.pipeline = self.base_model.pipeline
    
    def extract_features(self, optical_flow, mhi_features):
        """
        Extract features from optical flow and MHI
        
        Args:
            optical_flow: List of optical flow features across frames
            mhi_features: Motion history image features
            
        Returns:
            Extracted features
        """
        # Process optical flow
        flow = np.array(optical_flow)
        
        # Calculate statistics
        mean_flow = np.mean(flow, axis=0)
        std_flow = np.std(flow, axis=0)
        max_flow = np.max(flow, axis=0)
        
        # Combine with MHI features
        features = np.concatenate([
            mean_flow, std_flow, max_flow,
            np.array(mhi_features)
        ])
        
        return features


class VisualBasedModel(BaseMLModel):
    """Model specialized for visual features (histograms)"""
    
    def __init__(self, base_model_id="random_forest", select_k_best=100):
        """
        Initialize visual-based model
        
        Args:
            base_model_id: Base model ID to use
            select_k_best: Number of features to select
        """
        # Create feature selector
        feature_selector = SelectKBest(f_classif, k=select_k_best)
        
        super().__init__(feature_selector, model_name=f"Visual-{base_model_id}")
        
        # Create base model
        self.base_model = registry.create(base_model_id)
        
        # Use the base model's pipeline
        self.pipeline = self.base_model.pipeline
    
    def extract_features(self, histograms):
        """
        Extract features from histograms
        
        Args:
            histograms: List of histogram features across frames
            
        Returns:
            Extracted features
        """
        # Convert to numpy array
        hists = np.array(histograms)
        
        # Calculate statistics
        mean_hist = np.mean(hists, axis=0)
        std_hist = np.std(hists, axis=0)
        
        # Combine features
        features = np.concatenate([mean_hist, std_hist])
        
        return features


# Register models
registry.register(PoseBasedModel, "pose_model")
registry.register(MotionBasedModel, "motion_model")
registry.register(VisualBasedModel, "visual_model")