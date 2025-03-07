# ml_dataloader.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import csv
from tqdm import tqdm
from pathlib import Path
from utils.feature_extraction import FeatureExtractor

class MLFeatureDataset(Dataset):
    """Dataset for machine learning models with feature extraction"""
    
    def __init__(self, video_paths, labels=None, feature_dir=None, 
                 extractor=None, extract_on_load=False, feature_types=None):
        """
        Initialize ML feature dataset
        
        Args:
            video_paths: List of paths to video files
            labels: List of labels (0 for non-violence, 1 for violence, None to infer from filenames)
            feature_dir: Directory containing pre-extracted features
            extractor: Feature extractor instance
            extract_on_load: Whether to extract features on load if not found
            feature_types: List of feature types to load/extract
        """
        self.video_paths = video_paths
        self.feature_types = feature_types or ["pose", "optical_flow", "mhi", "histograms"]
        
        # Infer labels from filenames if not provided
        if labels is None:
            self.labels = []
            for path in video_paths:
                video_name = os.path.basename(path)
                if video_name.startswith('V_'):
                    self.labels.append(1)  # Violence
                elif video_name.startswith('NV_'):
                    self.labels.append(0)  # Non-violence
                else:
                    self.labels.append(None)  # Unknown
        else:
            self.labels = labels
        
        self.feature_dir = feature_dir
        self.extractor = extractor
        self.extract_on_load = extract_on_load
        
        # Create feature index
        self.feature_index = self._index_features()
    
    def _index_features(self):
        """Index available features"""
        feature_index = {}
        
        if self.feature_dir is None:
            return feature_index
        
        # Check for combined features first
        combined_dir = os.path.join(self.feature_dir, "combined")
        if os.path.exists(combined_dir):
            for class_dir in os.listdir(combined_dir):
                class_path = os.path.join(combined_dir, class_dir)
                if os.path.isdir(class_path):
                    for file in os.listdir(class_path):
                        if file.endswith('.pkl'):
                            video_name = os.path.splitext(file)[0]
                            feature_index[video_name] = {
                                "combined": os.path.join(class_path, file)
                            }
        
        # Check for individual feature types
        for feature_type in self.feature_types:
            feature_dir = os.path.join(self.feature_dir, feature_type)
            if os.path.exists(feature_dir):
                for class_dir in os.listdir(feature_dir):
                    class_path = os.path.join(feature_dir, class_dir)
                    if os.path.isdir(class_path):
                        for file in os.listdir(class_path):
                            video_name = file.split('_')[0]
                            if video_name not in feature_index:
                                feature_index[video_name] = {}
                            
                            if feature_type not in feature_index[video_name]:
                                feature_index[video_name][feature_type] = []
                            
                            feature_index[video_name][feature_type].append(
                                os.path.join(class_path, file)
                            )
        
        return feature_index
    
    def _load_features(self, video_path):
        """Load features for a video"""
        video_name = os.path.basename(video_path).split('.')[0]
        
        # Check if features are indexed
        if video_name in self.feature_index:
            # Try to load combined features first
            if "combined" in self.feature_index[video_name]:
                combined_path = self.feature_index[video_name]["combined"]
                with open(combined_path, 'rb') as f:
                    return pickle.load(f)
            
            # Load individual feature types
            features = {"video_name": video_name}
            
            for feature_type in self.feature_types:
                if feature_type in self.feature_index[video_name]:
                    feature_files = self.feature_index[video_name][feature_type]
                    
                    if feature_type == "pose":
                        # Load pose keypoints from CSV
                        pose_file = next((f for f in feature_files if f.endswith('.csv')), None)
                        if pose_file:
                            keypoints = []
                            with open(pose_file, 'r') as f:
                                reader = csv.reader(f)
                                next(reader)  # Skip header
                                for row in reader:
                                    if len(row) > 1:  # Skip empty rows
                                        # First element is frame_idx, rest are keypoints
                                        kp = list(map(float, row[1:]))
                                        keypoints.append(kp)
                            features["pose_keypoints"] = keypoints
                    
                    elif feature_type == "optical_flow":
                        # Load optical flow features from CSV
                        flow_file = next((f for f in feature_files if f.endswith('_flow.csv')), None)
                        if flow_file:
                            flow_features = []
                            with open(flow_file, 'r') as f:
                                reader = csv.reader(f)
                                next(reader)  # Skip header
                                for row in reader:
                                    if len(row) > 1:  # Skip empty rows
                                        # First element is frame_idx, rest are flow features
                                        flow = list(map(float, row[1:]))
                                        flow_features.append(flow)
                            features["optical_flow"] = flow_features
                    
                    elif feature_type == "mhi":
                        # Load MHI image and extract features
                        mhi_file = next((f for f in feature_files if f.endswith('_mhi.jpg')), None)
                        if mhi_file:
                            mhi = cv2.imread(mhi_file, cv2.IMREAD_GRAYSCALE)
                            
                            # Extract MHI features
                            mhi_features = []
                            if mhi is not None:
                                mhi_mean = np.mean(mhi)
                                mhi_std = np.std(mhi)
                                mhi_max = np.max(mhi)
                                
                                # Regional MHI features
                                h, w = mhi.shape
                                regions = [
                                    mhi[:h//2, :w//2],      # top-left
                                    mhi[:h//2, w//2:],      # top-right
                                    mhi[h//2:, :w//2],      # bottom-left
                                    mhi[h//2:, w//2:]       # bottom-right
                                ]
                                
                                region_means = [np.mean(region) for region in regions]
                                mhi_features = [mhi_mean, mhi_std, mhi_max] + region_means
                            
                            features["mhi_features"] = mhi_features
                    
                    elif feature_type == "histograms":
                        # Load histogram features from pickle file
                        hist_file = next((f for f in feature_files if f.endswith('_hist.pkl')), None)
                        if hist_file:
                            with open(hist_file, 'rb') as f:
                                histograms = pickle.load(f)
                            features["histograms"] = histograms
            
            return features
        
        # Extract features if needed
        if self.extract_on_load and self.extractor is not None:
            # Determine label
            idx = self.video_paths.index(video_path)
            label = self.labels[idx] if idx < len(self.labels) else None
            
            # Extract features
            return self.extractor.extract_features(video_path, label=label)
        
        # Couldn't load or extract features
        return {"video_name": video_name}
    
    def __len__(self):
        """Get number of videos"""
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """Get features for a video"""
        video_path = self.video_paths[idx]
        label = self.labels[idx] if idx < len(self.labels) else None
        
        # Load features
        features = self._load_features(video_path)
        
        # Add label
        features["label"] = label
        
        return features


def get_ml_dataloaders(train_video_paths, train_labels, val_video_paths, val_labels, 
                      test_video_paths, test_labels, feature_dir=None, 
                      batch_size=32, extract_on_load=False):
    """
    Create DataLoaders for ML models
    
    Args:
        train_video_paths, val_video_paths, test_video_paths: Lists of video paths
        train_labels, val_labels, test_labels: Lists of labels
        feature_dir: Directory containing pre-extracted features
        batch_size: Batch size for the DataLoader
        extract_on_load: Whether to extract features on load if not found
        
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    # Create feature extractor if needed
    extractor = None
    if extract_on_load:
        extractor = FeatureExtractor(output_dir=feature_dir or "./features")
    
    # Create datasets
    train_dataset = MLFeatureDataset(
        train_video_paths, train_labels, feature_dir,
        extractor, extract_on_load
    )
    
    val_dataset = MLFeatureDataset(
        val_video_paths, val_labels, feature_dir,
        extractor, extract_on_load
    )
    
    test_dataset = MLFeatureDataset(
        test_video_paths, test_labels, feature_dir,
        extractor, extract_on_load
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda x: x  # Don't collate, return list of dictionaries
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda x: x  # Don't collate, return list of dictionaries
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda x: x  # Don't collate, return list of dictionaries
    )
    
    return train_loader, val_loader, test_loader


def preprocess_and_extract_features(video_paths, labels=None, output_dir="./features", sample_rate=5):
    """
    Preprocess videos and extract features for ML models
    
    Args:
        video_paths: List of video paths
        labels: List of labels (optional)
        output_dir: Directory to save extracted features
        sample_rate: Number of frames to sample per video
        
    Returns:
        List of feature dictionaries
    """
    # Create feature extractor
    extractor = FeatureExtractor(output_dir=output_dir, sample_rate=sample_rate)
    
    # Process each video
    all_features = []
    for i, video_path in enumerate(tqdm(video_paths, desc="Extracting features")):
        # Determine label
        label = None
        if labels is not None and i < len(labels):
            label = labels[i]
        else:
            video_name = os.path.basename(video_path)
            if video_name.startswith('V_'):
                label = 1  # Violence
            elif video_name.startswith('NV_'):
                label = 0  # Non-violence
        
        # Extract features
        try:
            features = extractor.extract_features(video_path, label=label)
            if features:
                all_features.append(features)
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
    
    return all_features


def load_features_from_directory(feature_dir, feature_types=None):
    """
    Load all features from a directory
    
    Args:
        feature_dir: Directory containing features
        feature_types: List of feature types to load (None for all)
        
    Returns:
        List of feature dictionaries
    """
    if feature_types is None:
        feature_types = ["pose", "optical_flow", "mhi", "histograms"]
    
    # Create temporary dataset to index features
    dataset = MLFeatureDataset([], feature_dir=feature_dir, feature_types=feature_types)
    
    # Get list of all video names in the index
    video_names = list(dataset.feature_index.keys())
    
    # Create dummy video paths for the dataset
    dummy_paths = [f"{name}.mp4" for name in video_names]
    
    # Create new dataset with dummy paths
    full_dataset = MLFeatureDataset(dummy_paths, feature_dir=feature_dir, feature_types=feature_types)
    
    # Load all features
    all_features = []
    for i in range(len(full_dataset)):
        all_features.append(full_dataset[i])
    
    return all_features