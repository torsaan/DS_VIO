# Enhanced ViolenceDataset class
import os
import csv
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils.augmentor import VideoAugmenter
import random
import tqdm







class EnhancedViolenceDataset(Dataset):
    def __init__(self, video_paths, labels, pose_dir=None, 
                 transform=None, num_frames=32, target_fps=15,
                 normalize_pose=True, augment=True, model_type='3d_cnn',
                 frame_width=224, frame_height=224, training=True,
                 preload_to_ram=False):
        """
        Enhanced dataset for violence detection with both video and pose data.
        
        Args:
            preload_to_ram: Whether to preload videos into RAM
        """
        self.video_paths = video_paths
        self.labels = labels
        self.pose_dir = pose_dir
        self.transform = transform
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.normalize_pose = normalize_pose
        self.augment = augment
        self.model_type = model_type
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.training = training
        self.preload_to_ram = preload_to_ram
        
        # Initialize augmenters
        if self.augment and self.training:
            self.video_augmenter = VideoAugmenter(
                brightness_range=0.3,
                contrast_range=0.3,
                saturation_range=0.3,
                hue_range=0.1,
                rotation_angle=15,
                crop_percent=0.1
            )
            self.pose_augmenter = self.video_augmenter  # Same augmenter for consistency
        
        # Track current augmentation settings for consistency between video and pose
        self.current_augment_types = None
        
        # Pre-load videos to RAM if requested
        self.preloaded_frames = {}
        if self.preload_to_ram:
            print(f"Preloading {len(video_paths)} videos to RAM...")
            for i, video_path in enumerate(tqdm(video_paths, desc="Preloading videos")):
                try:
                    frames = self.read_video(video_path)
                    self.preloaded_frames[video_path] = frames
                except Exception as e:
                    print(f"Error preloading video {video_path}: {e}")
            print(f"Preloaded {len(self.preloaded_frames)} videos to RAM")

    def __len__(self):
        return len(self.video_paths)
    
    def read_video(self, video_path):
        """
        Read a fixed number of frames from a video, evenly distributed throughout.
        """
        from utils.video_standardizer import extract_fixed_frames
        
        # Extract frames with fixed count
        frames = extract_fixed_frames(
            video_path, 
            num_frames=self.num_frames, 
            resize_dim=(self.frame_width, self.frame_height)
        )
        
        if frames is None or len(frames) == 0:
            print(f"Error: Failed to extract frames from {video_path}")
            # Create empty frames as fallback
            frames = [np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8) 
                    for _ in range(self.num_frames)]
        
        return frames[:self.num_frames]  # Ensure we return exactly num_frames frames
    def process_frames(self, frames):
        """
        Process frames for the model by applying transformations and formatting.
        """
        # Convert frames to PIL Images for transformation
        processed_frames = []
        for frame in frames:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame)
            
            # Apply transformations if specified
            if self.transform:
                pil_image = self.transform(pil_image)
            
            processed_frames.append(pil_image)
        
        # Stack frames into a tensor
        frames_tensor = torch.stack(processed_frames)
        
        # Rearrange dimensions based on model type
        if self.model_type == '3d_cnn':
            # [T, C, H, W] -> [C, T, H, W] for 3D CNN
            frames_tensor = frames_tensor.permute(1, 0, 2, 3)
        
        return frames_tensor
    
    def __getitem__(self, idx):
        """
        Get a single video sample with its corresponding label and pose data.
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Read video frames, either from RAM or disk
            if self.preload_to_ram and video_path in self.preloaded_frames:
                frames = self.preloaded_frames[video_path]
            else:
                frames = self.read_video(video_path)
            
            # Process frames
            frames_tensor = self.process_frames(frames)
            
            # Load pose keypoints if needed
            if self.pose_dir:
                pose_keypoints = self.load_pose_keypoints(video_path)
                return frames_tensor, pose_keypoints, torch.tensor(label, dtype=torch.long)
            else:
                return frames_tensor, torch.tensor(label, dtype=torch.long)
                
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Return zeros with the correct shape as a fallback
            if self.model_type == '3d_cnn':
                frames_tensor = torch.zeros((3, self.num_frames, self.frame_height, self.frame_width), 
                                           dtype=torch.float32)
            else:
                frames_tensor = torch.zeros((self.num_frames, 3, self.frame_height, self.frame_width), 
                                           dtype=torch.float32)
            
            if self.pose_dir:
                pose_keypoints = torch.zeros((self.num_frames, 66), dtype=torch.float32)
                return frames_tensor, pose_keypoints, torch.tensor(label, dtype=torch.long)
            else:
                return frames_tensor, torch.tensor(label, dtype=torch.long)

def get_transforms(frame_height=224, frame_width=224):
    """
    Create transform pipelines for training and validation.
    """
    train_transform = transforms.Compose([
        transforms.Resize((frame_height, frame_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((frame_height, frame_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_dataloaders(train_video_paths, train_labels, val_video_paths, val_labels, 
                   test_video_paths, test_labels, pose_dir=None, batch_size=8,
                   num_workers=4, target_fps=10, num_frames=16, model_type='3d_cnn',
                   pin_memory=True, persistent_workers=True, prefetch_factor=2,
                   preload_to_ram=False):
    """
    Create DataLoaders for training, validation, and testing with optimizations.
    
    Args:
        preload_to_ram: Whether to preload the dataset into RAM (for small datasets)
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # If we want to preload to RAM, load a subset of the data
    if preload_to_ram:
        # Limit to a reasonable number of samples to avoid memory issues
        max_samples = 500
        train_video_paths = train_video_paths[:min(len(train_video_paths), max_samples)]
        train_labels = train_labels[:min(len(train_labels), max_samples)]
        print(f"Preloading {len(train_video_paths)} training samples to RAM...")
    
    # Create datasets with fixed frame count
    train_dataset = EnhancedViolenceDataset(
        train_video_paths, train_labels, pose_dir=pose_dir,
        transform=train_transform, num_frames=num_frames, 
        target_fps=target_fps, augment=True, model_type=model_type,
        training=True, preload_to_ram=preload_to_ram
    )
    
    val_dataset = EnhancedViolenceDataset(
        val_video_paths, val_labels, pose_dir=pose_dir,
        transform=val_transform, num_frames=num_frames, 
        target_fps=target_fps, augment=False, model_type=model_type,
        training=False, preload_to_ram=False  # Don't preload validation set
    )
    
    test_dataset = EnhancedViolenceDataset(
        test_video_paths, test_labels, pose_dir=pose_dir,
        transform=val_transform, num_frames=num_frames, 
        target_fps=target_fps, augment=False, model_type=model_type,
        training=False, preload_to_ram=False  # Don't preload test set
    )
    
    # Only use persistent_workers if num_workers > 0
    use_persistent = persistent_workers and num_workers > 0
    
    # Create loaders with performance optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True  # Drop last incomplete batch to ensure consistent batch size
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader
    