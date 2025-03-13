
import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils.augmentor import VideoAugmenter
import random

class EnhancedViolenceDataset(Dataset):
    def __init__(self, video_paths, labels, 
                 transform=None, num_frames=32, target_fps=15,
                 augment=True, model_type='3d_cnn',
                 frame_width=224, frame_height=224, training=True):
        """
        Enhanced dataset for violence detection with video data only.
        
        Args:
            video_paths: List of paths to video files
            labels: List of labels (0 for non-violence, 1 for violence)
            transform: Optional transforms to apply to video frames
            num_frames: Number of frames to sample from each video
            target_fps: Target frame rate for sampling (defaults to 15 FPS)
            augment: Whether to apply data augmentation
            model_type: Type of model ('3d_cnn', '2d_cnn_lstm', etc.)
            frame_width: Width for resizing frames
            frame_height: Height for resizing frames
            training: Whether this dataset is used for training
        """
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.augment = augment
        self.model_type = model_type
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.training = training
        
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
        
        # Track current augmentation settings
        self.current_augment_types = None

    def __len__(self):
        return len(self.video_paths)
    
    def read_video(self, video_path):
        """
        Read a fixed number of frames from a video, evenly distributed throughout.
        """
        from utils.video_standardizer import extract_fixed_frames
        from utils.dataprep import NUM_FRAMES
        
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
        
        return frames
    
    def process_frames(self, frames):
        """
        Process frames based on the model type and apply transformations.
        """
        processed_frames = []
        
        # Apply augmentation if needed (only during training)
        if self.augment and self.training:
            # Randomly decide which augmentations to apply
            available_augmentations = ['flip', 'rotate', 'brightness', 'contrast', 'saturation', 'hue', 'crop']
            num_augs = random.randint(1, 3)  # Apply 1-3 augmentations
            self.current_augment_types = random.sample(available_augmentations, num_augs)
            
            # Apply the same augmentations to all frames
            frames = self.video_augmenter.augment_video(frames, self.current_augment_types)
        else:
            self.current_augment_types = None
        
        # Apply transforms to each frame
        for frame in frames:
            if self.transform:
                # Convert to PIL Image for torchvision transforms
                pil_frame = Image.fromarray(frame)
                transformed_frame = self.transform(pil_frame)
            else:
                # Basic processing: resize and convert to tensor
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                transformed_frame = torch.from_numpy(frame.transpose(2, 0, 1).astype(np.float32)) / 255.0
            
            processed_frames.append(transformed_frame)
        
        # Stack frames based on model type
        if self.model_type == '3d_cnn':
            # For 3D CNN: [C, T, H, W]
            frames_tensor = torch.stack(processed_frames, dim=1)
        else:
            # For 2D CNN+LSTM, transformer, etc.: [T, C, H, W]
            frames_tensor = torch.stack(processed_frames, dim=0)
        
        return frames_tensor
    
    def __getitem__(self, idx):
        """
        Get a single video sample with its corresponding label.
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Read video frames
        try:
            frames = self.read_video(video_path)
            
            # Process frames
            frames_tensor = self.process_frames(frames)
            
            # Return frames and label
            return frames_tensor, torch.tensor(label, dtype=torch.long)
                
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Return zeros with the correct shape as a fallback
            if self.model_type == '3d_cnn':
                frames_tensor = torch.zeros((3, self.num_frames, self.frame_height, self.frame_width), dtype=torch.float32)
            else:
                frames_tensor = torch.zeros((self.num_frames, 3, self.frame_height, self.frame_width), dtype=torch.float32)
            
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
                   test_video_paths, test_labels, batch_size=8,
                   num_workers=4, target_fps=10, num_frames=16, model_type='3d_cnn',
                   pin_memory=True, persistent_workers=True, prefetch_factor=2):
    """
    Create DataLoaders for training, validation, and testing with optimizations.
    
    Args:
        train_video_paths, val_video_paths, test_video_paths: Lists of video paths
        train_labels, val_labels, test_labels: Lists of labels
        batch_size: Batch size
        num_workers: Number of worker processes for DataLoader
        target_fps: Target frame rate for sampling
        num_frames: Number of frames to sample per video
        model_type: Type of model ('3d_cnn', '2d_cnn_lstm', etc.)
        pin_memory: Whether to pin memory in DataLoader (speeds up GPU transfers)
        persistent_workers: Whether to keep worker processes alive between iterations
        prefetch_factor: Number of batches to prefetch per worker
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets without pose data
    train_dataset = EnhancedViolenceDataset(
        train_video_paths, train_labels,
        transform=train_transform, num_frames=num_frames, 
        target_fps=target_fps, augment=True, model_type=model_type,
        training=True
    )
    
    val_dataset = EnhancedViolenceDataset(
        val_video_paths, val_labels,
        transform=val_transform, num_frames=num_frames, 
        target_fps=target_fps, augment=False, model_type=model_type,
        training=False
    )
    
    test_dataset = EnhancedViolenceDataset(
        test_video_paths, test_labels,
        transform=val_transform, num_frames=num_frames, 
        target_fps=target_fps, augment=False, model_type=model_type,
        training=False
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