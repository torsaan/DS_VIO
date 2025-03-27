import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils.augmentor import VideoAugmenter
import random
from tqdm import tqdm
from pathlib import Path

class EnhancedViolenceDataset(Dataset):
    def __init__(self, video_paths, labels,
                 transform=None, num_frames=32, target_fps=15,
                 augment=True, model_type='3d_cnn',
                 frame_width=224, frame_height=224, training=True,
                 preload_to_ram=False):
        """
        Enhanced dataset for violence detection using only video frames.
        
        Args:
            preload_to_ram: Whether to preload videos into RAM.
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
        self.preload_to_ram = preload_to_ram

        # Initialize video augmenter (no separate pose augmenter needed)
        if self.augment and self.training:
            self.video_augmenter = VideoAugmenter(
                brightness_range=0.3,
                contrast_range=0.3,
                saturation_range=0.3,
                hue_range=0.1,
                rotation_angle=15,
                crop_percent=0.1
            )
        
        # Pre-load videos to RAM if requested
        self.preloaded_frames = {}
        if self.preload_to_ram:
            print(f"Preloading {len(video_paths)} videos to RAM...")
            for video_path in tqdm(video_paths, desc="Preloading videos"):
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
        
        frames = extract_fixed_frames(
            video_path, 
            num_frames=self.num_frames, 
            resize_dim=(self.frame_width, self.frame_height)
        )
        if frames is None or len(frames) == 0:
            print(f"Error: Failed to extract frames from {video_path}")
            frames = [np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8) 
                      for _ in range(self.num_frames)]
        return frames[:self.num_frames]
    
    def process_frames(self, frames):
        """
        Process frames by applying transformations and formatting.
        """
        processed_frames = []
        for frame in frames:
            pil_image = Image.fromarray(frame)
            if self.transform:
                pil_image = self.transform(pil_image)
            processed_frames.append(pil_image)
        
        frames_tensor = torch.stack(processed_frames)
        if self.model_type == '3d_cnn':
            frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]
        return frames_tensor
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        try:
            if self.preload_to_ram and video_path in self.preloaded_frames:
                frames = self.preloaded_frames[video_path]
            else:
                frames = self.read_video(video_path)
            frames_tensor = self.process_frames(frames)
            # Return only frames and label (pose data removed)
            return frames_tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            if self.model_type == '3d_cnn':
                frames_tensor = torch.zeros((3, self.num_frames, self.frame_height, self.frame_width), dtype=torch.float32)
            else:
                frames_tensor = torch.zeros((self.num_frames, 3, self.frame_height, self.frame_width), dtype=torch.float32)
            return frames_tensor, torch.tensor(label, dtype=torch.long)

def get_transforms(frame_height=224, frame_width=224):
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
                   pin_memory=True, persistent_workers=True, prefetch_factor=2,
                   preload_to_ram=False):
    train_transform, val_transform = get_transforms()
    
    if preload_to_ram:
        max_samples = 500
        train_video_paths = train_video_paths[:min(len(train_video_paths), max_samples)]
        train_labels = train_labels[:min(len(train_labels), max_samples)]
        print(f"Preloading {len(train_video_paths)} training samples to RAM...")
    
    train_dataset = EnhancedViolenceDataset(
        train_video_paths, train_labels, 
        transform=train_transform, num_frames=num_frames, 
        target_fps=target_fps, augment=True, model_type=model_type,
        training=True, preload_to_ram=preload_to_ram
    )
    
    val_dataset = EnhancedViolenceDataset(
        val_video_paths, val_labels, 
        transform=val_transform, num_frames=num_frames, 
        target_fps=target_fps, augment=False, model_type=model_type,
        training=False, preload_to_ram=False
    )
    
    test_dataset = EnhancedViolenceDataset(
        test_video_paths, test_labels,
        transform=val_transform, num_frames=num_frames, 
        target_fps=target_fps, augment=False, model_type=model_type,
        training=False, preload_to_ram=False
    )
    
    use_persistent = persistent_workers and num_workers > 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True
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
                   pin_memory=True, persistent_workers=True, prefetch_factor=2,
                   preload_to_ram=False, flow_dir=None):
    """
    Create data loaders for training, validation, and testing.
    Added flow_dir parameter to load pre-computed optical flow.
    """
    train_transform, val_transform = get_transforms()
    
    # Choose dataset class based on whether we have flow data
    if flow_dir and model_type == 'two_stream':
        dataset_class = EnhancedViolenceDatasetWithFlow
        print(f"Using dataset with pre-computed flow from {flow_dir}")
    else:
        dataset_class = EnhancedViolenceDataset
    
    if preload_to_ram:
        max_samples = 500
        train_video_paths = train_video_paths[:min(len(train_video_paths), max_samples)]
        train_labels = train_labels[:min(len(train_labels), max_samples)]
    
    # Create datasets
    train_dataset = dataset_class(
        train_video_paths, train_labels, flow_dir=flow_dir,
        transform=train_transform, num_frames=num_frames, 
        target_fps=target_fps, augment=True, model_type=model_type,
        training=True, preload_to_ram=preload_to_ram
    )
    
    val_dataset = dataset_class(
        val_video_paths, val_labels, flow_dir=flow_dir,
        transform=val_transform, num_frames=num_frames, 
        target_fps=target_fps, augment=False, model_type=model_type,
        training=False, preload_to_ram=False
    )
    
    test_dataset = dataset_class(
        test_video_paths, test_labels, flow_dir=flow_dir,
        transform=val_transform, num_frames=num_frames, 
        target_fps=target_fps, augment=False, model_type=model_type,
        training=False, preload_to_ram=False
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



class EnhancedViolenceDatasetWithFlow(Dataset):
    def __init__(self, video_paths, labels, flow_dir=None,
                 transform=None, num_frames=32, target_fps=15,
                 augment=True, model_type='3d_cnn',
                 frame_width=224, frame_height=224, training=True,
                 preload_to_ram=False):
        """
        Enhanced dataset that loads both video frames and pre-computed optical flow.
        
        Args:
            video_paths: List of video file paths
            labels: List of class labels
            flow_dir: Directory containing pre-computed optical flow (.flow.pt files)
            ...
        """
        super().__init__()
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
        self.preload_to_ram = preload_to_ram
        self.flow_dir = flow_dir

        # Initialize video augmenter
        if self.augment and self.training:
            self.video_augmenter = VideoAugmenter(
                brightness_range=0.3,
                contrast_range=0.3,
                saturation_range=0.3,
                hue_range=0.1,
                rotation_angle=15,
                crop_percent=0.1
            )
        
        # Pre-load videos to RAM if requested
        self.preloaded_frames = {}
        self.preloaded_flow = {}
        if self.preload_to_ram:
            print(f"Preloading {len(video_paths)} videos and flows to RAM...")
            for video_path in tqdm(video_paths, desc="Preloading"):
                try:
                    frames = self.read_video(video_path)
                    self.preloaded_frames[video_path] = frames
                    
                    if self.flow_dir:
                        flow_path = Path(self.flow_dir) / Path(video_path).with_suffix('.flow.pt').name
                        if flow_path.exists():
                            flow = torch.load(flow_path)
                            self.preloaded_flow[video_path] = flow
                except Exception as e:
                    print(f"Error preloading {video_path}: {e}")
            print(f"Preloaded {len(self.preloaded_frames)} videos to RAM")

    def __len__(self):
        return len(self.video_paths)

    def load_flow(self, video_path):
        """Load pre-computed optical flow for a video"""
        if self.flow_dir is None:
            return torch.zeros((2, self.num_frames-1, self.frame_height, self.frame_width), dtype=torch.float32)
            
        if video_path in self.preloaded_flow:
            return self.preloaded_flow[video_path]
            
        flow_path = Path(self.flow_dir) / Path(video_path).with_suffix('.flow.pt').name
        if flow_path.exists():
            return torch.load(flow_path)
        
        # Return empty tensor instead of None when flow file doesn't exist
        return torch.zeros((2, self.num_frames-1, self.frame_height, self.frame_width), dtype=torch.float32)

    def read_video(self, video_path):
        """
        Read a fixed number of frames from a video, evenly distributed throughout.
        """
        from utils.video_standardizer import extract_fixed_frames
        
        frames = extract_fixed_frames(
            video_path, 
            num_frames=self.num_frames, 
            resize_dim=(self.frame_width, self.frame_height)
        )
        if frames is None or len(frames) == 0:
            print(f"Error: Failed to extract frames from {video_path}")
            frames = [np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8) 
                      for _ in range(self.num_frames)]
        return frames[:self.num_frames]
    
    def process_frames(self, frames):
        """
        Process frames by applying transformations and formatting.
        """
        processed_frames = []
        for frame in frames:
            pil_image = Image.fromarray(frame)
            if self.transform:
                pil_image = self.transform(pil_image)
            processed_frames.append(pil_image)
        
        frames_tensor = torch.stack(processed_frames)
        if self.model_type == '3d_cnn':
            frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]
        return frames_tensor

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Get video frames
            if self.preload_to_ram and video_path in self.preloaded_frames:
                frames = self.preloaded_frames[video_path]
            else:
                frames = self.read_video(video_path)
            
            # Process frames
            frames_tensor = self.process_frames(frames)
            
            # Get optical flow
            flow_tensor = self.load_flow(video_path)
            
            # Return frames, flow, and label
            return frames_tensor, flow_tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Return empty tensors as fallback
            if self.model_type == '3d_cnn':
                frames_tensor = torch.zeros((3, self.num_frames, self.frame_height, self.frame_width), dtype=torch.float32)
            else:
                frames_tensor = torch.zeros((self.num_frames, 3, self.frame_height, self.frame_width), dtype=torch.float32)
            
            # Empty flow tensor
            flow_tensor = torch.zeros((2, self.num_frames-1, self.frame_height, self.frame_width), dtype=torch.float32)
            
            return frames_tensor, flow_tensor, torch.tensor(label, dtype=torch.long)

def get_dataloaders(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
                   batch_size=16, num_workers=4, target_fps=15, num_frames=16,
                   model_type='3d_cnn', pin_memory=True, flow_dir=None):
    """Get dataloaders for training, validation and testing"""
    
    # Create datasets based on model type
    if model_type == 'two_stream':
        print(f"Using dataset with pre-computed flow from {flow_dir}")
        train_dataset = EnhancedViolenceDatasetWithFlow(
            video_paths=train_paths,
            labels=train_labels,
            flow_dir=flow_dir,
            transform=get_train_transform(model_type),
            num_frames=num_frames,
            target_fps=target_fps,
            augment=True,
            model_type=model_type,
            training=True
        )
        
        val_dataset = EnhancedViolenceDatasetWithFlow(
            video_paths=val_paths,
            labels=val_labels,
            flow_dir=flow_dir,
            transform=get_val_transform(model_type),
            num_frames=num_frames,
            target_fps=target_fps,
            augment=False,
            model_type=model_type,
            training=False
        )
        
        test_dataset = EnhancedViolenceDatasetWithFlow(
            video_paths=test_paths,
            labels=test_labels,
            flow_dir=flow_dir,
            transform=get_val_transform(model_type),
            num_frames=num_frames,
            target_fps=target_fps,
            augment=False,
            model_type=model_type,
            training=False
        )
    else:
        # Standard dataset for all other models
        train_dataset = EnhancedViolenceDataset(
            video_paths=train_paths,
            labels=train_labels,
            transform=get_train_transform(model_type),
            num_frames=num_frames,
            target_fps=target_fps,
            augment=True,
            model_type=model_type,
            training=True
        )
        
        val_dataset = EnhancedViolenceDataset(
            video_paths=val_paths,
            labels=val_labels,
            transform=get_val_transform(model_type),
            num_frames=num_frames,
            target_fps=target_fps,
            augment=False,
            model_type=model_type,
            training=False
        )
        
        test_dataset = EnhancedViolenceDataset(
            video_paths=test_paths,
            labels=test_labels,
            transform=get_val_transform(model_type),
            num_frames=num_frames,
            target_fps=target_fps,
            augment=False,
            model_type=model_type,
            training=False
        )
    
    # Create and return data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader

def get_train_transform(model_type, frame_height=224, frame_width=224):
    """Get training transforms for the specified model type"""
    return transforms.Compose([
        transforms.Resize((frame_height, frame_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transform(model_type, frame_height=224, frame_width=224):
    """Get validation transforms for the specified model type"""
    return transforms.Compose([
        transforms.Resize((frame_height, frame_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
