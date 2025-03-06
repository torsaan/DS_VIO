# Enhanced ViolenceDataset class
import os
import csv
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils.augmentoor import VideoAugmenter
import random

class EnhancedViolenceDataset(Dataset):
    def __init__(self, video_paths, labels, pose_dir=None, 
                 transform=None, num_frames=32, target_fps=15,
                 normalize_pose=True, augment=True, model_type='3d_cnn',
                 frame_width=224, frame_height=224, training=True):
        """
        Enhanced dataset for violence detection with both video and pose data.
        
        Args:
            video_paths: List of paths to video files
            labels: List of labels (0 for non-violence, 1 for violence)
            pose_dir: Directory containing pose keypoint CSV files
            transform: Optional transforms to apply to video frames
            num_frames: Number of frames to sample from each video
            target_fps: Target frame rate for sampling (defaults to 15 FPS)
            normalize_pose: Whether to normalize pose keypoints
            augment: Whether to apply data augmentation
            model_type: Type of model ('3d_cnn', '2d_cnn_lstm', etc.)
            frame_width: Width for resizing frames
            frame_height: Height for resizing frames
            training: Whether this dataset is used for training
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

    def __len__(self):
        return len(self.video_paths)
    
    def read_video(self, video_path, num_frames = 32, frame_size =(112,112)):
        """
        Read video frames at the target FPS rate.
        """
        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if not orig_fps or orig_fps <= 0:
            orig_fps = 30  # fallback if FPS is not available
        
        # Calculate the interval to sample frames to achieve target_fps
        sample_interval = max(1, int(round(orig_fps / self.target_fps)))
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only add frames that are at the sampling interval
            if frame_idx % sample_interval == 0:
                #resize frame to the target size
                frame_resized = cv2.resize(frame, frame_size)
                # Convert frame from BGR to RGB
                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(frame_resized)
            
            frame_idx += 1
        
        cap.release()
        
        # Ensure we have exactly self.num_frames frames
        if len(frames) >= self.num_frames:
            # Use uniform sampling to get exactly num_frames
            indices = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        else:
            # Pad with the last frame if video is too short
            last_frame = frames[-1] if frames else np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            while len(frames) < self.num_frames:
                frames.append(last_frame.copy())
        
        return frames
    
    def load_pose_keypoints(self, video_path):
        """
        Load pose keypoints from CSV files.
        """
        if not self.pose_dir:
            return None
        
        # Extract the video filename without extension
        video_filename = os.path.basename(video_path)
        video_name = os.path.splitext(video_filename)[0]
        
        # Look for the pose CSV in different potential locations
        potential_paths = [
            os.path.join(self.pose_dir, f"{video_name}.csv"),
            os.path.join(self.pose_dir, "Violence", f"{video_name}.csv") if video_name.startswith("V_") else None,
            os.path.join(self.pose_dir, "NonViolence", f"{video_name}.csv") if video_name.startswith("NV_") else None,
        ]
        
        csv_path = next((p for p in potential_paths if p and os.path.exists(p)), None)
        
        if csv_path:
            try:
                keypoints = []
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)  # Skip header if present
                    
                    # Check if there's data after the header
                    for row in reader:
                        # Assuming format: frame_idx, x1, y1, x2, y2, ...
                        # Skip the frame index column if present
                        if len(row) >= 3:  # At least one keypoint (x,y) plus potential frame_idx
                            if row[0].isdigit():  # If first column is frame index
                                kp = list(map(float, row[1:]))
                            else:  # If all columns are keypoints
                                kp = list(map(float, row))
                            keypoints.append(kp)
                
                if not keypoints:
                    print(f"Warning: CSV file {csv_path} exists but contains no valid data")
                    return torch.zeros((self.num_frames, 66), dtype=torch.float32)
                    
                # Convert to numpy for easier processing
                keypoints = np.array(keypoints)
                
                # Sample keypoints to match self.num_frames
                if len(keypoints) >= self.num_frames:
                    indices = np.linspace(0, len(keypoints)-1, self.num_frames, dtype=int)
                    keypoints = keypoints[indices]
                else:
                    # Pad with the last keypoint if too few frames
                    padding = np.tile(keypoints[-1:], (self.num_frames - len(keypoints), 1))
                    keypoints = np.vstack([keypoints, padding])
                
                # Normalize keypoints if needed
                if self.normalize_pose:
                    keypoints = self.normalize_keypoints(keypoints)
                    
                # Apply augmentation if needed
                if self.augment and self.training and self.current_augment_types:
                    # Use the same augmentation types as for the video frames
                    keypoints = self.pose_augmenter.apply_to_keypoints(
                        keypoints, self.frame_width, self.frame_height, self.current_augment_types)
                
                return torch.tensor(keypoints, dtype=torch.float32)
                
            except Exception as e:
                print(f"Error loading pose keypoints from {csv_path}: {e}")
                return torch.zeros((self.num_frames, 66), dtype=torch.float32)
        else:
            print(f"Warning: No pose keypoints found for {video_name}")
            return torch.zeros((self.num_frames, 66), dtype=torch.float32)
    
    def normalize_keypoints(self, keypoints):
        """
        Normalize keypoints to the range [0, 1].
        """
        normalized = keypoints.copy()
        
        # Assuming alternating x, y coordinates
        for i in range(0, normalized.shape[1], 2):
            # Normalize x coordinates
            normalized[:, i] = normalized[:, i] / self.frame_width
            
            # Normalize y coordinates (if not at the end of the array)
            if i + 1 < normalized.shape[1]:
                normalized[:, i + 1] = normalized[:, i + 1] / self.frame_height
        
        return normalized
    
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
        Get a single video sample with its corresponding label and pose data.
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Read video frames
        try:
            frames = self.read_video(video_path)
            
            # Process frames
            frames_tensor = self.process_frames(frames)
            
            # Load pose keypoints
            pose_keypoints = self.load_pose_keypoints(video_path)
            
            # Return based on whether pose data is used
            if pose_keypoints is not None:
                return frames_tensor, pose_keypoints, torch.tensor(label, dtype=torch.long)
            else:
                return frames_tensor, torch.tensor(label, dtype=torch.long)
                
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Return zeros with the correct shape as a fallback
            if self.model_type == '3d_cnn':
                frames_tensor = torch.zeros((3, self.num_frames, self.frame_height, self.frame_width), dtype=torch.float32)
            else:
                frames_tensor = torch.zeros((self.num_frames, 3, self.frame_height, self.frame_width), dtype=torch.float32)
            
            pose_keypoints = torch.zeros((self.num_frames, 66), dtype=torch.float32)
            
            return frames_tensor, pose_keypoints, torch.tensor(label, dtype=torch.long)


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
                   test_video_paths, test_labels, pose_dir, batch_size=8,
                   num_workers=4, target_fps=15, num_frames=32, model_type='3d_cnn'):
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        train_video_paths, val_video_paths, test_video_paths: Lists of video paths
        train_labels, val_labels, test_labels: Lists of labels
        pose_dir: Directory containing pose keypoint CSVs
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for DataLoader
        target_fps: Target frame rate for video sampling
        num_frames: Number of frames to sample per video
        model_type: Type of model architecture
        
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = EnhancedViolenceDataset(
        train_video_paths, train_labels, pose_dir=pose_dir,
        transform=train_transform, num_frames=num_frames, 
        target_fps=target_fps, augment=True, model_type=model_type,
        training=True
    )
    
    val_dataset = EnhancedViolenceDataset(
        val_video_paths, val_labels, pose_dir=pose_dir,
        transform=val_transform, num_frames=num_frames, 
        target_fps=target_fps, augment=False, model_type=model_type,
        training=False
    )
    
    test_dataset = EnhancedViolenceDataset(
        test_video_paths, test_labels, pose_dir=pose_dir,
        transform=val_transform, num_frames=num_frames, 
        target_fps=target_fps, augment=False, model_type=model_type,
        training=False
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader