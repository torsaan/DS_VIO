# dataloader.py
# Defines a custom Dataset that loads video frames and, optionally, precomputed pose keypoints

import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np

# Example transforms (customize as needed)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class ViolenceDataset(Dataset):
    def __init__(self, video_paths, labels, pose_dir=None, transform=None, num_frames=32):
        self.video_paths = video_paths
        self.labels = labels
        self.pose_dir = pose_dir
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

def read_video(self, video_path):
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not orig_fps or orig_fps <= 0:
        orig_fps = 15  # fallback if FPS is not available
    
    # Calculate the interval to sample frames at 15 FPS
    sample_interval = max(1, int(round(orig_fps / 15)))
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only add frames that are at the sampling interval
        if frame_idx % sample_interval == 0:
            # Convert frame (if necessary) and apply transformations
            if self.transform:
                from PIL import Image
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = self.transform(frame)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
            frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    
    # Ensure a fixed number of frames per video clip (e.g., self.num_frames)
    if len(frames) >= self.num_frames:
        indices = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
        frames = [frames[i] for i in indices]
    else:
        # Pad with the last frame if video is too short
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
    
    return torch.stack(frames)  # [T, C, H, W]
def load_pose_keypoints(self, video_path):
        if not self.pose_dir:
            return None
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        csv_path = os.path.join(self.pose_dir, f"{video_name}.csv")
        keypoints = []
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    kp = list(map(float, row[1:]))  # assuming first column is frame index
                    keypoints.append(kp)
            # Sample keypoints to match self.num_frames
            if len(keypoints) >= self.num_frames:
                indices = np.linspace(0, len(keypoints)-1, self.num_frames, dtype=int)
                keypoints = [keypoints[i] for i in indices]
            else:
                while len(keypoints) < self.num_frames:
                    keypoints.append(keypoints[-1])
            return torch.tensor(keypoints, dtype=torch.float32)
        else:
            # Return zeros if not found (adjust feature size as needed)
            return torch.zeros((self.num_frames, 66))

def getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self.read_video(video_path)
        pose_keypoints = self.load_pose_keypoints(video_path)
        return frames, pose_keypoints, torch.tensor(label, dtype=torch.long)

def get_dataloaders(train_video_paths, train_labels, val_video_paths, val_labels, test_video_paths, test_labels, pose_dir, batch_size):
    train_dataset = ViolenceDataset(train_video_paths, train_labels, pose_dir, transform=get_train_transform())
    val_dataset = ViolenceDataset(val_video_paths, val_labels, pose_dir, transform=get_val_transform())
    test_dataset = ViolenceDataset(test_video_paths, test_labels, pose_dir, transform=get_val_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader

# You can test the dataloader by running this file directly.
if __name__ == '__main__':
    # Dummy paths and labels for testing
    train_paths = ['./Data/Violence/video1.mp4']
    train_labels = [1]
    val_paths = ['./Data/NonViolence/video2.mp4']
    val_labels = [0]
    test_paths = ['./Data/Violence/video3.mp4']
    test_labels = [1]
    
    tl, vl, _ = get_dataloaders(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, './Data/pose_keypoints', batch_size=2)
    for frames, pose, label in tl:
        print("Frames shape:", frames.shape, "Pose shape:", pose.shape, "Label:", label)
        break
