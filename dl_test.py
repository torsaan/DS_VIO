import torch
import numpy as np
from dataloader import get_dataloaders, EnhancedViolenceDataset

def dummy_read_video(self, video_path):
    # Create dummy frames: a list of num_frames random images of size (frame_height, frame_width, 3)
    frames = [np.random.randint(0, 255, (self.frame_height, self.frame_width, 3), dtype=np.uint8) 
              for _ in range(self.num_frames)]
    return frames

# Monkey-patch the read_video method of EnhancedViolenceDataset
EnhancedViolenceDataset.read_video = dummy_read_video

def test_dataloader():
    # Create dummy video paths and labels (using just two samples for testing)
    video_paths = ["dummy_video1.mp4", "dummy_video2.mp4"]
    labels = [0, 1]

    # Get dataloaders using the modified dataset (pose data is removed)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_video_paths=video_paths,
        train_labels=labels,
        val_video_paths=video_paths,
        val_labels=labels,
        test_video_paths=video_paths,
        test_labels=labels,
        batch_size=2,
        num_workers=0,  # Use 0 for debugging
        target_fps=10,
        num_frames=16,
        model_type='3d_cnn',  # For example, use 3D CNN configuration
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        preload_to_ram=False
    )

    # Retrieve one batch from the training dataloader
    for batch in train_loader:
        frames, labels_tensor = batch
        print("Frames shape:", frames.shape)
        print("Labels shape:", labels_tensor.shape)
        
        # For a '3d_cnn', frames are expected to be in shape [C, T, H, W] (i.e., [3, 16, 224, 224])
        assert frames.shape[1:] == (3, 16, 224, 224), f"Unexpected sample shape: {frames.shape[1:]}"

        # Labels should be a 1D tensor with batch size of 2
        assert labels_tensor.shape == (2,), f"Unexpected labels shape: {labels_tensor.shape}"
        
        print("Dataloader test passed!")
        break

if __name__ == '__main__':
    test_dataloader()
