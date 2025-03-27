# Add these imports if they're not already present
import os
import glob
from sklearn.model_selection import train_test_split
import json
import os
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
import json


RANDOM_SEED = 42  # Fixed seed for reproducible splits
TEST_SIZE = 0.2   # 20% for test set
VAL_SIZE = 0.15   # 15% of remaining for validation
NUM_FRAMES = 16   # Fixed number of frames per video


def prepare_violence_nonviolence_data(data_root, test_size=0.2, val_size=0.15, random_state=42):
    """
    Prepare data paths and labels from the VioNonVio directory structure.
    Ensures consistent splits using a fixed random seed.
    Uses pathlib.Path for cross-platform path handling.
    """
    # Convert input path to Path object for cross-platform compatibility
    data_root = Path(data_root)
    
    # Define paths to the VioNonVio folders
    violence_dir = data_root / "Violence"
    nonviolence_dir = data_root / "NonViolence"
        
    # Collect video paths and labels
    video_paths = []
    labels = []
    
    # Find Violence videos (with V_ prefix) - label 1
    violence_videos = list(violence_dir.glob("V_*.mp4"))
    video_paths.extend([str(path) for path in violence_videos])
    labels.extend([1] * len(violence_videos))
    
    # Find NonViolence videos (with NV_ prefix) - label 0
    nonviolence_videos = list(nonviolence_dir.glob("NV_*.mp4"))
    video_paths.extend([str(path) for path in nonviolence_videos])
    labels.extend([0] * len(nonviolence_videos))
    
    # Print summary of data found
    print(f"Found {len(violence_videos)} violence videos (V_*)")
    print(f"Found {len(nonviolence_videos)} non-violence videos (NV_*)")
    print(f"Total videos: {len(video_paths)}")
    
    # Split into train+val and test sets
    # Using a fixed random_state ensures the same split every time
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        video_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Further split train+val into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, 
        test_size=val_size/(1-test_size),  # Adjust validation size relative to train+val
        random_state=random_state,  # Same random_state for consistency
        stratify=train_val_labels
    )
    
    # Print split summary
    print(f"Training set: {len(train_paths)} videos")
    print(f"Validation set: {len(val_paths)} videos")
    print(f"Test set: {len(test_paths)} videos")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def save_data_splits(data_root, output_path="./data_splits.json"):
    """
    Save consistent train/val/test splits to a JSON file.
    This allows both team members to use exactly the same data splits.
    """
    # Get data splits
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        prepare_violence_nonviolence_data(data_root)
    
    # Create dictionary with splits
    data_splits = {
        "train": {
            "paths": train_paths,
            "labels": [int(label) for label in train_labels]
        },
        "val": {
            "paths": val_paths,
            "labels": [int(label) for label in val_labels]
        },
        "test": {
            "paths": test_paths,
            "labels": [int(label) for label in test_labels]
        },
        "metadata": {
            "random_seed": RANDOM_SEED,
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "num_frames": NUM_FRAMES
        }
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(data_splits, f, indent=2)
    
    print(f"Data splits saved to {output_path}")
    return output_path

def load_data_splits(splits_path="./data_splits.json"):
    """
    Load data splits from a JSON file.
    Normalizes paths for the current platform.
    """
    # Load from file
    with open(splits_path, 'r') as f:
        data_splits = json.load(f)
    
    # Extract splits
    train_paths = data_splits["train"]["paths"]
    train_labels = data_splits["train"]["labels"]
    val_paths = data_splits["val"]["paths"]
    val_labels = data_splits["val"]["labels"]
    test_paths = data_splits["test"]["paths"]
    test_labels = data_splits["test"]["labels"]
    
    # Normalize paths for current OS
    train_paths = [os.path.normpath(path) for path in train_paths]
    val_paths = [os.path.normpath(path) for path in val_paths]
    test_paths = [os.path.normpath(path) for path in test_paths]
    
    # Print split summary
    print(f"Training set: {len(train_paths)} videos")
    print(f"Validation set: {len(val_paths)} videos")
    print(f"Test set: {len(test_paths)} videos")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels