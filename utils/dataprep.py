# utils/data_preparation.py
import os
import glob
from sklearn.model_selection import train_test_split

def prepare_violence_nonviolence_data(data_root, test_size=0.2, val_size=0.15, random_state=42):
    """
    Prepare data paths and labels from the VioNonVio directory structure.
    
    Args:
        data_root: Root directory containing the data folders
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
    """
    # Define paths to the VioNonVio folders
    violence_dir = os.path.join(data_root, "VioNonVio", "Violence")
    nonviolence_dir = os.path.join(data_root, "VioNonVio", "NonViolence")
    
    # Collect video paths and labels
    video_paths = []
    labels = []
    
    # Find Violence videos (with V_ prefix) - label 1
    violence_videos = glob.glob(os.path.join(violence_dir, "V_*.mp4"))
    video_paths.extend(violence_videos)
    labels.extend([1] * len(violence_videos))
    
    # Find NonViolence videos (with NV_ prefix) - label 0
    nonviolence_videos = glob.glob(os.path.join(nonviolence_dir, "NV_*.mp4"))
    video_paths.extend(nonviolence_videos)
    labels.extend([0] * len(nonviolence_videos))
    
    # Print summary of data found
    print(f"Found {len(violence_videos)} violence videos (V_*)")
    print(f"Found {len(nonviolence_videos)} non-violence videos (NV_*)")
    print(f"Total videos: {len(video_paths)}")
    
    # Split into train+val and test sets
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        video_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Further split train+val into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, 
        test_size=val_size/(1-test_size),  # Adjust validation size relative to train+val
        random_state=random_state, 
        stratify=train_val_labels
    )
    
    # Print split summary
    print(f"Training set: {len(train_paths)} videos")
    print(f"Validation set: {len(val_paths)} videos")
    print(f"Test set: {len(test_paths)} videos")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

def check_pose_data_availability(video_paths, pose_dir):
    """
    Check which videos have corresponding pose keypoint data.
    
    Args:
        video_paths: List of video file paths
        pose_dir: Directory containing pose keypoint CSVs
        
    Returns:
        available_count: Number of videos with pose data
        total_count: Total number of videos
    """
    available_count = 0
    missing_videos = []
    
    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Check potential CSV locations
        potential_paths = [
            os.path.join(pose_dir, f"{video_name}.csv"),
            os.path.join(pose_dir, "Violence", f"{video_name}.csv") if video_name.startswith("V_") else None,
            os.path.join(pose_dir, "NonViolence", f"{video_name}.csv") if video_name.startswith("NV_") else None,
        ]
        
        if any(p and os.path.exists(p) for p in potential_paths if p):
            available_count += 1
        else:
            missing_videos.append(video_name)
    
    total_count = len(video_paths)
    print(f"Found pose data for {available_count}/{total_count} videos ({available_count/total_count*100:.1f}%)")
    
    if len(missing_videos) > 0:
        print(f"Missing pose data for {len(missing_videos)} videos")
        if len(missing_videos) <= 10:
            print("Missing for: ", ", ".join(missing_videos))
        else:
            print("First 10 missing: ", ", ".join(missing_videos[:10]))
    
    return available_count, total_count