#!/usr/bin/env python3
# setup_project.py
"""
Script to set up consistent data environment for both team members.
Run this once at the beginning of your project to generate consistent splits.
"""
import os
import argparse
from utils.dataprep import save_data_splits, load_data_splits
import glob
from tqdm import tqdm
import cv2
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Set up violence detection project data")
    parser.add_argument("--data_dir", type=str, default="./Data/VioNonVio", 
                        help="Directory containing original videos")
    parser.add_argument("--output_dir", type=str, default="./Data/Processed/standardized",
                        help="Directory to save standardized videos")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Fixed number of frames per video")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frame rate for standardized videos")
    parser.add_argument("--splits_path", type=str, default="./data_splits.json",
                        help="Path to save/load data splits")
    parser.add_argument("--verify_only", action="store_true",
                  help="Only verify standardized videos without regenerating them"),
    parser.add_argument("--skip_standardization", action="store_true",
                        help="Skip video standardization step")
    return parser.parse_args()


def standardize_video(video_path, output_path, target_width=224, target_height=224, 
                     num_frames=16, fps=10, verbose=False):
    """
    Standardize a video by extracting a fixed number of frames and saving as a new video.
    """
    from utils.video_standardizer import extract_fixed_frames
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract frames with fixed count
    frames = extract_fixed_frames(
        video_path, 
        num_frames=num_frames, 
        resize_dim=(target_width, target_height)
    )
    
    if frames is None or len(frames) == 0:
        print(f"Error: Failed to extract frames from {video_path}")
        return None
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    # Write frames to output video
    for frame in frames:
        out.write(frame)
    
    # Release resources
    out.release()
    
    return True


def standardize_videos(data_dir, output_dir, num_frames, fps):
    """Standardize all videos in data_dir to have a fixed number of frames."""
    # Find all videos
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(glob.glob(os.path.join(data_dir, '**', f'*{ext}'), recursive=True))
    
    print(f"Found {len(video_files)} videos to standardize")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "Violence"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "NonViolence"), exist_ok=True)
    
    # Standardize each video
    for video_path in tqdm(video_files, desc="Standardizing videos"):
        # Determine output path
        file_name = os.path.basename(video_path)
        if "Violence" in video_path or file_name.startswith("V_"):
            output_subdir = "Violence"
        elif "NonViolence" in video_path or file_name.startswith("NV_"):
            output_subdir = "NonViolence"
        else:
            print(f"Warning: Could not determine class for {video_path}, skipping")
            continue
        
        # Ensure filename starts with correct prefix
        if not (file_name.startswith("V_") or file_name.startswith("NV_")):
            if output_subdir == "Violence":
                file_name = "V_" + file_name
            else:
                file_name = "NV_" + file_name
        
        output_path = os.path.join(output_dir, output_subdir, file_name)
        
        # Standardize video
        try:
            standardize_video(
                video_path, 
                output_path, 
                num_frames=num_frames, 
                fps=fps
            )
        except Exception as e:
            print(f"Error standardizing {video_path}: {e}")


def verify_standardized_videos(output_dir, num_frames, fps):
    """Verify that all standardized videos have the expected properties."""
    # Find all videos
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(glob.glob(os.path.join(output_dir, '**', f'*{ext}'), recursive=True))
    
    print(f"Verifying {len(video_files)} standardized videos")
    
    issues = []
    
    for video_path in tqdm(video_files, desc="Verifying videos"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            issues.append(f"Could not open {video_path}")
            continue
        
        # Check frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        cap.release()
        
        # Check properties
        if frame_count != num_frames:
            issues.append(f"{video_path}: Expected {num_frames} frames, found {frame_count}")
        
        if abs(video_fps - fps) > 0.5:  # Allow small FPS variation
            issues.append(f"{video_path}: Expected {fps} FPS, found {video_fps}")
    
    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("All videos verified successfully!")
    
    return len(issues) == 0


def main():
    args = parse_args()
    if args.verify_only:
            print("Verifying standardized videos...")
            verify_standardized_videos(args.output_dir, args.num_frames, args.fps)
            return 
    
    
    # Standardize videos
    if not args.skip_standardization:
        print("Standardizing videos...")
        standardize_videos(args.data_dir, args.output_dir, args.num_frames, args.fps)
        
        # Verify standardized videos
        verify_standardized_videos(args.output_dir, args.num_frames, args.fps)
    
    # Generate consistent data splits
    print("\nGenerating data splits...")
    save_data_splits(args.output_dir, args.splits_path)
    
    # Test loading data splits
    print("\nTesting data splits...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        load_data_splits(args.splits_path)
    
    print("\nSetup complete! Share the following with your teammate:")
    print(f"1. Standardized videos directory: {args.output_dir}")
    print(f"2. Data splits file: {args.splits_path}")
    print(f"3. Configuration: {args.num_frames} frames per video, {args.fps} FPS")


if __name__ == "__main__":
    main()