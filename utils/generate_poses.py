#!/usr/bin/env python3
# generate_poses.py

import os
import argparse
import cv2
import mediapipe as mp
import numpy as np
import csv
from tqdm import tqdm
import glob

def setup_pose_detector():
    """Initialize MediaPipe pose detector"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # 0=Lite, 1=Full, 2=Heavy (choose based on your hardware)
    )
    return pose

def process_video(video_path, pose_detector, output_dir, target_fps=15):
    """
    Process a single video to extract pose keypoints.
    
    Args:
        video_path: Path to the video file
        pose_detector: MediaPipe pose detector instance
        output_dir: Directory to save the CSV output
        target_fps: Target frame rate for processing
        
    Returns:
        Path to the generated CSV file
    """
    # Extract video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Determine appropriate output subdirectory based on filename prefix
    if video_name.startswith('V_'):
        output_subdir = os.path.join(output_dir, 'Violence')
    elif video_name.startswith('NV_'):
        output_subdir = os.path.join(output_dir, 'NonViolence')
    else:
        output_subdir = output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_subdir, exist_ok=True)
    
    # Output CSV path
    csv_path = os.path.join(output_subdir, f"{video_name}.csv")
    
    # Skip if already processed
    if os.path.exists(csv_path):
        print(f"Skipping {video_name} - already processed")
        return csv_path
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None
    
    # Get video properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not orig_fps or orig_fps <= 0:
        orig_fps = 30  # Default assumption
    
    # Calculate frame interval to achieve target FPS
    frame_interval = max(1, int(round(orig_fps / target_fps)))
    
    # Get total frames for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_processed = total_frames // frame_interval + 1
    
    # Prepare CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        # MediaPipe pose model outputs 33 landmarks, each with x,y
        fieldnames = ['frame_idx'] + [f"{i}_{coord}" for i in range(33) for coord in ['x', 'y']]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process frames
        frame_idx = 0
        processed_idx = 0
        
        # Use tqdm for progress tracking
        with tqdm(total=expected_processed, desc=f"Processing {video_name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only frames at the specified interval
                if frame_idx % frame_interval == 0:
                    # Convert to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame with pose detector
                    results = pose_detector.process(rgb_frame)
                    
                    # Prepare row with frame index
                    row = {'frame_idx': processed_idx}
                    
                    # Extract keypoints if detected
                    if results.pose_landmarks:
                        for idx, landmark in enumerate(results.pose_landmarks.landmark):
                            # Store normalized coordinates (range 0-1)
                            row[f"{idx}_x"] = landmark.x
                            row[f"{idx}_y"] = landmark.y
                    else:
                        # If no pose detected, set all keypoints to 0
                        for idx in range(33):
                            row[f"{idx}_x"] = 0.0
                            row[f"{idx}_y"] = 0.0
                    
                    # Write to CSV
                    writer.writerow(row)
                    processed_idx += 1
                    pbar.update(1)
                
                frame_idx += 1
    
    # Release resources
    cap.release()
    
    print(f"Saved pose data for {video_name} - {processed_idx} frames processed")
    return csv_path

def batch_process_videos(video_dir, output_dir, target_fps=15):
    """
    Process all videos in the directory structure.
    
    Args:
        video_dir: Root directory containing videos
        output_dir: Directory to save pose keypoints
        target_fps: Target frame rate for processing
    """
    # Initialize pose detector
    pose_detector = setup_pose_detector()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Violence'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'NonViolence'), exist_ok=True)
    
    # Find all video files
    video_patterns = ['**/*.mp4', '**/*.avi', '**/*.mov', '**/*.mkv']
    video_paths = []
    
    for pattern in video_patterns:
        video_paths.extend(glob.glob(os.path.join(video_dir, pattern), recursive=True))
    
    print(f"Found {len(video_paths)} videos to process")
    
    # Filter videos based on naming conventions (V_ for Violence, NV_ for NonViolence)
    violence_videos = [v for v in video_paths if os.path.basename(v).startswith('V_')]
    nonviolence_videos = [v for v in video_paths if os.path.basename(v).startswith('NV_')]
    other_videos = [v for v in video_paths if not os.path.basename(v).startswith(('V_', 'NV_'))]
    
    print(f"Violence videos: {len(violence_videos)}")
    print(f"Non-violence videos: {len(nonviolence_videos)}")
    print(f"Other videos: {len(other_videos)}")
    
    # Process all videos
    processed_files = []
    
    try:
        for video_path in video_paths:
            try:
                csv_path = process_video(video_path, pose_detector, output_dir, target_fps)
                if csv_path:
                    processed_files.append(csv_path)
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
    finally:
        # Release resources
        pose_detector.close()
    
    print(f"Successfully processed {len(processed_files)}/{len(video_paths)} videos")
    return processed_files

def main():
    parser = argparse.ArgumentParser(description='Generate pose keypoints for violence detection dataset')
    parser.add_argument('--video_dir', type=str, default='./Data/VioNonVio', 
                        help='Directory containing violence/non-violence videos')
    parser.add_argument('--output_dir', type=str, default='./Data/Pose',
                        help='Directory to save pose keypoints')
    parser.add_argument('--fps', type=int, default=15,
                        help='Target FPS for pose extraction')
    parser.add_argument('--single_video', type=str, default=None,
                        help='Process a single video instead of batch processing')
    
    args = parser.parse_args()
    
    if args.single_video:
        # Process a single video
        if not os.path.exists(args.single_video):
            print(f"Error: Video file {args.single_video} not found")
            return
        
        pose_detector = setup_pose_detector()
        try:
            csv_path = process_video(args.single_video, pose_detector, args.output_dir, args.fps)
            if csv_path:
                print(f"Successfully saved pose data to {csv_path}")
        finally:
            pose_detector.close()
    else:
        # Batch process all videos
        if not os.path.exists(args.video_dir):
            print(f"Error: Video directory {args.video_dir} not found")
            return
        
        processed_files = batch_process_videos(args.video_dir, args.output_dir, args.fps)
        print(f"Completed pose extraction for {len(processed_files)} videos")
        
        
        
        # Add this to the top of your generate_poses.py script, just before the main function
def debug_script_execution():
    print("Script execution started")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if video directory exists
    video_dir = './Data/VioNonVio'
    if os.path.exists(video_dir):
        print(f"Video directory exists: {video_dir}")
        
        # Check for subdirectories
        subdirs = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]
        print(f"Subdirectories: {subdirs}")
    else:
        print(f"ERROR: Video directory does not exist: {video_dir}")
    
    # Check if output directory exists/can be created
    output_dir = './Data/Pose'
    if os.path.exists(output_dir):
        print(f"Output directory exists: {output_dir}")
    else:
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"ERROR creating output directory: {str(e)}")

# Call this at the beginning of your main function
debug_script_execution()

if __name__ == "__main__":
    main()