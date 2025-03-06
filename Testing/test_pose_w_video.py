#!/usr/bin/env python3
# visualize_poses.py

import os
import cv2
import numpy as np
import argparse
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_keypoints_from_csv(csv_path):
    """
    Load pose keypoints from CSV file
    
    Args:
        csv_path: Path to the CSV file containing pose keypoints
        
    Returns:
        Dictionary mapping frame_idx to keypoints array
    """
    frame_keypoints = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_idx = int(row['frame_idx'])
                keypoints = []
                
                # Extract all x,y coordinates
                for i in range(33):  # Mediapipe has 33 keypoints
                    if f"{i}_x" in row and f"{i}_y" in row:
                        x = float(row[f"{i}_x"])
                        y = float(row[f"{i}_y"])
                        keypoints.append((x, y))
                    else:
                        keypoints.append((0, 0))  # Default if missing
                
                frame_keypoints[frame_idx] = keypoints
        return frame_keypoints
    except Exception as e:
        print(f"Error loading keypoints from {csv_path}: {e}")
        return {}

def draw_pose_on_frame(frame, keypoints, threshold=0.01):
    """
    Draw pose keypoints and connections on a frame
    
    Args:
        frame: The video frame (numpy array)
        keypoints: List of (x,y) tuples for keypoints
        threshold: Minimum coordinate value to consider keypoint present
        
    Returns:
        Frame with drawn keypoints
    """
    h, w = frame.shape[:2]
    
    # Define connections between keypoints for visualization
    # Mediapipe pose connections
    pose_connections = [
        # Face connections
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        # Torso connections
        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        # Connectors
        (0, 9), (0, 10), (9, 11), (10, 12),
        # Legs
        (11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)
    ]
    
    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        # Skip if keypoint is missing or invalid (near zero value)
        if x < threshold and y < threshold:
            continue
            
        # Convert normalized coordinates to pixel coordinates
        px, py = int(x * w), int(y * h)
        
        # Draw circle for keypoint
        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
        
        # Optionally add keypoint number
        # cv2.putText(frame, str(i), (px+5, py+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw connections
    for connection in pose_connections:
        idx1, idx2 = connection
        
        # Skip if either keypoint is missing
        if (idx1 >= len(keypoints) or idx2 >= len(keypoints)):
            continue
            
        x1, y1 = keypoints[idx1]
        x2, y2 = keypoints[idx2]
        
        # Skip if either keypoint is invalid
        if (x1 < threshold and y1 < threshold) or (x2 < threshold and y2 < threshold):
            continue
            
        # Convert to pixel coordinates
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        
        # Draw line
        cv2.line(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
    
    return frame

def visualize_video_with_poses(video_path, pose_csv_path, output_path=None, target_fps=15):
    """
    Create a visualization of pose keypoints overlaid on the original video
    
    Args:
        video_path: Path to the original video
        pose_csv_path: Path to the CSV file with pose keypoints
        output_path: Path to save the visualization video (None for display only)
        target_fps: Target frame rate of the original processing
        
    Returns:
        True if successful, False otherwise
    """
    # Load keypoints
    frame_keypoints = load_keypoints_from_csv(pose_csv_path)
    if not frame_keypoints:
        print(f"No keypoints found in {pose_csv_path}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not orig_fps or orig_fps <= 0:
        orig_fps = 30  # Fallback value
    
    # Calculate frame interval to match target FPS used during processing
    frame_interval = max(1, int(round(orig_fps / target_fps)))
    
    # Setup output video writer if required
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    # Process frames
    frame_idx = 0
    keypoint_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc="Visualizing poses") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only frames at the target interval (same as during extraction)
            if frame_idx % frame_interval == 0:
                # Check if we have keypoints for this frame
                if keypoint_idx in frame_keypoints:
                    # Draw pose on frame
                    viz_frame = draw_pose_on_frame(frame.copy(), frame_keypoints[keypoint_idx])
                    
                    # Add frame indicators
                    cv2.putText(viz_frame, f"Frame: {frame_idx}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(viz_frame, f"Keypoint idx: {keypoint_idx}", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # Write to output video or display
                    if out:
                        out.write(viz_frame)
                    else:
                        # Resize for display if too large
                        if width > 1280 or height > 720:
                            display_scale = min(1280/width, 720/height)
                            display_width = int(width * display_scale)
                            display_height = int(height * display_scale)
                            viz_frame = cv2.resize(viz_frame, (display_width, display_height))
                        
                        cv2.imshow('Pose Visualization', viz_frame)
                        key = cv2.waitKey(30)  # Wait for 30ms
                        if key == 27:  # ESC key
                            break
                
                keypoint_idx += 1
            
            frame_idx += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    if output_path:
        print(f"Visualization saved to {output_path}")
    
    return True

def batch_visualize_videos(video_dir, pose_dir, output_dir, target_fps=15, sample_count=None):
    """
    Visualize poses for multiple videos
    
    Args:
        video_dir: Directory containing original videos
        pose_dir: Directory containing pose CSV files
        output_dir: Directory to save visualization videos
        target_fps: Target FPS used during extraction
        sample_count: Number of videos to sample (None for all)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files in pose directory
    pose_files = []
    for root, _, files in os.walk(pose_dir):
        for file in files:
            if file.endswith('.csv'):
                pose_files.append(os.path.join(root, file))
    
    print(f"Found {len(pose_files)} pose files")
    
    # Randomly sample if needed
    if sample_count and sample_count < len(pose_files):
        import random
        pose_files = random.sample(pose_files, sample_count)
        print(f"Sampled {sample_count} files for visualization")
    
    processed_count = 0
    
    for pose_file in pose_files:
        # Extract video name from pose file
        pose_filename = os.path.basename(pose_file)
        video_name = os.path.splitext(pose_filename)[0]
        
        # Look for matching video file
        video_path = None
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # Check in Violence directory
            potential_path = os.path.join(video_dir, 'Violence', video_name + ext)
            if os.path.exists(potential_path):
                video_path = potential_path
                break
                
            # Check in NonViolence directory
            potential_path = os.path.join(video_dir, 'NonViolence', video_name + ext)
            if os.path.exists(potential_path):
                video_path = potential_path
                break
                
            # Check directly in video_dir
            potential_path = os.path.join(video_dir, video_name + ext)
            if os.path.exists(potential_path):
                video_path = potential_path
                break
        
        if not video_path:
            print(f"Could not find video for {video_name}")
            continue
        
        # Create output path
        output_path = os.path.join(output_dir, f"{video_name}_pose_viz.mp4")
        
        print(f"Processing {video_name}...")
        success = visualize_video_with_poses(video_path, pose_file, output_path, target_fps)
        
        if success:
            processed_count += 1
        
        print(f"Processed {processed_count}/{len(pose_files)} videos")

def main():
    parser = argparse.ArgumentParser(description='Visualize pose keypoints on videos')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing original videos')
    parser.add_argument('--pose_dir', type=str, required=True, help='Directory containing pose CSV files')
    parser.add_argument('--output_dir', type=str, default='./visualization', help='Directory to save visualization videos')
    parser.add_argument('--fps', type=int, default=15, help='Target FPS used during extraction')
    parser.add_argument('--single_video', type=str, default=None, help='Process a single video file')
    parser.add_argument('--single_pose', type=str, default=None, help='Process a single pose CSV file')
    parser.add_argument('--sample', type=int, default=None, help='Number of videos to sample for batch visualization')
    parser.add_argument('--display_only', action='store_true', help='Display visualization without saving')
    
    args = parser.parse_args()
    
    if args.single_video and args.single_pose:
        # Process a single video
        output_path = None if args.display_only else os.path.join(args.output_dir, 'pose_visualization.mp4')
        os.makedirs(args.output_dir, exist_ok=True)
        
        print(f"Visualizing poses for {args.single_video}")
        visualize_video_with_poses(args.single_video, args.single_pose, output_path, args.fps)
    else:
        # Batch process videos
        batch_visualize_videos(args.video_dir, args.pose_dir, args.output_dir, args.fps, args.sample)

if __name__ == "__main__":
    main()