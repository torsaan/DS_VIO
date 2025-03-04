#!/usr/bin/env python3
# visualize_multi_person_poses.py

import os
import cv2
import numpy as np
import argparse
import csv
import glob
from tqdm import tqdm
import random

def load_person_keypoints(csv_path):
    """
    Load pose keypoints for a single person from CSV file
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Dictionary mapping frame_idx to keypoints
    """
    keypoints_by_frame = {}
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_idx = int(row['frame_idx'])
                keypoints = []
                
                # Extract all x,y coordinates
                for i in range(33):  # MediaPipe has 33 keypoints
                    if f"{i}_x" in row and f"{i}_y" in row:
                        x = float(row[f"{i}_x"])
                        y = float(row[f"{i}_y"])
                        keypoints.append((x, y))
                    else:
                        keypoints.append((0, 0))
                
                keypoints_by_frame[frame_idx] = keypoints
        
        return keypoints_by_frame
    except Exception as e:
        print(f"Error loading keypoints from {csv_path}: {e}")
        return {}

def load_all_persons_keypoints(csv_files):
    """
    Load pose keypoints for all persons in a video
    
    Args:
        csv_files: List of CSV files for different persons
        
    Returns:
        Dictionary mapping frame_idx to list of keypoints
    """
    all_keypoints = {}
    person_colors = {}
    
    for person_idx, csv_path in enumerate(csv_files):
        # Generate a random color for this person
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        person_colors[person_idx] = color
        
        keypoints_by_frame = load_person_keypoints(csv_path)
        
        for frame_idx, keypoints in keypoints_by_frame.items():
            if frame_idx not in all_keypoints:
                all_keypoints[frame_idx] = []
            
            all_keypoints[frame_idx].append((keypoints, color))
    
    return all_keypoints, person_colors

def load_distance_data(distances_csv):
    """
    Load inter-person distance data
    
    Args:
        distances_csv: Path to distances CSV file
        
    Returns:
        Dictionary mapping frame_idx to list of distance data
    """
    if not distances_csv or not os.path.exists(distances_csv):
        return {}
    
    distances_by_frame = {}
    
    try:
        with open(distances_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_idx = int(row['frame_idx'])
                
                if frame_idx not in distances_by_frame:
                    distances_by_frame[frame_idx] = []
                
                distances_by_frame[frame_idx].append({
                    'person_i': int(row['person_i']),
                    'person_j': int(row['person_j']),
                    'centroid_dist': float(row['centroid_dist']),
                    'hip_dist': float(row['hip_dist']),
                    'shoulder_dist': float(row['shoulder_dist'])
                })
        
        return distances_by_frame
    except Exception as e:
        print(f"Error loading distance data from {distances_csv}: {e}")
        return {}

def draw_pose_on_frame(frame, keypoints, color=(0, 255, 0), threshold=0.01):
    """
    Draw pose keypoints and connections on a frame
    
    Args:
        frame: The video frame
        keypoints: List of (x,y) tuples for keypoints
        color: Color for this person's skeleton
        threshold: Minimum value to consider keypoint present
        
    Returns:
        Frame with drawn keypoints
    """
    h, w = frame.shape[:2]
    
    # Define connections between keypoints for visualization (MediaPipe pose)
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
        # Skip if keypoint is missing or invalid
        if x < threshold and y < threshold:
            continue
            
        # Convert normalized coordinates to pixel coordinates
        px, py = int(x * w), int(y * h)
        
        # Draw circle for keypoint
        cv2.circle(frame, (px, py), 5, color, -1)
    
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
        cv2.line(frame, (px1, py1), (px2, py2), color, 2)
    
    return frame

def draw_distances_on_frame(frame, persons_keypoints, distances, person_colors, threshold=0.01):
    """
    Draw inter-person distances on frame
    
    Args:
        frame: The video frame
        persons_keypoints: List of (keypoints, color) for each person
        distances: List of distance data for this frame
        person_colors: Dictionary mapping person_idx to color
        threshold: Minimum value to consider keypoint present
        
    Returns:
        Frame with drawn distances
    """
    if not distances:
        return frame
    
    h, w = frame.shape[:2]
    
    for dist in distances:
        person_i = dist['person_i']
        person_j = dist['person_j']
        
        # Skip if we don't have both persons
        if person_i >= len(persons_keypoints) or person_j >= len(persons_keypoints):
            continue
        
        # Get keypoints and colors for both persons
        keypoints_i, color_i = persons_keypoints[person_i]
        keypoints_j, color_j = persons_keypoints[person_j]
        
        # Draw line between centroids (use central keypoint as approximation)
        # MediaPipe's central keypoint is approximately 0 (nose)
        nose_i = keypoints_i[0]
        nose_j = keypoints_j[0]
        
        # Skip if either nose keypoint is missing
        if (nose_i[0] < threshold and nose_i[1] < threshold) or (nose_j[0] < threshold and nose_j[1] < threshold):
            continue
        
        # Convert to pixel coordinates
        px1, py1 = int(nose_i[0] * w), int(nose_i[1] * h)
        px2, py2 = int(nose_j[0] * w), int(nose_j[1] * h)
        
        # Draw line between persons
        cv2.line(frame, (px1, py1), (px2, py2), (255, 255, 0), 1)
        
        # Calculate midpoint
        mid_x, mid_y = (px1 + px2) // 2, (py1 + py2) // 2
        
        # Display distance information
        dist_text = f"D:{dist['centroid_dist']:.2f}"
        cv2.putText(frame, dist_text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return frame

def visualize_multi_person_poses(video_path, csv_files, distances_csv=None, output_path=None, target_fps=15):
    """
    Create visualization of multi-person poses
    
    Args:
        video_path: Path to original video
        csv_files: List of CSV files with pose data for different persons
        distances_csv: Optional path to distances CSV file
        output_path: Path to save visualization video (None for display only)
        target_fps: Target frame rate used during extraction
        
    Returns:
        True if successful, False otherwise
    """
    # Load person keypoints
    all_keypoints, person_colors = load_all_persons_keypoints(csv_files)
    
    # Load distance data if available
    distances_by_frame = load_distance_data(distances_csv)
    
    # Check if we have any keypoints
    if not all_keypoints:
        print("No valid keypoints found in the provided CSV files")
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
        orig_fps = 30  # Fallback
    
    # Calculate frame interval to match target FPS
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
    
    with tqdm(total=total_frames, desc="Visualizing multi-person poses") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only frames at the target interval
            if frame_idx % frame_interval == 0:
                # Check if we have keypoints for this frame
                if keypoint_idx in all_keypoints:
                    viz_frame = frame.copy()
                    
                    # Draw all persons
                    for keypoints, color in all_keypoints[keypoint_idx]:
                        viz_frame = draw_pose_on_frame(viz_frame, keypoints, color)
                    
                    # Draw distances if available
                    if keypoint_idx in distances_by_frame:
                        viz_frame = draw_distances_on_frame(
                            viz_frame, 
                            all_keypoints[keypoint_idx], 
                            distances_by_frame[keypoint_idx],
                            person_colors
                        )
                    
                    # Add frame indicators
                    cv2.putText(viz_frame, f"Frame: {frame_idx}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # Draw person count
                    person_count = len(all_keypoints[keypoint_idx])
                    cv2.putText(viz_frame, f"Persons: {person_count}", (10, 70), 
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
                        
                        cv2.imshow('Multi-Person Pose Visualization', viz_frame)
                        key = cv2.waitKey(30)  # Wait for 30ms
                        if key == 27:  # ESC key
                            break
                
                keypoint_idx += 1
            
            frame_idx += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    if out:
        out.close()
    cv2.destroyAllWindows()
    
    if output_path:
        print(f"Visualization saved to {output_path}")
    
    return True

def find_video_for_pose_files(csv_files, video_dir):
    """
    Find the original video file for a set of pose CSV files
    
    Args:
        csv_files: List of pose CSV files
        video_dir: Directory to search for videos
        
    Returns:
        Path to the matching video file
    """
    if not csv_files:
        return None
    
    # Extract video name from first CSV file (removing _person0, etc.)
    csv_basename = os.path.basename(csv_files[0])
    video_name = csv_basename.split('_person')[0]
    
    # Look for matching video with various extensions
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        # Check recursively in video_dir
        for root, _, files in os.walk(video_dir):
            for file in files:
                if file.startswith(video_name) and file.endswith(ext):
                    return os.path.join(root, file)
    
    return None

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
    
    # Find all video groups (files sharing same base name)
    video_groups = {}
    
    for root, _, files in os.walk(pose_dir):
        for file in files:
            if file.endswith('.csv') and '_person' in file:
                # Extract video name from person file
                video_name = file.split('_person')[0]
                
                if video_name not in video_groups:
                    video_groups[video_name] = []
                
                video_groups[video_name].append(os.path.join(root, file))
    
    print(f"Found pose data for {len(video_groups)} videos")
    
    # Randomly sample if needed
    video_names = list(video_groups.keys())
    if sample_count and sample_count < len(video_names):
        import random
        video_names = random.sample(video_names, sample_count)
        print(f"Sampled {sample_count} videos for visualization")
    
    processed_count = 0
    
    for video_name in video_names:
        # Get all CSV files for this video
        csv_files = sorted(video_groups[video_name])
        
        # Check if we have a distances file
        distances_csv = os.path.join(os.path.dirname(csv_files[0]), f"{video_name}_distances.csv")
        if not os.path.exists(distances_csv):
            distances_csv = None
        
        # Find matching video file
        video_path = find_video_for_pose_files(csv_files, video_dir)
        
        if not video_path:
            print(f"Could not find video for {video_name}")
            continue
        
        # Create output path
        output_path = os.path.join(output_dir, f"{video_name}_multi_pose_viz.mp4")
        
        print(f"Processing {video_name}...")
        success = visualize_multi_person_poses(video_path, csv_files, distances_csv, output_path, target_fps)
        
        if success:
            processed_count += 1
        
        print(f"Processed {processed_count}/{len(video_names)} videos")

def main():
    parser = argparse.ArgumentParser(description='Visualize multi-person pose keypoints on videos')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing original videos')
    parser.add_argument('--pose_dir', type=str, required=True, help='Directory containing pose CSV files')
    parser.add_argument('--output_dir', type=str, default='./visualization/multi', help='Directory to save visualization videos')
    parser.add_argument('--fps', type=int, default=15, help='Target FPS used during extraction')
    parser.add_argument('--single_video', type=str, default=None, help='Process a single video file')
    parser.add_argument('--pose_files', nargs='+', default=None, help='List of pose CSV files for single video')
    parser.add_argument('--distances_csv', type=str, default=None, help='Distances CSV file for single video')
    parser.add_argument('--sample', type=int, default=None, help='Number of videos to sample for batch visualization')
    parser.add_argument('--display_only', action='store_true', help='Display visualization without saving')
    
    args = parser.parse_args()
    
    if args.single_video and args.pose_files:
        # Process a single video
        output_path = None if args.display_only else os.path.join(args.output_dir, 'multi_pose_visualization.mp4')
        os.makedirs(args.output_dir, exist_ok=True)
        
        print(f"Visualizing multi-person poses for {args.single_video}")
        visualize_multi_person_poses(args.single_video, args.pose_files, args.distances_csv, output_path, args.fps)
    else:
        # Batch process videos
        batch_visualize_videos(args.video_dir, args.pose_dir, args.output_dir, args.fps, args.sample)

if __name__ == "__main__":
    main()