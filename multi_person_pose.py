#!/usr/bin/env python3
# multi_person_pose.py

import os
import cv2
import torch
import numpy as np
import mediapipe as mp
import csv
import argparse
from tqdm import tqdm
import glob

def load_yolo():
    """
    Load YOLO model for person detection
    """
    # Load YOLOv5 model with PyTorch Hub
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    
    # Set model parameters
    model.conf = 0.25  # Confidence threshold
    model.iou = 0.45   # NMS IoU threshold
    model.classes = [0]  # Only detect persons (class 0 in COCO)
    model.max_det = 20   # Maximum number of detections per image
    
    # Move to GPU if available
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    return model

def setup_pose_detector():
    """Initialize MediaPipe pose detector"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # 0=Lite, 1=Full, 2=Heavy
    )
    return pose

def detect_persons_yolo(image, yolo_model):
    """
    Detect persons in an image using YOLO
    
    Args:
        image: BGR image
        yolo_model: YOLOv5 model
        
    Returns:
        List of bounding boxes [x1, y1, x2, y2, confidence]
    """
    # YOLOv5 expects RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run YOLO
    results = yolo_model(img_rgb)
    
    # Extract person detections
    persons = []
    predictions = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
    
    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred
        if int(cls) == 0:  # Class 0 is person in COCO
            persons.append([x1, y1, x2, y2, conf])
    
    return persons

def extract_pose_from_detection(image, detection, pose_detector, frame_width, frame_height):
    """
    Extract pose keypoints for a single detected person
    
    Args:
        image: BGR image
        detection: Bounding box [x1, y1, x2, y2, conf]
        pose_detector: MediaPipe pose detector
        frame_width, frame_height: Original frame dimensions
        
    Returns:
        List of normalized keypoints [(x1,y1), (x2,y2), ...]
    """
    # Extract bounding box
    x1, y1, x2, y2, _ = detection
    
    # Add some padding
    padding = 0.1
    width = x2 - x1
    height = y2 - y1
    
    # Apply padding with bounds checking
    x1_padded = max(0, x1 - width * padding)
    y1_padded = max(0, y1 - height * padding)
    x2_padded = min(frame_width, x2 + width * padding)
    y2_padded = min(frame_height, y2 + height * padding)
    
    # Crop image to person bounding box
    person_image = image[int(y1_padded):int(y2_padded), int(x1_padded):int(x2_padded)]
    
    # Skip if crop is empty
    if person_image.size == 0:
        return None
    
    # Convert to RGB for MediaPipe
    rgb_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
    
    # Process with pose detector
    results = pose_detector.process(rgb_image)
    
    # Extract keypoints
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            # Convert to coordinates relative to cropped image
            x_in_crop = landmark.x
            y_in_crop = landmark.y
            
            # Convert back to coordinates in original image
            x_in_orig = (x_in_crop * (x2_padded - x1_padded) + x1_padded) / frame_width
            y_in_orig = (y_in_crop * (y2_padded - y1_padded) + y1_padded) / frame_height
            
            # Ensure values are in [0, 1]
            x_in_orig = max(0, min(1, x_in_orig))
            y_in_orig = max(0, min(1, y_in_orig))
            
            keypoints.append((x_in_orig, y_in_orig))
    else:
        return None
    
    return keypoints

def calculate_inter_person_distances(detections, keypoints_list):
    """
    Calculate distances between detected persons
    
    Args:
        detections: List of person bounding boxes
        keypoints_list: List of keypoint lists for each person
        
    Returns:
        List of distance features [centroid_dist, hip_dist, shoulder_dist]
    """
    if len(detections) < 2 or len(keypoints_list) < 2:
        return []
    
    distance_features = []
    
    # Hip keypoints indices in MediaPipe
    left_hip_idx, right_hip_idx = 23, 24
    # Shoulder keypoints indices
    left_shoulder_idx, right_shoulder_idx = 11, 12
    
    for i in range(len(detections)):
        for j in range(i+1, len(detections)):
            if keypoints_list[i] is None or keypoints_list[j] is None:
                continue
                
            # Calculate centroid distances from bounding boxes
            x1_i, y1_i, x2_i, y2_i, _ = detections[i]
            x1_j, y1_j, x2_j, y2_j, _ = detections[j]
            
            centroid_i = ((x1_i + x2_i) / 2, (y1_i + y2_i) / 2)
            centroid_j = ((x1_j + x2_j) / 2, (y1_j + y2_j) / 2)
            
            centroid_dist = np.sqrt((centroid_i[0] - centroid_j[0])**2 + (centroid_i[1] - centroid_j[1])**2)
            
            # Calculate hip distances if keypoints exist
            hip_i = ((keypoints_list[i][left_hip_idx][0] + keypoints_list[i][right_hip_idx][0]) / 2,
                     (keypoints_list[i][left_hip_idx][1] + keypoints_list[i][right_hip_idx][1]) / 2)
            
            hip_j = ((keypoints_list[j][left_hip_idx][0] + keypoints_list[j][right_hip_idx][0]) / 2,
                     (keypoints_list[j][left_hip_idx][1] + keypoints_list[j][right_hip_idx][1]) / 2)
            
            hip_dist = np.sqrt((hip_i[0] - hip_j[0])**2 + (hip_i[1] - hip_j[1])**2)
            
            # Calculate shoulder distances
            shoulder_i = ((keypoints_list[i][left_shoulder_idx][0] + keypoints_list[i][right_shoulder_idx][0]) / 2,
                          (keypoints_list[i][left_shoulder_idx][1] + keypoints_list[i][right_shoulder_idx][1]) / 2)
            
            shoulder_j = ((keypoints_list[j][left_shoulder_idx][0] + keypoints_list[j][right_shoulder_idx][0]) / 2,
                          (keypoints_list[j][left_shoulder_idx][1] + keypoints_list[j][right_shoulder_idx][1]) / 2)
            
            shoulder_dist = np.sqrt((shoulder_i[0] - shoulder_j[0])**2 + (shoulder_i[1] - shoulder_j[1])**2)
            
            # Add to features
            distance_features.append({
                'person_i': i,
                'person_j': j,
                'centroid_dist': centroid_dist,
                'hip_dist': hip_dist,
                'shoulder_dist': shoulder_dist
            })
    
    return distance_features

def process_video_multi_person(video_path, output_dir, target_fps=15, include_distances=True, max_persons=10):
    """
    Process a video for multi-person pose extraction
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save CSV files
        target_fps: Target frame rate for processing
        include_distances: Whether to include inter-person distances
        max_persons: Maximum number of persons to track
        
    Returns:
        List of paths to generated CSV files
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
    
    # Load models
    yolo_model = load_yolo()
    pose_detector = setup_pose_detector()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not orig_fps or orig_fps <= 0:
        orig_fps = 30  # Default
    
    # Calculate frame interval to achieve target FPS
    frame_interval = max(1, int(round(orig_fps / target_fps)))
    
    # Setup CSV files
    csv_files = []
    csv_writers = []
    
    # Create individual CSVs for each potential person
    for i in range(max_persons):
        csv_path = os.path.join(output_subdir, f"{video_name}_person{i}.csv")
        csv_files.append(csv_path)
        
        with open(csv_path, 'w', newline='') as f:
            # MediaPipe pose model outputs 33 landmarks, each with x,y
            fieldnames = ['frame_idx'] + [f"{j}_{coord}" for j in range(33) for coord in ['x', 'y']]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            csv_writers.append((f, writer))
    
    # Create a CSV for inter-person distances if needed
    distances_csv = None
    distances_writer = None
    
    if include_distances:
        distances_path = os.path.join(output_subdir, f"{video_name}_distances.csv")
        csv_files.append(distances_path)
        
        distances_csv = open(distances_path, 'w', newline='')
        fieldnames = ['frame_idx', 'person_i', 'person_j', 'centroid_dist', 'hip_dist', 'shoulder_dist']
        distances_writer = csv.DictWriter(distances_csv, fieldnames=fieldnames)
        distances_writer.writeheader()
    
    # Process frames
    frame_idx = 0
    processed_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only frames at the specified interval
            if frame_idx % frame_interval == 0:
                # Detect persons with YOLO
                detections = detect_persons_yolo(frame, yolo_model)
                
                # Limit to max_persons
                detections = detections[:max_persons]
                
                # Extract pose for each person
                keypoints_list = []
                for det_idx, detection in enumerate(detections):
                    keypoints = extract_pose_from_detection(frame, detection, pose_detector, frame_width, frame_height)
                    keypoints_list.append(keypoints)
                    
                    # Write to person's CSV if valid pose
                    if keypoints is not None:
                        row = {'frame_idx': processed_idx}
                        for j, (x, y) in enumerate(keypoints):
                            row[f"{j}_x"] = x
                            row[f"{j}_y"] = y
                        
                        # Get file and writer
                        _, writer = csv_writers[det_idx]
                        writer.writerow(row)
                    
                # Calculate and write inter-person distances if needed
                if include_distances and distances_writer and len(detections) >= 2:
                    distance_features = calculate_inter_person_distances(detections, keypoints_list)
                    
                    for dist in distance_features:
                        dist['frame_idx'] = processed_idx
                        distances_writer.writerow(dist)
                
                processed_idx += 1
            
            frame_idx += 1
            pbar.update(1)
    
    # Close video and CSV files
    cap.release()
    
    for f, _ in csv_writers:
        f.close()
    
    if distances_csv:
        distances_csv.close()
    
    print(f"Saved multi-person pose data for {video_name} - {processed_idx} frames processed")
    
    # Clean up empty files (no detections for that person ID)
    valid_files = []
    for csv_path in csv_files:
        # Check if file has more than just the header
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            if len(lines) <= 1:  # Only header
                os.remove(csv_path)
                print(f"Removed empty file: {csv_path}")
            else:
                valid_files.append(csv_path)
    
    return valid_files

def batch_process_videos(video_dir, output_dir, target_fps=15, include_distances=True, max_persons=10):
    """
    Process all videos in a directory for multi-person pose extraction
    
    Args:
        video_dir: Directory containing videos
        output_dir: Directory to save pose data
        target_fps: Target FPS for extraction
        include_distances: Whether to include inter-person distances
        max_persons: Maximum number of persons to track per video
        
    Returns:
        List of all generated CSV files
    """
    # Find all video files
    video_paths = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_paths.extend(glob.glob(os.path.join(video_dir, '**', f'*{ext}'), recursive=True))
    
    print(f"Found {len(video_paths)} videos to process")
    
    all_csv_files = []
    
    for video_path in video_paths:
        try:
            csv_files = process_video_multi_person(
                video_path, output_dir, target_fps, include_distances, max_persons
            )
            all_csv_files.extend(csv_files)
            print(f"Generated {len(csv_files)} files for {os.path.basename(video_path)}")
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
    
    print(f"Generated a total of {len(all_csv_files)} CSV files")
    return all_csv_files

def main():
    parser = argparse.ArgumentParser(description='Extract multi-person pose keypoints from videos')
    parser.add_argument('--video_dir', type=str, default='C:\Github\DS_VIO\Data\VioNonVio', 
                        help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='./Data/Pose',
                        help='Directory to save pose keypoints')
    parser.add_argument('--fps', type=int, default=15,
                        help='Target FPS for extraction')
    parser.add_argument('--max_persons', type=int, default=10,
                        help='Maximum number of persons to track per video')
    parser.add_argument('--no_distances', action='store_true',
                        help='Disable inter-person distance calculation')
    parser.add_argument('--single_video', type=str, default=None,
                        help='Process a single video instead of batch processing')
    
    args = parser.parse_args()
    
    if args.single_video:
        # Process a single video
        if not os.path.exists(args.single_video):
            print(f"Error: Video file {args.single_video} not found")
            return
        
        csv_files = process_video_multi_person(
            args.single_video, args.output_dir, args.fps, 
            not args.no_distances, args.max_persons
        )
        print(f"Generated {len(csv_files)} CSV files")
    else:
        # Batch process all videos
        if not os.path.exists(args.video_dir):
            print(f"Error: Video directory {args.video_dir} not found")
            return
        
        batch_process_videos(
            args.video_dir, args.output_dir, args.fps,
            not args.no_distances, args.max_persons
        )

if __name__ == "__main__":
    main()