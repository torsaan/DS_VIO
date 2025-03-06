#!/usr/bin/env python3
# utils/mediapipe_pose.py
# Simple and reliable pose extraction using MediaPipe

import os
import cv2
import mediapipe as mp
import numpy as np
import csv
import argparse
from tqdm import tqdm
import glob
from pathlib import Path

class MediaPipePoseExtractor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Initialize MediaPipe pose detector"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1  # 0=Lite, 1=Full, 2=Heavy
        )
        
        # Define connections for visualization
        self.connections = self.mp_pose.POSE_CONNECTIONS
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def process_image(self, image):
        """Process a single image and return pose landmarks"""
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        return results
    
    def draw_pose(self, image, results):
        """Draw pose landmarks on image"""
        if results.pose_landmarks:
            # Draw the pose landmarks
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.connections,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return image
    
    def landmarks_to_array(self, landmarks, image_width, image_height):
        """Convert landmarks to numpy array with normalized coordinates"""
        if not landmarks:
            return np.zeros((33, 3))
        
        landmark_array = np.array([[
            landmark.x,
            landmark.y,
            landmark.visibility
        ] for landmark in landmarks.landmark])
        
        return landmark_array
    
    def process_video(self, video_path, output_dir, target_fps=15, visualize=False):
        """
        Process a video and extract pose landmarks
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save results
            target_fps: Target frame rate for processing
            visualize: Whether to create visualization video
            
        Returns:
            Path to output CSV file
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video filename
        video_name = Path(video_path).stem
        
        # Determine class subdirectory
        if video_name.startswith('V_'):
            class_dir = os.path.join(output_dir, 'Violence')
        elif video_name.startswith('NV_'):
            class_dir = os.path.join(output_dir, 'NonViolence')
        else:
            class_dir = output_dir
        
        os.makedirs(class_dir, exist_ok=True)
        
        # Output CSV path
        csv_path = os.path.join(class_dir, f"{video_name}.csv")
        
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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            fps = 30  # Default if FPS info is missing
        
        # Calculate sampling interval to match target FPS
        interval = max(1, round(fps / target_fps))
        
        # Setup video writer for visualization
        viz_video = None
        if visualize:
            viz_path = os.path.join(class_dir, f"{video_name}_pose.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            viz_video = cv2.VideoWriter(viz_path, fourcc, target_fps, (width, height))
        
        # Open CSV file for writing
        with open(csv_path, 'w', newline='') as csvfile:
            # Define CSV header: frame_idx, then x,y,visibility for each landmark
            fieldnames = ['frame_idx'] + [
                f"{i}_{coord}" 
                for i in range(33)  # MediaPipe outputs 33 landmarks
                for coord in ['x', 'y', 'vis']
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Process frames
            frame_idx = 0
            processed_idx = 0
            
            with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frames at the specified interval
                    if frame_idx % interval == 0:
                        # Get pose landmarks
                        results = self.process_image(frame)
                        
                        # Convert landmarks to array
                        if results.pose_landmarks:
                            landmarks = self.landmarks_to_array(
                                results.pose_landmarks, width, height
                            )
                        else:
                            # Empty array if no landmarks detected
                            landmarks = np.zeros((33, 3))
                        
                        # Prepare row with frame index
                        row = {'frame_idx': processed_idx}
                        
                        # Add landmark coordinates to row
                        for i, (x, y, vis) in enumerate(landmarks):
                            row[f"{i}_x"] = x
                            row[f"{i}_y"] = y
                            row[f"{i}_vis"] = vis
                        
                        # Write to CSV
                        writer.writerow(row)
                        
                        # Create visualization if requested
                        if visualize:
                            viz_frame = self.draw_pose(frame.copy(), results)
                            
                            # Add frame number
                            cv2.putText(viz_frame, f"Frame: {processed_idx}", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Write to visualization video
                            viz_video.write(viz_frame)
                        
                        processed_idx += 1
                    
                    frame_idx += 1
                    pbar.update(1)
        
        # Release resources
        cap.release()
        if viz_video:
            viz_video.release()
        
        print(f"Processed {processed_idx} frames from {video_name}")
        return csv_path
    
    def process_dataset(self, video_dir, output_dir, target_fps=15, visualize=False):
        """
        Process all videos in a dataset
        
        Args:
            video_dir: Directory containing videos
            output_dir: Directory to save results
            target_fps: Target frame rate
            visualize: Whether to create visualization videos
        """
        # Find all video files
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(glob.glob(os.path.join(video_dir, '**', f'*{ext}'), recursive=True))
        
        print(f"Found {len(video_files)} videos to process")
        
        # Process each video
        successful = 0
        failed = 0
        
        for video_path in video_files:
            try:
                result = self.process_video(
                    video_path, output_dir, target_fps, visualize
                )
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                failed += 1
        
        print(f"Processing complete! Successfully processed {successful}/{len(video_files)} videos")
        print(f"Failed: {failed}/{len(video_files)} videos")

def main():
    parser = argparse.ArgumentParser(description="Extract pose keypoints using MediaPipe")
    parser.add_argument('--video_dir', type=str, required=True,
                      help='Directory containing videos to process')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save pose keypoints')
    parser.add_argument('--fps', type=int, default=15,
                      help='Target frames per second for pose extraction')
    parser.add_argument('--visualize', action='store_true',
                      help='Generate visualization videos with pose overlays')
    parser.add_argument('--detection_confidence', type=float, default=0.5,
                      help='Minimum confidence for pose detection')
    parser.add_argument('--tracking_confidence', type=float, default=0.5,
                      help='Minimum confidence for pose tracking')
    parser.add_argument('--single_video', type=str, default=None,
                      help='Process a single video file instead of a directory')
    
    args = parser.parse_args()
    
    # Initialize the pose extractor
    extractor = MediaPipePoseExtractor(
        min_detection_confidence=args.detection_confidence,
        min_tracking_confidence=args.tracking_confidence
    )
    
    # Process single video or entire dataset
    if args.single_video:
        if not os.path.exists(args.single_video):
            print(f"Error: Video file {args.single_video} not found")
            return
        
        # Process a single video
        extractor.process_video(
            args.single_video, 
            args.output_dir, 
            args.fps, 
            args.visualize
        )
    else:
        # Process the entire dataset
        extractor.process_dataset(
            args.video_dir, 
            args.output_dir, 
            args.fps, 
            args.visualize
        )

if __name__ == "__main__":
    main()