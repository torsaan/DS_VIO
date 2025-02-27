# utils/pose_extraction.py
import os
import cv2
import csv
import numpy as np
import argparse
from tqdm import tqdm

def extract_pose_keypoints(video_path, output_dir, target_fps=15, use_mediapipe=True):
    """
    Extract pose keypoints from a video using MediaPipe or OpenPose.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save the CSV with keypoints
        target_fps: Target frame rate for extraction
        use_mediapipe: If True, use MediaPipe (simpler), else try OpenPose (more complex)
        
    Returns:
        Path to the saved CSV file
    """
    # Extract video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Determine output file path
    # If the filename starts with V_ (violence), put it in the Violence subfolder
    if video_name.startswith('V_'):
        output_subdir = os.path.join(output_dir, 'Violence')
    # If the filename starts with NV_ (non-violence), put it in the NonViolence subfolder
    elif video_name.startswith('NV_'):
        output_subdir = os.path.join(output_dir, 'NonViolence')
    else:
        output_subdir = output_dir
    
    # Ensure output directory exists
    os.makedirs(output_subdir, exist_ok=True)
    
    # Output CSV path
    csv_path = os.path.join(output_subdir, f"{video_name}.csv")
    
    # Skip if the file already exists
    if os.path.exists(csv_path):
        print(f"Skipping {video_name} - keypoints already extracted")
        return csv_path
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not orig_fps or orig_fps <= 0:
        orig_fps = 30  # fallback
    
    # Calculate sampling interval to achieve target_fps
    sample_interval = max(1, int(round(orig_fps / target_fps)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize pose model
    if use_mediapipe:
        # Import MediaPipe (install with: pip install mediapipe)
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1  # 0, 1 or 2
            )
        except ImportError:
            print("Error: MediaPipe not found. Install with: pip install mediapipe")
            return None
    else:
        # For OpenPose, we'd need to check if the OpenPose Python bindings are available
        # This is more complex and depends on your specific OpenPose installation
        try:
            # Try to import OpenPose Python bindings (adjust import based on installation)
            from openpose import pyopenpose as op
            
            # Initialize OpenPose
            params = {
                "model_folder": "./models/",  # Path to OpenPose models
                "number_people_max": 1,       # We expect only one person
                "disable_blending": True      # Speed optimization
            }
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()
            
        except ImportError:
            print("Error: OpenPose Python bindings not found.")
            print("Falling back to MediaPipe...")
            use_mediapipe = True
            
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
    
    # Open CSV file for writing
    with open(csv_path, 'w', newline='') as csvfile:
        # Define CSV header based on pose estimation method
        if use_mediapipe:
            # MediaPipe outputs 33 keypoints with x, y coordinates
            fieldnames = ['frame_idx'] + [f"{part}_{coord}" 
                                         for part in range(33) 
                                         for coord in ['x', 'y']]
        else:
            # OpenPose outputs 25 keypoints with x, y coordinates
            fieldnames = ['frame_idx'] + [f"{part}_{coord}" 
                                         for part in range(25) 
                                         for coord in ['x', 'y']]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process frames
        frame_idx = 0
        processed_count = 0
        
        # Process video frames with progress bar
        with tqdm(total=total_frames // sample_interval, desc=f"Extracting pose from {video_name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only frames at the target interval
                if frame_idx % sample_interval == 0:
                    # Convert BGR to RGB for pose estimation
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, _ = frame.shape
                    
                    if use_mediapipe:
                        # Process with MediaPipe
                        results = pose.process(frame_rgb)
                        
                        # Initialize row with frame index
                        row = {'frame_idx': processed_count}
                        
                        if results.pose_landmarks:
                            # Extract and store normalized keypoints
                            for i, landmark in enumerate(results.pose_landmarks.landmark):
                                row[f"{i}_x"] = landmark.x  # normalized [0, 1]
                                row[f"{i}_y"] = landmark.y  # normalized [0, 1]
                        else:
                            # If no landmarks detected, store zeros
                            for i in range(33):
                                row[f"{i}_x"] = 0.0
                                row[f"{i}_y"] = 0.0
                    else:
                        # Process with OpenPose
                        datum = op.Datum()
                        datum.cvInputData = frame
                        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                        
                        # Initialize row with frame index
                        row = {'frame_idx': processed_count}
                        
                        if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
                            # Extract and store normalized keypoints
                            keypoints = datum.poseKeypoints[0]  # First person
                            for i in range(25):  # OpenPose has 25 keypoints
                                if i < len(keypoints):
                                    # Normalize to [0, 1]
                                    row[f"{i}_x"] = keypoints[i][0] / w if keypoints[i][0] != 0 else 0.0
                                    row[f"{i}_y"] = keypoints[i][1] / h if keypoints[i][1] != 0 else 0.0
                                else:
                                    row[f"{i}_x"] = 0.0
                                    row[f"{i}_y"] = 0.0
                        else:
                            # If no keypoints detected, store zeros
                            for i in range(25):
                                row[f"{i}_x"] = 0.0
                                row[f"{i}_y"] = 0.0
                    
                    # Write row to CSV
                    writer.writerow(row)
                    processed_count += 1
                    pbar.update(1)
                
                frame_idx += 1
        
        print(f"Processed {processed_count} frames for {video_name}")
    
    # Release resources
    cap.release()
    if use_mediapipe:
        pose.close()
    
    return csv_path

def batch_extract_keypoints(video_dir, output_dir, target_fps=15, use_mediapipe=True):
    """
    Batch process all videos in a directory to extract pose keypoints.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save CSV files
        target_fps: Target frame rate for extraction
        use_mediapipe: If True, use MediaPipe, else try OpenPose
        
    Returns:
        List of paths to saved CSV files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files in the directory and subdirectories
    video_paths = []
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_paths.append(os.path.join(root, file))
    
    print(f"Found {len(video_paths)} videos to process")
    
    # Process each video
    csv_paths = []
    for video_path in video_paths:
        try:
            csv_path = extract_pose_keypoints(
                video_path, output_dir, target_fps, use_mediapipe
            )
            if csv_path:
                csv_paths.append(csv_path)
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
    
    print(f"Successfully processed {len(csv_paths)}/{len(video_paths)} videos")
    return csv_paths

def main():
    parser = argparse.ArgumentParser(description='Extract pose keypoints from videos')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for pose keypoints')
    parser.add_argument('--target_fps', type=int, default=15, help='Target FPS for keypoint extraction')
    parser.add_argument('--use_mediapipe', action='store_true', help='Use MediaPipe instead of OpenPose')
    parser.add_argument('--single_video', type=str, default=None, help='Process a single video file')
    
    args = parser.parse_args()
    
    if args.single_video:
        if not os.path.exists(args.single_video):
            print(f"Error: Video file {args.single_video} not found")
            return
        
        csv_path = extract_pose_keypoints(
            args.single_video, args.output_dir, args.target_fps, args.use_mediapipe
        )
        if csv_path:
            print(f"Keypoints saved to {csv_path}")
    else:
        if not os.path.exists(args.video_dir):
            print(f"Error: Video directory {args.video_dir} not found")
            return
        
        csv_paths = batch_extract_keypoints(
            args.video_dir, args.output_dir, args.target_fps, args.use_mediapipe
        )
        print(f"Processed {len(csv_paths)} videos")

if __name__ == "__main__":
    main()