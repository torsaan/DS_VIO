# utils/frame_extraction.py
import cv2
import numpy as np
import os
from tqdm import tqdm
import data_config as cfg

def extract_fixed_frames(video_path, num_frames=None, resize_shape=None):
    """
    Extract a fixed number of frames evenly distributed throughout the video.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract (defaults to cfg.NUM_FRAMES)
        resize_shape: Tuple of (width, height) to resize frames (default: cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT)
        
    Returns:
        List of extracted frames as numpy arrays
    """
    if num_frames is None:
        num_frames = cfg.NUM_FRAMES
        
    if resize_shape is None:
        resize_shape = (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle very short videos
    if total_frames <= 0:
        cap.release()
        print(f"Warning: Video has no frames: {video_path}")
        return []
    
    # Calculate frame indices to sample (evenly distributed)
    if total_frames <= num_frames:
        # If video has fewer frames than requested, use all frames and duplicate as needed
        indices = np.arange(total_frames)
        # Add duplicates of the last frame
        indices = np.append(indices, [total_frames-1] * (num_frames - total_frames))
    else:
        # Sample frames evenly
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    # Extract frames
    frames = []
    for frame_idx in indices:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Resize frame
            if resize_shape is not None:
                frame = cv2.resize(frame, resize_shape)
            frames.append(frame)
        else:
            # Handle frame read failure (use black frame as fallback)
            print(f"Warning: Failed to read frame {frame_idx} from {video_path}")
            black_frame = np.zeros((resize_shape[1], resize_shape[0], 3), dtype=np.uint8)
            frames.append(black_frame)
    
    # Release video
    cap.release()
    
    # Ensure we have exactly num_frames
    if len(frames) < num_frames:
        # Duplicate last frame if needed
        while len(frames) < num_frames:
            frames.append(frames[-1].copy() if frames else np.zeros((resize_shape[1], resize_shape[0], 3), dtype=np.uint8))
    
    return frames

def process_video_dataset(input_dir, output_dir=None, num_frames=None):
    """
    Process all videos in a dataset to extract fixed frames.
    
    Args:
        input_dir: Directory containing videos
        output_dir: Directory to save processed frames (if None, only returns frames)
        num_frames: Number of frames to extract per video
        
    Returns:
        Dictionary mapping video paths to extracted frames
    """
    if num_frames is None:
        num_frames = cfg.NUM_FRAMES
    
    # Find all video files
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(root, file))
    
    print(f"Found {len(video_files)} videos to process")
    
    # Process each video
    results = {}
    
    for video_path in tqdm(video_files, desc="Extracting frames"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Extract frames
        frames = extract_fixed_frames(video_path, num_frames)
        results[video_path] = frames
        
        # Save frames if output directory specified
        if output_dir is not None:
            # Determine class directory
            if video_name.startswith('V_'):
                class_dir = os.path.join(output_dir, 'Violence', video_name)
            elif video_name.startswith('NV_'):
                class_dir = os.path.join(output_dir, 'NonViolence', video_name)
            else:
                class_dir = os.path.join(output_dir, 'Unknown', video_name)
            
            os.makedirs(class_dir, exist_ok=True)
            
            # Save frames
            for i, frame in enumerate(frames):
                frame_path = os.path.join(class_dir, f"frame_{i:03d}.jpg")
                cv2.imwrite(frame_path, frame)
    
    return results

def process_single_video(video_path, output_dir=None, num_frames=None):
    """
    Process a single video to extract fixed frames.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames (optional)
        num_frames: Number of frames to extract
        
    Returns:
        List of extracted frames
    """
    if num_frames is None:
        num_frames = cfg.NUM_FRAMES
    
    # Extract frames
    frames = extract_fixed_frames(video_path, num_frames)
    
    # Save frames if output directory specified
    if output_dir is not None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame_path = os.path.join(output_dir, f"{video_name}_frame_{i:03d}.jpg")
            cv2.imwrite(frame_path, frame)
    
    return frames

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract fixed number of frames from videos")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing videos")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for extracted frames")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to extract per video")
    parser.add_argument("--single_video", type=str, default=None, help="Process a single video instead of a directory")
    
    args = parser.parse_args()
    
    if args.single_video:
        if not os.path.exists(args.single_video):
            print(f"Error: Video file {args.single_video} not found")
        else:
            frames = process_single_video(args.single_video, args.output_dir, args.num_frames)
            print(f"Extracted {len(frames)} frames from {args.single_video}")
    else:
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory {args.input_dir} not found")
        else:
            results = process_video_dataset(args.input_dir, args.output_dir, args.num_frames)
            print(f"Processed {len(results)} videos")