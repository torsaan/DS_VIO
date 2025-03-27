# utils/precompute_optical_flow.py
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import argparse
from pathlib import Path

def extract_frames(video_path, num_frames=16):
    """Extract fixed number of frames from video"""
    from video_standardizer import extract_fixed_frames
    
    frames = extract_fixed_frames(
        video_path, 
        num_frames=num_frames, 
        resize_dim=(224, 224)
    )
    
    if frames is None or len(frames) == 0:
        print(f"Error: Failed to extract frames from {video_path}")
        return None
    
    return frames

def compute_optical_flow(frames):
    """
    Compute optical flow using Farneback's algorithm.
    Args:
        frames: List of numpy arrays [T, H, W, C]
    Returns:
        Optical flow tensor [2, T-1, H, W]
    """
    if len(frames) < 2:
        return None
    
    flow_data = []
    gray_frames = []
    
    # Convert frames to grayscale
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_frames.append(gray)
    
    # Compute optical flow between consecutive frames
    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i+1], None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0
        )
        flow_data.append(flow)
    
    # Stack and transpose to [2, T-1, H, W]
    flow_stack = np.stack(flow_data, axis=0)  # [T-1, H, W, 2]
    flow_stack = flow_stack.transpose(3, 0, 1, 2)  # [2, T-1, H, W]
    
    return flow_stack

def process_dataset(video_dir, output_dir, num_frames=16):
    """Process all videos and save optical flow data"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files
    video_paths = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_paths.extend(Path(video_dir).glob(f'**/*{ext}'))
    
    print(f"Processing {len(video_paths)} videos...")
    
    for video_path in tqdm(video_paths):
        # Create output path with same structure
        rel_path = video_path.relative_to(video_dir)
        output_path = Path(output_dir) / rel_path.with_suffix('.flow.pt')
        
        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already processed
        if output_path.exists():
            continue
        
        # Extract frames
        frames = extract_frames(str(video_path), num_frames=num_frames)
        if frames is None:
            print(f"Skipping {video_path} due to frame extraction failure")
            continue
        
        # Compute optical flow
        flow = compute_optical_flow(frames)
        if flow is None:
            print(f"Skipping {video_path} due to optical flow computation failure")
            continue
        
        # Save as PyTorch tensor
        flow_tensor = torch.from_numpy(flow).float()
        torch.save(flow_tensor, output_path)
    
    print(f"Optical flow computation complete. Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute optical flow for videos")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save optical flow data")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to extract from each video")
    
    args = parser.parse_args()
    process_dataset(args.video_dir, args.output_dir, args.num_frames)