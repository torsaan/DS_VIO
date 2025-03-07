#!/usr/bin/env python3
# utils/video_standardizer.py

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
import shutil
import json

def get_video_properties(video_path):
    """
    Get the properties of a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video properties (width, height, fps, frame_count)
        or None if the video cannot be opened
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "path": video_path
    }

def standardize_video(video_path, output_path, target_width=224, target_height=224, 
                     num_frames=16, fps=15, verbose=False):
    """
    Standardize a video by extracting a fixed number of frames and saving as a new video.
    
    Args:
        video_path: Path to the input video file
        output_path: Path to save the standardized video
        target_width: Target width resolution
        target_height: Target height resolution
        num_frames: Fixed number of frames to extract
        fps: Output video frame rate
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with processing statistics or None if processing failed
    """
    import os
    import cv2
    import numpy as np
    from . import get_video_properties
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get video properties
    props = get_video_properties(video_path)
    if not props:
        return None
    
    if verbose:
        print(f"Processing {video_path}:")
        print(f"  Original: {props['width']}x{props['height']}, {props['fps']} FPS, {props['duration']:.2f}s")
    
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
    
    # Calculate stats
    actual_frame_count = len(frames)
    actual_duration = actual_frame_count / fps
    
    stats = {
        "original_width": props["width"],
        "original_height": props["height"],
        "original_fps": props["fps"],
        "original_duration": props["duration"],
        "standardized_width": target_width,
        "standardized_height": target_height,
        "standardized_fps": fps,
        "standardized_duration": actual_duration,
        "original_frame_count": props["frame_count"],
        "standardized_frame_count": actual_frame_count,
        "was_extended": actual_frame_count > props["frame_count"],
        "was_truncated": actual_frame_count < props["frame_count"]
    }
    
    if verbose:
        print(f"  Standardized: {target_width}x{target_height}, {fps} FPS, {actual_duration:.2f}s")
        if stats["was_extended"]:
            print("  Note: Video was extended to meet frame count")
        if stats["was_truncated"]:
            print("  Note: Video was truncated to meet frame count")
    
    return stats






def process_dataset(input_dir, output_dir, target_width=224, target_height=224, 
                   target_fps=15, max_duration=6, min_duration=2, 
                   max_workers=4, keep_structure=True, verbose=False):
    """
    Process all videos in a dataset, preserving directory structure.
    
    Args:
        input_dir: Input directory containing videos or subdirectories
        output_dir: Output directory to save standardized videos
        target_width, target_height: Target resolution
        target_fps: Target frame rate
        max_duration: Maximum duration in seconds
        min_duration: Minimum duration in seconds
        max_workers: Maximum number of worker processes
        keep_structure: Whether to maintain the directory structure
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with processing statistics
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files recursively
    video_files = []
    
    # Supported video extensions
    extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    # Recursively search for video files
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                video_path = os.path.join(root, file)
                video_files.append(video_path)
    
    print(f"Found {len(video_files)} video files in {input_dir}")
    
    # Setup for processing statistics
    all_stats = {}
    successful = 0
    failed = 0
    
    # Process videos with parallel workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Prepare tasks
        tasks = []
        
        for video_path in video_files:
            # Determine output path (preserving structure if requested)
            if keep_structure:
                # Preserve the original directory structure
                rel_path = os.path.relpath(video_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                
                # Ensure the parent directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            else:
                # Flatten the directory structure
                output_filename = os.path.basename(video_path)
                output_path = os.path.join(output_dir, output_filename)
            
            # Ensure output has .mp4 extension
            output_path = Path(output_path).with_suffix('.mp4')
            
            # Add task to list
            tasks.append((video_path, str(output_path)))
        
        # Submit tasks and track results
        futures = []
        for video_path, output_path in tasks:
            future = executor.submit(
                standardize_video,
                video_path,
                output_path,
                target_width,
                target_height,
                target_fps,
                max_duration,
                min_duration,
                verbose
            )
            futures.append((future, video_path, output_path))
        
        # Process results with progress tracking
        for future, video_path, output_path in tqdm(futures, desc="Standardizing videos"):
            try:
                stats = future.result()
                if stats:
                    all_stats[video_path] = stats
                    successful += 1
                else:
                    failed += 1
                    print(f"Failed to process {video_path}")
            except Exception as e:
                failed += 1
                print(f"Error processing {video_path}: {str(e)}")
    
    # Compute summary statistics
    summary = {
        "total_videos": len(video_files),
        "successful": successful,
        "failed": failed,
        "avg_original_duration": np.mean([stats["original_duration"] for stats in all_stats.values()]),
        "avg_standardized_duration": np.mean([stats["standardized_duration"] for stats in all_stats.values()]),
        "extended_count": sum(1 for stats in all_stats.values() if stats["was_extended"]),
        "truncated_count": sum(1 for stats in all_stats.values() if stats["was_truncated"]),
        "target_width": target_width,
        "target_height": target_height,
        "target_fps": target_fps,
        "max_duration": max_duration,
        "min_duration": min_duration
    }
    
    # Save statistics to output directory
    stats_path = os.path.join(output_dir, "standardization_stats.json")
    with open(stats_path, 'w') as f:
        json.dump({
            "summary": summary,
            "details": {k: v for k, v in all_stats.items()}
        }, f, indent=2)
    
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {successful}/{len(video_files)} videos")
    print(f"  Failed: {failed}/{len(video_files)} videos")
    print(f"  Extended {summary['extended_count']} videos to meet minimum duration")
    print(f"  Truncated {summary['truncated_count']} videos to meet maximum duration")
    print(f"  Average original duration: {summary['avg_original_duration']:.2f}s")
    print(f"  Average standardized duration: {summary['avg_standardized_duration']:.2f}s")
    print(f"  Statistics saved to: {stats_path}")
    
    return summary

def verify_dataset(data_dir):
    """Verify that all videos in the dataset have been properly standardized"""
    # Find all videos
    video_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    
    print(f"Checking {len(video_files)} videos for standardization...")
    
    issues = []
    properties = []
    
    for video_path in tqdm(video_files, desc="Verifying videos"):
        props = get_video_properties(video_path)
        if props:
            properties.append(props)
            # Check if video has expected properties
            if props['width'] != 224 or props['height'] != 224:
                issues.append(f"{video_path}: Incorrect resolution - {props['width']}x{props['height']}")
            
            if abs(props['fps'] - 15) > 0.5:  # Allow small FPS deviation
                issues.append(f"{video_path}: Incorrect FPS - {props['fps']}")
            
            if props['duration'] < 2 or props['duration'] > 6.5:  # Allow slight duration deviation
                issues.append(f"{video_path}: Incorrect duration - {props['duration']:.2f}s")
        else:
            issues.append(f"{video_path}: Could not read video properties")
    
    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("All videos verified successfully!")
    
    # Calculate statistics
    if properties:
        avg_width = np.mean([p['width'] for p in properties])
        avg_height = np.mean([p['height'] for p in properties])
        avg_fps = np.mean([p['fps'] for p in properties])
        avg_duration = np.mean([p['duration'] for p in properties])
        
        print(f"\nDataset statistics:")
        print(f"  Average resolution: {avg_width:.1f}x{avg_height:.1f}")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Average duration: {avg_duration:.2f}s")
    
    return len(issues) == 0


def extract_fixed_frames(video_path, num_frames=16, resize_dim=(224, 224)):
    """
    Extract a fixed number of frames evenly distributed throughout the video
    """
    import cv2
    import numpy as np
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to sample (evenly distributed)
    if total_frames <= num_frames:
        # If video has fewer frames than required, we'll duplicate some frames
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    else:
        # Sample frames evenly
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Resize frame
            if resize_dim:
                frame = cv2.resize(frame, resize_dim)
            frames.append(frame)
        else:
            # If seeking fails, use last valid frame or a black frame
            if frames:
                frames.append(frames[-1].copy())
            else:
                # Create a blank frame if no valid frames yet
                blank_frame = np.zeros((resize_dim[1], resize_dim[0], 3), dtype=np.uint8)
                frames.append(blank_frame)
    
    cap.release()
    
    return frames




















def main():
    parser = argparse.ArgumentParser(description='Standardize video dataset for violence detection')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing videos')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for standardized videos')
    parser.add_argument('--target_width', type=int, default=224,
                        help='Target width resolution')
    parser.add_argument('--target_height', type=int, default=224,
                        help='Target height resolution')
    parser.add_argument('--target_fps', type=int, default=15,
                        help='Target frames per second')
    parser.add_argument('--max_duration', type=float, default=6.0,
                        help='Maximum video duration in seconds')
    parser.add_argument('--min_duration', type=float, default=2.0,
                        help='Minimum video duration in seconds')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of parallel workers')
    parser.add_argument('--flatten', action='store_true',
                        help='Flatten directory structure (ignore original structure)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information during processing')
    parser.add_argument('--verify', action='store_true',
                        help='Verify dataset after processing')
    parser.add_argument('--verify_only', type=str, default=None,
                        help='Only verify the specified directory without processing')
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_dataset(args.verify_only)
    else:
        process_dataset(
            args.input_dir,
            args.output_dir,
            args.target_width,
            args.target_height,
            args.target_fps,
            args.max_duration,
            args.min_duration,
            args.max_workers,
            not args.flatten,
            args.verbose
        )
        
        if args.verify:
            print("\nVerifying processed dataset...")
            verify_dataset(args.output_dir)

if __name__ == "__main__":
    main()