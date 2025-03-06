#!/usr/bin/env python3
# run_standardization.py
# Runs the complete video standardization pipeline for violence detection

import os
import argparse
import subprocess
import time
import shutil
from pathlib import Path
import sys

def run_command(command, description=None):
    """Run a shell command and print output"""
    if description:
        print(f"\n{'-'*80}")
        print(f"{description}")
        print(f"{'-'*80}")
    
    print(f"Running: {' '.join(command)}")
    start_time = time.time()
    
    # Use direct call for Windows compatibility
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        encoding='utf-8',
        errors='replace'  # Handle encoding errors
    )
    
    # Print the output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"Command completed successfully in {end_time - start_time:.2f} seconds")
        return True
    else:
        print(f"Command failed with return code {result.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run complete video standardization pipeline')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing raw videos')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base output directory for processed data')
    parser.add_argument('--width', type=int, default=224,
                        help='Target video width')
    parser.add_argument('--height', type=int, default=224,
                        help='Target video height')
    parser.add_argument('--fps', type=int, default=15,
                        help='Target video FPS')
    parser.add_argument('--max_duration', type=float, default=6.0,
                        help='Maximum video duration in seconds')
    parser.add_argument('--min_duration', type=float, default=2.0,
                        help='Minimum video duration in seconds')
    parser.add_argument('--skip_analysis', action='store_true',
                        help='Skip dataset analysis step')
    parser.add_argument('--skip_standardization', action='store_true',
                        help='Skip video standardization step')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum worker processes for parallel processing')
    parser.add_argument('--pose_extraction', action='store_true',
                        help='Run pose extraction after standardization')
    
    args = parser.parse_args()
    
    # Create directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    
    standardized_dir = os.path.join(args.output_dir, 'standardized')
    analysis_dir = os.path.join(args.output_dir, 'analysis')
    pose_dir = os.path.join(args.output_dir, 'pose')
    
    os.makedirs(standardized_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Step 1: Analyze the original dataset
    if not args.skip_analysis:
        run_command(
            [
                'python', '-m', 'utils.dataset_analyzer',
                '--data_dir', args.input_dir,
                '--output_dir', analysis_dir,
                '--max_workers', str(args.max_workers)
            ],
            "Step 1: Analyzing original dataset"
        )
    
    # Step 2: Standardize videos
    if not args.skip_standardization:
        run_command(
            [
                'python', '-m', 'utils.video_standardizer',
                '--input_dir', args.input_dir,
                '--output_dir', standardized_dir,
                '--target_width', str(args.width),
                '--target_height', str(args.height),
                '--target_fps', str(args.fps),
                '--max_duration', str(args.max_duration),
                '--min_duration', str(args.min_duration),
                '--max_workers', str(args.max_workers),
                '--verify'
            ],
            "Step 2: Standardizing videos"
        )
    
    # Step 3: Analyze the standardized dataset and compare
    if not args.skip_analysis:
        run_command(
            [
                'python', '-m', 'utils.dataset_analyzer',
                '--data_dir', standardized_dir,
                '--output_dir', os.path.join(analysis_dir, 'standardized'),
                '--max_workers', str(args.max_workers),
                '--compare', args.input_dir
            ],
            "Step 3: Analyzing standardized dataset and comparing with original"
        )
    
    # Step 4: Extract poses if requested
    if args.pose_extraction:
        os.makedirs(pose_dir, exist_ok=True)
        
        # Check if we have a pose extraction script
        pose_script = None
        for potential_script in ['utils.pose_extraction', 'utils.multi_person_pose', 'utils.pose_on_frames']:
            try:
                if potential_script.endswith('.py'):
                    # Direct Python file
                    if os.path.exists(potential_script):
                        pose_script = potential_script
                        break
                else:
                    # Try importing module
                    __import__(potential_script)
                    pose_script = potential_script
                    break
            except (ImportError, ModuleNotFoundError):
                continue
        
        if pose_script:
            run_command(
                [
                    'python', '-m', pose_script,
                    '--video_dir' if 'pose.py' in pose_script else '--input_dir', standardized_dir,
                    '--output_dir', pose_dir,
                    '--fps', str(args.fps)
                ],
                "Step 4: Extracting pose keypoints"
            )
        else:
            print("\nWarning: Could not find pose extraction script. Please run pose extraction manually.")
            print("Suggested command:")
            print(f"python -m your_pose_extraction_script --input_dir {standardized_dir} --output_dir {pose_dir} --fps {args.fps}")
    
    print("\nStandardization pipeline completed!")
    print(f"Original videos: {args.input_dir}")
    print(f"Standardized videos: {standardized_dir}")
    print(f"Analysis results: {analysis_dir}")
    if args.pose_extraction:
        print(f"Pose keypoints: {pose_dir}")

if __name__ == "__main__":
    main()