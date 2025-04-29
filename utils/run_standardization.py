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
    parser.add_argument('--flow_extraction', action='store_true',
                        help='Run optical flow extraction after standardization')
    
    args = parser.parse_args()
    
    # Create directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    
    standardized_dir = os.path.join(args.output_dir, 'standardized')
    analysis_dir = os.path.join(args.output_dir, 'analysis')
    flow_dir = os.path.join(args.output_dir, 'flow')
    
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
    
    # Step 4: Extract optical flow if requested
    if args.flow_extraction:
        os.makedirs(flow_dir, exist_ok=True)
        
        # Check if we have a flow extraction script
        flow_script = 'utils.precompute_optical_flow'
        
        run_command(
            [
                'python', '-m', flow_script,
                '--video_dir', standardized_dir,
                '--output_dir', flow_dir,
                '--num_frames', str(16)  # Fixed to 16 frames for consistency
            ],
            "Step 4: Extracting optical flow"
        )
    
    print("\nStandardization pipeline completed!")
    print(f"Original videos: {args.input_dir}")
    print(f"Standardized videos: {standardized_dir}")
    print(f"Analysis results: {analysis_dir}")
    if args.flow_extraction:
        print(f"Optical flow data: {flow_dir}")

if __name__ == "__main__":
    main()