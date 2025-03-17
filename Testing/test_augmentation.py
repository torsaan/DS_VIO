#!/usr/bin/env python3
# Testing/test_augmentation.py
"""
Test script to verify the data augmentation pipeline.
Tests video and pose keypoint augmentation functions.
"""

import os
import sys
import torch
import numpy as np
import argparse
import cv2
import time
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.augmentor import VideoAugmenter

def display_frames(frames, title="Video Frames", rows=1, save_path=None):
    """Display a grid of video frames"""
    num_frames = len(frames)
    cols = int(np.ceil(num_frames / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < num_frames:
            ax.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
            ax.set_title(f"Frame {i}")
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def load_video_frames(video_path, num_frames=16):
    """Load a fixed number of frames from a video file"""
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found")
        return None
    
    try:
        from utils.video_standardizer import extract_fixed_frames
        
        # Extract frames with fixed count
        frames = extract_fixed_frames(
            video_path, 
            num_frames=num_frames,
            resize_dim=(224, 224)
        )
        
        if frames is None or len(frames) == 0:
            print(f"Error: Failed to extract frames from {video_path}")
            return None
        
        return frames
    except Exception as e:
        print(f"Error loading video: {str(e)}")
        return None

def test_video_augmentation(frames, output_dir=None):
    """Test video augmentation functions"""
    print("\n" + "="*50)
    print("Testing Video Augmentation")
    print("="*50)
    
    if frames is None or len(frames) == 0:
        print("Error: No frames to augment")
        return False
    
    # Create augmenter
    augmenter = VideoAugmenter(
        brightness_range=0.3,
        contrast_range=0.3,
        saturation_range=0.3,
        hue_range=0.1,
        rotation_angle=15,
        crop_percent=0.1
    )
    
    print(f"Created augmenter with parameters:")
    print(f"  Brightness range: ±{augmenter.brightness_range}")
    print(f"  Contrast range: ±{augmenter.contrast_range}")
    print(f"  Saturation range: ±{augmenter.saturation_range}")
    print(f"  Hue range: ±{augmenter.hue_range}")
    print(f"  Rotation angle: ±{augmenter.rotation_angle} degrees")
    print(f"  Crop percent: {augmenter.crop_percent * 100}%")
    
    # Test each augmentation type individually
    augmentation_types = [
        'flip', 'rotate', 'brightness', 'contrast', 'saturation', 'hue', 'crop'
    ]
    
    # Save original frames for comparison
    original_frames = frames.copy()
    
    # If output directory is provided, create it
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Test each augmentation type
    for aug_type in augmentation_types:
        print(f"Testing {aug_type} augmentation...")
        
        # Apply augmentation
        try:
            augmented_frames = augmenter.augment_video(frames, [aug_type])
            
            # Compare with original
            if len(augmented_frames) != len(original_frames):
                print(f"  Error: Augmented frames count {len(augmented_frames)} differs from original {len(original_frames)}")
                continue
            
            # Calculate difference
            diffs = []
            for i in range(len(original_frames)):
                diff = np.abs(augmented_frames[i].astype(float) - original_frames[i].astype(float)).mean()
                diffs.append(diff)
            avg_diff = np.mean(diffs)
            
            print(f"  Average pixel difference: {avg_diff:.2f}")
            
            # Check if augmentation had an effect (should be different from original)
            if avg_diff < 0.1 and aug_type != 'flip':  # Flip might not change all frames
                print(f"  Warning: {aug_type} augmentation may not be working correctly (low difference)")
            
            # Display or save comparison
            if output_dir:
                # Create a comparison of first frame before/after
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(cv2.cvtColor(original_frames[0], cv2.COLOR_BGR2RGB))
                ax[0].set_title("Original")
                ax[0].axis('off')
                
                ax[1].imshow(cv2.cvtColor(augmented_frames[0], cv2.COLOR_BGR2RGB))
                ax[1].set_title(f"Augmented ({aug_type})")
                ax[1].axis('off')
                
                plt.suptitle(f"{aug_type.capitalize()} Augmentation Comparison")
                plt.tight_layout()
                
                save_path = os.path.join(output_dir, f"aug_{aug_type}_comparison.png")
                plt.savefig(save_path)
                plt.close()
                
                # Save a grid of augmented frames
                display_frames(
                    augmented_frames, 
                    title=f"{aug_type.capitalize()} Augmented Frames", 
                    rows=2,
                    save_path=os.path.join(output_dir, f"aug_{aug_type}_frames.png")
                )
        
        except Exception as e:
            print(f"  Error testing {aug_type} augmentation: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Test random combination of augmentations
    print("\nTesting random combination of augmentations...")
    try:
        # Compare multiple random augmentations to see variety
        num_runs = 5
        all_augmented = []
        
        for i in range(num_runs):
            # Let the augmenter randomly select augmentations
            augmented = augmenter.augment_video(original_frames)
            all_augmented.append(augmented)
            
            # Calculate difference
            diff = np.abs(augmented[0].astype(float) - original_frames[0].astype(float)).mean()
            print(f"  Run {i+1}: Average pixel difference: {diff:.2f}")
        
        # Compare runs to ensure they're different
        for i in range(num_runs - 1):
            for j in range(i + 1, num_runs):
                diff = np.abs(all_augmented[i][0].astype(float) - all_augmented[j][0].astype(float)).mean()
                print(f"  Difference between runs {i+1} and {j+1}: {diff:.2f}")
                if diff < 0.1:
                    print(f"  Warning: Runs {i+1} and {j+1} look similar, randomization may not be working")
        
        if output_dir:
            # Create a grid comparison of first frame from each run
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes = axes.flatten()
            
            # Original
            axes[0].imshow(cv2.cvtColor(original_frames[0], cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            # Each augmented run
            for i in range(num_runs):
                if i+1 < len(axes):
                    axes[i+1].imshow(cv2.cvtColor(all_augmented[i][0], cv2.COLOR_BGR2RGB))
                    axes[i+1].set_title(f"Random Run {i+1}")
                    axes[i+1].axis('off')
            
            plt.suptitle("Random Augmentation Comparison")
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, "aug_random_comparison.png")
            plt.savefig(save_path)
            plt.close()
    
    except Exception as e:
        print(f"  Error testing random augmentations: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("Video augmentation tests completed")
    return True

def create_dummy_keypoints(num_frames=16, num_keypoints=33):
    """Create dummy pose keypoints for testing"""
    # Create dummy keypoints [num_frames, num_keypoints*2]
    # where each keypoint has x, y coordinates in range [0, 1]
    keypoints = np.random.rand(num_frames, num_keypoints * 2)
    return keypoints

def visualize_keypoints(keypoints, frame_width=224, frame_height=224, title="Pose Keypoints", save_path=None):
    """Visualize pose keypoints on a blank frame"""
    # Create a blank frame
    frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
    
    # Reshape keypoints to [num_keypoints, 2]
    keypoints_reshaped = keypoints.reshape(-1, 2)
    
    # Convert normalized coordinates to pixel coordinates
    points = []
    for i in range(len(keypoints_reshaped)):
        x = int(keypoints_reshaped[i, 0] * frame_width)
        y = int(keypoints_reshaped[i, 1] * frame_height)
        points.append((x, y))
    
    # Draw keypoints
    for p in points:
        cv2.circle(frame, p, 3, (0, 0, 255), -1)
    
    # Draw connections (simplified for testing)
    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i+1], (0, 255, 0), 1)
    
    # Display or save
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return frame

def test_keypoint_augmentation(output_dir=None):
    """Test keypoint augmentation functions"""
    print("\n" + "="*50)
    print("Testing Keypoint Augmentation")
    print("="*50)
    
    # Create augmenter
    augmenter = VideoAugmenter(
        brightness_range=0.3,
        contrast_range=0.3,
        saturation_range=0.3,
        hue_range=0.1,
        rotation_angle=15,
        crop_percent=0.1
    )
    
    # Create dummy keypoints
    keypoints = create_dummy_keypoints(num_frames=5, num_keypoints=33)
    print(f"Created dummy keypoints: {keypoints.shape}")
    
    # Define frame dimensions
    frame_width = 224
    frame_height = 224
    
    # Save original keypoints for comparison
    original_keypoints = keypoints.copy()
    
    # If output directory is provided, create it
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize original keypoints
        for i in range(min(2, len(original_keypoints))):
            visualize_keypoints(
                original_keypoints[i], 
                frame_width, 
                frame_height, 
                title=f"Original Keypoints (Frame {i})",
                save_path=os.path.join(output_dir, f"keypoints_original_{i}.png")
            )
    
    # Test each compatible augmentation type
    augmentation_types = ['flip', 'rotate', 'crop']
    
    for aug_type in augmentation_types:
        print(f"Testing {aug_type} augmentation on keypoints...")
        try:
            # Apply augmentation to keypoints
            augmented_keypoints = augmenter.apply_to_keypoints(
                keypoints.copy(), 
                frame_width, 
                frame_height, 
                [aug_type]
            )
            
            # Compare with original
            if augmented_keypoints.shape != original_keypoints.shape:
                print(f"  Error: Augmented keypoints shape {augmented_keypoints.shape} differs from original {original_keypoints.shape}")
                continue
            
            # Calculate difference
            diff = np.abs(augmented_keypoints - original_keypoints).mean()
            print(f"  Average coordinate difference: {diff:.4f}")
            
            # Check if augmentation had an effect
            if diff < 0.01:
                print(f"  Warning: {aug_type} augmentation may not be working correctly (low difference)")
            
            # Visualize augmented keypoints
            if output_dir:
                for i in range(min(2, len(augmented_keypoints))):
                    visualize_keypoints(
                        augmented_keypoints[i], 
                        frame_width, 
                        frame_height, 
                        title=f"{aug_type.capitalize()} Augmented Keypoints (Frame {i})",
                        save_path=os.path.join(output_dir, f"keypoints_{aug_type}_{i}.png")
                    )
                    
                    # Create comparison visualization
                    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                    
                    # Original
                    orig_frame = visualize_keypoints(original_keypoints[i], frame_width, frame_height)
                    ax[0].imshow(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB))
                    ax[0].set_title("Original Keypoints")
                    ax[0].axis('off')
                    
                    # Augmented
                    aug_frame = visualize_keypoints(augmented_keypoints[i], frame_width, frame_height)
                    ax[1].imshow(cv2.cvtColor(aug_frame, cv2.COLOR_BGR2RGB))
                    ax[1].set_title(f"{aug_type.capitalize()} Augmented Keypoints")
                    ax[1].axis('off')
                    
                    plt.suptitle(f"{aug_type.capitalize()} Keypoint Augmentation Comparison (Frame {i})")
                    plt.tight_layout()
                    
                    plt.savefig(os.path.join(output_dir, f"keypoints_{aug_type}_comparison_{i}.png"))
                    plt.close()
            
        except Exception as e:
            print(f"  Error testing {aug_type} keypoint augmentation: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Test combination of augmentations
    print("\nTesting combination of keypoint augmentations...")
    try:
        combined_types = ['flip', 'rotate', 'crop']
        augmented_keypoints = augmenter.apply_to_keypoints(
            original_keypoints.copy(),
            frame_width,
            frame_height,
            combined_types
        )
        
        # Calculate difference
        diff = np.abs(augmented_keypoints - original_keypoints).mean()
        print(f"  Average coordinate difference (combined): {diff:.4f}")
        
        # Visualize augmented keypoints
        if output_dir:
            for i in range(min(2, len(augmented_keypoints))):
                visualize_keypoints(
                    augmented_keypoints[i], 
                    frame_width, 
                    frame_height, 
                    title=f"Combined Augmented Keypoints (Frame {i})",
                    save_path=os.path.join(output_dir, f"keypoints_combined_{i}.png")
                )
                
                # Create comparison visualization
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                
                # Original
                orig_frame = visualize_keypoints(original_keypoints[i], frame_width, frame_height)
                ax[0].imshow(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB))
                ax[0].set_title("Original Keypoints")
                ax[0].axis('off')
                
                # Augmented
                aug_frame = visualize_keypoints(augmented_keypoints[i], frame_width, frame_height)
                ax[1].imshow(cv2.cvtColor(aug_frame, cv2.COLOR_BGR2RGB))
                ax[1].set_title("Combined Augmented Keypoints")
                ax[1].axis('off')
                
                plt.suptitle(f"Combined Keypoint Augmentation Comparison (Frame {i})")
                plt.tight_layout()
                
                plt.savefig(os.path.join(output_dir, f"keypoints_combined_comparison_{i}.png"))
                plt.close()
        
    except Exception as e:
        print(f"  Error testing combined keypoint augmentation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("Keypoint augmentation tests completed")
    return True

def test_dataset_augmentation(data_dir=None, output_dir=None):
    """Test augmentation within the dataset class"""
    print("\n" + "="*50)
    print("Testing Augmentation in Dataset")
    print("="*50)
    
    if not data_dir or not os.path.exists(data_dir):
        print("No valid data directory provided, skipping dataset augmentation test")
        return False
    
    # Add parent directory to path to import modules
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    try:
        from dataloader import EnhancedViolenceDataset
        from torchvision import transforms
        
        # Find the first video in the data directory
        video_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    video_paths.append(os.path.join(root, file))
                    if len(video_paths) >= 5:  # Limit to 5 videos
                        break
            if len(video_paths) >= 5:
                break
        
        if not video_paths:
            print("No video files found in the data directory")
            return False
        
        print(f"Found {len(video_paths)} videos for testing")
        
        # Create test dataset with augmentation enabled
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Create dataset with augmentation
        augmented_dataset = EnhancedViolenceDataset(
            video_paths,
            [0] * len(video_paths),  # Dummy labels
            transform=transform,
            num_frames=16,
            augment=True,
            model_type='3d_cnn',
            training=True
        )
        
        # Create dataset without augmentation
        non_augmented_dataset = EnhancedViolenceDataset(
            video_paths,
            [0] * len(video_paths),  # Dummy labels
            transform=transform,
            num_frames=16,
            augment=False,
            model_type='3d_cnn',
            training=True
        )
        
        print("Loading samples from both datasets...")
        
        # Load the same sample from both datasets multiple times
        num_tries = 3
        all_samples = []
        
        for i in range(num_tries):
            # Get a sample from each dataset
            aug_sample, aug_label = augmented_dataset[0]
            non_aug_sample, non_aug_label = non_augmented_dataset[0]
            
            # Convert to numpy for visualization
            aug_np = aug_sample.permute(0, 2, 3, 1).cpu().numpy()
            non_aug_np = non_aug_sample.permute(0, 2, 3, 1).cpu().numpy()
            
            # Normalize to [0, 255] range for visualization
            aug_np = (aug_np * 255).astype(np.uint8)
            non_aug_np = (non_aug_np * 255).astype(np.uint8)
            
            all_samples.append((aug_np, non_aug_np))
            
            print(f"  Try {i+1}: Loaded samples, augmented shape {aug_sample.shape}, non-augmented shape {non_aug_sample.shape}")
        
        # Compare augmented samples to check for different augmentations
        has_variations = False
        for i in range(num_tries - 1):
            diff = np.abs(all_samples[i][0].astype(float) - all_samples[i+1][0].astype(float)).mean()
            print(f"  Difference between augmented samples {i+1} and {i+2}: {diff:.2f}")
            if diff > 5.0:  # Small threshold to detect variations
                has_variations = True
        
        if not has_variations:
            print("Warning: Augmented samples don't appear to vary between calls, augmentation may not be working")
        
        # Compare augmented vs non-augmented
        for i in range(num_tries):
            diff = np.abs(all_samples[i][0].astype(float) - all_samples[i][1].astype(float)).mean()
            print(f"  Difference between augmented and non-augmented (try {i+1}): {diff:.2f}")
            if diff < 5.0:
                print("Warning: Augmented and non-augmented samples look similar, augmentation may not be working")
        
        # Visualize samples
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            for i in range(num_tries):
                # Create a comparison of the first frame
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                
                ax[0].imshow(all_samples[i][0][0])
                ax[0].set_title(f"Augmented (try {i+1})")
                ax[0].axis('off')
                
                ax[1].imshow(all_samples[i][1][0])
                ax[1].set_title("Non-Augmented")
                ax[1].axis('off')
                
                plt.suptitle("Dataset Augmentation Comparison (First Frame)")
                plt.tight_layout()
                
                plt.savefig(os.path.join(output_dir, f"dataset_augmentation_comparison_{i+1}.png"))
                plt.close()
                
                # Save grid of augmented frames
                display_frames(
                    all_samples[i][0], 
                    title=f"Augmented Frames (Try {i+1})", 
                    rows=2,
                    save_path=os.path.join(output_dir, f"dataset_augmented_frames_{i+1}.png")
                )
        
        print("Dataset augmentation tests completed")
        return True
        
    except Exception as e:
        print(f"Error testing dataset augmentation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test data augmentation functions")
    parser.add_argument("--video_path", type=str, default=None,
                      help="Path to a video file for testing")
    parser.add_argument("--data_dir", type=str, default=None,
                      help="Path to data directory for dataset augmentation tests")
    parser.add_argument("--output_dir", type=str, default="./Testing/augmentation_output",
                      help="Directory to save visualizations")
    parser.add_argument("--skip_video", action="store_true",
                      help="Skip video augmentation tests")
    parser.add_argument("--skip_keypoints", action="store_true",
                      help="Skip keypoint augmentation tests")
    parser.add_argument("--skip_dataset", action="store_true",
                      help="Skip dataset augmentation tests")
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Test video augmentation
    video_aug_ok = True
    if not args.skip_video:
        if args.video_path and os.path.exists(args.video_path):
            # Load video frames
            frames = load_video_frames(args.video_path)
            
            video_aug_ok = test_video_augmentation(frames, args.output_dir)
        else:
            print("No valid video path provided, creating dummy frames for video augmentation test")
            # Create dummy frames for testing
            dummy_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
            video_aug_ok = test_video_augmentation(dummy_frames, args.output_dir)
    
    # Test keypoint augmentation
    keypoint_aug_ok = True
    if not args.skip_keypoints:
        keypoint_aug_ok = test_keypoint_augmentation(args.output_dir)
    
    # Test dataset augmentation
    dataset_aug_ok = True
    if not args.skip_dataset:
        dataset_aug_ok = test_dataset_augmentation(args.data_dir, args.output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("Augmentation Tests Summary")
    print("="*50)
    
    if not args.skip_video:
        print(f"Video Augmentation: {'✓ PASSED' if video_aug_ok else '✗ FAILED'}")
    
    if not args.skip_keypoints:
        print(f"Keypoint Augmentation: {'✓ PASSED' if keypoint_aug_ok else '✗ FAILED'}")
    
    if not args.skip_dataset:
        print(f"Dataset Augmentation: {'✓ PASSED' if dataset_aug_ok else '✗ FAILED'}")
    
    # Return success only if all tests pass
    if video_aug_ok and keypoint_aug_ok and dataset_aug_ok:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())