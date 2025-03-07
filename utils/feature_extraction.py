# utils/feature_extraction.py
import os
import cv2
import numpy as np
import mediapipe as mp
import csv
from tqdm import tqdm
from pathlib import Path
import pickle

class FeatureExtractor:
    """Feature extractor for violence detection"""
    
    def __init__(self, output_dir="./features", sample_rate=None, 
                frame_width=224, frame_height=224):
        """
        Initialize feature extractor
        
        Args:
            output_dir: Directory to save extracted features
            sample_rate: Number of frames to sample per video (defaults to NUM_FRAMES from dataprep)
            frame_width: Width for resizing frames
            frame_height: Height for resizing frames
        """
        from utils.dataprep import NUM_FRAMES
        
        self.output_dir = output_dir
        self.sample_rate = sample_rate if sample_rate is not None else NUM_FRAMES
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "poses"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "motion"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "histograms"), exist_ok=True)
        
        # Initialize MediaPipe pose detector
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
    
    def extract_frames(self, video_path):
        """Extract fixed number of frames from video"""
        video_name = Path(video_path).stem
        
        # Use the updated extract_fixed_frames function from video_standardizer.py
        from utils.video_standardizer import extract_fixed_frames
        
        # Use the same fixed number of frames as standardization
        frames = extract_fixed_frames(
            video_path, 
            num_frames=self.sample_rate, 
            resize_dim=(self.frame_width, self.frame_height)
        )
        
        if frames is None:
            print(f"Error extracting frames from {video_path}")
            # Return empty frames with correct dimensions
            frames = [np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8) 
                    for _ in range(self.sample_rate)]
        
        # Ensure we have exactly the requested number of frames
        while len(frames) < self.sample_rate:
            # Duplicate the last frame if needed
            frames.append(frames[-1].copy() if frames else 
                        np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8))
        
        return frames[:self.sample_rate]  # Ensure we return exactly sample_rate frames
    
    def compute_optical_flow(self, frames):
        """Compute optical flow between consecutive frames"""
        if len(frames) < 2:
            return []
        
        flow_features = []
        prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        for i in range(1, len(frames)):
            curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, curr_frame, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Create features from optical flow
            mean_magnitude = np.mean(magnitude)
            std_magnitude = np.std(magnitude)
            max_magnitude = np.max(magnitude)
            
            # Calculate motion direction histograms
            angle_bins = np.linspace(0, 2*np.pi, 8+1)
            angle_hist, _ = np.histogram(angle, bins=angle_bins, weights=magnitude)
            angle_hist = angle_hist / (np.sum(angle_hist) + 1e-10)  # Normalize
            
            # Combine features
            frame_flow_features = [mean_magnitude, std_magnitude, max_magnitude] + angle_hist.tolist()
            flow_features.append(frame_flow_features)
            
            # Update previous frame
            prev_frame = curr_frame
        
        # Add zero features for the first frame
        flow_features.insert(0, [0] * len(flow_features[0]))
        
        return flow_features
    
    def create_motion_history_image(self, frames, tau=0.75):
        """Create motion history image from frames"""
        if len(frames) < 2:
            return np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        
        # Initialize MHI
        mhi = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        
        # Convert frames to grayscale
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        
        # Process frames
        for i in range(1, len(gray_frames)):
            # Calculate absolute difference
            frame_diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
            
            # Threshold difference
            _, motion_mask = cv2.threshold(frame_diff, 25, 1, cv2.THRESH_BINARY)
            
            # Update MHI
            mhi = mhi * tau
            mhi[motion_mask == 1] = 1.0
        
        return mhi
    
    def extract_pose_keypoints(self, frames):
        """Extract pose keypoints from frames"""
        keypoints_list = []
        
        for frame in frames:
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(frame_rgb)
            
            # Extract keypoints
            keypoints = []
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y])
            else:
                # Fill with zeros if no pose detected
                keypoints = [0.0] * 66  # 33 landmarks × 2 coordinates
            
            keypoints_list.append(keypoints)
        
        return keypoints_list
    
    def compute_histograms(self, frames, bins=32):
        """Compute color histograms from frames"""
        histograms = []
        
        for frame in frames:
            # Split into channels
            channels = cv2.split(frame)
            hist_features = []
            
            # Compute histogram for each channel
            for channel in channels:
                hist = cv2.calcHist([channel], [0], None, [bins], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                hist_features.extend(hist)
            
            histograms.append(hist_features)
        
        return histograms
    
    def extract_features(self, video_path, label=None, save=True):
        """
        Extract all features from a video
        
        Args:
            video_path: Path to video file
            label: Class label (0 for non-violence, 1 for violence, None if unknown)
            save: Whether to save extracted features to disk
            
        Returns:
            Dictionary containing extracted features
        """
        video_name = Path(video_path).stem
        
        # Extract frames
        frames = self.extract_frames(video_path)
        if frames is None:
            print(f"Error extracting frames from {video_path}")
            return None
        
        # Extract different feature types
        optical_flow = self.compute_optical_flow(frames)
        mhi = self.create_motion_history_image(frames)
        pose_keypoints = self.extract_pose_keypoints(frames)
        histograms = self.compute_histograms(frames)
        
        # Compute MHI features
        mhi_features = []
        mhi_mean = np.mean(mhi)
        mhi_std = np.std(mhi)
        mhi_max = np.max(mhi)
        
        # Simple regional MHI features
        h, w = mhi.shape
        regions = [
            mhi[:h//2, :w//2],      # top-left
            mhi[:h//2, w//2:],      # top-right
            mhi[h//2:, :w//2],      # bottom-left
            mhi[h//2:, w//2:]       # bottom-right
        ]
        
        region_means = [np.mean(region) for region in regions]
        mhi_features = [mhi_mean, mhi_std, mhi_max] + region_means
        
        # Combine features
        features = {
            "video_name": video_name,
            "label": label,
            "frames": frames,
            "optical_flow": optical_flow,
            "mhi": mhi,
            "mhi_features": mhi_features,
            "pose_keypoints": pose_keypoints,
            "histograms": histograms
        }
        
        # Save features if requested
        if save:
            self._save_features(features, video_name)
        
        return features
    
    def _save_features(self, features, video_name):
        """Save extracted features to disk"""
        # Determine class folder (Violence or NonViolence)
        if video_name.startswith('V_'):
            class_dir = 'Violence'
        elif video_name.startswith('NV_'):
            class_dir = 'NonViolence'
        else:
            class_dir = 'Unknown'
        
        # Define output paths
        frames_dir = os.path.join(self.output_dir, "frames", class_dir)
        poses_dir = os.path.join(self.output_dir, "poses", class_dir)
        motion_dir = os.path.join(self.output_dir, "motion", class_dir)
        hist_dir = os.path.join(self.output_dir, "histograms", class_dir)
        
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(poses_dir, exist_ok=True)
        os.makedirs(motion_dir, exist_ok=True)
        os.makedirs(hist_dir, exist_ok=True)
        
        # Save extracted frames
        for i, frame in enumerate(features["frames"]):
            frame_path = os.path.join(frames_dir, f"{video_name}_frame{i}.jpg")
            cv2.imwrite(frame_path, frame)
        
        # Save pose keypoints
        pose_path = os.path.join(poses_dir, f"{video_name}.csv")
        with open(pose_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_idx'] + [f'kp{i}_{coord}' for i in range(33) for coord in ['x', 'y']])
            for i, keypoints in enumerate(features["pose_keypoints"]):
                writer.writerow([i] + keypoints)
        
        # Save motion history image
        mhi_path = os.path.join(motion_dir, f"{video_name}_mhi.jpg")
        cv2.imwrite(mhi_path, (features["mhi"] * 255).astype(np.uint8))
        
        # Save optical flow features
        flow_path = os.path.join(motion_dir, f"{video_name}_flow.csv")
        with open(flow_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_idx', 'mean_mag', 'std_mag', 'max_mag'] + 
                            [f'dir_hist{i}' for i in range(8)])
            for i, flow in enumerate(features["optical_flow"]):
                writer.writerow([i] + flow)
        
        # Save histogram features
        hist_path = os.path.join(hist_dir, f"{video_name}_hist.pkl")
        with open(hist_path, 'wb') as f:
            pickle.dump(features["histograms"], f)
        
        # Save all combined features for convenience
        combined_dir = os.path.join(self.output_dir, "combined", class_dir)
        os.makedirs(combined_dir, exist_ok=True)
        combined_path = os.path.join(combined_dir, f"{video_name}.pkl")
        
        with open(combined_path, 'wb') as f:
            # Save a smaller version without the raw frames
            features_small = features.copy()
            features_small.pop("frames")
            pickle.dump(features_small, f)
    
    def batch_extract_features(self, video_dir, label_map=None):
        """
        Extract features from all videos in a directory
        
        Args:
            video_dir: Directory containing videos
            label_map: Dictionary mapping filenames to labels (optional)
            
        Returns:
            List of extracted feature dictionaries
        """
        # Find all video files
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(list(Path(video_dir).glob(f'**/*{ext}')))
        
        print(f"Found {len(video_files)} videos in {video_dir}")
        
        # Process videos
        all_features = []
        for video_file in tqdm(video_files, desc="Extracting features"):
            video_name = video_file.stem
            
            # Determine label
            label = None
            if label_map is not None and video_name in label_map:
                label = label_map[video_name]
            elif video_name.startswith('V_'):
                label = 1  # Violence
            elif video_name.startswith('NV_'):
                label = 0  # Non-violence
            
            # Extract features
            try:
                features = self.extract_features(str(video_file), label=label)
                if features:
                    all_features.append(features)
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")
        
        return all_features