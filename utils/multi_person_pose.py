#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
import csv
import argparse
from tqdm import tqdm
import glob
import mediapipe as mp

class PersonTracker:
    def __init__(self, target_fps=15, max_persons=10, visualization_dir=None):
        """
        Initialize the person tracker with advanced detection and tracking capabilities.
        
        Args:
            target_fps (int): Target frames per second to process
            max_persons (int): Maximum number of persons to track
            visualization_dir (str): Directory to save visualization outputs
        """
        # YOLO for person detection
        from ultralytics import YOLO
        
        # Use the recommended YOLOv5lu model
        model_path = 'C:\\DS_VIO\\yolov5lu.pt'
        
        self.yolo_model = YOLO(model_path)
        self.yolo_model.conf = 0.25  # Confidence threshold
        self.yolo_model.iou = 0.45   # NMS IoU threshold
        self.yolo_model.classes = [0]  # Only detect persons (COCO class 0)
        self.yolo_model.max_det = max_persons  # Maximum detections per image
        
        # MediaPipe for pose estimation
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1  # 0 is fastest, 2 is most accurate
        )
        
        self.target_fps = target_fps
        self.max_persons = max_persons
        self.visualization_dir = visualization_dir
        
        # Tracking parameters
        self.tracks = []
        self.next_track_id = 0
        self.iou_threshold = 0.3

    def resize_image(self, image, target_size=(640, 640)):
        """
        Resize image maintaining aspect ratio with padding
        
        Args:
            image: Input image
            target_size: Target size for YOLO model
        
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        new_w, new_h = target_size
        
        # Calculate scaling factor
        scale = min(new_w / w, new_h / h)
        
        # Calculate new dimensions
        resized_w = int(w * scale)
        resized_h = int(h * scale)
        
        # Resize image
        resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded_image = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
        
        # Calculate coordinates to paste resized image
        start_x = (new_w - resized_w) // 2
        start_y = (new_h - resized_h) // 2
        
        padded_image[start_y:start_y+resized_h, start_x:start_x+resized_w] = resized_image
        
        return padded_image

    def detect_persons_yolo(self, image):
        """
        Detect persons in an image using YOLO.
        
        Args:
            image: BGR image
        
        Returns:
            List of bounding boxes [x1, y1, x2, y2, confidence]
        """
        # Resize image to match YOLO model input
        resized_image = self.resize_image(image)
        
        # YOLO expects RGB
        img_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = self.yolo_model(img_rgb)
        predictions = results[0].boxes.xyxy.cpu().numpy()
        
        # Adjust bounding boxes back to original image scale
        h, w = image.shape[:2]
        new_h, new_w = resized_image.shape[:2]
        scale_x = w / new_w
        scale_y = h / new_h
        
        persons = []
        for pred in predictions:
            # Check the number of elements in each prediction
            if pred.size == 6:
                x1, y1, x2, y2, conf, cls = pred
            elif pred.size == 4:
                x1, y1, x2, y2 = pred
                conf = 1.0  # default confidence
                cls = 0     # assume person
            else:
                continue
            
            if int(cls) == 0:  # Only persons
                # Scale back to original image coordinates
                scaled_x1 = max(0, x1 * scale_x)
                scaled_y1 = max(0, y1 * scale_y)
                scaled_x2 = min(w, x2 * scale_x)
                scaled_y2 = min(h, y2 * scale_y)
                
                persons.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2, conf])
        
        return persons

    # Rest of the methods remain the same as in the previous implementation
#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
import csv
import argparse
from tqdm import tqdm
import glob
import mediapipe as mp

class PersonTracker:
    def __init__(self, target_fps=15, max_persons=10, visualization_dir=None):
        """
        Initialize the person tracker with advanced detection and tracking capabilities.
        
        Args:
            target_fps (int): Target frames per second to process
            max_persons (int): Maximum number of persons to track
            visualization_dir (str): Directory to save visualization outputs
        """
        # YOLO for person detection
        from ultralytics import YOLO
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)
        model_path = os.path.join(project_dir, 'yolov5l.pt')
        
        self.yolo_model = YOLO(model_path)
        self.yolo_model.conf = 0.25  # Confidence threshold
        self.yolo_model.iou = 0.45   # NMS IoU threshold
        self.yolo_model.classes = [0]  # Only detect persons (COCO class 0)
        self.yolo_model.max_det = max_persons  # Maximum detections per image
        
        # MediaPipe for pose estimation
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.target_fps = target_fps
        self.max_persons = max_persons
        self.visualization_dir = visualization_dir
        
        # Tracking parameters
        self.tracks = []
        self.next_track_id = 0
        self.iou_threshold = 0.3
        
    def compute_iou(self, box1, box2):
        """
        Compute Intersection over Union (IoU) between two boxes.
        Box format: [x1, y1, x2, y2]
        """
        x1_int = max(box1[0], box2[0])
        y1_int = max(box1[1], box2[1])
        x2_int = min(box1[2], box2[2])
        y2_int = min(box1[3], box2[3])
        
        inter_area = max(0, x2_int - x1_int) * max(0, y2_int - y1_int)
        box1_area = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
        box2_area = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def track_detections(self, detections, frame_idx):
        """
        Assign track IDs to detections using IoU matching.
        
        Args:
            detections: List of detections [x1, y1, x2, y2, conf]
            frame_idx: Current frame index
        
        Returns:
            List of tracked detections with track IDs
        """
        tracked_detections = []
        
        for det in detections:
            best_iou = 0
            best_track = None
            
            # Try to match with existing tracks
            for track in self.tracks:
                iou = self.compute_iou(det[:4], track['bbox'])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track = track
            
            if best_track is not None:
                # Match found: assign existing track id and update track
                det_dict = {
                    'bbox': det[:4], 
                    'conf': det[4], 
                    'track_id': best_track['id']
                }
                tracked_detections.append(det_dict)
                best_track['bbox'] = det[:4]
                best_track['last_seen'] = frame_idx
            else:
                # No matching track: create a new track
                det_dict = {
                    'bbox': det[:4], 
                    'conf': det[4], 
                    'track_id': self.next_track_id
                }
                tracked_detections.append(det_dict)
                self.tracks.append({
                    'id': self.next_track_id, 
                    'bbox': det[:4], 
                    'last_seen': frame_idx
                })
                self.next_track_id += 1
        
        # Remove tracks that haven't been updated in the last 10 frames
        self.tracks = [t for t in self.tracks if frame_idx - t['last_seen'] <= 10]
        
        return tracked_detections

    def detect_persons_yolo(self, image):
        """
        Detect persons in an image using YOLO.
        
        Args:
            image: BGR image
        
        Returns:
            List of bounding boxes [x1, y1, x2, y2, confidence]
        """
        # YOLO expects RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.yolo_model(img_rgb)
        predictions = results[0].boxes.xyxy.cpu().numpy()
        
        persons = []
        for pred in predictions:
            # Check the number of elements in each prediction
            if pred.size == 6:
                x1, y1, x2, y2, conf, cls = pred
            elif pred.size == 4:
                x1, y1, x2, y2 = pred
                conf = 1.0  # default confidence
                cls = 0     # assume person
            else:
                continue
            
            if int(cls) == 0:  # Only persons
                persons.append([x1, y1, x2, y2, conf])
        return persons

    def estimate_pose(self, image, tracked_detections):
        """
        Estimate pose for each tracked person.
        
        Args:
            image: Input image
            tracked_detections: List of tracked person detections
        
        Returns:
            List of pose landmarks for each tracked person
        """
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store pose results
        pose_results = []
        
        for detection in tracked_detections:
            # Crop the bounding box
            x1, y1, x2, y2 = map(int, detection['bbox'])
            person_crop = image_rgb[y1:y2, x1:x2]
            
            # Process the cropped image
            results = self.pose.process(person_crop)
            
            if results.pose_landmarks:
                # Convert landmarks to image coordinates
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    # Scale landmarks back to original image coordinates
                    lm_x = int(landmark.x * person_crop.shape[1] + x1)
                    lm_y = int(landmark.y * person_crop.shape[0] + y1)
                    landmarks.append({
                        'x': lm_x,
                        'y': lm_y,
                        'visibility': landmark.visibility
                    })
                
                pose_results.append({
                    'track_id': detection['track_id'],
                    'landmarks': landmarks
                })
        
        return pose_results

    def analyze_pose_for_violence(self, pose_results):
        """
        Analyze pose landmarks to detect potential violent actions.
        
        Args:
            pose_results: List of pose results for tracked persons
        
        Returns:
            List of potential violence indicators
        """
        violence_indicators = []
        
        for person_pose in pose_results:
            landmarks = person_pose['landmarks']
            
            # Example violence detection heuristics 
            # These are simplistic and should be refined with domain expertise
            
            # Check for aggressive arm positioning
            if landmarks:
                left_wrist = landmarks[self.mp_pose.PoseLandmarks.LEFT_WRIST]
                right_wrist = landmarks[self.mp_pose.PoseLandmarks.RIGHT_WRIST]
                left_elbow = landmarks[self.mp_pose.PoseLandmarks.LEFT_ELBOW]
                right_elbow = landmarks[self.mp_pose.PoseLandmarks.RIGHT_ELBOW]
                
                # Example: High arm elevation might indicate aggressive posture
                if (left_wrist['y'] < left_elbow['y'] or 
                    right_wrist['y'] < right_elbow['y']):
                    violence_indicators.append({
                        'track_id': person_pose['track_id'],
                        'type': 'high_arm_position',
                        'confidence': 0.6
                    })
        
        return violence_indicators

    def process_video(self, video_path, output_dir):
        """
        Process a single video for multi-person tracking and pose estimation.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save output files
        
        Returns:
            Path to the generated CSV file
        """
        # Extract video name
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Determine output subdirectory
        if video_name.startswith('V_'):
            output_subdir = os.path.join(output_dir, 'Violence')
        elif video_name.startswith('NV_'):
            output_subdir = os.path.join(output_dir, 'NonViolence')
        else:
            output_subdir = output_dir
        os.makedirs(output_subdir, exist_ok=True)
        
        # Visualization directory
        if self.visualization_dir:
            vis_dir = os.path.join(self.visualization_dir, video_name)
            os.makedirs(vis_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return []
        
        # Get video properties
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if not orig_fps or orig_fps <= 0:
            orig_fps = 30  # Default FPS
        frame_interval = max(1, int(round(orig_fps / self.target_fps)))
        
        # Create CSV file for tracking data
        csv_path = os.path.join(output_subdir, f"{video_name}_tracked.csv")
        csv_file = open(csv_path, 'w', newline='')
        fieldnames = [
            'frame_idx', 'track_id', 'x1', 'y1', 'x2', 'y2', 'conf', 
            'violence_indicator', 'violence_confidence'
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        # Reset tracking for this video
        self.tracks = []
        self.next_track_id = 0
        
        frame_idx = 0
        processed_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    # Detect persons
                    detections = self.detect_persons_yolo(frame)
                    
                    # Track detections
                    tracked_detections = self.track_detections(detections, frame_idx)
                    
                    # Estimate poses
                    pose_results = self.estimate_pose(frame, tracked_detections)
                    
                    # Analyze poses for violence indicators
                    violence_indicators = self.analyze_pose_for_violence(pose_results)
                    
                    # Prepare visualization if enabled
                    if self.visualization_dir:
                        vis_frame = frame.copy()
                        for detection in tracked_detections:
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, detection['bbox'])
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(vis_frame, f"ID: {detection['track_id']}", 
                                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                        (0, 255, 0), 2)
                        
                        # Save visualization frame
                        vis_path = os.path.join(vis_dir, f"frame_{processed_idx:04d}.jpg")
                        cv2.imwrite(vis_path, vis_frame)
                    
                    # Write tracking and violence data to CSV
                    for detection in tracked_detections:
                        # Check if this detection has a violence indicator
                        violence_info = next(
                            (ind for ind in violence_indicators 
                             if ind['track_id'] == detection['track_id']), 
                            None
                        )
                        
                        row = {
                            'frame_idx': processed_idx,
                            'track_id': detection['track_id'],
                            'x1': detection['bbox'][0],
                            'y1': detection['bbox'][1],
                            'x2': detection['bbox'][2],
                            'y2': detection['bbox'][3],
                            'conf': detection['conf'],
                            'violence_indicator': violence_info['type'] if violence_info else None,
                            'violence_confidence': violence_info['confidence'] if violence_info else None
                        }
                        csv_writer.writerow(row)
                    
                    processed_idx += 1
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        csv_file.close()
        
        print(f"Saved tracked detections for {video_name} - {processed_idx} frames processed")
        return [csv_path]

def main():
    parser = argparse.ArgumentParser(description='Advanced Person Tracking for Violence Detection')
    parser.add_argument('--video_dir', type=str, default='./videos',
                        help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save tracking data')
    parser.add_argument('--visualization_dir', type=str, default='./visualizations',
                        help='Directory to save visualization frames')
    parser.add_argument('--fps', type=int, default=15,
                        help='Target FPS for processing')
    parser.add_argument('--max_persons', type=int, default=10,
                        help='Maximum number of persons to track per video')
    parser.add_argument('--single_video', type=str, default=None,
                        help='Process a single video instead of batch processing')
    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.visualization_dir, exist_ok=True)
    
    # Initialize tracker
    tracker = PersonTracker(
        target_fps=args.fps, 
        max_persons=args.max_persons, 
        visualization_dir=args.visualization_dir
    )
    
    # Process single video or batch of videos
    if args.single_video:
        if not os.path.exists(args.single_video):
            print(f"Error: Video file {args.single_video} not found")
            return
        
        csv_files = tracker.process_video(args.single_video, args.output_dir)
        print(f"Generated {len(csv_files)} CSV file(s)")
    else:
        if not os.path.exists(args.video_dir):
            print(f"Error: Video directory {args.video_dir} not found")
            return
        
        # Find all video files
        video_paths = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_paths.extend(
                glob.glob(os.path.join(args.video_dir, '**', f'*{ext}'), recursive=True)
            )
        
        print(f"Found {len(video_paths)} videos to process")
        all_csv_files = []
        
        # Process each video
        for video_path in video_paths:
            try:
                csv_files = tracker.process_video(video_path, args.output_dir)
                all_csv_files.extend(csv_files)
                print(f"Generated {len(csv_files)} file(s) for {os.path.basename(video_path)}")
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
        
        print(f"Generated a total of {len(all_csv_files)} CSV file(s)")

if __name__ == "__main__":
    main()