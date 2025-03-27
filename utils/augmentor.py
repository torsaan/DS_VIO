# utils/video_augmentation.py
import cv2
import numpy as np
import random

class VideoAugmenter:
    def __init__(self, brightness_range=0.3, contrast_range=0.3, 
                 saturation_range=0.3, hue_range=0.1, 
                 rotation_angle=15, crop_percent=0.1):
        """
        Initialize the video augmenter with configurable parameters.
        
        Args:
            brightness_range: Range for random brightness adjustment
            contrast_range: Range for random contrast adjustment
            saturation_range: Range for random saturation adjustment
            hue_range: Range for random hue adjustment
            rotation_angle: Maximum rotation angle in degrees
            crop_percent: Percentage of frame to crop (0.1 = 10%)
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.rotation_angle = rotation_angle
        self.crop_percent = crop_percent
    
    def augment_video(self, frames, augment_types=None):
        """
        Apply a series of augmentations to video frames.
        
        Args:
            frames: List of numpy arrays (RGB frames)
            augment_types: List of augmentation types to apply.
                           If None, randomly selects a combination.
        
        Returns:
            List of augmented frames
        """
        if augment_types is None:
            # Randomly choose a subset of augmentations
            available_augmentations = [
                'flip', 'rotate', 'brightness', 'contrast', 
                'saturation', 'hue', 'crop'
            ]
            num_augs = random.randint(1, 3)  # Apply 1-3 augmentations
            augment_types = random.sample(available_augmentations, num_augs)
        
        augmented_frames = frames.copy()
        
        # We'll apply the same transformations to all frames in the sequence
        # to maintain temporal consistency
        
        # Apply horizontal flip
        if 'flip' in augment_types and random.random() < 0.5:
            augmented_frames = [cv2.flip(frame, 1) for frame in augmented_frames]
        
        # Apply rotation
        if 'rotate' in augment_types:
            angle = random.uniform(-self.rotation_angle, self.rotation_angle)
            h, w = augmented_frames[0].shape[:2]
            center = (w // 2, h // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented_frames = [cv2.warpAffine(frame, rot_matrix, (w, h), 
                               borderMode=cv2.BORDER_REPLICATE) for frame in augmented_frames]
        
        # Apply crop and resize
        if 'crop' in augment_types:
            h, w = augmented_frames[0].shape[:2]
            crop_h = int(h * (1 - self.crop_percent))
            crop_w = int(w * (1 - self.crop_percent))
            
            # Random crop offsets
            y_offset = random.randint(0, h - crop_h) if h > crop_h else 0
            x_offset = random.randint(0, w - crop_w) if w > crop_w else 0
            
            cropped_frames = []
            for frame in augmented_frames:
                crop = frame[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w]
                resized = cv2.resize(crop, (w, h))
                cropped_frames.append(resized)
            augmented_frames = cropped_frames
        
        # Convert to HSV for color adjustments
        if any(x in augment_types for x in ['brightness', 'contrast', 'saturation', 'hue']):
            hsv_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32) for frame in augmented_frames]
            
            # Brightness adjustment (V channel)
            if 'brightness' in augment_types:
                brightness_factor = random.uniform(1-self.brightness_range, 1+self.brightness_range)
                for hsv in hsv_frames:
                    hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
                    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            
            # Saturation adjustment (S channel)
            if 'saturation' in augment_types:
                saturation_factor = random.uniform(1-self.saturation_range, 1+self.saturation_range)
                for hsv in hsv_frames:
                    hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            
            # Hue adjustment (H channel)
            if 'hue' in augment_types:
                hue_shift = random.uniform(-self.hue_range * 180, self.hue_range * 180)
                for hsv in hsv_frames:
                    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            
            # Convert back to RGB
            augmented_frames = [cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB) for hsv in hsv_frames]
        
        # Contrast adjustment (in RGB space)
        if 'contrast' in augment_types:
            contrast_factor = random.uniform(1-self.contrast_range, 1+self.contrast_range)
            for i in range(len(augmented_frames)):
                mean = np.mean(augmented_frames[i], axis=(0, 1), keepdims=True)
                augmented_frames[i] = np.clip((augmented_frames[i].astype(np.float32) - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        
        return augmented_frames