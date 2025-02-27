# Utils/augmentation.py
import numpy as np

def augment_keypoints_jitter(keypoints, jitter_amount=0.01):
    """
    Add small random noise to keypoints.
    keypoints: numpy array (num_frames, num_features)
    """
    noise = np.random.normal(0, jitter_amount, keypoints.shape)
    return keypoints + noise

def augment_keypoints_scaling(keypoints, scale_factor=0.1):
    """
    Slightly scale keypoints.
    """
    return keypoints * (1 + np.random.uniform(-scale_factor, scale_factor))
