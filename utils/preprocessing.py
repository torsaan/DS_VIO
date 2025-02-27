# Utils/preprocessing.py
import numpy as np

def normalize_keypoints(keypoints, frame_width, frame_height):
    """
    Normalize keypoint coordinates to the [0, 1] range.
    keypoints: numpy array (num_frames, num_features)
    Assumes keypoints format: [x1, y1, x2, y2, ...]
    """
    normalized = keypoints.copy()
    for i in range(0, normalized.shape[1], 2):
        normalized[:, i] /= frame_width
        normalized[:, i+1] /= frame_height
    return normalized

def detect_outliers(keypoints, threshold=0.1):
    """
    Detect outliers in keypoints.
    Returns a mask of frames where keypoints may be missing or extreme.
    """
    return (keypoints == 0).any(axis=1)

def sample_frames(data, num_frames):
    """
    Uniformly sample num_frames from the data (list or array).
    """
    if len(data) >= num_frames:
        indices = np.linspace(0, len(data)-1, num_frames, dtype=int)
        return [data[i] for i in indices]
    else:
        while len(data) < num_frames:
            data.append(data[-1])
        return data
