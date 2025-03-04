# test_mediapipe.py
import mediapipe as mp
import cv2

def test_mediapipe():
    print("Testing MediaPipe installation...")
    
    # Initialize pose detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Create a simple black image
    img = cv2.imread('screen.png') if cv2.imread('screen.png') is not None else cv2.imread('screen.png')
    
    # If no image found, create a black image
    if img is None:
        print("No test image found, creating a blank image")
        img = cv2.cvtColor(cv2.imread('screen.png') if cv2.imread('screen.png') is not None else cv2.imread('screen.png'), cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = pose.process(img)
    
    print("MediaPipe test completed without errors")
    pose.close()

if __name__ == "__main__":
    test_mediapipe()