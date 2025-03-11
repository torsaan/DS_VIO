#optical

import cv2
import numpy as np
import glob
import os
import argparse

def compute_optical_flow(prev_frame, next_frame):
    """
    Compute optical flow between two consecutive frames using the Farneback method.
    
    Args:
        prev_frame (numpy.ndarray): Previous grayscale frame.
        next_frame (numpy.ndarray): Next grayscale frame.
        
    Returns:
        flow (numpy.ndarray): Optical flow vector field.
        flow_vis (numpy.ndarray): Colored visualization of the optical flow.
    """
    # Convert frames to grayscale if not already
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert flow to HSV for visualization
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255  # Set saturation to max

    # Compute magnitude and angle of flow
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Use angle to set hue (color) and magnitude for brightness
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue based on flow direction
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value based on magnitude

    # Convert HSV to RGB for visualization
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow, flow_vis

def process_video(video_path, output_folder):
    """Process a single video and display optical flow."""
    print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)  # Load video
    ret, prev_frame = cap.read()  # Read the first frame

    if not ret:
        print(f"Error: Couldn't read first frame of {video_path}")
        cap.release()
        return

    if "Violence" in video_path:
        output_folder = os.path.join(output_folder, "Violence")  # Save in Violence folder
    else:
        output_folder = os.path.join(output_folder, "NonViolence")  # Save in NonViolence folder

    video_name = os.path.basename(video_path).split(".")[0]
    output_path = os.path.join(output_folder, f"{video_name}_flow.mp4")

    os.makedirs(output_folder, exist_ok=True)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format,,,,,,,,,,,,

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 60:
        fps = 30
    print(f"Using FPS: {fps}") 

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


    while cap.isOpened():
        ret, next_frame = cap.read()
        if not ret:
            print("End of video or frame read error.")
            break  # End of video

        # Compute optical flow
        flow, flow_vis = compute_optical_flow(prev_frame, next_frame)

        # Show results
        #cv2.imshow("Original Video", next_frame)
        out.write(flow_vis)  # Save frame to video
        
        # Update previous frame
        prev_frame = next_frame

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    print(f"Saved optical flow video: {output_path}")

# Argument parser setup
parser = argparse.ArgumentParser(description="Optical Flow Video Processing")
parser.add_argument("mode", type=str, choices=["vid"], help="Set to 'vid' to process videos.")
parser.add_argument("num_videos", type=int, help="Number of videos to process (1 or 2000)")

args = parser.parse_args()

# Path to videos
#violenceFolder = "./Data/Processed/standardized/Violence"
#nonviolenceFolder = "./Data/Processed/standardized/NonViolence"

violenceFolder = "./Data/VioNonVio/Violence"
nonviolenceFolder = "./Data/VioNonVio/NonViolence"

outputFolder = "./Data/OpticalOutput"

# Get video list
video_files = glob.glob(os.path.join(violenceFolder, "*.mp4")) + glob.glob(os.path.join(nonviolenceFolder, "*.mp4"))

if not video_files: 
    print("No videos found!")
else:
    num_to_process = min(args.num_videos, len(video_files))  # Limit to available videos

    print(f"Processing {num_to_process} videos...")

    for i in range(num_to_process):
        process_video(video_files[i], outputFolder)  # Process the selected number of videos


cv2.destroyAllWindows()

print("Processing complete.")