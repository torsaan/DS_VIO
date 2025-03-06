# test_videos.py
import os
import glob

def find_videos(video_dir):
    print(f"Checking directory: {os.path.abspath(video_dir)}")
    
    # Check if directory exists
    if not os.path.exists(video_dir):
        print(f"ERROR: Directory {video_dir} does not exist!")
        return
    
    # List subdirectories
    subdirs = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]
    print(f"Found subdirectories: {subdirs}")
    
    # Look for video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos = []
    
    for root, _, files in os.walk(video_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                videos.append(os.path.join(root, file))
    
    print(f"Found {len(videos)} video files")
    
    # Show some of the videos found
    if videos:
        print("Sample videos:")
        for video in videos[:5]:  # Show first 5 videos
            print(f"  - {video}")
    
    # Check for V_ and NV_ prefix naming
    violence_videos = [v for v in videos if os.path.basename(v).startswith('V_')]
    nonviolence_videos = [v for v in videos if os.path.basename(v).startswith('NV_')]
    other_videos = [v for v in videos if not (os.path.basename(v).startswith('V_') or os.path.basename(v).startswith('NV_'))]
    
    print(f"Videos with V_ prefix: {len(violence_videos)}")
    print(f"Videos with NV_ prefix: {len(nonviolence_videos)}")
    print(f"Videos with other prefixes: {len(other_videos)}")
    
    if other_videos:
        print("Sample videos with incorrect naming:")
        for video in other_videos[:5]:  # Show first 5 incorrectly named videos
            print(f"  - {video}")

if __name__ == "__main__":
    video_dir = "./Data/VioNonVio"
    find_videos(video_dir)