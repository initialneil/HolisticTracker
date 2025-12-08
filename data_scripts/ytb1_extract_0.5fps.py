#!/usr/bin/env python3
"""
Extract frames from videos at specified FPS using OpenCV.
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add EHM-Tracker to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.general_utils import parallel_foreach


def create_preview_grid(output_dir, video_name, keyframes_dir):
    """
    Create a preview grid image from extracted frames.
    
    Args:
        output_dir: Directory containing extracted frames
        video_name: Name of the video
        keyframes_dir: Root directory for saving preview
    
    Returns:
        True if successful, False otherwise
    """
    # Get all jpg files
    frame_files = sorted(list(output_dir.glob('*.jpg')))
    
    if len(frame_files) == 0:
        return False
    
    # Resize frames to small thumbnails (max 128x128, maintaining aspect ratio)
    max_size = 128
    thumbnails = []
    
    for frame_path in frame_files:
        img = cv2.imread(str(frame_path))
        if img is not None:
            # Calculate scaling to fit within max_size x max_size
            h, w = img.shape[:2]
            scale = min(max_size / w, max_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize to thumbnail
            thumb = cv2.resize(img, (new_w, new_h))
            
            # Add frame index text
            frame_idx = frame_path.stem
            cv2.putText(thumb, frame_idx, (5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            thumbnails.append(thumb)
    
    if len(thumbnails) == 0:
        return False
    
    # Get thumbnail dimensions (all should have same size after scaling)
    thumb_h, thumb_w = thumbnails[0].shape[:2]
    
    # Calculate grid dimensions for roughly 4:3 aspect ratio
    n_images = len(thumbnails)
    grid_cols = int(np.ceil(np.sqrt(n_images * 4 / 3)))
    grid_rows = (n_images + grid_cols - 1) // grid_cols
    
    # Create grid
    rows = []
    for i in range(grid_rows):
        row_images = thumbnails[i * grid_cols:(i + 1) * grid_cols]
        # Pad row if needed
        while len(row_images) < grid_cols:
            row_images.append(np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8))
        rows.append(np.hstack(row_images))
    
    grid = np.vstack(rows)
    
    # Save grid
    output_path = keyframes_dir / f"{video_name}.jpg"
    cv2.imwrite(str(output_path), grid)
    
    return True


def extract_frames_from_video(video_path, output_dir, target_fps):
    """
    Extract frames from a video at target FPS.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        target_fps: Target FPS for extraction (e.g., 0.5 for one frame every 2 seconds)
    
    Returns:
        Tuple of (num_frames, frames_keys)
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0, []
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval (how many frames to skip)
    frame_interval = int(video_fps / target_fps)
    if frame_interval < 1:
        frame_interval = 1
    
    print(f"  Video FPS: {video_fps:.2f}, Total frames: {total_frames}")
    print(f"  Target FPS: {target_fps}, Frame interval: {frame_interval}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames
    frame_idx = 0
    saved_count = 0
    frames_keys = []
    
    with tqdm(total=total_frames, desc=f"  Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame at the specified interval
            if frame_idx % frame_interval == 0:
                # Use continuous counting for saved frames
                output_path = output_dir / f"{saved_count:06d}.jpg"
                cv2.imwrite(str(output_path), frame)
                frames_keys.append(f"{saved_count:06d}")
                saved_count += 1
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    
    print(f"  Extracted {saved_count} frames")
    return saved_count, frames_keys


def process_video(func_args):
    """
    Process a single video with the given arguments.
    
    Args:
        func_args: Dictionary containing:
            - video_path: Path to video file
            - keyframes_dir: Output directory for extracted frames
            - target_fps: Target FPS for extraction
    
    Returns:
        Number of frames extracted
    """
    video_path = func_args['video_path']
    keyframes_dir = func_args['keyframes_dir']
    target_fps = func_args['target_fps']
    
    # Extract video name (first part before '.')
    video_name = video_path.stem.split('.')[0]
    
    # Create output directory for this video
    output_dir = keyframes_dir / video_name
    output_json = output_dir / f"{video_name}.json"
    preview_image = keyframes_dir / f"{video_name}.jpg"
    
    # Skip if preview image already exists
    if preview_image.exists():
        # Try to load frame count from JSON if it exists
        if output_json.exists():
            with open(output_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            num_existing = data.get(video_name, {}).get('frames_num', 0)
        else:
            # Count frames from directory
            num_existing = len(list(output_dir.glob('*.jpg'))) if output_dir.exists() else 0
        print(f"Skipping {video_name}: Already processed ({num_existing} frames)")
        return num_existing
    
    print(f"Processing: {video_path.name}")
    print(f"  Video name: {video_name}")
    
    # Extract frames
    num_frames, frames_keys = extract_frames_from_video(video_path, output_dir, target_fps)
    if num_frames == 0:
        print(f"  No frames extracted for {video_name}, skipping metadata and preview.")
        return 0
    
    # Write JSON metadata
    metadata = {
        video_name: {
            "frames_num": num_frames,
            "frames_keys": frames_keys,
            "fps": target_fps
        }
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved metadata to {output_json}")
    
    # Create preview grid
    if create_preview_grid(output_dir, video_name, keyframes_dir):
        print(f"  Created preview grid: {video_name}.jpg")
    
    print()
    return num_frames


def process_videos(videos_dir, keyframes_dir, target_fps):
    """
    Process all MP4 videos in the directory.
    
    Args:
        videos_dir: Directory containing MP4 videos
        keyframes_dir: Output directory for extracted frames
        target_fps: Target FPS for extraction
    """
    videos_dir = Path(videos_dir)
    keyframes_dir = Path(keyframes_dir)
    
    # List all MP4 files
    video_files = sorted(list(videos_dir.glob('*.mp4')))
    
    if len(video_files) == 0:
        print(f"No MP4 files found in {videos_dir}")
        return
    
    print(f"Found {len(video_files)} video(s)")
    print(f"Target FPS: {target_fps}")
    print()
    
    # Prepare arguments for parallel processing
    func_args_list = [
        {
            'video_path': video_path,
            'keyframes_dir': keyframes_dir,
            'target_fps': target_fps
        }
        for video_path in video_files
    ]
    
    # Process videos in parallel
    frame_counts = parallel_foreach(process_video, func_args_list)
    total_frames = sum(frame_counts)
    
    print(f"{'='*60}")
    print(f"Processing complete!")
    print(f"Total videos processed: {len(video_files)}")
    print(f"Total frames extracted: {total_frames}")
    print(f"Output directory: {keyframes_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos at specified FPS')
    parser.add_argument('--videos_dir', type=str, required=True,
                        help='Directory containing MP4 videos')
    parser.add_argument('--keyframes_dir', type=str, required=True,
                        help='Output directory for extracted frames')
    parser.add_argument('--fps', type=float, default=0.5,
                        help='Target FPS for extraction (default: 0.5)')
    
    args = parser.parse_args()
    
    # Validate inputs
    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists():
        raise ValueError(f"Videos directory does not exist: {videos_dir}")
    
    # Process videos
    process_videos(args.videos_dir, args.keyframes_dir, args.fps)


if __name__ == '__main__':
    main()
