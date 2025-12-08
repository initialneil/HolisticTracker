#!/usr/bin/env python3
"""
Generate videos_info.json from existing frame directories.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def process_video_folder(video_folder):
    """
    Process a single video folder and generate videos_info.json.
    
    Args:
        video_folder: Path to the video folder containing frame subdirectories
        
    Returns:
        Number of frames processed
    """
    video_folder = Path(video_folder)
    video_name = video_folder.name
    
    # Find all JPG frames in all subdirectories
    all_frames = sorted(list(video_folder.glob('**/*.jpg')))
    
    if not all_frames:
        print(f"Warning: No JPG frames found in {video_folder}")
        return 0
    
    # Group frames by segment (subdirectory)
    segments = {}
    for frame_path in all_frames:
        # Get the segment directory (parent of the frame)
        seg_dir = frame_path.parent
        seg_id = seg_dir.name
        
        if seg_id not in segments:
            segments[seg_id] = []
        
        # Extract frame index from filename (without extension)
        frame_idx = frame_path.stem
        # Create frame key as seg_id_frameIdx
        frame_key = f"{seg_id}_{frame_idx}"
        segments[seg_id].append(frame_key)
    
    # Sort frame keys within each segment
    for seg_id in segments:
        segments[seg_id].sort()
    
    # If there's only one segment, use its frames directly
    # Otherwise, concatenate all segments in order
    if len(segments) == 1:
        frames_keys = list(segments.values())[0]
    else:
        # Sort segments by name and concatenate
        frames_keys = []
        for seg_id in sorted(segments.keys()):
            frames_keys.extend(segments[seg_id])
    
    frames_num = len(frames_keys)
    
    # Create videos_info.json
    videos_info = {
        video_name: {
            "frames_num": frames_num,
            "frames_keys": frames_keys
        }
    }
    
    info_path = video_folder / "videos_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(videos_info, f, ensure_ascii=False, indent=4)
    
    print(f"Processed {video_name}: {frames_num} frames -> {info_path}")
    
    return frames_num


def main():
    parser = argparse.ArgumentParser(description='Generate videos_info.json from existing frame directories')
    parser.add_argument('--frames_root', type=str, required=True,
                        help='Root directory containing video folders with frames')
    
    args = parser.parse_args()
    
    frames_root = Path(args.frames_root)
    
    if not frames_root.exists():
        raise ValueError(f"Path {frames_root} does not exist")
    
    if not frames_root.is_dir():
        raise ValueError(f"Path {frames_root} is not a directory")
    
    # List all subdirectories as video folders
    video_folders = [d for d in frames_root.iterdir() if d.is_dir()]
    
    if not video_folders:
        raise ValueError(f"No subdirectories found in {frames_root}")
    
    video_folders = sorted(video_folders)
    print(f"Found {len(video_folders)} video folder(s)")
    
    # Process each video folder
    total_frames = 0
    for video_folder in tqdm(video_folders, desc="Processing video folders"):
        num_frames = process_video_folder(video_folder)
        total_frames += num_frames
    
    print(f"\nProcessed {len(video_folders)} video folders")
    print(f"Total frames: {total_frames}")


if __name__ == '__main__':
    main()
