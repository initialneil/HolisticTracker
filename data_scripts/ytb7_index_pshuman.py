#!/usr/bin/env python3
"""
ytb7_index_pshuman.py - Update JSON index with PSHuman multi-view images

This script processes video JSON files and appends PSHuman multi-view image paths
to the frames_keys list for each frame that has corresponding PSHuman data.
"""
import os
import numpy as np
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def load_video_json(video_name, images_dir):
    """
    Load the JSON file for a video.
    
    Args:
        video_name: Name of the video
        images_dir: Root directory containing video folders
    
    Returns:
        Dictionary with video data or None if file doesn't exist
    """
    json_path = os.path.join(images_dir, video_name, f"{video_name}.json")
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r') as f:
        return json.load(f)


def check_pshuman_frame(pshuman_dir, video_name, frame_key, view_keys):
    """
    Check if PSHuman data exists for a frame and return available view paths.
    
    Args:
        pshuman_dir: Root directory containing PSHuman data
        video_name: Name of the video
        frame_key: Frame key in format "shotid/frameid"
        view_keys: List of view keys to check (e.g., ['03', '05'])
    
    Returns:
        List of valid PSHuman image paths relative to the frame
    """
    frame_dir = os.path.join(pshuman_dir, video_name, frame_key)
    
    # Check if frame directory exists
    if not os.path.isdir(frame_dir):
        return []
    
    valid_views = []
    for view_key in view_keys:
        image_path = os.path.join(frame_dir, f"color_{view_key}.jpg")
        if os.path.exists(image_path):
            # Return relative path format: frame_key/color_view.jpg
            valid_views.append(f"{frame_key}/pshuman_{view_key}")
    
    return valid_views


def process_video(video_name, images_dir, pshuman_dir, list_pshuman):
    """
    Process a single video and update its JSON with PSHuman image paths.
    
    Args:
        video_name: Name of the video
        images_dir: Root directory containing video folders
        pshuman_dir: Root directory containing PSHuman data
        list_pshuman: List of view keys to include (e.g., ['03'])
    
    Returns:
        True if successful, False otherwise
    """
    # Load original JSON
    video_data = load_video_json(video_name, images_dir)
    if video_data is None:
        print(f"  Warning: JSON not found for {video_name}")
        return False
    
    # Extract video info (assuming JSON has video_name as key)
    if video_name not in video_data:
        print(f"  Warning: Video key '{video_name}' not found in JSON")
        return False
    
    video_info = video_data[video_name]
    frames_keys = video_info.get('frames_keys', [])
    
    if not frames_keys:
        print(f"  Warning: No frames_keys found for {video_name}")
        return False
    
    # Create new frames_keys list with PSHuman views appended
    new_frames_keys = []
    pshuman_count = 0
    
    for frame_key in tqdm(frames_keys, desc=f"  Processing {video_name}", leave=False):
        # Add original frame key
        new_frames_keys.append(frame_key)
        
        # Check and add PSHuman views
        pshuman_views = check_pshuman_frame(pshuman_dir, video_name, frame_key, list_pshuman)
        if pshuman_views:
            new_frames_keys.extend(pshuman_views)
            pshuman_count += len(pshuman_views)
    
    # Update video info with new frames_keys
    video_info['frames_keys'] = new_frames_keys
    video_info['frames_num'] = len(frames_keys)  # Keep original frame count
    # video_info['pshuman_dir'] = pshuman_dir
    
    # Create output JSON
    output_data = {video_name: video_info}
    output_path = os.path.join(images_dir, video_name, f"{video_name}_pshuman.json")
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"  ✓ Processed {video_name}: {len(frames_keys)} frames, {pshuman_count} PSHuman views")
    print(f"    Saved to: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Update JSON index with PSHuman multi-view images')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Root directory containing video folders with JSON files')
    parser.add_argument('--pshuman_dir', type=str, required=True,
                        help='Root directory containing PSHuman multi-view data')
    parser.add_argument('--list_pshuman', type=str, nargs='+', default=['03'],
                        help='List of view keys to include (default: ["03"])')
    parser.add_argument('--video_idxs', type=int, nargs='+', default=None,
                        help='Specific videos to process (default: all videos in images_dir)')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.images_dir):
        print(f"Error: images_dir not found: {args.images_dir}")
        return
    
    if not os.path.exists(args.pshuman_dir):
        print(f"Error: pshuman_dir not found: {args.pshuman_dir}")
        return
    
    # Get video list
    # List all directories in images_dir
    video_list = [d for d in os.listdir(args.images_dir) 
                  if os.path.isdir(os.path.join(args.images_dir, d))]
    video_list.sort()
    
    if args.video_idxs:
        video_list = np.array(video_list)[args.video_idxs]
    
    print(f"Found {len(video_list)} video(s) to process")
    print(f"PSHuman views to include: {args.list_pshuman}")
    print("")
    
    # Process each video
    success_count = 0
    for video_name in video_list:
        try:
            if process_video(video_name, args.images_dir, args.pshuman_dir, args.list_pshuman):
                success_count += 1
        except Exception as e:
            print(f"  ✗ Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("")
    print("="*80)
    print(f"Processing complete: {success_count}/{len(video_list)} videos processed successfully")
    print("="*80)


if __name__ == '__main__':
    main()
