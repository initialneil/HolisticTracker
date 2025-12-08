#!/usr/bin/env python3
"""
Filter keyframes from videos based on DWPose detection criteria.
Selects frames with single person detection and sufficient boundary margin.
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

from src.utils.io import load_config
from src.utils.helper import instantiate_from_config
from src.modules.dwpose import inference_detector


def create_dwpose_detector(dwpose_cfg_path):
    """Create and warm up DWPose detector."""
    print("Initializing DWPose detector...")
    dwpose_detector = instantiate_from_config(load_config(dwpose_cfg_path))
    dwpose_detector.warmup()
    print("DWPose detector ready")
    return dwpose_detector


def get_bbox_area(bbox):
    """Calculate area of a bounding box."""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def check_bbox_boundary(bbox, img_shape, min_margin=50):
    """
    Check if bbox is at least min_margin pixels away from image boundaries.
    
    Args:
        bbox: [x1, y1, x2, y2]
        img_shape: (height, width, channels)
        min_margin: Minimum distance in pixels from boundary
        
    Returns:
        True if bbox satisfies margin requirement
    """
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Check all four boundaries
    if x1 < min_margin or y1 < min_margin:
        return False
    if x2 > w - min_margin or y2 > h - min_margin:
        return False
    
    return True


def check_bbox_intersection(bbox1, bbox2):
    """
    Check if two bounding boxes intersect.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
        
    Returns:
        True if bboxes intersect
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Check if one bbox is completely to the left/right/above/below the other
    if x2_1 <= x1_2 or x2_2 <= x1_1:
        return False
    if y2_1 <= y1_2 or y2_2 <= y1_1:
        return False
    
    return True


def process_video(video_name, frames_dir, output_dir, dwpose_detector, min_boundary_margin=50):
    """
    Process frames from a video folder: detect poses and filter valid keyframes.
    
    Args:
        video_name: Name of the video
        frames_dir: Root directory containing frame folders
        output_dir: Output directory for filtered frames
        dwpose_detector: DWPose detector instance
        min_boundary_margin: Minimum pixels from image boundary
        
    Returns:
        Number of valid frames found
    """
    # Input paths
    input_frames_dir = Path(frames_dir) / video_name
    if not input_frames_dir.exists():
        print(f"Error: Frame directory {input_frames_dir} does not exist")
        return 0
    
    # List all JPG files
    frame_files = sorted(list(input_frames_dir.glob('*.jpg')))
    if len(frame_files) == 0:
        print(f"Error: No JPG files found in {input_frames_dir}")
        return 0
    
    # Output paths
    output_video_dir = Path(output_dir) / video_name
    output_json_path = Path(output_dir) / f"{video_name}.json"
    
    # # Skip if JSON already exists
    # if output_json_path.exists():
    #     print(f"Skipping {video_name}: JSON already exists")
    #     return 0
    
    # Create output directory
    output_video_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {video_name}: {len(frame_files)} frames")
    
    valid_frames = []
    
    with tqdm(total=len(frame_files), desc=f"Filtering {video_name}") as pbar:
        for frame_idx, frame_file in enumerate(frame_files):
            # Read frame
            frame = cv2.imread(str(frame_file))
            if frame is None:
                pbar.update(1)
                continue
            
            try:
                # Run DWPose detector to get full detection info
                det_info, det_raw_info = dwpose_detector(frame)
                
                # Check criteria
                # 1. Must have exactly one bbox with valid face detection
                if det_info['bbox'] is None or det_info['faces'] is None:
                    pbar.update(1)
                    continue
                
                bbox = det_info['bbox']
                
                # 2. Bbox must be at least 50 pixels away from boundary
                if not check_bbox_boundary(bbox, frame.shape, min_boundary_margin):
                    pbar.update(1)
                    continue
                
                # All criteria passed - save valid frame
                frame_filename = f"{frame_idx:06d}.jpg"
                frame_path = output_video_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                valid_frames.append(f"{frame_idx:06d}")
                
            except Exception as e:
                # Skip frames with detection errors
                pass
            
            pbar.update(1)
    
    # Write JSON metadata
    if len(valid_frames) > 0:
        video_info = {
            video_name: {
                "frames_num": len(valid_frames),
                "frames_keys": valid_frames
            }
        }
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(video_info, f, ensure_ascii=False, indent=4)
        
        print(f"  Saved {len(valid_frames)} valid frames to {output_video_dir}")
        print(f"  Saved metadata to {output_json_path}")
    else:
        print(f"  Warning: No valid frames found for {video_name}")
    
    return len(valid_frames)


def main():
    parser = argparse.ArgumentParser(description='Filter keyframes from videos using DWPose detection')
    parser.add_argument('--keyframes_dir', type=str, required=True,
                        help='Directory containing frame folders')
    parser.add_argument('--keyframes_dwpose', type=str, required=True,
                        help='Output directory for filtered keyframes')
    parser.add_argument('--dwpose_cfg', type=str, 
                        default='src/configs/model_configs/dwpose_onnx_config.yaml',
                        help='Path to DWPose config file')
    parser.add_argument('--min_boundary_margin', type=int, default=10,
                        help='Minimum pixels from image boundary (default: 10)')
    
    args = parser.parse_args()
    
    # Get list of video folders (by finding MP4 files and extracting video names)
    keyframes_dir = Path(args.keyframes_dir)
    if not keyframes_dir.exists():
        raise ValueError(f"Input directory {keyframes_dir} does not exist")
    
    # Find thumbnail mp4
    video_list = sorted(list(keyframes_dir.glob('*.mp4')))
    video_list = ["-nKdufEaL8k"]
    
    if len(video_list) == 0:
        raise ValueError(f"No MP4 files found in {keyframes_dir}")
    
    print(f"Found {len(video_list)} video file(s)")
    
    # Create DWPose detector
    dwpose_detector = create_dwpose_detector(args.dwpose_cfg)
    
    # Create output directory
    output_dir = Path(args.keyframes_dwpose)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each video
    total_valid_frames = 0
    for video_name in video_list:
        num_valid = process_video(
            video_name,
            keyframes_dir,
            output_dir, 
            dwpose_detector,
            args.min_boundary_margin
        )
        total_valid_frames += num_valid
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total videos processed: {len(video_list)}")
    print(f"Total valid frames extracted: {total_valid_frames}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
