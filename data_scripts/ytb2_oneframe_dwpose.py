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
    Process frames from a video folder and save the first valid frame.
    
    Args:
        video_name: Name of the video
        frames_dir: Root directory containing frame folders
        output_dir: Output directory for the selected frame
        dwpose_detector: DWPose detector instance
        min_boundary_margin: Minimum pixels from image boundary
        
    Returns:
        1 if valid frame found and saved, 0 otherwise
    """
    # Output path
    output_frame_path = Path(output_dir) / f"{video_name}.jpg"
    
    # # Skip if output frame already exists
    # if output_frame_path.exists():
    #     print(f"Skipping {video_name}: Output frame already exists")
    #     return 0
    
    # Input paths
    input_frames_dir = Path(frames_dir) / video_name
    if not input_frames_dir.exists():
        print(f"Error: Frame directory {input_frames_dir} does not exist")
        return 0
    
    # List all JPG files
    frame_files = sorted(list(input_frames_dir.glob('*.jpg')))
    frame_files = frame_files[len(frame_files)//2:] + frame_files[len(frame_files)//2::-1]  # Start from middle frame
    if len(frame_files) == 0:
        print(f"Error: No JPG files found in {input_frames_dir}")
        return 0
    
    print(f"\nProcessing {video_name}: {len(frame_files)} frames")
    
    # Search for first valid frame
    for frame_idx, frame_file in enumerate(tqdm(frame_files, desc=f"Searching {video_name}")):
        # Read frame
        frame = cv2.imread(str(frame_file))
        if frame is None:
            continue
        
        try:
            # Run DWPose detector to get bboxes
            bboxes = inference_detector(dwpose_detector.pose_estimation.session_det, frame)
            
            # Check criteria
            # 1. Must have at least one bbox
            if len(bboxes) == 0:
                continue
            
            # 2. Sort bboxes by area (largest first)
            bbox_areas = [(bbox, get_bbox_area(bbox)) for bbox in bboxes]
            bbox_areas.sort(key=lambda x: x[1], reverse=True)
            
            largest_bbox = bbox_areas[0][0]
            largest_area = bbox_areas[0][1]
            
            # 3. Largest bbox must be 50% larger than second, or only one bbox
            if len(bboxes) > 1:
                second_largest_area = bbox_areas[1][1]
                if largest_area < 1.5 * second_largest_area:
                    continue
            
            # 4. Bbox height must be larger than 1/3 of image height
            x1, y1, x2, y2 = largest_bbox
            bbox_height = y2 - y1
            img_height = frame.shape[0]
            if bbox_height <= img_height / 3:
                continue
            
            # # 5. Bbox must be at least 50 pixels away from boundary
            # if not check_bbox_boundary(largest_bbox, frame.shape, min_boundary_margin):
            #     continue
            
            # 6. Crop image with the bbox and verify face detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped_frame = frame[y1:y2, x1:x2]
            
            # Run full dwpose detector on cropped frame to check face validity
            det_info, det_raw_info = dwpose_detector(cropped_frame)
            
            # Must have valid face detection
            if det_info['bbox'] is None or det_info['faces'] is None and -1 not in det_info['bodies']['subset']:
                continue

            if det_raw_info['scores'].min() < 0.5:
                continue
            
            # 7. Check if face is front-facing
            # Use face landmarks to determine orientation
            if det_info['faces'] is not None and len(det_info['faces']) > 0:
                face_lmks = det_info['faces']  # 68x2 standard face landmarks
                # For 68-point landmarks: 
                # 36-41: right eye contour, 42-47: left eye contour
                # 30: nose tip
                # Use eye centers and nose tip to check symmetry
                
                # Calculate right eye center (points 36-41)
                right_eye_points = face_lmks[36:42]
                if np.any(right_eye_points == -1):
                    continue
                right_eye_center_x = np.mean(right_eye_points[:, 0])
                
                # Calculate left eye center (points 42-47)
                left_eye_points = face_lmks[42:48]
                if np.any(left_eye_points == -1):
                    continue
                left_eye_center_x = np.mean(left_eye_points[:, 0])
                
                # Nose tip (point 30)
                nose_x = face_lmks[30, 0]
                if nose_x == -1:
                    continue
                
                # Distance from nose to each eye center
                right_dist = abs(nose_x - right_eye_center_x)
                left_dist = abs(nose_x - left_eye_center_x)
                
                # Eyes should be roughly equidistant from nose for front face
                # Allow 30% difference (ratio between 0.7 and 1.3)
                if right_dist > 0 and left_dist > 0:
                    eye_ratio = min(right_dist, left_dist) / max(right_dist, left_dist)
                    if eye_ratio < 0.7:
                        # Face is too much turned to side
                        continue
            
            # All criteria passed - save this frame and stop
            cv2.imwrite(str(output_frame_path), frame)
            print(f"  Saved valid frame (frame {frame_idx}) to {output_frame_path}")
            return 1
            
        except Exception as e:
            # Skip frames with detection errors
            continue
    
    # No valid frame found
    print(f"  Warning: No valid frame found for {video_name}")
    return 0


def main():
    parser = argparse.ArgumentParser(description='Filter keyframes from videos using DWPose detection')
    parser.add_argument('--keyframes_dir', type=str, required=True,
                        help='Directory containing frame folders')
    parser.add_argument('--oneframe_dwpose', type=str, required=True,
                        help='Output directory for one frame')
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
    
    if len(video_list) == 0:
        raise ValueError(f"No MP4 files found in {keyframes_dir}")
    
    print(f"Found {len(video_list)} video file(s)")
    
    # Create DWPose detector
    dwpose_detector = create_dwpose_detector(args.dwpose_cfg)
    
    # Create output directory
    output_dir = Path(args.oneframe_dwpose)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each video
    total_success = 0
    for video_path in video_list:
        video_name = video_path.stem
        success = process_video(
            video_name,
            keyframes_dir,
            output_dir, 
            dwpose_detector,
            args.min_boundary_margin
        )
        total_success += success
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total videos processed: {len(video_list)}")
    print(f"Successfully extracted frames: {total_success}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
