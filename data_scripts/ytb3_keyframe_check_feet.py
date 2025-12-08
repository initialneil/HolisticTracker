#!/usr/bin/env python3
"""
Filter keyframes based on DWPose detection and face recognition.
"""

import os
import sys
import json
import cv2
import numpy as np
import time
import shlex
import subprocess
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Add EHM-Tracker to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.io import load_config
from src.utils.helper import instantiate_from_config
from src.modules.dwpose import inference_detector, det_info_to_pose, draw_pose


def create_dwpose_detector(dwpose_cfg_path):
    """Create and warm up DWPose detector."""
    print("Initializing DWPose detector...")
    dwpose_detector = instantiate_from_config(load_config(dwpose_cfg_path))
    dwpose_detector.warmup()
    print("DWPose detector ready")
    return dwpose_detector


def check_body_legs_feet(det_info):
    """Check if body has visible legs and feet."""
    if det_info['bodies'] is None:
        return False
    
    bodies = det_info['bodies']
    subset = bodies['subset'].squeeze()
    
    # Check if subset has valid entries (not all -1)
    if isinstance(subset, np.ndarray):
        if subset.ndim == 0 or (subset.ndim == 1 and len(subset) == 0):
            return False
        # Body keypoints: indices 8-13 are legs
        # 8,9: right hip, left hip
        # 10: right knee
        # 11: left knee
        # 12: right ankle
        # 13: left ankle
        # Check if all 6 leg keypoints exist
        leg_indices = [8, 9, 10, 11, 12, 13]
        for idx in leg_indices:
            if subset[idx] < 0:
                return False
    
    # Check feet
    feet = det_info.get('feet')
    if feet is None:
        return False
    
    if isinstance(feet, np.ndarray):
        # Check if at least one foot keypoint is visible
        visible_feet = np.sum(feet[:, 0] != -1)
        if visible_feet < 1:
            return False
    
    return True


def get_bbox_area(bbox):
    """Calculate area of a bounding box."""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def calculate_bbox_overlap(bbox1, bbox2):
    """Calculate overlap ratio between two bboxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    
    return intersection_area / bbox1_area if bbox1_area > 0 else 0.0


def check_bbox_boundary(bbox, img_shape, min_margin):
    """Check if bbox is at least min_margin pixels away from boundaries."""
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox
    
    if x1 < min_margin or y1 < min_margin:
        return False
    if x2 > w - min_margin or y2 > h - min_margin:
        return False
    
    return True


def extend_bbox_to_square(bbox, scale=1.0):
    """Extend bbox to a square and optionally enlarge by scale."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    # Make it square
    size = max(w, h)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    # Apply scale
    size = size * scale
    
    x1_new = cx - size / 2
    y1_new = cy - size / 2
    x2_new = cx + size / 2
    y2_new = cy + size / 2
    
    return [x1_new, y1_new, x2_new, y2_new]


def crop_with_padding(image, bbox):
    """Crop image with bbox, padding with zeros if bbox exceeds boundaries."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(b) for b in bbox]
    
    # Calculate padding needed
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    
    # Adjust bbox to image boundaries
    x1_img = max(0, x1)
    y1_img = max(0, y1)
    x2_img = min(w, x2)
    y2_img = min(h, y2)
    
    # Crop
    cropped = image[y1_img:y2_img, x1_img:x2_img]
    
    # Add padding if needed
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        cropped = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    
    return cropped


def create_visualization_grid(images_with_text):
    """Create a grid of images with roughly 4:3 aspect ratio."""
    if len(images_with_text) == 0:
        return None
    
    # Calculate grid dimensions for roughly 4:3 aspect ratio
    n_images = len(images_with_text)
    # For 4:3 ratio, cols/rows ≈ 4/3, so cols ≈ sqrt(n * 4/3)
    grid_cols = int(np.ceil(np.sqrt(n_images * 4 / 3)))
    grid_rows = (n_images + grid_cols - 1) // grid_cols
    
    # Create grid
    rows = []
    for i in range(grid_rows):
        row_images = images_with_text[i * grid_cols:(i + 1) * grid_cols]
        # Pad row if needed
        while len(row_images) < grid_cols:
            row_images.append(np.zeros_like(images_with_text[0]))
        rows.append(np.hstack(row_images))
    
    grid = np.vstack(rows)
    return grid


def build_command(args, video_idx, gpu_id):
    """Build command to run single video processing on specific GPU."""
    cmd_parts = [
        f'CUDA_VISIBLE_DEVICES={gpu_id}',
        'python', __file__
    ]
    
    # Add all arguments except distribute_gpus and video_idx
    cmd_parts.extend([
        '--keyframes_dir', shlex.quote(str(args.keyframes_dir)),
        '--filtered_dir', shlex.quote(str(args.filtered_dir)),
        '--dwpose_cfg', shlex.quote(str(args.dwpose_cfg)),
        '--min_boundary_margin', str(args.min_boundary_margin),
        '--video_idx', str(video_idx)
    ])
    
    return ' '.join(cmd_parts)


def process_video(video_name, keyframes_dir, filtered_dir, 
                  dwpose_detector, min_boundary_margin):
    """Process one video and filter frames based on body/leg/feet detection."""
    
    # Output paths
    output_json = Path(filtered_dir) / f"{video_name}.json"
    output_vis = Path(filtered_dir) / f"{video_name}.jpg"
    
    # Skip if already processed
    if output_json.exists():
        print(f"Skipping {video_name}: Already processed")
        return 0
    
    # Input frames directory
    input_frames_dir = Path(keyframes_dir) / video_name
    if not input_frames_dir.exists():
        print(f"Error: Frame directory not found for {video_name}")
        return 0
    
    # List all JPG files
    frame_files = sorted(list(input_frames_dir.glob('*.jpg')))
    if len(frame_files) == 0:
        print(f"Error: No JPG files found for {video_name}")
        return 0
    
    print(f"\nProcessing {video_name}: {len(frame_files)} frames")
    
    # Storage for valid frames
    valid_frames = {}
    vis_images = []
    
    for frame_file in tqdm(frame_files, desc=f"Processing {video_name}"):
        frame_idx = frame_file.stem
        frame = cv2.imread(str(frame_file))

        # if int(frame_idx) == 185:
        #     pass
        
        if frame is None:
            continue
        
        h, w = frame.shape[:2]
        
        try:
            # DWPose detection
            bboxes = inference_detector(dwpose_detector.pose_estimation.session_det, frame)
            
            if len(bboxes) == 0:
                continue
            
            # Sort by area and get largest
            bbox_areas = [(bbox, get_bbox_area(bbox)) for bbox in bboxes]
            bbox_areas.sort(key=lambda x: x[1], reverse=True)
            
            largest_bbox = bbox_areas[0][0]
            x1, y1, x2, y2 = largest_bbox
            bbox_height = y2 - y1
            
            # Criteria 2: bbox height > 1/3 of image height
            if bbox_height < h / 3:
                continue
            
            # Criteria 3: min_boundary_margin away from boundary
            if not check_bbox_boundary(largest_bbox, frame.shape, min_boundary_margin):
                continue
            
            # Criteria 4: overlap with other bboxes < 10%
            overlap_ok = True
            for i in range(1, len(bboxes)):
                overlap = calculate_bbox_overlap(largest_bbox, bbox_areas[i][0])
                if overlap >= 0.1:
                    overlap_ok = False
                    break
            
            if not overlap_ok:
                continue
            
            # Criteria 5: Check body legs and feet exist
            # x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
            # cropped = frame[y1_int:y2_int, x1_int:x2_int]

            # Extend bbox to square with scale 1.1
            square_bbox = extend_bbox_to_square(largest_bbox, scale=1.1)
            
            # Create visualization using square bbox
            cropped_square = crop_with_padding(frame, square_bbox)
            
            # Run full dwpose detector on cropped frame
            det_info, det_raw_info = dwpose_detector(cropped_square)
            
            if not check_body_legs_feet(det_info):
                continue

            # All criteria passed - store result
            valid_frames[frame_idx] = {
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "square_bbox_xyxy": [float(b) for b in square_bbox]
            }

            # Draw DWPose on cropped_square before resizing
            pose = det_info_to_pose(det_info, cropped_square.shape[0], cropped_square.shape[1])
            dwpose_vis = draw_pose(pose, canvas=cropped_square.copy())

            subset = det_info['bodies']['subset']
            candidate = det_info['bodies']['candidate']
            for idx in range(subset.shape[-1]):
                if subset[idx] >= 0:
                    x_coord = int(candidate[idx][0])
                    y_coord = int(candidate[idx][1])
                    cv2.putText(dwpose_vis, str(idx), (x_coord, y_coord), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            vis_img = cv2.resize(dwpose_vis, (128, 128))

            # Add text
            cv2.putText(vis_img, frame_idx, (5, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(vis_img, "pass", (5, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            vis_images.append(vis_img)
            
        except Exception as e:
            continue
    
    # Write results
    if len(valid_frames) > 0:
        result = {
            video_name: {
                "frames_num": len(valid_frames),
                "frames_keys": valid_frames
            }
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        # Create visualization grid
        if len(vis_images) > 0:
            grid = create_visualization_grid(vis_images)
            cv2.imwrite(str(output_vis), grid)
        
        print(f"  Saved {len(valid_frames)} valid frames for {video_name}")
        return len(valid_frames)
    else:
        print(f"  No valid frames found for {video_name}")
        return 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter keyframes using body/leg/feet detection')
    parser.add_argument('--keyframes_dir', type=str, required=True,
                        help='Directory containing frame folders')
    parser.add_argument('--filtered_dir', type=str, required=True,
                        help='Output directory for filtered results')
    parser.add_argument('--dwpose_cfg', type=str,
                        default='src/configs/model_configs/dwpose_onnx_config.yaml',
                        help='Path to DWPose config file')
    parser.add_argument('--min_boundary_margin', type=int, default=10,
                        help='Minimum pixels from image boundary (default: 10)')
    parser.add_argument('--distribute_gpus', type=str, default=None,
                        help='Comma-separated GPU IDs for parallel processing (e.g., "0,1,2")')
    parser.add_argument('--video_idx', type=int, default=None,
                        help='Process specific video index (used internally for parallel processing)')
    
    args = parser.parse_args()
    
    # Setup
    keyframes_dir = Path(args.keyframes_dir)
    filtered_dir = Path(args.filtered_dir)
    
    if not keyframes_dir.exists():
        raise ValueError(f"Input directory {keyframes_dir} does not exist")
    
    filtered_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video list from preview images in keyframes_dir
    video_list = [f.stem for f in sorted(keyframes_dir.glob('*.jpg'))]
    
    if len(video_list) == 0:
        raise ValueError(f"No video directories found in {keyframes_dir}")
    
    print(f"Found {len(video_list)} videos to process")
    
    # Check if parallel GPU distribution is requested
    if args.distribute_gpus is not None:
        # Parallel mode: spawn subprocesses for each GPU
        visible_gpus = [int(x) for x in args.distribute_gpus.split(',') if x.strip() != '']
        
        print(f"Running in parallel mode with GPUs: {visible_gpus}")
        
        all_procs = []
        counter = 0
        
        for video_idx in range(len(video_list)):
            gpu_id = visible_gpus[video_idx % len(visible_gpus)]
            
            cmd = build_command(args, video_idx, gpu_id)
            print(f"Launching process for video {video_idx} ({video_list[video_idx]}) on GPU {gpu_id}")
            all_procs.append(subprocess.Popen(cmd, shell=True))
            counter += 1
            
            if counter % len(visible_gpus) == 0:
                # Sleep 30 seconds after every visible_gpus processes to avoid memory overload during warm-up
                print("Start sleeping for 30 seconds.......")
                time.sleep(30)
                print("Finish sleeping.......")
        
        # Wait for all processes to complete
        for p in all_procs:
            p.wait()
        
        print(f"\n{'='*60}")
        print(f"Parallel processing complete!")
        print(f"Total videos processed: {len(video_list)}")
        print(f"Output directory: {filtered_dir}")
        
    elif args.video_idx is not None:
        # Single video mode (called from parallel subprocess)
        if args.video_idx < 0 or args.video_idx >= len(video_list):
            print(f"Error: video_idx {args.video_idx} out of range (0-{len(video_list)-1})")
            return
        
        video_name = video_list[args.video_idx]
        print(f"Processing single video {args.video_idx}: {video_name}")
        
        # Initialize DWPose detector
        dwpose_detector = create_dwpose_detector(args.dwpose_cfg)
        
        # Process single video
        num_valid = process_video(
            video_name, keyframes_dir, filtered_dir,
            dwpose_detector, args.min_boundary_margin
        )
        
        print(f"\nVideo {video_name} complete: {num_valid} valid frames")
        
    else:
        # Sequential mode: process all videos
        print("Running in sequential mode")
        
        # Initialize DWPose detector
        dwpose_detector = create_dwpose_detector(args.dwpose_cfg)
        
        # Process each video
        total_valid = 0
        for video_name in video_list:
            num_valid = process_video(
                video_name, keyframes_dir, filtered_dir,
                dwpose_detector, args.min_boundary_margin
            )
            total_valid += num_valid
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Total videos processed: {len(video_list)}")
        print(f"Total valid frames: {total_valid}")
        print(f"Output directory: {filtered_dir}")


if __name__ == '__main__':
    main()

