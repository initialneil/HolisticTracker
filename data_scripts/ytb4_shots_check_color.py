#!/usr/bin/env python3
"""
Extract and segment frames based on color histogram similarity.
Creates visualization grids with scene segmentation indicators.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add EHM-Tracker to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.general_utils import parallel_foreach


def compute_color_histogram(image, bins=(8, 8, 8)):
    """Compute normalized 3D color histogram in HSV color space."""
    if image is None or image.size == 0:
        return None
    
    # Convert to HSV for better color representation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Compute 3D histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, 
                        [0, 180, 0, 256, 0, 256])
    
    # Normalize histogram
    cv2.normalize(hist, hist)
    
    return hist.flatten()


def compare_color_histograms(hist1, hist2):
    """Compare two color histograms using correlation method."""
    if hist1 is None or hist2 is None:
        return 0.0
    
    # Use correlation method (returns value in [-1, 1], higher is more similar)
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return correlation


def crop_image_with_bbox(image, bbox):
    """Crop image with bbox, clamping to image boundaries."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(b) for b in bbox]
    
    # Clamp to boundaries
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return image[y1:y2, x1:x2]


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


def generate_random_color():
    """Generate a random BGR color."""
    return tuple(np.random.randint(0, 256, 3).tolist())


def add_scene_indicator(image, scene_color):
    """Add 10-line scene indicator at bottom (9 lines scene color, 1 line white)."""
    h, w = image.shape[:2]
    
    # Create indicator strip (10 pixels tall)
    indicator = np.zeros((10, w, 3), dtype=np.uint8)
    
    # Fill first 9 lines with scene color
    indicator[:9, :] = scene_color
    
    # Fill last line with white
    indicator[9:, :] = (255, 255, 255)
    
    # Concatenate to bottom of image
    result = np.vstack([image, indicator])
    
    return result


def segment_frames_by_scene(frame_infos, keyframes_dir, video_name, scene_threshold=0.3):
    """
    Segment frames into scenes based on frame continuity and color histogram similarity.
    
    Returns:
        List of (frame_idx, frame_info, scene_id, cropped_image, similarity_score),
        scene_colors dict,
        shots dict mapping scene_id to list of frame indices
    """
    results = []
    current_scene_id = 0
    prev_histogram = None
    prev_frame_num = None
    
    # Generate random color for first scene
    scene_colors = {0: generate_random_color()}
    
    # Track shots (scenes)
    shots = {0: []}
    
    for frame_idx, frame_info in frame_infos:
        # Load frame
        frame_path = Path(keyframes_dir) / video_name / f"{frame_idx}.jpg"
        if not frame_path.exists():
            continue
        
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        
        # Crop with bbox_xyxy
        bbox = frame_info['bbox_xyxy']
        cropped = crop_image_with_bbox(frame, bbox)
        if cropped is None:
            continue
        
        # Compute color histogram
        curr_histogram = compute_color_histogram(cropped)
        if curr_histogram is None:
            continue
        
        # Check for scene change
        similarity_score = None
        current_frame_num = int(frame_idx)
        
        if prev_frame_num is not None:
            # Check if frames are continuous
            is_continuous = (current_frame_num - prev_frame_num == 1)
            
            if not is_continuous:
                # Not continuous -> new scene
                current_scene_id += 1
                scene_colors[current_scene_id] = generate_random_color()
                shots[current_scene_id] = []
            else:
                # Continuous -> check color similarity
                similarity = compare_color_histograms(prev_histogram, curr_histogram)
                similarity_score = similarity
                if similarity < scene_threshold:
                    # Color change detected -> new scene
                    current_scene_id += 1
                    scene_colors[current_scene_id] = generate_random_color()
                    shots[current_scene_id] = []
        
        # Crop with square_bbox_xyxy for visualization
        square_bbox = frame_info['square_bbox_xyxy']
        cropped_square = crop_with_padding(frame, square_bbox)
        
        results.append((frame_idx, frame_info, current_scene_id, cropped_square, similarity_score))
        shots[current_scene_id].append(frame_idx)
        prev_histogram = curr_histogram
        prev_frame_num = current_frame_num
    
    return results, scene_colors, shots


def create_visualization_grid(frame_data, scene_colors):
    """
    Create a grid of images with scene indicators.
    
    Args:
        frame_data: List of (frame_idx, frame_info, scene_id, cropped_image, similarity_score)
        scene_colors: Dict mapping scene_id to color
    """
    if len(frame_data) == 0:
        return None
    
    # Resize all images and add scene indicators
    vis_images = []
    for frame_idx, frame_info, scene_id, cropped_img, similarity_score in frame_data:
        # Resize to 128x128
        vis_img = cv2.resize(cropped_img, (128, 128))
        
        # Add frame index text
        cv2.putText(vis_img, frame_idx, (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add similarity score (change value) on second and subsequent images
        if similarity_score is not None:
            score_text = f"chg:{similarity_score:.3f}"
            cv2.putText(vis_img, score_text, (5, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Add scene indicator at bottom
        scene_color = scene_colors[scene_id]
        vis_img = add_scene_indicator(vis_img, scene_color)
        
        vis_images.append(vis_img)
    
    # Calculate grid dimensions for roughly 4:3 aspect ratio
    n_images = len(vis_images)
    grid_cols = int(np.ceil(np.sqrt(n_images * 4 / 3)))
    grid_rows = (n_images + grid_cols - 1) // grid_cols
    
    # Create grid
    rows = []
    for i in range(grid_rows):
        row_images = vis_images[i * grid_cols:(i + 1) * grid_cols]
        # Pad row if needed
        while len(row_images) < grid_cols:
            # Create empty placeholder with same height (128 + 10 for indicator)
            row_images.append(np.zeros((138, 128, 3), dtype=np.uint8))
        rows.append(np.hstack(row_images))
    
    grid = np.vstack(rows)
    return grid


def process_video(func_args):
    """Process one video and create segmented frame visualization.
    
    Args:
        func_args: Dictionary containing:
            - video_name: Name of the video
            - keyframes_dir: Directory containing frame folders
            - filtered_dir: Directory containing filtered JSON files
            - shots_dir: Output directory for shot visualizations
            - scene_threshold: Threshold for scene change detection
    
    Returns:
        Number of frames processed
    """
    video_name = func_args['video_name']
    keyframes_dir = func_args['keyframes_dir']
    filtered_dir = func_args['filtered_dir']
    shots_dir = func_args['shots_dir']
    scene_threshold = func_args['scene_threshold']
    
    # Input JSON path
    input_json = Path(filtered_dir) / f"{video_name}.json"
    if not input_json.exists():
        print(f"Error: JSON not found for {video_name}")
        return 0
    
    # Output paths
    output_vis = Path(shots_dir) / f"{video_name}.jpg"
    output_json = Path(shots_dir) / f"{video_name}.json"
    
    # Skip if already processed
    if output_vis.exists() and output_json.exists():
        print(f"Skipping {video_name}: Already processed")
        return 0
    
    # Load JSON
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if video_name not in data:
        print(f"Error: Video name not found in JSON for {video_name}")
        return 0
    
    video_data = data[video_name]
    frames_keys = video_data.get('frames_keys', {})
    
    if len(frames_keys) == 0:
        print(f"Warning: No frames found for {video_name}")
        return 0
    
    # Convert to sorted list
    frame_infos = [(frame_idx, frame_info) for frame_idx, frame_info in sorted(frames_keys.items())]
    
    print(f"\nProcessing {video_name}: {len(frame_infos)} frames")
    
    # Segment frames by scene
    frame_data, scene_colors, shots = segment_frames_by_scene(
        frame_infos, keyframes_dir, video_name, scene_threshold
    )
    
    if len(frame_data) == 0:
        print(f"  Warning: No valid frames processed for {video_name}")
        return 0
    
    # Create shot_keys list and update frames_keys with shot_key
    shot_keys = []
    updated_frames_keys = {}
    
    for scene_id, frame_indices in shots.items():
        if len(frame_indices) > 0:
            # Create shot_key as <first_frame>_<last_frame>
            shot_key = f"{frame_indices[0]}_{frame_indices[-1]}"
            shot_keys.append(shot_key)
            
            # Add shot_key to each frame in this shot
            for frame_idx in frame_indices:
                if frame_idx in frames_keys:
                    frame_info = frames_keys[frame_idx].copy()
                    frame_info['shot_key'] = shot_key
                    updated_frames_keys[frame_idx] = frame_info
    
    # Create output JSON with all original data plus shot information
    output_data = data.copy()
    output_data[video_name]['shot_keys'] = shot_keys
    output_data[video_name]['frames_keys'] = updated_frames_keys
    
    # Save output JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Create visualization grid
    grid = create_visualization_grid(frame_data, scene_colors)
    if grid is not None:
        cv2.imwrite(str(output_vis), grid)
        
        # Count scenes
        num_scenes = len(scene_colors)
        print(f"  Saved {len(frame_data)} frames in {num_scenes} shot(s) to {output_vis}")
        print(f"  Saved shot metadata to {output_json}")
        return len(frame_data)
    else:
        print(f"  Error: Failed to create visualization for {video_name}")
        return 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract and segment frames with scene detection')
    parser.add_argument('--keyframes_dir', type=str, required=True,
                        help='Directory containing frame folders')
    parser.add_argument('--filtered_dir', type=str, required=True,
                        help='Directory containing filtered JSON files')
    parser.add_argument('--shots_dir', type=str, required=True,
                        help='Output directory for shot visualizations and JSON files')
    parser.add_argument('--scene_threshold', type=float, default=0.3,
                        help='Color histogram correlation threshold for scene change (default: 0.3)')
    
    args = parser.parse_args()
    
    # Setup
    keyframes_dir = Path(args.keyframes_dir)
    filtered_dir = Path(args.filtered_dir)
    shots_dir = Path(args.shots_dir)
    
    if not keyframes_dir.exists():
        raise ValueError(f"Keyframes directory {keyframes_dir} does not exist")
    if not filtered_dir.exists():
        raise ValueError(f"Filtered directory {filtered_dir} does not exist")
    
    shots_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video list from JSON files in filtered_dir
    video_list = [f.stem for f in sorted(filtered_dir.glob('*.json'))]
    
    if len(video_list) == 0:
        raise ValueError(f"No JSON files found in {filtered_dir}")
    
    print(f"Found {len(video_list)} videos to process")
    
    # Prepare arguments for parallel processing
    func_args_list = [
        {
            'video_name': video_name,
            'keyframes_dir': keyframes_dir,
            'filtered_dir': filtered_dir,
            'shots_dir': shots_dir,
            'scene_threshold': args.scene_threshold
        }
        for video_name in video_list
    ]
    
    # Process videos in parallel
    frame_counts = parallel_foreach(process_video, func_args_list)
    total_frames = sum(frame_counts)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total videos processed: {len(video_list)}")
    print(f"Total frames processed: {total_frames}")
    print(f"Output directory: {shots_dir}")


if __name__ == '__main__':
    main()
