#!/usr/bin/env python3
"""
Filter keyframes based on DWPose detection and face recognition.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Add EHM-Tracker to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.io import load_config
from src.utils.helper import instantiate_from_config
from src.modules.dwpose import inference_detector

import torch
import torchvision.transforms as transforms


def create_dwpose_detector(dwpose_cfg_path):
    """Create and warm up DWPose detector."""
    print("Initializing DWPose detector...")
    dwpose_detector = instantiate_from_config(load_config(dwpose_cfg_path))
    dwpose_detector.warmup()
    print("DWPose detector ready")
    return dwpose_detector


def create_dinov2_model(device):
    """Create DINOv2 model for feature extraction."""
    print("Loading DINOv2 model...")
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
    dino_model = dino_model.to(device).eval()
    dino_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    print("DINOv2 model ready")
    return dino_model, dino_normalize


def extract_dino_feature(image, bbox, dino_model, dino_normalize, device):
    """Extract DINOv2 global feature from cropped bbox region."""
    x1, y1, x2, y2 = [int(b) for b in bbox]
    h, w = image.shape[:2]
    
    # Clamp bbox to image boundaries
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Crop region
    cropped = image[y1:y2, x1:x2]
    
    # Convert to RGB and resize to 224x224
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    cropped_pil = Image.fromarray(cropped_rgb)
    cropped_resized = cropped_pil.resize((224, 224), Image.BILINEAR)
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(np.array(cropped_resized)).permute(2, 0, 1).float() / 255.0
    img_tensor = dino_normalize(img_tensor).unsqueeze(0).to(device)
    
    # Extract global feature
    with torch.no_grad():
        features = dino_model.get_intermediate_layers(img_tensor, 1)
        global_feature = features[0][:, 0]  # CLS token
    
    return global_feature


def check_body_legs_feet(det_info):
    """Check if body has visible legs and feet."""
    if det_info['bodies'] is None:
        return False
    
    bodies = det_info['bodies']
    subset = bodies['subset']
    
    # Check if subset has valid entries (not all -1)
    if isinstance(subset, np.ndarray):
        if subset.ndim == 0 or (subset.ndim == 1 and len(subset) == 0):
            return False
        # Body keypoints: indices 11-16 are legs (11,12: hips, 13,14: knees, 15,16: ankles)
        # Check if at least some leg keypoints are visible
        leg_indices = [11, 12, 13, 14, 15, 16]
        visible_legs = sum([1 for idx in leg_indices if subset[idx] >= 0])
        if visible_legs < 3:  # At least 3 leg keypoints should be visible
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

def process_video(video_name, keyframes_dir, oneframe_dwpose, filtered_dir, 
                  dwpose_detector, dino_model, dino_normalize, device, min_boundary_margin, 
                  similarity_threshold):
    """Process one video and filter frames based on DINOv2 similarity."""
    
    # Output paths
    output_json = Path(filtered_dir) / f"{video_name}.json"
    output_vis = Path(filtered_dir) / f"{video_name}.jpg"
    
    # # Skip if already processed
    # if output_json.exists():
    #     print(f"Skipping {video_name}: Already processed")
    #     return 0
    
    # Load reference image
    reference_img_path = Path(oneframe_dwpose) / f"{video_name}.jpg"
    if not reference_img_path.exists():
        print(f"Error: Reference image not found for {video_name}")
        return 0
    
    reference_img = cv2.imread(str(reference_img_path))
    
    # Get reference bbox using DWPose
    ref_bboxes = inference_detector(dwpose_detector.pose_estimation.session_det, reference_img)
    if len(ref_bboxes) == 0:
        print(f"Error: No bbox detected in reference image for {video_name}")
        return 0
    
    # Use largest bbox as reference
    ref_bbox_areas = [(bbox, get_bbox_area(bbox)) for bbox in ref_bboxes]
    ref_bbox_areas.sort(key=lambda x: x[1], reverse=True)
    ref_bbox = ref_bbox_areas[0][0]
    
    # Extract reference DINOv2 feature
    ref_feature = extract_dino_feature(reference_img, ref_bbox, dino_model, dino_normalize, device)
    if ref_feature is None:
        print(f"Error: Cannot extract DINOv2 feature from reference image for {video_name}")
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
            x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
            cropped = frame[y1_int:y2_int, x1_int:x2_int]
            
            # Run full dwpose detector on cropped frame
            det_info, det_raw_info = dwpose_detector(cropped)
            
            if not check_body_legs_feet(det_info):
                continue
            
            # Extend bbox to square with scale 1.1
            square_bbox = extend_bbox_to_square(largest_bbox, scale=1.1)
            
            # Criteria 6: Extract DINOv2 feature and calculate similarity
            curr_feature = extract_dino_feature(frame, largest_bbox, dino_model, dino_normalize, device)
            if curr_feature is None:
                continue
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(curr_feature, ref_feature).item()
            
            if similarity >= similarity_threshold:
                # Valid frame - store result
                valid_frames[frame_idx] = {
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "square_bbox_xyxy": [float(b) for b in square_bbox],
                    "dino_similarity": float(similarity)
                }
            
            # Create visualization using square bbox
            cropped_square = crop_with_padding(frame, square_bbox)
            vis_img = cv2.resize(cropped_square, (128, 128))
            # Add text with color based on threshold
            text_color = (0, 255, 0) if similarity >= similarity_threshold else (0, 0, 255)  # Green if pass, red if fail
            cv2.putText(vis_img, frame_idx, (5, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(vis_img, f"sim:{similarity:.3f}", (5, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
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
    
    parser = argparse.ArgumentParser(description='Filter keyframes using DINOv2 similarity')
    parser.add_argument('--keyframes_dir', type=str, required=True,
                        help='Directory containing frame folders')
    parser.add_argument('--oneframe_dwpose', type=str, required=True,
                        help='Directory with reference images')
    parser.add_argument('--filtered_dir', type=str, required=True,
                        help='Output directory for filtered results')
    parser.add_argument('--dwpose_cfg', type=str,
                        default='src/configs/model_configs/dwpose_onnx_config.yaml',
                        help='Path to DWPose config file')
    parser.add_argument('--min_boundary_margin', type=int, default=10,
                        help='Minimum pixels from image boundary (default: 10)')
    parser.add_argument('--similarity_threshold', type=float, default=0.7,
                        help='DINOv2 similarity threshold (default: 0.7)')
    
    args = parser.parse_args()
    
    # Setup
    keyframes_dir = Path(args.keyframes_dir)
    oneframe_dwpose = Path(args.oneframe_dwpose)
    filtered_dir = Path(args.filtered_dir)
    
    if not keyframes_dir.exists():
        raise ValueError(f"Input directory {keyframes_dir} does not exist")
    if not oneframe_dwpose.exists():
        raise ValueError(f"Reference directory {oneframe_dwpose} does not exist")
    
    filtered_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video list from reference faces
    video_list = [f.stem for f in sorted(oneframe_dwpose.glob('*.jpg'))]
    
    if len(video_list) == 0:
        raise ValueError(f"No reference faces found in {oneframe_dwpose}")
    
    print(f"Found {len(video_list)} videos to process")
    
    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dwpose_detector = create_dwpose_detector(args.dwpose_cfg)
    
    dino_model, dino_normalize = create_dinov2_model(device)
    
    # Process each video
    total_valid = 0
    for video_name in video_list:
        num_valid = process_video(
            video_name, keyframes_dir, oneframe_dwpose, filtered_dir,
            dwpose_detector, dino_model, dino_normalize, device,
            args.min_boundary_margin, args.similarity_threshold
        )
        total_valid += num_valid
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total videos processed: {len(video_list)}")
    print(f"Total valid frames: {total_valid}")
    print(f"Output directory: {filtered_dir}")


if __name__ == '__main__':
    main()

