#!/usr/bin/env python3
"""
Apply SAM3 segmentation to extracted shot images.
Processes all images in shot directories and saves alpha masks.
Creates alpha-blended preview videos with white background.
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image

# Add EHM-Tracker to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def load_sam3_model(device):
    """
    Load the SAM3 model.
    
    Args:
        device: Device to load model on
    
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading SAM3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print(f"SAM3 model loaded successfully")
    return model, processor


def process_image(img_path, processor, text_prompt="The speechmaker."):
    """
    Apply SAM3 segmentation to a single image.
    
    Args:
        img_path: Path to input image
        processor: SAM3 processor
        text_prompt: Text prompt for segmentation
    
    Returns:
        Alpha matte as numpy array (H, W) in range [0, 255]
    """
    # Load image with PIL
    image = Image.open(str(img_path))
    if image is None:
        return None
    
    # Get image dimensions
    img_w, img_h = image.size
    
    # Run SAM3 segmentation
    with torch.no_grad():
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
    
    # Get the masks, bounding boxes, and scores
    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]
    
    # If no masks found, return None
    if masks is None or len(masks) == 0:
        return None
    
    # Use the mask with highest score
    best_mask = masks[0]  # Assuming masks are sorted by score
    
    # Convert mask to numpy array (H, W) in range [0, 255]
    if torch.is_tensor(best_mask):
        alpha_np = best_mask.cpu().numpy()
    else:
        alpha_np = np.array(best_mask)
    
    # Ensure mask is 2D
    if alpha_np.ndim == 3:
        alpha_np = alpha_np[0]  # Take first channel if multi-channel
    
    # Convert boolean mask to uint8
    if alpha_np.dtype == bool:
        alpha_np = alpha_np.astype(np.uint8) * 255
    else:
        alpha_np = np.clip(alpha_np, 0, 1)
        alpha_np = (alpha_np * 255).round().astype(np.uint8)
    
    return alpha_np


def create_alpha_blended_video(shot_dir, mattes_dir, output_video_path, fps=10):
    """
    Create alpha-blended video with white background from shot images and mattes.
    
    Args:
        shot_dir: Directory containing original shot images
        mattes_dir: Directory containing corresponding alpha mattes
        output_video_path: Path to save output MP4 video
        fps: Frame rate for output video
    
    Returns:
        True if successful, False otherwise
    """
    # Get list of images
    image_files = sorted(shot_dir.glob('*.jpg'))
    if len(image_files) == 0:
        return False
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        return False
    
    h, w = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))
    
    if not video_writer.isOpened():
        return False
    
    # Process each frame
    for img_file in image_files:
        # Load original image
        img_bgr = cv2.imread(str(img_file))
        if img_bgr is None:
            continue
        
        # Load corresponding matte
        matte_file = mattes_dir / img_file.name.replace('.jpg', '.png')
        if not matte_file.exists():
            # If matte doesn't exist, use original image
            video_writer.write(img_bgr)
            continue
        
        alpha = cv2.imread(str(matte_file), cv2.IMREAD_GRAYSCALE)
        if alpha is None:
            video_writer.write(img_bgr)
            continue
        
        # Normalize alpha to [0, 1]
        alpha_norm = alpha.astype(np.float32) / 255.0
        alpha_3ch = alpha_norm[:, :, np.newaxis]
        
        # Create white background
        white_bg = np.ones_like(img_bgr, dtype=np.float32) * 255
        
        # Alpha blend: result = foreground * alpha + background * (1 - alpha)
        img_float = img_bgr.astype(np.float32)
        blended = img_float * alpha_3ch + white_bg * (1 - alpha_3ch)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        video_writer.write(blended)
    
    video_writer.release()
    return True


def process_shot(shot_dir, matte_output_dir, processor, fps=10):
    """
    Process all images in a shot directory.
    
    Args:
        shot_dir: Directory containing shot images
        matte_output_dir: Directory to save alpha mattes
        video_output_dir: Directory to save preview video
        processor: SAM3 processor
        fps: Frame rate for output video
    
    Returns:
        Number of images processed
    """
    # Create output directory
    matte_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all jpg images in shot directory
    image_files = sorted(shot_dir.glob('*.jpg'))
    
    if len(image_files) == 0:
        return 0
    
    # Process each image
    processed_count = 0
    for img_file in tqdm(image_files, desc=f"Processing {shot_dir.name}", leave=False):
        # Output matte path
        matte_path = matte_output_dir / img_file.name.replace('.jpg', '.png')
        
        # Skip if already processed
        if matte_path.exists():
            processed_count += 1
            continue
        
        # Process image
        alpha = process_image(img_file, processor)
        
        if alpha is not None:
            # Save single-channel PNG
            cv2.imwrite(str(matte_path), alpha)
            processed_count += 1
    
    # Create alpha-blended video
    shot_name = shot_dir.name
    video_path = matte_output_dir.parent / f"{shot_name}.mp4"
    
    if not video_path.exists():
        success = create_alpha_blended_video(shot_dir, matte_output_dir, video_path, fps)
        if success:
            print(f"    Created video: {video_path.name}")
    
    return processed_count


def process_video(video_name, images_dir, mattes_dir, processor):
    """
    Process all shots in a video directory.
    
    Args:
        video_name: Name of the video
        images_dir: Root directory containing video folders
        mattes_dir: Root directory for matte output
        processor: SAM3 processor
    
    Returns:
        Number of shots processed
    """
    video_dir = Path(images_dir) / video_name
    
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        return 0
    
    # Load FPS from video JSON metadata
    video_json = video_dir / f"{video_name}.json"
    fps = 10  # Default fallback
    if video_json.exists():
        try:
            import json
            with open(video_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            video_data = data.get(video_name, {})
            fps = video_data.get('fps', 10)
            if fps is None:
                fps = 10
            print(f"  Loaded FPS from JSON: {fps}")
        except Exception as e:
            print(f"  Warning: Could not load FPS from JSON, using default (10): {e}")
    else:
        print(f"  Warning: JSON not found, using default FPS (10)")
    
    # Create output directories
    matte_video_dir = Path(mattes_dir) / video_name
    matte_video_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all shot directories (subdirectories with naming pattern like 000010_000050)
    shot_dirs = [d for d in sorted(video_dir.iterdir()) 
                 if d.is_dir() and '_' in d.name]
    
    if len(shot_dirs) == 0:
        print(f"Warning: No shot directories found in {video_dir}")
        return 0
    
    print(f"\nProcessing {video_name}: {len(shot_dirs)} shot(s)")
    
    shots_processed = 0
    total_images = 0
    
    for shot_dir in shot_dirs:
        shot_name = shot_dir.name
        matte_output_dir = matte_video_dir / shot_name
        
        # Count images
        num_images = len(list(shot_dir.glob('*.jpg')))
        
        if num_images == 0:
            continue
        
        # Process shot
        num_processed = process_shot(shot_dir, matte_output_dir, 
                                     processor, fps)
        
        if num_processed > 0:
            print(f"  {shot_name}: {num_processed} images")
            shots_processed += 1
            total_images += num_processed
    
    print(f"  Total: {shots_processed} shot(s), {total_images} images")
    return shots_processed


def main():
    parser = argparse.ArgumentParser(description='Apply SAM3 segmentation to shot images')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing video folders with shot images')
    parser.add_argument('--mattes_dir', type=str, required=True,
                        help='Output directory for alpha mattes')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup
    images_dir = Path(args.images_dir)
    mattes_dir = Path(args.mattes_dir)
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    
    mattes_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SAM3 model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, processor = load_sam3_model(device)
    
    # Get list of video folders
    video_folders = [d.name for d in sorted(images_dir.iterdir()) 
                     if d.is_dir()]
    
    if len(video_folders) == 0:
        raise ValueError(f"No video folders found in {images_dir}")
    
    print(f"Found {len(video_folders)} video folder(s)")
    print(f"Output directory: {mattes_dir}")
    
    # Process each video
    total_shots = 0
    for video_name in video_folders:
        num_shots = process_video(video_name, images_dir, mattes_dir, 
                                  processor)
        total_shots += num_shots
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total videos processed: {len(video_folders)}")
    print(f"Total shots processed: {total_shots}")
    print(f"Output directory: {mattes_dir}")


if __name__ == '__main__':
    main()
