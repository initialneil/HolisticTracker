#!/usr/bin/env python3
"""
Apply RobustVideoMatting (RVM) to extracted shot images.
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
import onnxruntime

# Add EHM-Tracker to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_rvm_model(model_path):
    """
    Load the RVM ONNX model.
    
    Args:
        model_path: Path to ONNX model file
    
    Returns:
        ONNX runtime inference session
    """
    print(f"Loading RVM ONNX model from: {model_path}")
    session = onnxruntime.InferenceSession(model_path)
    print(f"RVM model loaded successfully")
    return session


def process_image(img_path, session, rec, downsample_ratio=0.25):
    """
    Apply RVM matting to a single image.
    
    Args:
        img_path: Path to input image
        session: ONNX runtime inference session
        rec: Recurrent states [r1, r2, r3, r4]
        downsample_ratio: Downsample ratio for RVM
    
    Returns:
        Tuple of (alpha matte, updated recurrent states)
        Alpha matte as numpy array (H, W) in range [0, 255]
    """
    # Read image
    im = cv2.imread(str(img_path))
    if im is None:
        return None, rec
    
    # Convert BGR to RGB and normalize to [0, 1]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(np.float32) / 255.0
    
    # Get original dimensions
    im_h, im_w, _ = im.shape
    
    # Prepare input: [B, C, H, W]
    src = np.transpose(im, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
    
    # Prepare downsample_ratio input
    downsample_ratio_input = np.array([downsample_ratio], dtype=np.float32)
    
    # Run inference with recurrent states
    outputs = session.run([], {
        'src': src,
        'r1i': rec[0],
        'r2i': rec[1],
        'r3i': rec[2],
        'r4i': rec[3],
        'downsample_ratio': downsample_ratio_input
    })
    
    # Extract outputs: [fgr, pha, r1o, r2o, r3o, r4o]
    fgr, pha, r1o, r2o, r3o, r4o = outputs
    
    # Update recurrent states
    new_rec = [r1o, r2o, r3o, r4o]
    
    # Convert alpha to uint8
    matte = (np.squeeze(pha) * 255).astype(np.uint8)
    
    return matte, new_rec


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


def process_shot(shot_dir, matte_output_dir, session, fps=10):
    """
    Process all images in a shot directory.
    
    Args:
        shot_dir: Directory containing shot images
        matte_output_dir: Directory to save alpha mattes
        video_output_dir: Directory to save preview video
        session: RVM ONNX session
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
    
    # Initialize recurrent states to zero tensors
    rec = [np.zeros([1, 1, 1, 1], dtype=np.float32) for _ in range(4)]
    
    # Calculate downsample_ratio based on image size
    # Read first image to get dimensions
    first_img = cv2.imread(str(image_files[0]))
    if first_img is not None:
        h, w = first_img.shape[:2]
        # Set downsample_ratio so downsampled resolution is between 256 and 512
        max_dim = max(h, w)
        if max_dim <= 512:
            downsample_ratio = 1.0
        elif max_dim <= 1024:
            downsample_ratio = 0.5
        elif max_dim <= 1536:
            downsample_ratio = 0.375
        else:
            downsample_ratio = 0.25
    else:
        downsample_ratio = 0.25
    
    # Process each image
    processed_count = 0
    for img_file in tqdm(image_files, desc=f"Processing {shot_dir.name}", leave=False):
        # Output matte path
        matte_path = matte_output_dir / img_file.name.replace('.jpg', '.png')
        
        # Skip if already processed
        if matte_path.exists():
            processed_count += 1
            continue
        
        # Process image with recurrent states
        alpha, rec = process_image(img_file, session, rec, downsample_ratio)
        
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


def process_video(video_name, images_dir, mattes_dir, session):
    """
    Process all shots in a video directory.
    
    Args:
        video_name: Name of the video
        images_dir: Root directory containing video folders
        mattes_dir: Root directory for matte output
        session: RVM ONNX session
    
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
                                     session, fps)
        
        if num_processed > 0:
            print(f"  {shot_name}: {num_processed} images")
            shots_processed += 1
            total_images += num_processed
    
    print(f"  Total: {shots_processed} shot(s), {total_images} images")
    return shots_processed


def main():
    parser = argparse.ArgumentParser(description='Apply RVM matting to shot images')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing video folders with shot images')
    parser.add_argument('--mattes_dir', type=str, required=True,
                        help='Output directory for alpha mattes')
    parser.add_argument('--model_path', type=str, default='pretrained/matting/rvm_resnet50_fp32.onnx',
                        help='Path to RVM ONNX model')
    
    args = parser.parse_args()
    
    # Setup
    images_dir = Path(args.images_dir)
    mattes_dir = Path(args.mattes_dir)
    model_path = Path(args.model_path)
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    
    if not model_path.exists():
        raise ValueError(f"Model file not found: {model_path}")
    
    mattes_dir.mkdir(parents=True, exist_ok=True)
    
    # Load RVM model
    session = load_rvm_model(str(model_path))
    
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
                                  session)
        total_shots += num_shots
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total videos processed: {len(video_folders)}")
    print(f"Total shots processed: {total_shots}")
    print(f"Output directory: {mattes_dir}")


if __name__ == '__main__':
    main()
