#!/usr/bin/env python3
"""
Apply RMBG-2.0 background removal to extracted shot images.
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
from modelscope import AutoModelForImageSegmentation
import torchvision.transforms as transforms
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader

# Add EHM-Tracker to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_rmbg2_model(model_name):
    """
    Load the RMBG-2.0 model from HuggingFace.
    
    Args:
        model_name: Model name or path
    
    Returns:
        Loaded model
    """
    print(f"Loading RMBG-2.0 model from: {model_name}")
    model = AutoModelForImageSegmentation.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model.eval()
    print(f"RMBG-2.0 model loaded successfully")
    return model


def process_image(img_path, model, image_size=1024):
    """
    Apply RMBG-2.0 background removal to a single image.
    
    Args:
        img_path: Path to input image
        model: RMBG-2.0 model
        image_size: Image size for processing
    
    Returns:
        Alpha matte as numpy array (H, W) in range [0, 255]
    """
    # Load image with PIL
    image = Image.open(str(img_path)).convert('RGB')
    if image is None:
        return None
    
    # Get original size
    orig_size = image.size
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Transform image
    input_tensor = transform(image).unsqueeze(0).to(model.device)
    
    # Run inference
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    
    # Get the mask
    pred = preds[0].squeeze()
    
    # Resize mask back to original size
    mask_pil = transforms.ToPILImage()(pred)
    mask_pil = mask_pil.resize(orig_size, Image.BILINEAR)
    
    # Convert to numpy array
    matte = np.array(mask_pil)
    
    return matte


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


def process_shot(shot_dir, matte_output_dir, model, fps=10):
    """
    Process all images in a shot directory.
    
    Args:
        shot_dir: Directory containing shot images
        matte_output_dir: Directory to save alpha mattes
        video_output_dir: Directory to save preview video
        model: RMBG-2.0 model
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
        alpha = process_image(img_file, model)
        
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


def process_video(video_name, images_dir, mattes_dir, model):
    """
    Process all shots in a video directory.
    
    Args:
        video_name: Name of the video
        images_dir: Root directory containing video folders
        mattes_dir: Root directory for matte output
        model: RMBG-2.0 model
    
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
                                     model, fps)
        
        if num_processed > 0:
            print(f"  {shot_name}: {num_processed} images")
            shots_processed += 1
            total_images += num_processed
    
    print(f"  Total: {shots_processed} shot(s), {total_images} images")
    return shots_processed


class VideoDataset(Dataset):
    """Simple dataset for video folder names."""
    def __init__(self, video_folders, images_dir, mattes_dir):
        self.video_folders = video_folders
        self.images_dir = images_dir
        self.mattes_dir = mattes_dir
    
    def __len__(self):
        return len(self.video_folders)
    
    def __getitem__(self, idx):
        return {
            'video_name': self.video_folders[idx],
            'images_dir': self.images_dir,
            'mattes_dir': self.mattes_dir
        }


def main():
    parser = argparse.ArgumentParser(description='Apply RMBG-2.0 background removal to shot images')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing video folders with shot images')
    parser.add_argument('--mattes_dir', type=str, required=True,
                        help='Output directory for alpha mattes')
    parser.add_argument('--model_name', type=str, default='AI-ModelScope/RMBG-2.0',
                        help='ModelScope model name for RMBG-2.0')
    
    args = parser.parse_args()
    
    # Setup
    images_dir = Path(args.images_dir)
    mattes_dir = Path(args.mattes_dir)
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    
    mattes_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load RMBG-2.0 model
    model = load_rmbg2_model(args.model_name)
    
    # Prepare model with accelerator
    model = accelerator.prepare(model)
    
    # Get list of video folders
    video_folders = [d.name for d in sorted(images_dir.iterdir()) 
                     if d.is_dir()]
    
    if len(video_folders) == 0:
        raise ValueError(f"No video folders found in {images_dir}")
    
    if accelerator.is_main_process:
        print(f"Found {len(video_folders)} video folder(s)")
        print(f"Output directory: {mattes_dir}")
        print(f"Using {accelerator.num_processes} process(es)")
    
    # Create dataset and dataloader
    dataset = VideoDataset(video_folders, str(images_dir), str(mattes_dir))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    dataloader = accelerator.prepare(dataloader)
    
    # Process each video in parallel
    total_shots = 0
    for batch in dataloader:
        video_name = batch['video_name'][0]
        batch_images_dir = batch['images_dir'][0]
        batch_mattes_dir = batch['mattes_dir'][0]
        
        num_shots = process_video(video_name, batch_images_dir, batch_mattes_dir, model)
        total_shots += num_shots
    
    # Wait for all processes
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Total videos processed: {len(video_folders)}")
        print(f"Total shots processed: {total_shots}")
        print(f"Output directory: {mattes_dir}")


if __name__ == '__main__':
    main()
