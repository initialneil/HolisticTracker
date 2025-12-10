#!/usr/bin/env python3
"""
infer_ehmx_flame.py - Run FLAME refiner on track-based processed results
Loads base tracking results and refines FLAME parameters.
"""
import os
import sys
import json
import argparse
import pickle
from pathlib import Path
import torch
import numpy as np
import cv2
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image as PILImage

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.configs.data_prepare_config import DataPreparationConfig
from src.ehmx_refine_flame import RefineFlamePipeline
from src.utils.io import load_dict_pkl, write_dict_pkl
from src.utils.draw import draw_landmarks


def load_image(image_path):
    """Load image from disk as RGB numpy array."""
    return np.array(PILImage.open(image_path).convert('RGB'))


def load_matte(matte_path):
    """Load single-channel matte image from disk as alpha mask."""
    if not os.path.exists(matte_path):
        return None
    matte = np.array(PILImage.open(matte_path).convert('L')).astype(np.float32) / 255.0
    return matte


def apply_matte_to_image(img_rgb, matte, background_color=1.0):
    """Apply matte alpha mask to RGB image with white background."""
    img_float = img_rgb.astype(np.float32) / 255.0
    matte_3ch = matte[:, :, None]
    matted = img_float * matte_3ch + background_color * (1 - matte_3ch)
    return (np.clip(matted, 0, 1) * 255).round().astype(np.uint8)


def load_track_results(track_pkl_path):
    """Load base tracking results from pickle file."""
    if not os.path.exists(track_pkl_path):
        raise FileNotFoundError(f"Track results not found: {track_pkl_path}")
    return load_dict_pkl(track_pkl_path)


def load_id_share_params(id_share_pkl_path):
    """Load identity-shared parameters from pickle file."""
    if os.path.exists(id_share_pkl_path):
        return load_dict_pkl(id_share_pkl_path)
    else:
        # Initialize empty id_share_params if not exists
        return {
            'flame_shape': [],
            'smplx_shape': [],
            'left_mano_shape': [],
            'right_mano_shape': []
        }


def load_head_images(video_name, frames_keys, base_results, images_dir, mattes_dir=None):
    """
    Load and crop head images for all frames.
    
    Args:
        video_name: Name of the video
        frames_keys: List of frame keys
        base_results: Base tracking results with crop info
        images_dir: Directory containing original images
        mattes_dir: Optional directory containing matte images
    
    Returns:
        Dictionary mapping frame_key to cropped head image tensors
    """
    from src.utils.crop import _transform_img
    
    head_images = {}
    for frame_key in tqdm(frames_keys, desc="Loading head images"):
        if frame_key not in base_results:
            continue
        
        # Load original image
        img_path = os.path.join(images_dir, video_name, f"{frame_key}.jpg")
        if not os.path.exists(img_path):
            continue
        
        img_rgb = load_image(img_path)
        
        # Apply matte if available
        if mattes_dir:
            matte_path = os.path.join(mattes_dir, video_name, f"{frame_key}.png")
            matte = load_matte(matte_path)
            if matte is not None:
                img_rgb = apply_matte_to_image(img_rgb, matte)
        
        # Crop head image using saved M_o2c transformation matrix
        head_crop_info = base_results[frame_key]['head_crop']
        M_o2c = head_crop_info['M_o2c']
        crop_size = 512  # Head crop size is always 512
        
        # Apply affine transformation
        head_img_cropped = _transform_img(img_rgb, M_o2c, dsize=crop_size)
        
        # Convert to tensor (C, H, W)
        head_img_tensor = torch.from_numpy(head_img_cropped).permute(2, 0, 1).float()
        head_images[frame_key] = head_img_tensor
    
    return head_images


def generate_preview(video_name, frames_keys, base_results, refined_results, head_images, out_dir):
    """
    Generate preview visualization grid with all frames in 2:1 aspect ratio layout.
    
    Args:
        video_name: Name of the video
        frames_keys: List of frame keys
        base_results: Base tracking results
        refined_results: Refined tracking results
        head_images: Dictionary of head image tensors
        out_dir: Output directory
    """
    video_out_dir = os.path.join(out_dir, video_name)
    preview_path = os.path.join(video_out_dir, 'track_flame.jpg')
    
    print(f"  Generating preview grid...")
    
    # Collect all valid frames
    preview_images = []
    for frame_key in frames_keys:
        if frame_key not in head_images or frame_key not in refined_results:
            continue
        
        # Get head image
        head_img = head_images[frame_key].numpy().transpose(1, 2, 0).astype(np.uint8).copy()
        
        # Draw base landmarks (green)
        if 'head_lmk_203' in base_results[frame_key]:
            base_lmks = base_results[frame_key]['head_lmk_203']
            head_img = draw_landmarks(base_lmks, head_img, color=(0, 255, 0), radius=1)
        
        # Draw refined landmarks (red) - would need to project from refined flame params
        # For now just show the image with base landmarks
        
        preview_images.append(head_img)
    
    if len(preview_images) == 0:
        print(f"  Warning: No valid frames for preview")
        return
    
    # Create 2:1 aspect ratio grid layout
    # Calculate grid dimensions: width should be 2x height
    total_frames = len(preview_images)
    
    # Find optimal grid size with 2:1 aspect ratio
    # For n frames, we want grid_cols * grid_rows >= n and grid_cols ~= 2 * grid_rows
    grid_rows = max(1, int(np.sqrt(total_frames / 2)))
    grid_cols = int(np.ceil(total_frames / grid_rows))
    
    # Adjust to maintain 2:1 ratio
    while grid_cols < 2 * grid_rows and grid_rows > 1:
        grid_rows -= 1
        grid_cols = int(np.ceil(total_frames / grid_rows))
    
    # Get image dimensions (assuming all images are same size)
    img_height, img_width = preview_images[0].shape[:2]
    
    # Create blank grid
    grid_height = grid_rows * img_height
    grid_width = grid_cols * img_width
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Place images in grid
    for idx, img in enumerate(preview_images):
        row = idx // grid_cols
        col = idx % grid_cols
        
        y_start = row * img_height
        y_end = y_start + img_height
        x_start = col * img_width
        x_end = x_start + img_width
        
        grid[y_start:y_end, x_start:x_end] = img
    
    # Save grid
    cv2.imwrite(preview_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  Saved preview grid ({grid_rows}x{grid_cols}, {total_frames} frames) to: {preview_path}")


def process_video_flame(video_name, video_data, args, flame_pipeline, cfg, optim_cfg):
    """
    Run FLAME refinement for a single video.
    
    Args:
        video_name: Name of the video
        video_data: Dictionary with 'frames_num' and 'frames_keys'
        args: Command line arguments
        flame_pipeline: RefineFlamePipeline instance
        cfg: DataPreparationConfig instance
        optim_cfg: Optimization config from YAML
    
    Returns:
        Tuple of (refined_results, id_share_params)
    """
    frames_keys = video_data['frames_keys']
    frames_num = video_data['frames_num']
    
    print(f"\nProcessing video: {video_name}")
    print(f"  Frames: {frames_num}")
    
    # Define paths
    video_out_dir = os.path.join(args.ehmx_dir, video_name)
    track_pkl_path = os.path.join(video_out_dir, 'base_tracking.pkl')
    id_share_pkl_path = os.path.join(video_out_dir, 'id_share_params.pkl')
    refined_pkl_path = os.path.join(video_out_dir, 'track_flame.pkl')
    preview_path = os.path.join(video_out_dir, 'track_flame.jpg')
    
    # Check if already processed
    if os.path.exists(refined_pkl_path) and not args.overwrite:
        # If preview doesn't exist, generate it
        if not os.path.exists(preview_path):
            print(f"  Loading existing results to generate preview...")
            refined_results = load_dict_pkl(refined_pkl_path)
            id_share_params = load_id_share_params(id_share_pkl_path)
            base_results = load_track_results(track_pkl_path)
            
            # Load head images
            head_images = load_head_images(video_name, frames_keys, base_results, 
                                         args.images_dir, args.mattes_dir)
            
            # Generate preview
            generate_preview(video_name, frames_keys, base_results, refined_results, 
                           head_images, args.ehmx_dir)
            
            return refined_results, id_share_params
        else:
            print(f"  Skipping: Already processed (use --overwrite to reprocess)")
            refined_results = load_dict_pkl(refined_pkl_path)
            id_share_params = load_id_share_params(id_share_pkl_path)
            return refined_results, id_share_params
    
    # Load base tracking results
    print(f"  Loading track results from: {track_pkl_path}")
    base_results = load_track_results(track_pkl_path)
    
    # Load identity-shared parameters
    id_share_params = load_id_share_params(id_share_pkl_path)
    
    # Load head images from original images
    print(f"  Loading head images from: {args.images_dir}")
    head_images = load_head_images(video_name, frames_keys, base_results, 
                                   args.images_dir, args.mattes_dir)
    
    # Detect frame interval from frame naming
    frame_interval = 1
    if len(frames_keys) > 1:
        try:
            # Try to parse frame numbers - handle formats like "000032" or "000032/000000"
            frame_nums = []
            for k in frames_keys:
                # Get last part after underscore, then take first part before any slash
                last_part = k.split('_')[-1].split('/')[0]
                frame_nums.append(int(last_part))
            
            if len(frame_nums) > 1:
                diffs = np.diff(frame_nums)
                if len(set(diffs)) == 1:
                    frame_interval = int(diffs[0])
        except (ValueError, IndexError):
            # If parsing fails, use default interval of 1
            frame_interval = 1
    
    print(f"  Frame interval: {frame_interval}")
    
    # Set saving root for flame pipeline
    flame_pipeline.saving_root = video_out_dir
    
    print(f"  Running FLAME optimization...")
    print(f"    Steps: {optim_cfg.steps}")
    print(f"    Batch size: {optim_cfg.mini_batch_size}")
    
    try:
        # Run FLAME refinement
        opt_flame_coeff, id_share_params = flame_pipeline.run(
            base_results,
            id_share_params,
            optim_cfg,
            head_images,
            frame_interval
        )
        
        # Update results with refined FLAME coefficients
        refined_results = {k: v.copy() for k, v in base_results.items()}
        for key in refined_results.keys():
            if key in opt_flame_coeff:
                refined_results[key]['flame_coeffs'] = opt_flame_coeff[key]
        
        # Save refined results
        print(f"  Saving refined results to: {refined_pkl_path}")
        write_dict_pkl(refined_pkl_path, refined_results)
        
        # Update and save id_share_params
        write_dict_pkl(id_share_pkl_path, id_share_params)
        
        # Generate preview
        generate_preview(video_name, frames_keys, base_results, refined_results, 
                        head_images, args.ehmx_dir)
        
        print(f"  ✓ FLAME refinement completed for {video_name}")
        
        return refined_results, id_share_params
        
    except Exception as e:
        print(f"  ✗ FLAME refinement failed: {e}")
        import traceback
        traceback.print_exc()
        return None, id_share_params


def main():
    parser = argparse.ArgumentParser(description='Run FLAME refiner on track-based processed results')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Root directory containing original images')
    parser.add_argument('--index_json', type=str, required=True,
                        help='JSON file with videos and frames to process')
    parser.add_argument('--mattes_dir', type=str, default=None,
                        help='Directory containing matte images (optional)')
    parser.add_argument('--ehmx_dir', type=str, required=True,
                        help='Directory for EHM-X tracking results')
    parser.add_argument('--config', type=str,
                        default='src/configs/optim_configs/ehm_fbody_ytb.yaml',
                        help='Path to optimization config file')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing refined results')
    
    args = parser.parse_args()
    
    # Load index JSON
    print(f"Loading index from: {args.index_json}")
    with open(args.index_json, 'r') as f:
        videos_info = json.load(f)
    
    print(f"Found {len(videos_info)} video(s) to process")
    
    # Load optimization config using OmegaConf
    print(f"Loading optimization config from: {args.config}")
    if not os.path.exists(args.config):
        print(f"Error: Optimization config not found: {args.config}")
        return
    
    full_optim_config = OmegaConf.load(args.config)
    optim_cfg = full_optim_config.optim_flame
    
    print(f"Optimization config loaded:")
    print(f"  Steps: {optim_cfg.steps}")
    print(f"  Batch size: {optim_cfg.mini_batch_size}")
    print(f"  Share ID: {optim_cfg.share_id}")
    
    # Load data preparation config
    print(f"\nInitializing FLAME refinement pipeline...")
    cfg = DataPreparationConfig()
    
    # Initialize FLAME refinement pipeline
    flame_pipeline = RefineFlamePipeline(cfg)
    
    print(f"FLAME pipeline initialized on {cfg.device}")
    
    # Process each video
    for video_idx, (video_name, video_data) in enumerate(videos_info.items(), 1):
        print(f"\n[{video_idx}/{len(videos_info)}] Processing: {video_name}")
        
        try:
            refined_results, id_share_params = process_video_flame(
                video_name, video_data, args, flame_pipeline, cfg, optim_cfg
            )
            
            if refined_results is None:
                print(f"  Skipping {video_name} due to error")
                continue
                
        except Exception as e:
            print(f"  Error processing {video_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("FLAME refinement complete!")
    print("="*80)


if __name__ == '__main__':
    main()
