#!/usr/bin/env python3
"""
ehmx1_track_base.py - Process frames using TrackBasePipeline
Loads frames from shots directory, processes them, and saves base tracking results.
"""
import os
import sys
import json
import argparse
import pickle
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image as PILImage

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ehmx_track_base import TrackBasePipeline
from src.configs.data_prepare_config import DataPreparationConfig


def load_image(image_path):
    """Load image from disk as RGB numpy array."""
    return np.array(PILImage.open(image_path).convert('RGB'))


def load_matte(matte_path):
    """Load single-channel matte image from disk as alpha mask."""
    if not os.path.exists(matte_path):
        return None
    # Load as grayscale and normalize to [0, 1]
    matte = np.array(PILImage.open(matte_path).convert('L')).astype(np.float32) / 255.0
    return matte


def apply_matte_to_image(img_rgb, matte, background_color=1.0):
    """Apply matte alpha mask to RGB image with white background.
    
    Args:
        img_rgb: RGB image (H, W, 3) with values in [0, 255]
        matte: Alpha mask (H, W) with values in [0, 1]
        background_color: Background color (0-1 range), default white (1.0)
    
    Returns:
        Matted RGB image (H, W, 3) with values in [0, 255]
    """
    img_float = img_rgb.astype(np.float32) / 255.0
    matte_3ch = matte[:, :, None]  # (H, W, 1)
    matted = img_float * matte_3ch + background_color * (1 - matte_3ch)
    return (np.clip(matted, 0, 1) * 255).round().astype(np.uint8)


def process_video(video_name, video_data, args, pipeline):
    """
    Process all frames for a single video.
    
    Args:
        video_name: Name of the video
        video_data: Dictionary with 'frames_num' and 'frames_keys'
        args: Command line arguments
        pipeline: TrackBasePipeline instance
    
    Returns:
        Tuple of (base_results, id_share_params, ret_images_dict)
    """
    frames_keys = video_data['frames_keys']
    frames_num = video_data['frames_num']
    
    print(f"\nProcessing video: {video_name}")
    print(f"  Frames: {frames_num}")
    
    base_results = {}
    id_share_params = {}
    ret_images_dict = {}
    
    # Accumulators for identity-shared parameters
    shape_accum = {
        'smplx_shape': [],
        'flame_shape': [],
        'left_mano_shape': [],
        'right_mano_shape': []
    }
    
    valid_count = 0
    
    for frame_key in tqdm(frames_keys, desc=f"Processing {video_name}"):
        # Construct image path
        img_path = os.path.join(args.images_dir, video_name, f"{frame_key}.jpg")
        
        if not os.path.exists(img_path):
            print(f"  Warning: Image not found: {img_path}")
            continue
        
        # Load image
        img_rgb = load_image(img_path)
        
        # Apply matte if available
        if args.mattes_dir:
            matte_path = os.path.join(args.mattes_dir, video_name, f"{frame_key}.png")
            matte = load_matte(matte_path)
            if matte is not None:
                img_rgb = apply_matte_to_image(img_rgb, matte)
        
        # Process frame
        ret_images, frame_base_results, mean_shape_results = pipeline.track_base(img_rgb)
        
        if ret_images is None or frame_base_results is None:
            print(f"  Warning: Failed to process frame: {frame_key}")
            continue
        
        # Store results
        base_results[frame_key] = frame_base_results
        ret_images_dict[frame_key] = ret_images
        
        # Accumulate shape parameters
        for key in shape_accum:
            if key in mean_shape_results:
                shape_accum[key].append(mean_shape_results[key])
        
        valid_count += 1
    
    print(f"  Successfully processed {valid_count}/{frames_num} frames")
    
    # Average shape parameters
    for key, values in shape_accum.items():
        if len(values) > 0:
            id_share_params[key] = np.mean(values, axis=0)
    
    return base_results, id_share_params, ret_images_dict


def save_results(video_name, base_results, id_share_params, out_dir):
    """
    Save base tracking results to pickle files.
    
    Args:
        video_name: Name of the video
        base_results: Dictionary of per-frame base tracking results
        id_share_params: Dictionary of identity-shared parameters
        out_dir: Output directory
    """
    video_out_dir = os.path.join(out_dir, video_name)
    os.makedirs(video_out_dir, exist_ok=True)
    
    base_track_path = os.path.join(video_out_dir, 'base_tracking.pkl')
    id_share_path = os.path.join(video_out_dir, 'id_share_params.pkl')
    
    # Save base tracking results
    with open(base_track_path, 'wb') as f:
        pickle.dump(base_results, f)
    
    # Save identity-shared parameters
    with open(id_share_path, 'wb') as f:
        pickle.dump(id_share_params, f)
    
    print(f"  Saved base tracking to: {base_track_path}")
    print(f"  Saved id share params to: {id_share_path}")


def generate_visualization(video_name, video_data, ret_images_dict, base_results, out_dir, pipeline):
    """
    Generate and save visualization grid.
    
    Args:
        video_name: Name of the video
        video_data: Dictionary with 'frames_keys'
        ret_images_dict: Dictionary mapping frame_key to ret_images
        base_results: Dictionary of base tracking results for landmark drawing
        out_dir: Output directory
        pipeline: TrackBasePipeline instance
    """
    video_out_dir = os.path.join(out_dir, video_name)
    vis_path = os.path.join(video_out_dir, 'track_base.jpg')
    
    print(f"  Generating visualization grid...")
    frames_keys = video_data['frames_keys']
    grid = pipeline.create_visualization_grid(ret_images_dict, frames_keys, base_results)
    
    cv2.imwrite(vis_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  Saved visualization to: {vis_path}")


def main():
    parser = argparse.ArgumentParser(description='Process frames using TrackBasePipeline')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Root directory containing shot images')
    parser.add_argument('--index_json', type=str, required=True,
                        help='JSON file with frames to process')
    parser.add_argument('--mattes_dir', type=str, default=None,
                        help='Directory containing single-channel matte images (optional)')
    parser.add_argument('--ehmx_dir', type=str, required=True,
                        help='Directory for EHM-X tracking results')
    
    args = parser.parse_args()
    
    # Load index JSON
    print(f"Loading index from: {args.index_json}")
    with open(args.index_json, 'r') as f:
        index_data = json.load(f)
    
    print(f"Found {len(index_data)} videos in index")
    
    # Initialize pipeline
    print("\nInitializing TrackBasePipeline...")
    config = DataPreparationConfig()
    pipeline = TrackBasePipeline(config)
    
    # Process each video
    for video_name, video_data in index_data.items():
        video_out_dir = os.path.join(args.ehmx_dir, video_name)
        base_track_path = os.path.join(video_out_dir, 'base_tracking.pkl')
        vis_path = os.path.join(video_out_dir, 'track_base.jpg')
        
        # Skip if both files exist
        if os.path.exists(base_track_path) and os.path.exists(vis_path):
            print(f"\nSkipping {video_name}: both base_tracking.pkl and track_base.jpg exist")
            continue
        
        # If base_track exists but vis doesn't, generate vis from saved pkl
        if os.path.exists(base_track_path) and not os.path.exists(vis_path):
            print(f"\nGenerating visualization for {video_name} from existing base_tracking.pkl")
            
            # Load existing results
            with open(base_track_path, 'rb') as f:
                base_results = pickle.load(f)
            
            # Reload images and crop face/hands using saved transformation matrices
            frames_keys = video_data['frames_keys']
            ret_images_dict = {}
            
            for frame_key in frames_keys:
                if frame_key not in base_results:
                    continue
                
                img_path = os.path.join(args.images_dir, video_name, f"{frame_key}.jpg")
                if not os.path.exists(img_path):
                    continue
                
                # Load original image
                img_rgb = load_image(img_path)
                
                # Reconstruct cropped body, face and hands using M_o2c from base_results
                ret_images_dict[frame_key] = pipeline.reconstruct_cropped_images(
                    img_rgb, base_results[frame_key]
                )
            
            # Generate visualization with landmarks
            generate_visualization(video_name, video_data, ret_images_dict, base_results, args.ehmx_dir, pipeline)
            continue
        
        # Process video (base_track doesn't exist)
        print(f"\nProcessing {video_name} (no existing base_tracking.pkl)")
        base_results, id_share_params, ret_images_dict = process_video(
            video_name, video_data, args, pipeline
        )
        
        if len(base_results) == 0:
            print(f"  Warning: No frames processed for {video_name}")
            continue
        
        # Save results
        save_results(video_name, base_results, id_share_params, args.ehmx_dir)
        
        # Generate visualization with landmarks
        generate_visualization(video_name, video_data, ret_images_dict, base_results, args.ehmx_dir, pipeline)
    
    print("\nProcessing complete!")


if __name__ == '__main__':
    main()
