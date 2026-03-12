#!/usr/bin/env python3
"""
infer_ehmx_smplx.py - Run SMPL-X refiner on track-based processed results
Loads base tracking results and refines SMPL-X body parameters.
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
from src.ehmx_track_base import load_frame_image, load_matte, apply_matte_to_image
from src.ehmx_refine_smplx import RefineSmplxPipeline
from src.utils.io import load_dict_pkl, write_dict_pkl
from src.utils.draw import draw_landmarks


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
            'right_mano_shape': [],
            'head_scale': np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
            'hand_scale': np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
            'joints_offset': np.zeros((1, 55, 3), dtype=np.float32)
        }


def load_body_images(video_name, frames_keys, base_results, images_dir, mattes_dir=None, pshuman_dir=None):
    """
    Load and crop body images for all frames.
    
    Args:
        video_name: Name of the video
        frames_keys: List of frame keys
        base_results: Base tracking results with crop info
        images_dir: Directory containing original images
        mattes_dir: Optional directory containing matte images
    
    Returns:
        Dictionary mapping frame_key to cropped body image tensors
    """
    from src.utils.crop import _transform_img
    
    body_images = {}
    for frame_key in tqdm(frames_keys, desc="Loading body images"):
        if frame_key not in base_results:
            continue
        
        # Load original image
        img_rgb = load_frame_image(images_dir, video_name, frame_key, pshuman_dir)
        
        # Apply matte if available
        if mattes_dir:
            matte_path = os.path.join(mattes_dir, video_name, f"{frame_key}.png")
            matte = load_matte(matte_path)
            if matte is not None:
                img_rgb = apply_matte_to_image(img_rgb, matte)
        
        # Crop body image using saved M_o2c transformation matrix
        body_crop_info = base_results[frame_key]['body_crop']
        M_o2c_hd = body_crop_info['M_o2c-hd']
        crop_size = 1024  # Body HD crop size is 1024
        
        # Apply affine transformation
        body_img_cropped = _transform_img(img_rgb, M_o2c_hd, dsize=crop_size)
        
        # Convert to tensor (C, H, W)
        body_img_tensor = torch.from_numpy(body_img_cropped).permute(2, 0, 1).float()
        body_images[frame_key] = body_img_tensor
    
    return body_images


def process_video_smplx(video_name, video_data, args, smplx_pipeline, cfg, optim_cfgs):
    """
    Run SMPL-X refinement for a single video.
    
    Args:
        video_name: Name of the video
        video_data: Dictionary with 'frames_num' and 'frames_keys'
        args: Command line arguments
        smplx_pipeline: RefineSmplxPipeline instance
        cfg: DataPreparationConfig instance
        optim_cfg: Optimization config from YAML
    
    Returns:
        Tuple of (refined_results, id_share_params)
    """
    frames_keys = video_data['frames_keys']
    frames_num = video_data['frames_num']
    fps = video_data['fps']
    frame_interval = 24 / fps
    
    print(f"\nProcessing video: {video_name}")
    print(f"  Frames: {frames_num}")
    print(f"  FPS: {fps}")
    print(f"  Frame interval: {frame_interval}")
    
    # Define paths
    video_out_dir = os.path.join(args.ehmx_dir, video_name)
    track_flame_pkl_path = os.path.join(video_out_dir, 'track_flame.pkl')
    id_share_pkl_path = os.path.join(video_out_dir, 'id_share_params.pkl')
    refined_pkl_path = os.path.join(video_out_dir, 'optim_tracking_ehm.pkl')
    preview_path = os.path.join(video_out_dir, 'track_smplx.jpg')
    
    # Check if already processed
    if os.path.exists(refined_pkl_path) and not args.overwrite:
        # If preview doesn't exist, load data and regenerate visualization
        if not os.path.exists(preview_path):
            print(f"  Loading existing results to regenerate visualization...")
            refined_results = load_dict_pkl(refined_pkl_path)
            id_share_params = load_id_share_params(id_share_pkl_path)
            base_results = load_track_results(track_flame_pkl_path)
            
            # Load body images
            body_images = load_body_images(video_name, frames_keys, base_results,
                                           args.images_dir, args.mattes_dir, args.pshuman_dir)
            
            # Set saving root for visualization rewrite
            smplx_pipeline.saving_root = video_out_dir
            
            # Collate all frames for visualization
            valid_keys = [key for key in frames_keys if key in base_results and key in refined_results]
            if len(valid_keys) > 0:
                batch_data = [base_results[key] for key in valid_keys]
                batch_data = torch.utils.data.default_collate(batch_data)
                from src.ehmx_refine_smplx import data_to_device
                batch_data = data_to_device(batch_data, device=cfg.device)
                
                # Prepare batch images
                batch_body_imgs = [body_images[key] for key in valid_keys if key in body_images]
                if len(batch_body_imgs) > 0:
                    batch_body_imgs = torch.stack(batch_body_imgs).to(cfg.device)
                    
                    # Run a single optimization step to trigger visualization with saved data
                    print(f"  Regenerating visualization from saved data...")
                    # Extract optimized coefficients and update batch_data
                    for idx, key in enumerate(valid_keys):
                        if key in refined_results and 'smplx_coeffs' in refined_results[key]:
                            for param_key in ['body_pose', 'camera_RT_params', 'global_pose', 
                                            'left_hand_pose', 'right_hand_pose']:
                                if param_key in refined_results[key]['smplx_coeffs']:
                                    batch_data['smplx_coeffs'][param_key][idx] = torch.from_numpy(
                                        refined_results[key]['smplx_coeffs'][param_key]).to(cfg.device)
                    
                    # Trigger visualization by running optimize with 1 step
                    vis_cfg = OmegaConf.create({
                        'steps': 1,
                        'share_id': True,
                        'optim_camera': False,
                        'lambda_3d_head': 0.0,
                        'lambda_3d_hand_l': 0.0,
                        'lambda_3d_hand_r': 0.0,
                        'lambda_2d_kpt': 0.0,
                        'lambda_prior': 0.0,
                        'lambda_mtn_body_pose': 0.0,
                    })
                    smplx_pipeline.optimize(
                        valid_keys, batch_data, id_share_params, vis_cfg,
                        batch_id=0, batch_imgs=batch_body_imgs, interval=frame_interval
                    )
                    
                    # Copy the generated visualization to preview path
                    vis_path = os.path.join(video_out_dir, "visual_results", "vis_fit_smplx_bid-0_stp-0.png")
                    if os.path.exists(vis_path):
                        import shutil
                        shutil.copy(vis_path, preview_path)
                        print(f"  ✓ Visualization regenerated: {preview_path}")
            
            return refined_results, id_share_params
        else:
            print(f"  Skipping: Already processed (use --overwrite to reprocess)")
            refined_results = load_dict_pkl(refined_pkl_path)
            id_share_params = load_id_share_params(id_share_pkl_path)
            return refined_results, id_share_params
    
    # Load FLAME-refined tracking results as base
    print(f"  Loading FLAME-refined results from: {track_flame_pkl_path}")
    if not os.path.exists(track_flame_pkl_path):
        print(f"  Error: FLAME-refined results not found. Please run infer_ehmx_flame.py first.")
        return None, None
    
    base_results = load_track_results(track_flame_pkl_path)
    
    # Load identity-shared parameters
    id_share_params = load_id_share_params(id_share_pkl_path)
    
    # Load body images from original images
    print(f"  Loading body images from: {args.images_dir}")
    body_images = load_body_images(video_name, frames_keys, base_results, 
                                   args.images_dir, args.mattes_dir, args.pshuman_dir)
    
    # Set saving root for smplx pipeline
    smplx_pipeline.saving_root = video_out_dir
    
    print(f"  Running SMPL-X optimization ({len(optim_cfgs)} rounds)...")
    
    current_base_results = base_results
    current_id_share_params = id_share_params
    
    try:
        for round_idx, optim_cfg in enumerate(optim_cfgs):
            print(f"    Round {round_idx}: Steps={optim_cfg.steps}")
            print(f"    Optimize camera: {optim_cfg.get('optim_camera', False)}")
            
            # Run SMPL-X refinement
            opt_smplx_coeff, current_id_share_params = smplx_pipeline.run(
                current_base_results,
                current_id_share_params,
                optim_cfg,
                body_images,
                frame_interval
            )
            
            # Update current_base_results with refined coefficients for next round
            for key in current_base_results.keys():
                if key in opt_smplx_coeff:
                    current_base_results[key]['smplx_coeffs'] = opt_smplx_coeff[key]
        
        # Final results
        refined_results = current_base_results
        
        # Save refined results
        print(f"  Saving refined results to: {refined_pkl_path}")
        write_dict_pkl(refined_pkl_path, refined_results)
        
        # Update and save id_share_params
        write_dict_pkl(id_share_pkl_path, current_id_share_params)
        
        # Copy the final visualization to preview path
        last_optim_cfg = optim_cfgs[-1]
        final_vis_base = os.path.join(video_out_dir, "visual_results",
                                     f"vis_fit_smplx_bid-0_stp-{last_optim_cfg.steps - 1}")
        # Try both .jpg and .png extensions
        final_vis_path = None
        for ext in ['.jpg', '.png']:
            if os.path.exists(final_vis_base + ext):
                final_vis_path = final_vis_base + ext
                break
        if final_vis_path is not None:
            import shutil
            shutil.copy(final_vis_path, preview_path)
            print(f"  Saved preview: {preview_path}")
        else:
            print(f"  Warning: Final visualization not found at {final_vis_path}")
        
        print(f"  ✓ SMPL-X refinement completed for {video_name}")
        
        return refined_results, id_share_params
        
    except Exception as e:
        print(f"  ✗ SMPL-X refinement failed: {e}")
        import traceback
        traceback.print_exc()
        return None, id_share_params


def main():
    parser = argparse.ArgumentParser(description='Run SMPL-X refiner on track-based processed results')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Root directory containing original images')
    parser.add_argument('--index_json', type=str, required=True,
                        help='JSON file with videos and frames to process')
    parser.add_argument('--mattes_dir', type=str, default=None,
                        help='Directory containing matte images (optional)')
    parser.add_argument('--pshuman_dir', type=str, default=None,
                        help='Root directory containing pshuman images')
    parser.add_argument('--ehmx_dir', type=str, required=True,
                        help='Directory for EHM-X tracking results')
    parser.add_argument('--config', type=str,
                        default='src/configs/optim_configs/ehm_fbody_ytb.yaml',
                        help='Path to optimization config file')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing refined results')
    parser.add_argument('--body_landmark_type', type=str, default='sapiens',
                        choices=['sapiens', 'dwpose'],
                        help='Body landmark detector type (default: sapiens)')
    
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
    
    # Parse optimization rounds
    optim_cfgs = []
    if 'optim_ehm' in full_optim_config:
        optim_cfgs.append(full_optim_config.optim_ehm)
    else:
        # Look for optim_ehm_round0, optim_ehm_round1, ...
        round_idx = 0
        while True:
            round_key = f'optim_ehm_round{round_idx}'
            if round_key in full_optim_config:
                optim_cfgs.append(full_optim_config[round_key])
                round_idx += 1
            else:
                break
    
    if not optim_cfgs:
        print("Error: No optim_ehm or optim_ehm_roundX found in config")
        return
        
    print(f"Optimization config loaded: {len(optim_cfgs)} rounds")
    for i, cfg_round in enumerate(optim_cfgs):
        print(f"  Round {i}: Steps={cfg_round.steps}, Share ID={cfg_round.share_id}")
    
    # Load data preparation config
    print(f"\nInitializing SMPL-X refinement pipeline...")
    cfg = DataPreparationConfig()
    cfg.body_landmark_type = args.body_landmark_type
    print(f"Body landmark type: {cfg.body_landmark_type}")
    
    # Initialize SMPL-X refinement pipeline
    smplx_pipeline = RefineSmplxPipeline(cfg)
    
    print(f"SMPL-X pipeline initialized on {cfg.device}")
    
    # Process each video
    for video_idx, (video_name, video_data) in enumerate(videos_info.items(), 1):
        print(f"\n[{video_idx}/{len(videos_info)}] Processing: {video_name}")
        
        try:
            refined_results, id_share_params = process_video_smplx(
                video_name, video_data, args, smplx_pipeline, cfg, optim_cfgs
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
    print("SMPL-X refinement complete!")
    print("="*80)


if __name__ == '__main__':
    main()
