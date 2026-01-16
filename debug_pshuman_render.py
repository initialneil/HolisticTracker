#!/usr/bin/env python3
"""
debug_pshuman_render.py - Debug PSHuman camera transformation by rendering SMPL-X
Renders SMPL-X from both original and PSHuman camera views for comparison.
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
from src.ehmx_track_base import load_frame_image, load_matte, apply_matte_to_image, split_frame_key
from src.ehmx_refine_smplx import RefineSmplxPipeline, make_pshuman_camera, data_to_device
from src.utils.io import load_dict_pkl, write_dict_pkl
from src.utils.draw import draw_landmarks
from src.utils.graphics import GS_Camera
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from pytorch3d.renderer import PointLights


def find_first_pshuman_frame(videos_info):
    """Find the first frame with a PSHuman view."""
    for video_name, video_data in videos_info.items():
        frames_keys = video_data['frames_keys']
        for frame_key in frames_keys:
            shot_id, frame_id, view_id = split_frame_key(frame_key)
            if view_id and ('pshuman' in view_id or 'color' in view_id):
                # Also find the corresponding original frame
                original_key = f"{shot_id}/{frame_id}"
                if original_key in frames_keys:
                    return video_name, original_key, frame_key, view_id
    return None, None, None, None


def render_smplx_debug(smplx_pipeline, smplx_coeffs, flame_coeffs, camera_RT, save_path, img_bg=None):
    """
    Render SMPL-X with given camera parameters.
    
    Args:
        smplx_pipeline: RefineSmplxPipeline instance
        smplx_coeffs: SMPL-X coefficients
        flame_coeffs: FLAME coefficients
        camera_RT: Camera RT parameters (3x4 or 4x4)
        save_path: Path to save rendered image
        img_bg: Optional background image
    
    Returns:
        Rendered image as numpy array
    """
    device = smplx_pipeline.device
    
    # Prepare batch data
    batch_smplx = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else torch.from_numpy(v).to(device) 
                   for k, v in smplx_coeffs.items()}
    # batch_flame = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else torch.from_numpy(v).to(device)
    #                for k, v in flame_coeffs.items()}
    
    # Forward through EHM
    with torch.no_grad():
        smplx_dict = smplx_pipeline.ehm(batch_smplx, None, pose_type='aa')
    
    # Setup camera
    batch_size = 1
    focal_length = smplx_pipeline.body_focal_length
    cameras_kwargs = smplx_pipeline.build_cameras_kwargs(batch_size, focal_length)
    cameras = GS_Camera(**cameras_kwargs).to(device)
    
    # Extract R and T from camera_RT
    if camera_RT.shape[0] == 3:
        RT_4x4 = torch.eye(4, device=device, dtype=torch.float32)
        RT_4x4[:3, :4] = camera_RT
        camera_RT = RT_4x4
    
    R = camera_RT[:3, :3].unsqueeze(0)
    T = camera_RT[:3, 3].unsqueeze(0)
    
    # Build camera with R and T
    t_camera = GS_Camera(**cameras_kwargs, R=R, T=T)
    
    # Render mesh
    lights = PointLights(device=device, location=[[0.0, -1.0, -10.0]])
    mesh_img = smplx_pipeline.body_renderer.render_mesh(
        smplx_dict['vertices'],
        t_camera,
        lights=lights
    )
    mesh_img = mesh_img[:, :3].detach().cpu().numpy().clip(0, 255).astype(np.uint8)[0].transpose(1, 2, 0)
    
    # Overlay on background if provided
    if img_bg is not None:
        mesh_img = cv2.addWeighted(img_bg, 0.3, mesh_img, 0.7, 0)
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(mesh_img, cv2.COLOR_RGB2BGR))
    
    return mesh_img


def debug_pshuman_rendering(video_name, original_key, pshuman_key, view_id, args, smplx_pipeline, cfg):
    """
    Debug PSHuman camera transformation by rendering SMPL-X from both views.
    
    Args:
        video_name: Name of the video
        original_key: Frame key for original view
        pshuman_key: Frame key for PSHuman view
        view_id: PSHuman view identifier (e.g., 'pshuman_02', 'pshuman_03')
        args: Command line arguments
        smplx_pipeline: RefineSmplxPipeline instance
        cfg: DataPreparationConfig instance
    """
    print(f"\nDebug PSHuman Rendering:")
    print(f"  Video: {video_name}")
    print(f"  Original frame: {original_key}")
    print(f"  PSHuman frame: {pshuman_key}")
    print(f"  View ID: {view_id}")
    
    # Define paths
    video_out_dir = os.path.join(args.ehmx_dir, video_name)
    track_base_pkl_path = os.path.join(video_out_dir, 'base_tracking.pkl')
    visual_dir = os.path.join(video_out_dir, 'visual_results')
    os.makedirs(visual_dir, exist_ok=True)
    
    # Load base tracking results
    print(f"  Loading base tracking results from: {track_base_pkl_path}")
    if not os.path.exists(track_base_pkl_path):
        print(f"  Error: Base tracking results not found. Please run infer_ehmx_track_base.py first.")
        return
    
    base_results = load_dict_pkl(track_base_pkl_path)
    
    # Check if both frames exist
    if original_key not in base_results or pshuman_key not in base_results:
        print(f"  Error: Frame data not found in base results")
        return
    
    original_data = base_results[original_key]
    pshuman_data = base_results[pshuman_key]
    
    # Load original image
    print(f"  Loading original image...")
    from src.utils.crop import _transform_img
    original_img = load_frame_image(args.images_dir, video_name, original_key, None)
    if args.mattes_dir:
        matte_path = os.path.join(args.mattes_dir, video_name, f"{original_key}.png")
        matte = load_matte(matte_path)
        if matte is not None:
            original_img = apply_matte_to_image(original_img, matte)
    
    # Crop original image
    M_o2c_hd = original_data['body_crop']['M_o2c-hd']
    original_img_cropped = _transform_img(original_img, M_o2c_hd, dsize=1024)
    
    # Load PSHuman image
    print(f"  Loading PSHuman image...")
    pshuman_img = load_frame_image(args.images_dir, video_name, pshuman_key, args.pshuman_dir)
    if args.mattes_dir:
        matte_path = os.path.join(args.mattes_dir, video_name, f"{pshuman_key}.png")
        matte = load_matte(matte_path)
        if matte is not None:
            pshuman_img = apply_matte_to_image(pshuman_img, matte)
    
    # Crop PSHuman image
    M_o2c_hd_pshuman = pshuman_data['body_crop']['M_o2c-hd']
    pshuman_img_cropped = _transform_img(pshuman_img, M_o2c_hd_pshuman, dsize=1024)
    
    # Get SMPL-X coefficients from original frame
    smplx_coeffs = original_data['smplx_coeffs']
    flame_coeffs = original_data['flame_coeffs']
    
    # Get original camera RT
    original_camera_RT = smplx_coeffs['camera_RT_params']
    if isinstance(original_camera_RT, np.ndarray):
        original_camera_RT = torch.from_numpy(original_camera_RT).to(cfg.device)
    original_camera_RT = original_camera_RT.squeeze(0)
    
    print(f"  Rendering SMPL-X with original camera...")
    original_render_path = os.path.join(visual_dir, f"debug_original_{original_key.replace('/', '_')}.jpg")
    render_smplx_debug(smplx_pipeline, smplx_coeffs, flame_coeffs, 
                       original_camera_RT, original_render_path, original_img_cropped)
    print(f"  ✓ Saved original render: {original_render_path}")
    
    # Calculate PSHuman camera RT
    print(f"  Calculating PSHuman camera RT...")
    pshuman_camera_RT = make_pshuman_camera(original_camera_RT[:3, :4], view_id, cfg.device)
    
    print(f"  Original camera RT:")
    print(f"    {original_camera_RT[:3, :4]}")
    print(f"  PSHuman camera RT:")
    print(f"    {pshuman_camera_RT}")
    
    # Render SMPL-X with PSHuman camera
    print(f"  Rendering SMPL-X with PSHuman camera...")
    pshuman_render_path = os.path.join(visual_dir, f"debug_pshuman_{pshuman_key.replace('/', '_')}.jpg")
    render_smplx_debug(smplx_pipeline, smplx_coeffs, flame_coeffs,
                      pshuman_camera_RT, pshuman_render_path, pshuman_img_cropped)
    print(f"  ✓ Saved PSHuman render: {pshuman_render_path}")
    
    # Create side-by-side comparison
    print(f"  Creating comparison image...")
    original_render = cv2.imread(original_render_path)
    pshuman_render = cv2.imread(pshuman_render_path)
    
    # Add text labels
    cv2.putText(original_render, f"Original: {original_key}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(pshuman_render, f"PSHuman: {view_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    comparison = np.hstack([original_render, pshuman_render])
    comparison_path = os.path.join(visual_dir, f"debug_comparison.jpg")
    cv2.imwrite(comparison_path, comparison)
    print(f"  ✓ Saved comparison: {comparison_path}")
    
    print(f"\n✓ PSHuman debug rendering complete!")
    print(f"  Results saved to: {visual_dir}")


def main():
    parser = argparse.ArgumentParser(description='Debug PSHuman camera transformation rendering')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Root directory containing original images')
    parser.add_argument('--index_json', type=str, required=True,
                        help='JSON file with videos and frames to process')
    parser.add_argument('--mattes_dir', type=str, default=None,
                        help='Directory containing matte images (optional)')
    parser.add_argument('--pshuman_dir', type=str, default=None,
                        help='Root directory containing PSHuman images')
    parser.add_argument('--ehmx_dir', type=str, required=True,
                        help='Directory for EHM-X tracking results')
    
    args = parser.parse_args()
    
    # Load index JSON
    print(f"Loading index from: {args.index_json}")
    with open(args.index_json, 'r') as f:
        videos_info = json.load(f)
    
    print(f"Found {len(videos_info)} video(s)")
    
    # Find first PSHuman frame
    print(f"\nSearching for first PSHuman frame...")
    video_name, original_key, pshuman_key, view_id = find_first_pshuman_frame(videos_info)
    
    if video_name is None:
        print(f"Error: No PSHuman frames found in index")
        return
    
    # Load data preparation config
    print(f"\nInitializing rendering pipeline...")
    cfg = DataPreparationConfig()
    
    # Initialize SMPL-X refinement pipeline (for rendering)
    smplx_pipeline = RefineSmplxPipeline(cfg)
    print(f"Pipeline initialized on {cfg.device}")
    
    # Debug PSHuman rendering
    try:
        debug_pshuman_rendering(video_name, original_key, pshuman_key, view_id, 
                               args, smplx_pipeline, cfg)
    except Exception as e:
        print(f"Error during debug rendering: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("PSHuman debug rendering complete!")
    print("="*80)


if __name__ == '__main__':
    main()
