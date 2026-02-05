#!/usr/bin/env python3
"""
convert_lhm_track.py - Convert LHM track results to EHM format and visualize
Loads LHM tracking results, converts to EHM format, and generates visualization.
"""
import os
import sys
import json
import argparse
import shutil
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
from src.ehmx_refine_smplx import RefineSmplxPipeline, data_to_device
from src.utils.io import load_dict_pkl, write_dict_pkl
from src.utils.crop import _transform_img


def load_track_results(track_pkl_path):
    """Load tracking results from pickle file."""
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
        images_dir: Directory containing original images (should already point to video folder)
        mattes_dir: Optional directory containing matte images
        pshuman_dir: Optional directory containing pshuman images
    
    Returns:
        Dictionary mapping frame_key to cropped body image tensors
    """
    body_images = {}
    for frame_key in tqdm(frames_keys, desc="Loading body images"):
        if frame_key not in base_results:
            continue
        
        # Load original image - try different path formats
        shot_id, frame_id = frame_key.split('/')[:2]
        
        # Try shot/frame.jpg format
        img_path = os.path.join(images_dir, shot_id, f"{frame_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, shot_id, f"{frame_id}.png")
        
        if not os.path.exists(img_path):
            print(f"    Warning: Image not found for {frame_key}, skipping")
            continue
        
        img_rgb = np.array(PILImage.open(img_path).convert('RGB'))
        
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


def visualize_tracking(shot_key, video_data, output_dir, args, smplx_pipeline, cfg):
    """
    Generate visualization for tracking results using direct rendering.
    
    Args:
        shot_key: Shot identifier (e.g., 'video_name/shot_name')
        video_data: Dictionary with 'frames_num', 'frames_keys', and 'fps'
        output_dir: Directory where tracking results are saved
        args: Command line arguments
        smplx_pipeline: RefineSmplxPipeline instance
        cfg: DataPreparationConfig instance
    """
    frames_keys = video_data['frames_keys']
    frames_num = video_data['frames_num']
    fps = video_data.get('fps', 24)
    
    video_name = shot_key.split('/')[0]
    
    print(f"\nGenerating visualization for: {shot_key}")
    print(f"  Frames: {frames_num}")
    print(f"  FPS: {fps}")
    
    # Define paths
    track_pkl_path = os.path.join(output_dir, 'optim_tracking_ehm.pkl')
    id_share_pkl_path = os.path.join(output_dir, 'id_share_params.pkl')
    preview_path = os.path.join(output_dir, 'track_smplx.jpg')
    
    # Check if preview already exists
    if os.path.exists(preview_path) and not args.overwrite:
        print(f"  Skipping: Visualization already exists (use --overwrite to regenerate)")
        return
    
    # Load tracking and id_share_params
    print(f"  Loading tracking results...")
    if not os.path.exists(track_pkl_path):
        print(f"  Error: Tracking results not found: {track_pkl_path}")
        return
    
    tracking_data = load_dict_pkl(track_pkl_path)
    id_share_params = load_id_share_params(id_share_pkl_path)
    
    # Load body images
    print(f"  Loading body images...")
    # For nested structure, images_dir is the parent, need to add video_name
    # For direct structure, video_name would be a shot_id, so this still works
    video_images_dir = os.path.join(args.images_dir, video_name)
    if not os.path.exists(video_images_dir):
        # Fallback: maybe images_dir already points to the video
        video_images_dir = args.images_dir
    
    body_images = load_body_images(video_name, frames_keys, tracking_data,
                                   video_images_dir, args.mattes_dir, args.pshuman_dir)
    
    if len(body_images) == 0:
        print(f"  Error: No body images loaded")
        return
    
    # Call render function (same as render_ehmx_smplx.py)
    success = render_smplx_visualization(
        shot_key, tracking_data, body_images, preview_path, cfg, smplx_pipeline, id_share_params
    )
    
    if success:
        print(f"  ✓ Visualization saved: {preview_path}")
    else:
        print(f"  Error: Visualization generation failed")


def render_smplx_visualization(shot_key, tracking_data, body_images, output_path, cfg, smplx_pipeline, id_share_params):
    """
    Render SMPL-X visualization directly without optimization.
    (Copied from render_ehmx_smplx.py)
    """
    print(f"  Rendering SMPL-X visualization...")
    
    # Get all valid frame keys
    frames_keys = sorted(tracking_data.keys())
    valid_keys = [key for key in frames_keys if key in body_images and 'smplx_coeffs' in tracking_data[key]]
    
    if len(valid_keys) == 0:
        print(f"  Error: No valid frames found for visualization")
        return False
    
    print(f"  Rendering {len(valid_keys)} frames...")
    
    # Prepare batch data
    batch_size = min(len(valid_keys), 16)  # Limit to 16 frames for visualization
    selected_keys = valid_keys[::max(1, len(valid_keys) // batch_size)][:batch_size]
    
    vis_imgs = []
    
    for frame_key in tqdm(selected_keys, desc="  Rendering frames"):
        frame_data = tracking_data[frame_key]
        body_img = body_images[frame_key]
        
        # Convert image to numpy (H, W, C)
        img_np = body_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # Get SMPL-X and FLAME coefficients
        smplx_coeffs = frame_data['smplx_coeffs']
        flame_coeffs = frame_data.get('flame_coeffs', None)
        
        # Prepare batch_smplx dictionary
        batch_smplx = {
            'body_pose': torch.from_numpy(smplx_coeffs['body_pose']).unsqueeze(0).to(cfg.device),
            'global_pose': torch.from_numpy(smplx_coeffs['global_pose']).unsqueeze(0).to(cfg.device),
            'left_hand_pose': torch.from_numpy(smplx_coeffs['left_hand_pose']).unsqueeze(0).to(cfg.device),
            'right_hand_pose': torch.from_numpy(smplx_coeffs['right_hand_pose']).unsqueeze(0).to(cfg.device),
            'body_cam': torch.from_numpy(smplx_coeffs['body_cam']).unsqueeze(0).to(cfg.device),
            'camera_RT_params': torch.from_numpy(smplx_coeffs['camera_RT_params']).unsqueeze(0).to(cfg.device),
            'shape': torch.tensor(id_share_params['smplx_shape'], device=cfg.device).float(),
            'head_scale': torch.tensor(id_share_params['head_scale'], device=cfg.device),
            'hand_scale': torch.tensor(id_share_params['hand_scale'], device=cfg.device),
            'joints_offset': torch.tensor(id_share_params['joints_offset'], device=cfg.device),
        }
        
        # Add expression params - use FLAME if available, else zeros
        if flame_coeffs is not None:
            batch_smplx['exp'] = torch.from_numpy(flame_coeffs['expression_params']).unsqueeze(0).to(cfg.device)
        else:
            batch_smplx['exp'] = torch.zeros(1, 50, device=cfg.device, dtype=torch.float32)
        
        # Prepare batch_flame dictionary if available
        batch_flame = None
        if flame_coeffs is not None:
            batch_smplx['exp'] = torch.from_numpy(flame_coeffs['expression_params']).unsqueeze(0).to(cfg.device)
            batch_flame = {
                'shape_params': torch.tensor(id_share_params['flame_shape'], device=cfg.device).float(),
                'expression_params': torch.from_numpy(flame_coeffs['expression_params']).unsqueeze(0).to(cfg.device),
                'jaw_params': torch.from_numpy(flame_coeffs['jaw_params']).unsqueeze(0).to(cfg.device),
                'pose_params': torch.from_numpy(flame_coeffs['pose_params']).unsqueeze(0).to(cfg.device),
                'eye_pose_params': torch.from_numpy(flame_coeffs['eye_pose_params']).unsqueeze(0).to(cfg.device),
                'eyelid_params': torch.from_numpy(flame_coeffs['eyelid_params']).unsqueeze(0).to(cfg.device),
            }
        
        # Get camera parameters
        camera_RT = batch_smplx['camera_RT_params']
        
        # Forward pass through EHM model
        with torch.no_grad():
            smplx_dict = smplx_pipeline.ehm(
                batch_smplx,
                batch_flame,
                pose_type='aa'
            )
            
            # Get 3D vertices in world space
            smplx_vertices = smplx_dict['vertices']  # (1, 10475, 3)
            smplx_joints = smplx_dict['joints']  # (1, N, 3)
        
        # Convert image to numpy (H, W, C)
        img_np = body_img.permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()
        img_size = 1024
        
        # Build camera for projection and rendering
        from src.utils.graphics import GS_Camera
        from pytorch3d.renderer import PointLights
        
        # Extract R and T from camera_RT_params [1, 3, 4]
        R, T = camera_RT.split([3, 1], dim=-1)
        T = T.squeeze(-1)
        
        # Create camera for projection
        cameras_kwargs = smplx_pipeline.build_cameras_kwargs(1, smplx_pipeline.body_focal_length)
        cameras = GS_Camera(**cameras_kwargs).to(cfg.device)
        
        # Project vertices and joints
        proj_vertices = cameras.transform_points_screen(smplx_vertices, R=R, T=T)
        proj_joints = cameras.transform_points_screen(smplx_joints, R=R, T=T)
        
        # Panel 1: Original image with keypoints
        img_panel1 = img_np.copy()
        smplx_kpts_2d_np = proj_joints[0, :, :2].cpu().numpy()
        for kpt in smplx_kpts_2d_np:
            x, y = int(kpt[0]), int(kpt[1])
            if 0 <= x < img_size and 0 <= y < img_size:
                cv2.circle(img_panel1, (x, y), 3, (255, 0, 0), -1)
        
        # Panel 2: Mesh rendering
        lights = PointLights(device=cfg.device, location=[[0.0, -1.0, -10.0]])
        t_camera = GS_Camera(**cameras_kwargs, R=R, T=T)
        mesh_img = smplx_pipeline.body_renderer.render_mesh(
            smplx_vertices, t_camera, lights=lights
        )
        mesh_img = (mesh_img[:, :3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1, 2, 0)
        img_panel2 = cv2.addWeighted(img_np, 0.3, mesh_img, 0.7, 0)
        
        # Panel 3: Original image with keypoints
        img_panel3 = img_np.copy()
        for kpt in smplx_kpts_2d_np:
            x, y = int(kpt[0]), int(kpt[1])
            if 0 <= x < img_size and 0 <= y < img_size:
                cv2.circle(img_panel3, (x, y), 2, (0, 255, 0), -1)
        
        # Concatenate three panels horizontally
        vis_img = np.concatenate([img_panel1, img_panel2, img_panel3], axis=1)
        vis_imgs.append(vis_img)
    
    # Create grid layout
    if len(vis_imgs) > 0:
        total_frames = len(vis_imgs)
        grid_rows = max(1, int(np.sqrt(total_frames * 2)))
        grid_cols = int(np.ceil(total_frames / grid_rows))
        
        # Adjust to maintain 1:2 ratio
        while grid_rows < 2 * grid_cols and grid_cols > 1:
            grid_cols -= 1
            grid_rows = int(np.ceil(total_frames / grid_cols))
        
        img_height, img_width = vis_imgs[0].shape[:2]
        grid_height = grid_rows * img_height
        grid_width = grid_cols * img_width
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        for idx, img in enumerate(vis_imgs):
            row = idx // grid_cols
            col = idx % grid_cols
            y_start = row * img_height
            y_end = y_start + img_height
            x_start = col * img_width
            x_end = x_start + img_width
            grid[y_start:y_end, x_start:x_end] = img
        
        cv2.imwrite(output_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        return True
    
    return False


def load_lhm_shot_data(lhm_shot_dir, shot_name):
    """
    Load LHM tracking data from a shot directory and convert to EHM format.
    
    Args:
        lhm_shot_dir: Path to the shot directory containing flame_params and smplx_params
        shot_name: Name of the shot (e.g., '005472_005712')
    
    Returns:
        Dictionary with converted tracking data for each frame
    """
    flame_params_dir = os.path.join(lhm_shot_dir, 'flame_params')
    smplx_params_dir = os.path.join(lhm_shot_dir, 'smplx_params')
    
    print(f"  Loading LHM data from shot: {shot_name}")
    
    # Check if required directories exist
    if not os.path.exists(smplx_params_dir):
        print(f"    Warning: smplx_params directory not found, skipping shot")
        return {}
    
    # Get all json files
    smplx_files = sorted([f for f in os.listdir(smplx_params_dir) if f.endswith('.json')])
    
    if not smplx_files:
        print(f"    Warning: No SMPL-X param files found, skipping shot")
        return {}
    
    print(f"    Found {len(smplx_files)} SMPL-X param files")
    
    # Build frame tracking data
    frame_tracking = {}
    
    for smplx_file in tqdm(smplx_files, desc=f"    Converting {shot_name}"):
        # Extract frame number from filename (00001.json -> 0, 00002.json -> 1, etc.)
        frame_num = int(smplx_file.split('.')[0]) - 1
        frame_key = f"{shot_name}/{frame_num:06d}"
        
        # Load SMPL-X params
        smplx_path = os.path.join(smplx_params_dir, smplx_file)
        with open(smplx_path, 'r') as f:
            smplx_data = json.load(f)
        
        # Load corresponding FLAME params if exists
        flame_data = None
        if os.path.exists(flame_params_dir):
            flame_file = smplx_file  # Same numbering
            flame_path = os.path.join(flame_params_dir, flame_file)
            if os.path.exists(flame_path):
                with open(flame_path, 'r') as f:
                    flame_data = json.load(f)
        
        # Convert to EHM format
        frame_data = convert_lhm_to_ehm_format(smplx_data, flame_data, frame_key)
        frame_tracking[frame_key] = frame_data
    
    return frame_tracking


def convert_lhm_to_ehm_format(smplx_data, flame_data, frame_key):
    """
    Convert LHM parameter format to EHM tracking format.
    
    Args:
        smplx_data: Dictionary with LHM SMPL-X parameters
        flame_data: Dictionary with LHM FLAME parameters (optional)
        frame_key: Frame identifier (e.g., '005472_005712/000000')
    
    Returns:
        Dictionary in EHM tracking format
    """
    # Initialize frame data structure
    frame_data = {}
    
    # Convert SMPL-X coefficients
    smplx_coeffs = {}
    
    # Expression parameters (first 50 from FLAME if available, else zeros)
    if flame_data and 'expcode' in flame_data:
        smplx_coeffs['exp'] = np.array(flame_data['expcode'][:50], dtype=np.float32)
    else:
        smplx_coeffs['exp'] = np.zeros(50, dtype=np.float32)
    
    # Global pose (root_pose from SMPL-X)
    smplx_coeffs['global_pose'] = np.array(smplx_data['root_pose'], dtype=np.float32)
    
    # Body pose (21 joints x 3)
    body_pose_flat = smplx_data['body_pose']
    if isinstance(body_pose_flat[0], list):
        # Already in [21, 3] format
        smplx_coeffs['body_pose'] = np.array(body_pose_flat, dtype=np.float32)
    else:
        # Flat format, reshape to [21, 3]
        smplx_coeffs['body_pose'] = np.array(body_pose_flat, dtype=np.float32).reshape(21, 3)
    
    # Hand poses
    lhand_pose_flat = smplx_data['lhand_pose']
    if isinstance(lhand_pose_flat[0], list):
        smplx_coeffs['left_hand_pose'] = np.array(lhand_pose_flat, dtype=np.float32)
    else:
        smplx_coeffs['left_hand_pose'] = np.array(lhand_pose_flat, dtype=np.float32).reshape(15, 3)
    
    rhand_pose_flat = smplx_data['rhand_pose']
    if isinstance(rhand_pose_flat[0], list):
        smplx_coeffs['right_hand_pose'] = np.array(rhand_pose_flat, dtype=np.float32)
    else:
        smplx_coeffs['right_hand_pose'] = np.array(rhand_pose_flat, dtype=np.float32).reshape(15, 3)
    
    # Camera parameters - following make_guava_scan.py conversion
    transl = np.array(smplx_data['trans'], dtype=np.float32)  # SMPLX translation in world space
    focal = np.array(smplx_data['focal'], dtype=np.float32)
    princpt = np.array(smplx_data['princpt'], dtype=np.float32)
    
    # Get image dimensions from LHM data if available, otherwise use 1024x1024
    if 'img_size_wh' in smplx_data:
        width, height = smplx_data['img_size_wh']
    else:
        width, height = 1024.0, 1024.0
    
    # Body cam: [invtanfov, tx, ty] similar to make_guava_scan.py
    # invtanfov = focal / (width * 0.5)
    invtanfov = focal[0] / (width * 0.5)
    tx = (princpt[0] - width / 2) / (width * 0.5)
    ty = (princpt[1] - height / 2) / (height * 0.5)
    smplx_coeffs['body_cam'] = np.array([invtanfov, tx, ty], dtype=np.float32)
    
    # Camera RT params: Following make_guava_scan.py logic
    # 1. Start with identity camera at origin
    cam_R = np.eye(3, dtype=np.float32)
    cam_t = np.zeros(3, dtype=np.float32)
    
    # 2. Compute camera center and adjust by SMPLX transl
    # cam_c = -cam_R.T @ cam_t = [0, 0, 0]
    cam_c = np.zeros(3, dtype=np.float32)
    cam_c -= transl  # Key step: subtract SMPLX transl from camera center
    
    # 3. Recompute camera translation
    cam_t = -cam_R @ cam_c
    
    # 4. Build world-to-camera matrix
    w2c_cam = np.eye(4, dtype=np.float32)
    w2c_cam[:3, :3] = cam_R
    w2c_cam[:3, 3] = cam_t
    
    # 5. Apply coordinate conversion (to PyTorch3D coordinate)
    # Flip x and y axes: c2c_mat inverse @ w2c_cam
    c2c_mat = np.array([[-1, 0, 0, 0],
                        [ 0,-1, 0, 0],
                        [ 0, 0, 1, 0],
                        [ 0, 0, 0, 1]], dtype=np.float32)
    c2c_inv = np.linalg.inv(c2c_mat)
    w2c_cam = c2c_inv @ w2c_cam
    
    camera_RT = w2c_cam[:3, :4]
    smplx_coeffs['camera_RT_params'] = camera_RT
    
    frame_data['smplx_coeffs'] = smplx_coeffs
    
    # Convert FLAME coefficients - ALWAYS create this entry for full EHM support
    flame_coeffs = {}
    
    if flame_data:
        # Use actual FLAME parameters when available
        # Expression params (50) - enables facial expressions
        flame_coeffs['expression_params'] = np.array(flame_data['expcode'][:50], dtype=np.float32)
        
        # Jaw params (3) - enables jaw movement
        flame_coeffs['jaw_params'] = np.array(flame_data['posecode'][3:6], dtype=np.float32)
        
        # Neck pose params (3)
        flame_coeffs['neck_pose_params'] = np.array(flame_data['neckcode'], dtype=np.float32)
        
        # Eye pose params (6) - enables eye ball movement
        flame_coeffs['eye_pose_params'] = np.array(flame_data['eyecode'], dtype=np.float32)
        
        # Eyelid params (2) - enables eyelid movement
        flame_coeffs['eyelid_params'] = np.zeros(2, dtype=np.float32)
        
        # Pose params (first 3 from posecode) - head global pose
        flame_coeffs['pose_params'] = np.array(flame_data['posecode'][:3], dtype=np.float32)
    else:
        # Use neutral/zero FLAME parameters when FLAME data not available
        flame_coeffs['expression_params'] = np.zeros(50, dtype=np.float32)  # Neutral expression
        flame_coeffs['jaw_params'] = np.zeros(3, dtype=np.float32)  # Closed jaw
        flame_coeffs['neck_pose_params'] = np.zeros(3, dtype=np.float32)  # Neutral neck
        flame_coeffs['eye_pose_params'] = np.zeros(6, dtype=np.float32)  # Forward gaze
        flame_coeffs['eyelid_params'] = np.zeros(2, dtype=np.float32)  # Open eyes
        flame_coeffs['pose_params'] = np.zeros(3, dtype=np.float32)  # Neutral head pose
    
    # Camera RT (same as SMPL-X)
    flame_coeffs['camera_RT_params'] = camera_RT.copy()
    
    # Cam params (same as body_cam)
    flame_coeffs['cam'] = smplx_coeffs['body_cam'].copy()
    
    # ALWAYS add flame_coeffs to enable full EHM (SMPL-X + FLAME)
    frame_data['flame_coeffs'] = flame_coeffs
    
    # Add placeholder data for other required fields
    # These would need actual data from preprocessing pipelines
    frame_data['body_crop'] = {
        'M_o2c': np.eye(3, dtype=np.float32),
        'M_c2o': np.eye(3, dtype=np.float32),
        'M_o2c-hd': np.eye(3, dtype=np.float32),
        'M_c2o-hd': np.eye(3, dtype=np.float32),
    }
    
    frame_data['head_crop'] = {
        'M_o2c': np.eye(3, dtype=np.float32),
        'M_c2o': np.eye(3, dtype=np.float32),
    }
    
    frame_data['left_hand_crop'] = {
        'M_o2c': np.eye(3, dtype=np.float32),
        'M_c2o': np.eye(3, dtype=np.float32),
    }
    
    frame_data['right_hand_crop'] = {
        'M_o2c': np.eye(3, dtype=np.float32),
        'M_c2o': np.eye(3, dtype=np.float32),
    }
    
    # Placeholder dwpose data
    frame_data['dwpose_raw'] = {
        'keypoints': np.zeros((134, 2), dtype=np.float32),
        'scores': np.zeros(134, dtype=np.float32),
        'bbox': np.zeros(4, dtype=np.float32),
    }
    
    frame_data['dwpose_rlt'] = {
        'keypoints': np.zeros((134, 2), dtype=np.float32),
        'scores': np.zeros(134, dtype=np.float32),
        'faces': np.zeros((68, 2), dtype=np.float32),
        'hands': np.zeros((2, 21, 2), dtype=np.float32),
    }
    
    # Placeholder landmarks
    frame_data['head_lmk_203'] = np.zeros((203, 2), dtype=np.float32)
    frame_data['head_lmk_70'] = np.zeros((70, 2), dtype=np.float32)
    frame_data['head_lmk_mp'] = np.zeros((468, 3), dtype=np.float32)
    frame_data['head_lmk_valid'] = True
    
    # Hand validity
    frame_data['left_hand_valid'] = True
    frame_data['right_hand_valid'] = True
    
    # Placeholder MANO coeffs
    frame_data['left_mano_coeffs'] = {}
    frame_data['right_mano_coeffs'] = {}
    
    return frame_data


def convert_shot_to_ehm(video_lhm_dir, video_name, shot_name, args):
    """
    Convert single shot LHM tracking data to EHM format.
    
    Args:
        video_lhm_dir: Video LHM directory
        video_name: Name of the video
        shot_name: Name of the shot
        args: Command line arguments
    
    Returns:
        Tuple of (tracking_data, id_share_params, output_info)
    """
    shot_dir = os.path.join(video_lhm_dir, shot_name)
    
    print(f"\n{'='*80}")
    print(f"Converting shot: {video_name}/{shot_name}")
    print(f"{'='*80}")
    
    if not os.path.exists(shot_dir):
        print(f"  Error: Shot directory not found: {shot_dir}")
        return None, None, None
    
    # Process this shot
    shot_tracking = load_lhm_shot_data(shot_dir, shot_name)
    
    if len(shot_tracking) == 0:
        print(f"Warning: No frames converted for shot {shot_name}. Skipping...")
        return None, None, None
    
    print(f"\nTotal frames converted: {len(shot_tracking)}")
    
    # Create id_share_params from this shot's first frame
    id_share_params = {
        'flame_shape': np.zeros((1, 300), dtype=np.float32),
        'smplx_shape': np.zeros((1, 10), dtype=np.float32),
        'left_mano_shape': np.zeros((1, 10), dtype=np.float32),
        'right_mano_shape': np.zeros((1, 10), dtype=np.float32),
        'head_scale': np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        'hand_scale': np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        'joints_offset': np.zeros((1, 55, 3), dtype=np.float32)
    }
    
    # Get first SMPL-X file from this shot
    smplx_params_dir = os.path.join(shot_dir, 'smplx_params')
    if os.path.exists(smplx_params_dir):
        smplx_files = sorted([f for f in os.listdir(smplx_params_dir) if f.endswith('.json')])
        if smplx_files:
            first_smplx_path = os.path.join(smplx_params_dir, smplx_files[0])
            with open(first_smplx_path, 'r') as f:
                first_smplx = json.load(f)
            id_share_params['smplx_shape'] = np.array([first_smplx['betas']], dtype=np.float32)
    
    # Get first FLAME file from this shot
    flame_params_dir = os.path.join(shot_dir, 'flame_params')
    if os.path.exists(flame_params_dir):
        flame_files = sorted([f for f in os.listdir(flame_params_dir) if f.endswith('.json')])
        if flame_files:
            first_flame_path = os.path.join(flame_params_dir, flame_files[0])
            with open(first_flame_path, 'r') as f:
                first_flame = json.load(f)
            id_share_params['flame_shape'] = np.array([first_flame['shapecode']], dtype=np.float32)
    
    # Create output info for this shot
    shot_key = f"{video_name}/{shot_name}"
    output_info = {
        'videos_info': {
            video_name: {
                'frames_num': len(shot_tracking),
                'frames_keys': sorted(shot_tracking.keys())
            }
        },
        'extra_info': {
            'frames_root': args.images_dir,
            'matte_root': args.mattes_dir,
            'pshuman_root': args.pshuman_dir
        }
    }
    
    return shot_tracking, id_share_params, output_info


def save_ehm_outputs(output_dir, tracking_data, id_share_params, output_info):
    """
    Save all EHM format outputs.
    
    Args:
        output_dir: Directory to save outputs
        tracking_data: Frame tracking dictionary
        id_share_params: Identity shared parameters
        output_info: Dictionary with videos_info and extra_info
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tracking data
    tracking_pkl_path = os.path.join(output_dir, 'optim_tracking_ehm.pkl')
    print(f"\n  Saving tracking data to: {tracking_pkl_path}")
    write_dict_pkl(tracking_pkl_path, tracking_data)
    
    # Save id_share_params
    id_share_pkl_path = os.path.join(output_dir, 'id_share_params.pkl')
    print(f"  Saving id_share_params to: {id_share_pkl_path}")
    write_dict_pkl(id_share_pkl_path, id_share_params)
    
    # Save videos_info.json
    videos_info_path = os.path.join(output_dir, 'videos_info.json')
    print(f"  Saving videos_info to: {videos_info_path}")
    with open(videos_info_path, 'w') as f:
        json.dump(output_info['videos_info'], f, indent=2)
    
    # Save extra_info.json
    extra_info_path = os.path.join(output_dir, 'extra_info.json')
    print(f"  Saving extra_info to: {extra_info_path}")
    with open(extra_info_path, 'w') as f:
        json.dump(output_info['extra_info'], f, indent=4)
    
    print(f"\n  ✓ All outputs saved to: {output_dir}")


def analyze_lhm_data(lhm_dir, video_name):
    """
    Analyze LHM directory structure.
    
    Args:
        lhm_dir: Root directory containing LHM tracking results
        video_name: Name of the video
    """
    video_lhm_dir = os.path.join(lhm_dir, video_name)
    
    print(f"\n{'='*80}")
    print(f"Analyzing LHM data for: {video_name}")
    print(f"LHM directory: {video_lhm_dir}")
    print(f"{'='*80}")
    
    if not os.path.exists(video_lhm_dir):
        print(f"  Error: LHM directory not found: {video_lhm_dir}")
        return None
    
    # List all shot folders
    shot_folders = sorted([d for d in os.listdir(video_lhm_dir) 
                          if os.path.isdir(os.path.join(video_lhm_dir, d))])
    
    print(f"\nFound {len(shot_folders)} shot folders:")
    for shot_folder in shot_folders:
        print(f"  - {shot_folder}")
    
    # Analyze first shot
    if shot_folders:
        first_shot = shot_folders[0]
        shot_dir = os.path.join(video_lhm_dir, first_shot)
        
        flame_params_dir = os.path.join(shot_dir, 'flame_params')
        smplx_params_dir = os.path.join(shot_dir, 'smplx_params')
        
        flame_files = []
        smplx_files = []
        
        if os.path.exists(flame_params_dir):
            flame_files = sorted([f for f in os.listdir(flame_params_dir) if f.endswith('.json')])
        if os.path.exists(smplx_params_dir):
            smplx_files = sorted([f for f in os.listdir(smplx_params_dir) if f.endswith('.json')])
        
        print(f"\n  First shot: {first_shot}")
        print(f"    FLAME params: {len(flame_files)} files")
        print(f"    SMPL-X params: {len(smplx_files)} files")
    
    return {
        'video_lhm_dir': video_lhm_dir,
        'shot_folders': shot_folders
    }


def main():
    parser = argparse.ArgumentParser(description='Convert LHM track results to EHM format (per shot) and visualize')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Root directory containing original images')
    parser.add_argument('--lhm_dir', type=str, required=True,
                        help='Root directory containing LHM tracking results (can be video dir or parent dir)')
    parser.add_argument('--mattes_dir', type=str, default=None,
                        help='Directory containing matte images (optional)')
    parser.add_argument('--pshuman_dir', type=str, default=None,
                        help='Root directory containing pshuman images')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing outputs')
    parser.add_argument('--analyze_only', action='store_true',
                        help='Only analyze LHM data structure without conversion')
    
    args = parser.parse_args()
    
    # Detect if lhm_dir is a video directory or parent directory
    print(f"\nDetecting LHM directory structure...")
    print(f"LHM directory: {args.lhm_dir}")
    
    subdirs = sorted([d for d in os.listdir(args.lhm_dir) 
                     if os.path.isdir(os.path.join(args.lhm_dir, d))])
    
    if not subdirs:
        print(f"Error: No subdirectories found in {args.lhm_dir}")
        return
    
    # Check first subdir - if it has smplx_params, we're in a video directory
    first_subdir = subdirs[0]
    first_subdir_path = os.path.join(args.lhm_dir, first_subdir)
    has_smplx_params = os.path.exists(os.path.join(first_subdir_path, 'smplx_params'))
    
    # Build list of (video_name, video_dir, shot_list)
    video_shot_list = []
    
    if has_smplx_params:
        # We're in a video directory - subdirs are shot folders
        print(f"  Detected: Single video directory (shots in subdirectories)")
        video_name = os.path.basename(args.lhm_dir)
        shot_names = subdirs
        video_shot_list.append((video_name, args.lhm_dir, shot_names))
        print(f"  Video: {video_name}")
        print(f"  Shots: {shot_names}")
    else:
        # We're in parent directory - subdirs are video folders
        print(f"  Detected: Parent directory (multiple videos)")
        for video_name in subdirs:
            video_dir = os.path.join(args.lhm_dir, video_name)
            shot_names = sorted([d for d in os.listdir(video_dir) 
                               if os.path.isdir(os.path.join(video_dir, d))])
            if shot_names:
                video_shot_list.append((video_name, video_dir, shot_names))
        print(f"  Found {len(video_shot_list)} video(s)")
        for video_name, _, shot_names in video_shot_list:
            print(f"    {video_name}: {len(shot_names)} shot(s)")
    
    # If analyze_only mode, analyze all shots and exit
    if args.analyze_only:
        for video_name, video_dir, shot_names in video_shot_list:
            analyze_lhm_data(os.path.dirname(video_dir), video_name)
        print("\n" + "="*80)
        print("LHM data analysis complete!")
        print("="*80)
        return
    
    # Initialize pipeline once for all videos/shots
    print("\nInitializing pipeline for visualization...")
    cfg = DataPreparationConfig()
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smplx_pipeline = RefineSmplxPipeline(cfg)
    
    # Process each video's shots
    total_shots = sum(len(shot_names) for _, _, shot_names in video_shot_list)
    processed_shots = 0
    
    for video_name, video_dir, shot_names in video_shot_list:
        print(f"\n{'='*80}")
        print(f"Processing video: {video_name} ({len(shot_names)} shots)")
        print(f"{'='*80}")
        
        for shot_name in shot_names:
            print(f"\n{'-'*80}")
            print(f"Processing shot: {video_name}/{shot_name}")
            print(f"{'-'*80}")
            
            # Convert shot to EHM format
            tracking_data, id_share_params, output_info = convert_shot_to_ehm(
                video_dir,
                video_name, 
                shot_name,
                args
            )
            
            if tracking_data is None:
                print(f"Failed to convert shot {video_name}/{shot_name}, skipping...")
                continue
            
            # Save EHM outputs to shot directory
            output_dir = os.path.join(video_dir, shot_name)
            save_ehm_outputs(output_dir, tracking_data, id_share_params, output_info)
            
            print(f"\nShot conversion complete: {video_name}/{shot_name}")
            print(f"Output directory: {output_dir}")
            
            # Check if final visualization already exists
            shot_key = f"{video_name}/{shot_name}"
            copy_dest = os.path.join(video_dir, f"{video_name}_{shot_name}_track_ehmx.jpg")
            if os.path.exists(copy_dest) and not args.overwrite:
                print(f"  Skipping visualization: {copy_dest} already exists (use --overwrite to regenerate)")
            else:
                # Generate visualization
                video_data = output_info['videos_info'][video_name].copy()
                video_data['fps'] = video_data.get('fps', 24)  # Default to 24 fps
                
                visualize_tracking(shot_key, video_data, output_dir, args, smplx_pipeline, cfg)
                
                # Copy visualization to video dir with renamed filename
                track_smplx_path = os.path.join(output_dir, 'track_smplx.jpg')
                if os.path.exists(track_smplx_path):
                    shutil.copy2(track_smplx_path, copy_dest)
                    print(f"  ✓ Copied visualization to: {copy_dest}")
            
            processed_shots += 1
    
    print(f"\n{'='*80}")
    print(f"All conversions complete! Processed {processed_shots}/{total_shots} shot(s)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
