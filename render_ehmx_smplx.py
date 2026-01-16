#!/usr/bin/env python3
"""
render_ehmx_smplx.py - Render SMPL-X tracking visualization
Loads EHM tracking results and generates track_smplx.jpg visualization.
"""
import os
import sys
import json
import argparse
import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image as PILImage

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.configs.data_prepare_config import DataPreparationConfig
from src.ehmx_refine_smplx import RefineSmplxPipeline
from src.utils.io import load_dict_pkl
from src.utils.crop import _transform_img
from src.utils.draw import draw_landmarks


def load_frame_image(images_dir, video_name, frame_key, pshuman_dir=None):
    """Load a frame image from disk."""
    if pshuman_dir:
        img_path = os.path.join(pshuman_dir, video_name, f"{frame_key}.png")
        if os.path.exists(img_path):
            img = PILImage.open(img_path).convert('RGB')
            return np.array(img)
    
    # Try direct path first (frame_key might include shot_name)
    img_path = os.path.join(images_dir, f"{frame_key}.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(images_dir, f"{frame_key}.jpg")
    
    # If frame_key includes shot_name (e.g., "005472_005712/000000"), images are under images_dir/shot/frame.jpg
    if not os.path.exists(img_path) and '/' in frame_key:
        parts = frame_key.split('/')
        if len(parts) == 2:
            shot_name, frame_num = parts
            img_path = os.path.join(images_dir, shot_name, f"{frame_num}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(images_dir, shot_name, f"{frame_num}.jpg")
        else:
            # Handle paths like "shot/frame/pshuman_02" - just use shot and frame
            shot_name, frame_num = parts[0], parts[1]
            img_path = os.path.join(images_dir, shot_name, f"{frame_num}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(images_dir, shot_name, f"{frame_num}.jpg")
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img = PILImage.open(img_path).convert('RGB')
    return np.array(img)


def load_body_images(video_name, frames_keys, tracking_results, images_dir, crop_size=1024):
    """
    Load and crop body images for all frames.
    
    Args:
        video_name: Name of the video
        frames_keys: List of frame keys
        tracking_results: Tracking results with crop info
        images_dir: Directory containing original images
        crop_size: Size of the cropped image (default: 1024)
    
    Returns:
        Dictionary mapping frame_key to cropped body image tensors
    """
    body_images = {}
    
    print(f"  Loading {len(frames_keys)} body images...")
    for frame_key in tqdm(frames_keys):
        if frame_key not in tracking_results:
            continue
        
        # Load original image
        try:
            img_rgb = load_frame_image(images_dir, video_name, frame_key)
        except FileNotFoundError:
            print(f"    Warning: Image not found for {frame_key}, skipping")
            continue
        
        # Crop body image using saved M_o2c transformation matrix
        body_crop_info = tracking_results[frame_key]['body_crop']
        M_o2c_hd = body_crop_info['M_o2c-hd']
        
        # Apply affine transformation
        body_img_cropped = _transform_img(img_rgb, M_o2c_hd, dsize=crop_size)
        
        # Convert to tensor (C, H, W)
        body_img_tensor = torch.from_numpy(body_img_cropped).permute(2, 0, 1).float()
        body_images[frame_key] = body_img_tensor
    
    print(f"  Loaded {len(body_images)} body images")
    return body_images


def render_smplx_visualization(video_name, tracking_data, body_images, output_path, cfg, smplx_pipeline, id_share_params):
    """
    Render SMPL-X visualization directly without optimization.
    
    Args:
        video_name: Name of the video
        tracking_data: Dictionary with tracking results
        body_images: Dictionary of body image tensors
        output_path: Path to save the visualization
        cfg: DataPreparationConfig instance
        smplx_pipeline: RefineSmplxPipeline instance
        id_share_params: Dictionary with identity-shared parameters
    """
    print(f"\n  Rendering SMPL-X visualization...")
    
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
        flame_coeffs = frame_data['flame_coeffs']
        
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
            'exp': torch.from_numpy(flame_coeffs['expression_params']).unsqueeze(0).to(cfg.device),  # Use FLAME expression
        }
        
        # Prepare batch_flame dictionary
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
        
        # Forward pass through EHM model (combines SMPL-X and FLAME)
        with torch.no_grad():
            smplx_dict = smplx_pipeline.ehm(
                batch_smplx,
                batch_flame,
                pose_type='aa'  # axis-angle format
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
        R, T = camera_RT.split([3, 1], dim=-1)  # R: [1, 3, 3], T: [1, 3, 1]
        T = T.squeeze(-1)  # T: [1, 3]
        
        # Create camera for projection
        cameras_kwargs = smplx_pipeline.build_cameras_kwargs(1, smplx_pipeline.body_focal_length)
        cameras = GS_Camera(**cameras_kwargs).to(cfg.device)
        
        # Project vertices and joints using the same method as optimization
        proj_vertices = cameras.transform_points_screen(smplx_vertices, R=R, T=T)
        proj_joints = cameras.transform_points_screen(smplx_joints, R=R, T=T)
        
        # Panel 1: Original image with keypoints
        img_panel1 = img_np.copy()
        
        # Draw projected SMPL-X joints
        smplx_kpts_2d_np = proj_joints[0, :, :2].cpu().numpy()
        for kpt in smplx_kpts_2d_np:
            x, y = int(kpt[0]), int(kpt[1])
            if 0 <= x < img_size and 0 <= y < img_size:
                cv2.circle(img_panel1, (x, y), 3, (255, 0, 0), -1)
        
        # Panel 2: Mesh rendering
        # Create camera for rendering
        lights = PointLights(device=cfg.device, location=[[0.0, -1.0, -10.0]])
        t_camera = GS_Camera(**cameras_kwargs, R=R, T=T)
        
        # Render mesh
        mesh_img = smplx_pipeline.body_renderer.render_mesh(
            smplx_vertices, t_camera, lights=lights
        )
        mesh_img = (mesh_img[:, :3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1, 2, 0)
        
        # Blend mesh with original image
        img_panel2 = cv2.addWeighted(img_np, 0.3, mesh_img, 0.7, 0)
        
        # Panel 3: Original image (hand details - simplified for now)
        img_panel3 = img_np.copy()
        
        # Draw SMPL-X keypoints on panel 3 as well
        for kpt in smplx_kpts_2d_np:
            x, y = int(kpt[0]), int(kpt[1])
            if 0 <= x < img_size and 0 <= y < img_size:
                cv2.circle(img_panel3, (x, y), 2, (0, 255, 0), -1)
        
        # Concatenate three panels horizontally
        vis_img = np.concatenate([img_panel1, img_panel2, img_panel3], axis=1)
        vis_imgs.append(vis_img)
    
    # Create grid layout
    if len(vis_imgs) > 0:
        # Calculate 1:2 aspect ratio grid layout
        total_frames = len(vis_imgs)
        grid_rows = max(1, int(np.sqrt(total_frames * 2)))
        grid_cols = int(np.ceil(total_frames / grid_rows))
        
        # Adjust to maintain 1:2 ratio
        while grid_rows < 2 * grid_cols and grid_cols > 1:
            grid_cols -= 1
            grid_rows = int(np.ceil(total_frames / grid_cols))
        
        # Get dimensions
        img_height, img_width = vis_imgs[0].shape[:2]
        grid_height = grid_rows * img_height
        grid_width = grid_cols * img_width
        
        # Create blank grid
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Place images in grid
        for idx, img in enumerate(vis_imgs):
            row = idx // grid_cols
            col = idx % grid_cols
            y_start = row * img_height
            y_end = y_start + img_height
            x_start = col * img_width
            x_end = x_start + img_width
            grid[y_start:y_end, x_start:x_end] = img
        
        # Save grid image
        cv2.imwrite(output_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  ✓ Visualization saved: {output_path}")
        return True
    
    return False


def main():
    parser = argparse.ArgumentParser(description='Render SMPL-X tracking visualization')
    parser.add_argument('--ehmx_dir', type=str, required=True,
                        help='Directory containing EHM tracking results')
    parser.add_argument('--images_dir', type=str, default=None,
                        help='Directory containing original images (will use extra_info.json if not provided)')
    parser.add_argument('--output_name', type=str, default='track_smplx.jpg',
                        help='Output filename (default: track_smplx.jpg)')
    
    args = parser.parse_args()
    
    # Check if ehmx_dir exists
    if not os.path.exists(args.ehmx_dir):
        print(f"Error: EHM-X directory not found: {args.ehmx_dir}")
        return
    
    # Extract video name from path
    video_name = os.path.basename(args.ehmx_dir)
    print(f"\n{'='*80}")
    print(f"Rendering visualization for: {video_name}")
    print(f"EHM-X directory: {args.ehmx_dir}")
    print(f"{'='*80}")
    
    # Load tracking results
    tracking_pkl_path = os.path.join(args.ehmx_dir, 'optim_tracking_ehm.pkl')
    if not os.path.exists(tracking_pkl_path):
        print(f"Error: Tracking results not found: {tracking_pkl_path}")
        return
    
    print(f"\nLoading tracking results from: {tracking_pkl_path}")
    tracking_data = load_dict_pkl(tracking_pkl_path)
    print(f"  Loaded {len(tracking_data)} frames")
    
    # Load extra_info to get images directory
    extra_info_path = os.path.join(args.ehmx_dir, 'extra_info.json')
    if args.images_dir is None:
        if os.path.exists(extra_info_path):
            with open(extra_info_path, 'r') as f:
                extra_info = json.load(f)
            args.images_dir = extra_info.get('frames_root', None)
            print(f"  Using images_dir from extra_info: {args.images_dir}")
        
        if args.images_dir is None:
            print(f"Error: images_dir not provided and not found in extra_info.json")
            return
    
    # Load body images
    frames_keys = sorted(tracking_data.keys())
    body_images = load_body_images(video_name, frames_keys, tracking_data, args.images_dir)
    
    if len(body_images) == 0:
        print(f"Error: No body images loaded")
        return
    
    # Initialize pipeline
    print(f"\nInitializing SMPL-X pipeline...")
    cfg = DataPreparationConfig()
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {cfg.device}")
    
    smplx_pipeline = RefineSmplxPipeline(cfg)
    
    # Load id_share_params
    id_share_pkl_path = os.path.join(args.ehmx_dir, 'id_share_params.pkl')
    if not os.path.exists(id_share_pkl_path):
        print(f"Error: id_share_params not found: {id_share_pkl_path}")
        return
    
    print(f"\nLoading id_share_params from: {id_share_pkl_path}")
    id_share_params = load_dict_pkl(id_share_pkl_path)
    print(f"  Loaded identity parameters:")
    for k, v in id_share_params.items():
        if hasattr(v, 'shape'):
            print(f"    {k}: shape={v.shape}")
    
    # Render visualization
    output_path = os.path.join(args.ehmx_dir, args.output_name)
    success = render_smplx_visualization(
        video_name, tracking_data, body_images, output_path, cfg, smplx_pipeline, id_share_params
    )
    
    if success:
        print(f"\n{'='*80}")
        print(f"Visualization complete!")
        print(f"Output: {output_path}")
        print(f"{'='*80}")
    else:
        print(f"\nError: Visualization failed")


if __name__ == '__main__':
    main()
