"""
ehmx_refine_flame.py - FLAME Refinement Pipeline
Refines FLAME parameters from base tracking results.
"""
import torch
import numpy as np
import cv2
import os
import torch.nn.functional as nnfunc
from tqdm.auto import tqdm
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from .modules.flame import FLAME
from .modules.renderer.head_renderer import Renderer
from .utils.rprint import rlog as log
from .utils.graphics import GS_Camera
from .utils.helper import build_minibatch
from .losses import Landmark2DLoss
from .utils.draw import draw_landmarks

np.random.seed(0)


def parse_frame_key(frame_key):
    """
    Parse frame key into video_name, shot_id, frame_id.
    
    Args:
        frame_key: String in format "video_name_shotid/frameid" or "video_name_shotid"
    
    Returns:
        Tuple of (video_name, shot_id, frame_id)
    """
    parts = frame_key.split('_')
    last_part = parts[-1]
    
    # Check if last part contains frame_id (with /)
    if '/' in last_part:
        shot_and_frame = last_part.split('/')
        shot_id = shot_and_frame[0]
        frame_id = shot_and_frame[1]
        video_name = '_'.join(parts[:-1])
    else:
        shot_id = last_part
        frame_id = '000000'
        video_name = '_'.join(parts[:-1])
    
    return video_name, shot_id, frame_id


def group_frames_by_shots(frame_keys):
    """
    Group frame indices by shots for temporal regularization.
    
    Args:
        frame_keys: List of frame keys
    
    Returns:
        List of lists, where each inner list contains frame indices belonging to the same shot
    """
    shot_groups = {}
    for idx, frame_key in enumerate(frame_keys):
        video_name, shot_id, frame_id = parse_frame_key(frame_key)
        shot_key = f"{video_name}_{shot_id}"
        if shot_key not in shot_groups:
            shot_groups[shot_key] = []
        shot_groups[shot_key].append(idx)
    
    # Return as list of lists
    return list(shot_groups.values())


class RefineFlamePipeline(object):
    """Pipeline for refining FLAME parameters from base tracking results."""
    
    def __init__(self, cfg):
        """
        Initialize FLAME refinement pipeline.
        
        Args:
            cfg: DataPreparationConfig instance
        """
        self.cfg = cfg
        self.device = cfg.device
        self.image_size = cfg.head_crop_size
        self.focal_length = 1 / cfg.tanfov
        
        # Initialize FLAME model and renderer
        self.flame = FLAME(cfg.flame_assets_dir).to(self.device)
        self.renderer = Renderer(cfg.flame_assets_dir, self.image_size, 
                                 focal_length=self.focal_length).to(self.device)
        
        # Initialize landmark loss
        self.face2d_lmk_distance = Landmark2DLoss(
            self.flame.lmk_203_left_indices,
            self.flame.lmk_203_right_indices,
            self.flame.lmk_203_front_indices,
            self.flame.lmk_mp_indices
        )
        
        self.saving_root = None

        # NOTE: pixel3dmm refinement was tested but removed — it shifts FLAME params
        # too far from TEASER init, causing landmark loss to start 10x higher (14426 vs 1103).
        # The landmark optimizer can't fully recover, resulting in worse mesh alignment.
        # TEASER init + landmark optimization alone gives better results.
    
    def build_cameras_kwargs(self, batch_size):
        """Build camera parameters for rendering."""
        screen_size = torch.tensor([self.image_size, self.image_size], 
                                   device=self.device).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': torch.zeros(batch_size, 2, device=self.device).float(),
            'focal_length': self.focal_length,
            'image_size': screen_size,
            'device': self.device,
        }
        return cameras_kwargs
    
    def _visualize_frames(self, batch_imgs, ret_dict, landmark_lst_dct, cameras, R, T,
                         head_lmk_valid, frame_indices=None):
        """
        Visualize frames with landmarks and mesh overlay.
        
        Args:
            batch_imgs: Batch of images tensors
            ret_dict: FLAME forward pass results
            landmark_lst_dct: Dictionary of ground truth landmarks
            cameras: Camera object
            R: Rotation matrices
            T: Translation vectors
            head_lmk_valid: Boolean mask of valid landmarks
            frame_indices: Optional list of frame indices to visualize (if None, visualize all)
        
        Returns:
            List of visualization images
        """
        if frame_indices is None:
            frame_indices = range(len(batch_imgs))
        
        vis_imgs = []
        k = 'lmk_mp'  # Use mediapipe landmarks
        
        for im_idx in frame_indices:
            t_lmk = cameras.transform_points_screen(ret_dict[k], R=R, T=T)[..., :2]
            _img = batch_imgs[im_idx].clone().numpy().transpose(1, 2, 0).astype(np.uint8).copy()
            _t_lmk_mp = t_lmk[im_idx].detach().cpu().numpy()
            _landmark_mp = landmark_lst_dct[k][im_idx, self.face2d_lmk_distance.selected_mp_indices].detach().cpu().numpy()
            
            # Render mesh
            t_cameras = GS_Camera(R=R[im_idx][None], T=T[im_idx][None],
                                 **self.build_cameras_kwargs(1)).to(self.device)
            mesh_img = self.renderer.render_mesh(ret_dict['vertices'][im_idx, None], cameras=t_cameras)
            mesh_img = (mesh_img[:, :3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1, 2, 0)
            
            # Blend mesh with image
            blended_img = cv2.addWeighted(_img, 0.6, mesh_img, 0.4, 0)
            
            # Draw landmarks
            if head_lmk_valid[im_idx]:
                blended_img = draw_landmarks(_t_lmk_mp, blended_img, color=(255, 0, 0), radius=1)
            blended_img = draw_landmarks(_landmark_mp, blended_img, color=(0, 255, 0), radius=1)
            
            vis_imgs.append(blended_img)
        
        return vis_imgs
    
    def _compute_temporal_loss(self, params_dict, shot_indices, lambdas, interval=1):
        """
        Compute temporal regularization loss within shots.
        
        Args:
            params_dict: Dictionary of parameters to regularize
            shot_indices: List of lists containing frame indices for each shot
            lambdas: Dictionary of lambda weights for each parameter
            interval: Frame interval for scaling
        
        Returns:
            Total temporal loss
        """
        temp_loss = 0
        
        for shot_idx_list in shot_indices:
            if len(shot_idx_list) <= 1:
                continue
            
            # Sort indices to ensure temporal order
            shot_idx_list = sorted(shot_idx_list)
            
            for param_name, param_tensor in params_dict.items():
                if param_name not in lambdas or lambdas[param_name] == 0:
                    continue
                
                # Extract parameters for this shot
                shot_params = param_tensor[shot_idx_list]
                
                # Compute temporal difference
                if param_name in ['gl_R_6d', 'gl_T', 'gl_T_z']:
                    # Use L1 loss for camera parameters
                    if param_name == 'gl_T_z':
                        temp_loss += torch.mean(nnfunc.l1_loss(shot_params[1:] - shot_params[:-1], 
                                                               shot_params[1:] * 0)) * lambdas[param_name] / interval
                    else:
                        temp_loss += torch.mean(nnfunc.l1_loss(shot_params[1:] - shot_params[:-1], 
                                                               shot_params[1:] * 0)) * lambdas[param_name] / interval
                else:
                    # Use L2 loss for FLAME parameters
                    temp_loss += torch.mean(torch.square(shot_params[1:] - shot_params[:-1])) * lambdas[param_name] / interval
        
        return temp_loss
    
    def _run_optimization_loop(self, opt_config, batch_flame, g_flame_shape, cameras, 
                              landmark_lst_dct, head_lmk_valid, shot_indices, 
                              batch_id, interval, batch_imgs=None, **extra_params):
        """
        Run optimization loop with given configuration.
        
        Args:
            opt_config: Dictionary with optimization configuration:
                - params: Dict mapping param name to (param_tensor, learning_rate)
                - lambda_config: Dict of lambda weights
                - steps: Number of optimization steps
                - scheduler_config: Optional scheduler configuration
                - save_prefix: Prefix for visualization saving
            batch_flame: Dictionary of FLAME parameters
            g_flame_shape: Global shape parameters
            cameras: Camera object
            landmark_lst_dct: Dictionary of ground truth landmarks
            head_lmk_valid: Boolean mask of valid landmarks
            shot_indices: List of lists of frame indices per shot
            batch_id: Batch ID for logging
            interval: Frame interval for temporal regularization
            batch_imgs: Optional batch of images for visualization
            **extra_params: Additional parameters (e.g., gl_T, gl_R_6d, eye_pose_code)
        
        Returns:
            Dictionary of optimized parameters
        """
        steps = opt_config['steps']
        params_config = opt_config['params']
        lambda_config = opt_config['lambda_config']
        # Per-set landmark validity (fallback frames only have lmk_fan from Sapiens)
        lmk_valid_per_set = opt_config.get('lmk_valid_per_set', {})
        
        batch_size = batch_flame['camera_RT_params'].shape[0]
        
        # Collect parameters to optimize
        opt_params = []
        param_dict = {}
        
        for param_name, (param_tensor, lr) in params_config.items():
            param_tensor.requires_grad = True
            param_dict[param_name] = param_tensor
            opt_params.append({'params': [param_tensor], 'lr': lr})
        
        # Create optimizer
        optimizer = torch.optim.AdamW(opt_params)
        
        # Create scheduler if specified
        scheduler = None
        if 'scheduler_config' in opt_config:
            sched_cfg = opt_config['scheduler_config']
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=sched_cfg['step_size'], gamma=sched_cfg['gamma']
            )
        
        # Optimization loop
        desc = opt_config.get('desc', 'Optimizing')
        t_bar = tqdm(range(steps), desc=desc)
        
        for i_step in t_bar:
            # Update batch_flame with current parameters
            if 'update_flame' in opt_config:
                opt_config['update_flame'](batch_flame, param_dict, batch_size)
            
            # Forward pass
            ret_dict = self.flame(batch_flame)
            
            # Compute landmark loss
            lmk_losses = []
            R = param_dict.get('gl_R', batch_flame['camera_RT_params'][:, :3, :3])
            T = param_dict.get('gl_T', batch_flame['camera_RT_params'][:, :3, 3])
            
            if 'gl_R_6d' in param_dict:
                R = rotation_6d_to_matrix(param_dict['gl_R_6d'])
            
            for k in landmark_lst_dct.keys():
                # Per-set validity mask (lmk_203/lmk_mp need MediaPipe, lmk_fan can use Sapiens fallback)
                k_valid = lmk_valid_per_set.get(k, head_lmk_valid)
                if k_valid.sum() == 0:
                    continue
                # Get landmark key for gaze optimization
                lmk_key = k
                if 'gaze_mode' in opt_config and opt_config['gaze_mode']:
                    t_gaze_landmarks, t_gaze_idx = opt_config['get_gaze_landmarks'](ret_dict)
                    t_lmk = cameras.transform_points_screen(t_gaze_landmarks[k], R=R, T=T)[..., :2]
                    t_lmk = t_lmk[k_valid]
                    g_lmk = landmark_lst_dct[k][k_valid][:, t_gaze_idx[k], :]
                else:
                    t_lmk = cameras.transform_points_screen(ret_dict[k], R=R, T=T)[..., :2]
                    t_lmk = t_lmk[k_valid]
                    g_lmk = landmark_lst_dct[k][k_valid]

                cam = batch_flame['cam'][k_valid]
                t_w = lambda_config.get('lambda_lmk_2d', 1.0) * 2 * 5
                if '203' in k:
                    t_w = lambda_config.get('lambda_lmk_203', 1.0) * 10 * 5
                lmk_losses.append(t_w * self.face2d_lmk_distance(t_lmk, g_lmk, cam=cam))
            
            landmark_loss = sum(lmk_losses)
            
            # Compute temporal loss
            temp_loss_params = {}
            temp_loss_lambdas = {}
            
            for param_name in opt_config.get('temporal_params', []):
                if param_name in param_dict:
                    temp_loss_params[param_name] = param_dict[param_name]
                    lambda_key = f'lambda_{param_name}_motion'
                    temp_loss_lambdas[param_name] = lambda_config.get(lambda_key, 0)
            
            temp_loss = self._compute_temporal_loss(temp_loss_params, shot_indices, 
                                                   temp_loss_lambdas, interval)
            
            # Compute regularization loss
            reg_loss = 0
            if 'expression_params' in param_dict:
                reg_loss += torch.mean(torch.square(param_dict['expression_params'])) * 0.5 * lambda_config.get('lambda_exp_reg', 0.01)
            if 'g_flame_shape' in param_dict:
                reg_loss += torch.mean(torch.square(param_dict['g_flame_shape'])) * 5 * lambda_config.get('lambda_shape_reg', 0.01)
            if 'jaw_params' in param_dict:
                reg_loss += torch.mean(torch.square(param_dict['jaw_params'][..., 1:])) * 1e4 * 0.5 * lambda_config.get('lambda_pose_reg', 1.0)
            
            total_loss = landmark_loss + reg_loss + temp_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # Logging
            if i_step % 50 == 0:
                t_bar.set_description(
                    f'Batch {batch_id} | Iter {i_step} => Loss {total_loss:.4f} | '
                    f'Lmk {landmark_loss:.4f} | Reg {reg_loss:.4f} | Temp {temp_loss:.4f}'
                )
            
            # Visualization
            if batch_imgs is not None and i_step % (steps - 1) == 0 and self.saving_root is not None:
                vis_imgs = self._visualize_frames(batch_imgs, ret_dict, landmark_lst_dct, 
                                                 cameras, R, T, head_lmk_valid)
                self._save_visualization_grid(vis_imgs, batch_id, i_step, 
                                             opt_config.get('save_prefix', 'optim'))
        
        return param_dict
    
    def optimize(self, track_frames, batch_base, id_share_params_result, optim_cfg,
                 batch_id=0, batch_imgs=None, interval=1):
        """
        Optimize FLAME parameters for a batch of frames.
        
        Args:
            track_frames: List of frame keys
            batch_base: Dictionary with flame_coeffs and landmarks
            id_share_params_result: Shared identity parameters
            optim_cfg: Optimization configuration
            batch_id: Batch index for logging
            batch_imgs: Optional batch of images for visualization
            interval: Frame interval for temporal regularization
        
        Returns:
            Tuple of (optimized_results, updated_id_share_params)
        """
        steps = optim_cfg.get('steps', 1001)
        share_id = optim_cfg.get('share_id', True)
        share_pose = optim_cfg.get('share_pose', False)
        
        # Extract lambda weights from config
        lambda_lmk_2d = optim_cfg.get('lambda_lmk_2d', 1.0)
        lambda_lmk_203 = optim_cfg.get('lambda_lmk_203', 1.0)
        lambda_shape_reg = optim_cfg.get('lambda_shape_reg', 0.01)
        lambda_exp_reg = optim_cfg.get('lambda_exp_reg', 0.01)
        lambda_pose_reg = optim_cfg.get('lambda_pose_reg', 1.0)
        lambda_motion_reg = optim_cfg.get('lambda_motion_reg', 10.0)

        batch_size = len(track_frames)
        batch_flame = batch_base['flame_coeffs']
        gt_lmk_203 = batch_base['head_lmk_203']
        gt_lmk_fan = batch_base['head_lmk_70']
        gt_lmk_mp = batch_base['head_lmk_mp']
        
        g_flame_shape = torch.tensor(id_share_params_result['flame_shape'], device=self.device)
        if share_id:
            g_flame_shape = g_flame_shape.mean(dim=0, keepdim=True)
            batch_flame['shape_params'] = g_flame_shape.expand(batch_size, -1)
        
        landmark_lst_dct = dict(lmk_203=gt_lmk_203, lmk_fan=gt_lmk_fan, lmk_mp=gt_lmk_mp)
        head_lmk_valid = batch_base['head_lmk_valid']
        # Per-set validity: lmk_fan uses Sapiens (valid for most frames),
        # lmk_203/lmk_mp use MediaPipe (only valid when MP detected the face)
        lmk_valid_per_set = {
            'lmk_203': batch_base.get('head_lmk_valid_mp', head_lmk_valid).clone(),
            'lmk_fan': head_lmk_valid.clone(),
            'lmk_mp': batch_base.get('head_lmk_valid_mp', head_lmk_valid).clone(),
        }

        # Early return if no valid landmarks
        if head_lmk_valid.sum() == 0:
            print(f"Batch {batch_id}: No valid landmarks, skipping optimization")
            batch_flame = {k: v.squeeze(1) for k, v in batch_flame.items()}
            optim_results = {}
            for idx, name in enumerate(track_frames):
                optim_results[name] = {
                    'expression_params': batch_flame['expression_params'][idx].detach().float().cpu().numpy(),
                    'jaw_params': batch_flame['jaw_params'][idx].detach().float().cpu().numpy(),
                    'neck_pose_params': batch_flame['pose_params'][idx].detach().float().cpu().numpy() * 0,
                    'eye_pose_params': torch.zeros_like(torch.cat([batch_flame['jaw_params'][idx]] * 2)).detach().float().cpu().numpy(),
                    'eyelid_params': batch_flame['eyelid_params'][idx].detach().float().cpu().numpy(),
                    'pose_params': batch_flame['pose_params'][idx].detach().float().cpu().numpy(),
                    'camera_RT_params': batch_flame['camera_RT_params'][idx].detach().float().cpu().numpy(),
                    'cam': batch_flame['cam'][idx].detach().float().cpu().numpy(),
                }
            return optim_results, id_share_params_result
        
        # Group frames by shots for temporal regularization
        shot_indices = group_frames_by_shots(track_frames)
        
        batch_flame = {k: v.squeeze(1) for k, v in batch_flame.items()}
        expression_params = batch_flame['expression_params']
        eyelid_params = batch_flame['eyelid_params']
        jaw_params = batch_flame['jaw_params']
        pose_params = batch_flame['pose_params']

        if share_pose and head_lmk_valid.sum() > 0:
            expression_params = expression_params[head_lmk_valid].mean(dim=0, keepdim=True)
            eyelid_params = eyelid_params[head_lmk_valid].mean(dim=0, keepdim=True)
            jaw_params = jaw_params[head_lmk_valid].mean(dim=0, keepdim=True)
            batch_flame['expression_params'] = expression_params.expand(batch_size, -1)
            batch_flame['eyelid_params'] = eyelid_params.expand(batch_size, -1)
            batch_flame['jaw_params'] = jaw_params.expand(batch_size, -1)

        # Prepare camera parameters
        gl_T = batch_flame['camera_RT_params'][:, :3, 3].detach().clone()
        gl_R_6d = matrix_to_rotation_6d(batch_flame['camera_RT_params'][:, :3, :3]).detach().clone()
        
        cameras_kwargs = self.build_cameras_kwargs(batch_size)
        cameras = GS_Camera(**cameras_kwargs).to(self.device)
        
        # Lambda configuration
        lambda_config = {
            'lambda_lmk_2d': lambda_lmk_2d,
            'lambda_lmk_203': lambda_lmk_203,
            'lambda_shape_reg': lambda_shape_reg,
            'lambda_exp_reg': lambda_exp_reg,
            'lambda_pose_reg': lambda_pose_reg,
            'lambda_motion_reg': lambda_motion_reg,
            'lambda_expression_params_motion': 10 * lambda_motion_reg if share_id else 0,
            'lambda_jaw_params_motion': 1e4 * lambda_motion_reg if share_id else 0,
            'lambda_pose_params_motion': 1e4 * 0.5 * lambda_motion_reg if share_id else 0,
            'lambda_gl_R_6d_motion': 1e3 * 0.5 * lambda_motion_reg if share_id else 0,
            'lambda_gl_T_motion': 1e3 * lambda_motion_reg if share_id else 0,
            'lambda_gl_T_z_motion': 5e3 * lambda_motion_reg if share_id else 0,
        }
        
        # FLAME parameter optimization configuration
        flame_opt_config = {
            'params': {
                'g_flame_shape': (g_flame_shape, 1e-4),
                'expression_params': (expression_params, 1e-3),
                'eyelid_params': (eyelid_params, 2e-4),
                'jaw_params': (jaw_params, 1e-3),
                'pose_params': (pose_params, 1e-3),
                'gl_T': (gl_T, 1e-5),
                'gl_R_6d': (gl_R_6d, 1e-5),
            },
            'temporal_params': ['expression_params', 'jaw_params', 'pose_params', 'gl_R_6d', 'gl_T', 'gl_T_z'],
            'lambda_config': lambda_config,
            'steps': steps,
            'desc': 'Tuning FLAME params',
            'save_prefix': 'flame',
            'update_flame': lambda bf, pd, bs: self._update_flame_params(bf, pd, bs),
            'lmk_valid_per_set': lmk_valid_per_set,
        }

        # Run FLAME optimization
        param_dict = self._run_optimization_loop(
            flame_opt_config, batch_flame, g_flame_shape, cameras,
            landmark_lst_dct, head_lmk_valid, shot_indices, batch_id, interval, batch_imgs
        )
        
        # Update camera parameters
        batch_flame['camera_RT_params'][:, :3, 3] = param_dict['gl_T'].detach()
        batch_flame['camera_RT_params'][:, :3, :3] = rotation_6d_to_matrix(param_dict['gl_R_6d']).detach()
        g_flame_shape = param_dict['g_flame_shape']
        batch_flame['shape_params'] = g_flame_shape.expand(batch_size, -1)
        # Detach all parameters
        for k in batch_flame.keys():
            if isinstance(batch_flame[k], torch.Tensor):
                batch_flame[k] = batch_flame[k].clone().detach()
        
        # Eye pose optimization configuration
        eye_pose_code = torch.zeros_like(torch.cat([batch_flame['jaw_params']] * 2, dim=1))
        if share_pose and head_lmk_valid.sum() > 0:
            eye_pose_code = eye_pose_code[head_lmk_valid].mean(dim=0, keepdim=True)
        batch_flame['eye_pose_params'] = eye_pose_code.expand(batch_size, -1)
        
        def get_gaze_landmarks(ret_dict):
            pred_vertices = ret_dict['vertices']
            pred_lmk_203_gaze, index_gaze_203 = self.flame.reselect_eyes(pred_vertices, 'lmks203')
            pred_lmk_mp_gaze, index_gaze_mp = self.flame.reselect_eyes(pred_vertices, 'lmks_mp')
            pred_lmk_fan_gaze, index_gaze_fan = self.flame.reselect_eyes(pred_vertices, 'lmks68')
            t_gaze_landmarks = {'lmk_203': pred_lmk_203_gaze, 'lmk_mp': pred_lmk_mp_gaze, 'lmk_fan': pred_lmk_fan_gaze}
            t_gaze_idx = {'lmk_203': index_gaze_203, 'lmk_mp': index_gaze_mp, 'lmk_fan': index_gaze_fan}
            return t_gaze_landmarks, t_gaze_idx
        
        gaze_opt_config = {
            'params': {
                'eye_pose_code': (eye_pose_code, 0.01),
            },
            'temporal_params': ['eye_pose_code'],
            'lambda_config': {
                **lambda_config,
                'lambda_eye_pose_code_motion': 1e6 if share_id else 0,
            },
            'steps': steps // 2 + 1,
            'desc': 'Tuning FLAME eyepose params',
            'save_prefix': 'gaze',
            'scheduler_config': {'step_size': 501, 'gamma': 0.1},
            'gaze_mode': True,
            'get_gaze_landmarks': get_gaze_landmarks,
            'update_flame': lambda bf, pd, bs: bf.update({'eye_pose_params': pd['eye_pose_code'].expand(bs, -1)}),
            'lmk_valid_per_set': lmk_valid_per_set,
        }
        
        log('Start tuning FLAME eyepose params')
        param_dict = self._run_optimization_loop(
            gaze_opt_config, batch_flame, g_flame_shape, cameras,
            landmark_lst_dct, head_lmk_valid, shot_indices, batch_id, interval, batch_imgs
        )

        # Package optimized results
        optim_results = {}
        for idx, name in enumerate(track_frames):
            if head_lmk_valid.sum() == 0:
                _idx = 0
            else:
                if share_pose:
                    _idx = head_lmk_valid.nonzero(as_tuple=False)[0][0]
                else:
                    _idx = idx
                    if not head_lmk_valid[idx]:
                        _idx = head_lmk_valid.nonzero(as_tuple=False)[0][0]
            
            optim_results[name] = {
                'expression_params': batch_flame['expression_params'][_idx].detach().float().cpu().numpy(),
                'jaw_params': batch_flame['jaw_params'][_idx].detach().float().cpu().numpy(),
                'neck_pose_params': batch_flame['pose_params'][_idx].detach().float().cpu().numpy() * 0,
                'eye_pose_params': batch_flame['eye_pose_params'][_idx].detach().float().cpu().numpy(),
                'eyelid_params': batch_flame['eyelid_params'][_idx].detach().float().cpu().numpy(),
                'pose_params': batch_flame['pose_params'][idx].detach().float().cpu().numpy(),
                'camera_RT_params': batch_flame['camera_RT_params'][idx].detach().float().cpu().numpy(),
                'cam': batch_flame['cam'][idx].detach().float().cpu().numpy(),
            }
        
        id_share_params_result['flame_shape'] = g_flame_shape.detach().float().cpu().numpy()
        return optim_results, id_share_params_result
    
    def _update_flame_params(self, batch_flame, param_dict, batch_size):
        """Update batch_flame with optimized parameters."""
        batch_flame['shape_params'] = param_dict['g_flame_shape'].expand(batch_size, -1)
        batch_flame['expression_params'] = param_dict['expression_params'].expand(batch_size, -1)
        batch_flame['eyelid_params'] = param_dict['eyelid_params'].expand(batch_size, -1)
        batch_flame['jaw_params'] = param_dict['jaw_params'].expand(batch_size, -1)
        batch_flame['pose_params'] = param_dict['pose_params'].expand(batch_size, -1)
    
    def _save_visualization_grid(self, vis_imgs, batch_id, i_step, prefix):
        """Save visualization images as a grid."""
        save_path = os.path.join(self.saving_root, "visual_results")
        os.makedirs(save_path, exist_ok=True)
        
        # Calculate grid layout (2:1 aspect ratio)
        n_imgs = len(vis_imgs)
        grid_rows = max(1, int(np.sqrt(n_imgs / 2)))
        grid_cols = int(np.ceil(n_imgs / grid_rows))
        
        while grid_cols < 2 * grid_rows and grid_rows > 1:
            grid_rows -= 1
            grid_cols = int(np.ceil(n_imgs / grid_rows))
        
        # Get image dimensions
        img_height, img_width = vis_imgs[0].shape[:2]
        
        # Create grid
        grid = np.zeros((grid_rows * img_height, grid_cols * img_width, 3), dtype=np.uint8)
        
        for idx, img in enumerate(vis_imgs):
            row = idx // grid_cols
            col = idx % grid_cols
            y_start = row * img_height
            y_end = y_start + img_height
            x_start = col * img_width
            x_end = x_start + img_width
            grid[y_start:y_end, x_start:x_end] = img
        
        cv2.imwrite(os.path.join(save_path, f"vis_fit_{prefix}_bid-{batch_id}_stp-{i_step}.jpg"),
                   cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    
    def generate_preview(self, frames_keys, base_results, refined_results, head_images, save_path, id_share_params=None):
        """
        Generate preview visualization grid with FLAME rendering.
        
        Args:
            frames_keys: List of frame keys
            base_results: Base tracking results
            refined_results: Refined tracking results with flame_coeffs
            head_images: Dictionary of head image tensors (C, H, W)
            save_path: Path to save the preview image
            id_share_params: Optional identity shared parameters (for shape_params)
        """
        print(f"  Generating preview grid with FLAME rendering...")
        
        # Collect all valid frames
        preview_images = []
        valid_keys = []
        
        for frame_key in frames_keys:
            if frame_key not in head_images or frame_key not in refined_results:
                continue
            valid_keys.append(frame_key)
        
        if len(valid_keys) == 0:
            print(f"  Warning: No valid frames for preview")
            return
        
        # Calculate 2:1 aspect ratio grid layout
        total_frames = len(valid_keys)
        grid_rows = max(1, int(np.sqrt(total_frames / 2)))
        grid_cols = int(np.ceil(total_frames / grid_rows))
        
        # Adjust to maintain 2:1 ratio
        while grid_cols < 2 * grid_rows and grid_rows > 1:
            grid_rows -= 1
            grid_cols = int(np.ceil(total_frames / grid_rows))
        
        # Build camera kwargs
        cameras_kwargs = self.build_cameras_kwargs(1)
        
        # Get shape params from id_share_params if available
        shape_params = None
        if id_share_params is not None and 'flame_shape' in id_share_params:
            flame_shape = id_share_params['flame_shape']
            if isinstance(flame_shape, np.ndarray) and flame_shape.size > 0:
                shape_params = torch.from_numpy(flame_shape).float().to(self.device)
                if shape_params.ndim == 2:
                    shape_params = shape_params[0]  # Take first if multiple
        
        # Render all frames
        for frame_key in tqdm(valid_keys, desc="Rendering preview frames"):
            head_img = head_images[frame_key].numpy().transpose(1, 2, 0).astype(np.uint8).copy()
            
            # Render refined FLAME mesh if available
            if 'flame_coeffs' in refined_results[frame_key]:
                flame_coeffs = refined_results[frame_key]['flame_coeffs']
                
                # Build batch_flame dict
                batch_flame = {}
                for key, value in flame_coeffs.items():
                    if isinstance(value, np.ndarray):
                        batch_flame[key] = torch.from_numpy(value).float().to(self.device)[None]
                    else:
                        batch_flame[key] = torch.tensor(value).float().to(self.device)[None]
                
                if 'shape_params' not in batch_flame and shape_params is not None:
                    batch_flame['shape_params'] = shape_params[None]
                
                with torch.no_grad():
                    ret_dict = self.flame(batch_flame)
                    vertices = ret_dict['vertices']
                    lmk_mp = ret_dict['lmk_mp']
                    
                    R = batch_flame['camera_RT_params'][:, :3, :3]
                    T = batch_flame['camera_RT_params'][:, :3, 3]
                    camera = GS_Camera(R=R, T=T, **cameras_kwargs).to(self.device)
                    
                    # Render and blend
                    mesh_img = self.renderer.render_mesh(vertices, cameras=camera)
                    mesh_img = (mesh_img[:, :3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1, 2, 0)
                    blended_img = cv2.addWeighted(head_img, 0.6, mesh_img, 0.4, 0)
                    
                    # Draw refined landmarks (red) - use MediaPipe landmarks with selected indices
                    refined_lmk_2d = camera.transform_points_screen(lmk_mp, R=R, T=T)[0, :, :2]
                    refined_lmk_2d = refined_lmk_2d.detach().cpu().numpy()
                    blended_img = draw_landmarks(refined_lmk_2d, blended_img, color=(255, 0, 0), radius=1)
                    
                    # Draw base landmarks (green) - use MediaPipe landmarks with selected indices
                    if 'head_lmk_mp' in base_results[frame_key]:
                        base_lmks_mp = base_results[frame_key]['head_lmk_mp'][self.face2d_lmk_distance.selected_mp_indices]
                        blended_img = draw_landmarks(base_lmks_mp, blended_img, color=(0, 255, 0), radius=1)
                    
                    preview_images.append(blended_img)
            else:
                # Draw base landmarks only
                if 'head_lmk_mp' in base_results[frame_key]:
                    base_lmks_mp = base_results[frame_key]['head_lmk_mp'][self.face2d_lmk_distance.selected_mp_indices]
                    head_img = draw_landmarks(base_lmks_mp, head_img, color=(0, 255, 0), radius=1)
                preview_images.append(head_img)
        
        # Get image dimensions
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  Saved preview grid ({grid_rows}x{grid_cols}, {total_frames} frames) to: {save_path}")
    
    def run(self, tracked_rlt, id_share_params_result, optim_cfg, batch_imgs_dict=None, frame_interval=1):
        """
        Run FLAME refinement on tracking results.
        
        Args:
            tracked_rlt: Dictionary of per-frame tracking results
            id_share_params_result: Shared identity parameters
            optim_cfg: Optimization configuration
            batch_imgs_dict: Optional dictionary mapping frame_key to images
            frame_interval: Frame interval for temporal regularization
        
        Returns:
            Tuple of (optimized_results, updated_id_share_params)
        """
        # Optimize all frames together
        all_frame_keys = list(tracked_rlt.keys())
        
        # Prepare batch images if provided
        batch_face_imgs = None
        if batch_imgs_dict is not None:
            batch_face_imgs = [batch_imgs_dict[key] for key in all_frame_keys if key in batch_imgs_dict]

        # pixel3dmm refinement removed — see note in __init__

        # Collate all frames (strip _debug and _projected_verts_2d which can't be collated)
        _skip_keys = {'_debug', '_projected_verts_2d'}
        def _clean_dict(d):
            out = {}
            for k, v in d.items():
                if k in _skip_keys:
                    continue
                if isinstance(v, dict):
                    out[k] = _clean_dict(v)
                else:
                    out[k] = v
            return out

        batch_flame_lmk = []
        for key in all_frame_keys:
            batch_flame_lmk.append(_clean_dict(tracked_rlt[key]))

        # special fix for dwpose_raw.bbox
        for idx, item in enumerate(batch_flame_lmk):
            if 'dwpose_raw' in item and 'bbox' in item['dwpose_raw']:
                if item['dwpose_raw']['bbox'].shape[0] == 0:
                    print(f"Warning: frame [{idx}] {all_frame_keys[idx]} has empty dwpose_raw.bbox, fix to [0,0,0,0]")
                    item['dwpose_raw']['bbox'] = np.zeros((4))

        batch_flame_lmk = torch.utils.data.default_collate(batch_flame_lmk)
        batch_flame_lmk = data_to_device(batch_flame_lmk, device=self.device)

        # Use Sapiens face landmarks for ALL frames in lmk_fan (more robust than MediaPipe 70-pt)
        # Also enables FLAME optimization for frames where MediaPipe failed entirely
        batch_flame_lmk['head_lmk_valid_mp'] = batch_flame_lmk['head_lmk_valid'].clone()
        if 'body_lmk_rlt' in batch_flame_lmk:
            sapiens_faces = batch_flame_lmk['body_lmk_rlt'].get('faces')  # (B, 68, 2) original image space
            if sapiens_faces is not None:
                M_o2c = batch_flame_lmk['head_crop']['M_o2c']  # (B, 3, 3)
                B = sapiens_faces.shape[0]
                # Check per-frame validity
                sap_valid = sapiens_faces.abs().sum(dim=(1, 2)) > 0  # (B,)
                if sap_valid.sum() > 0:
                    # Transform ALL valid sapiens landmarks from original image space to face crop space
                    ones = torch.ones(*sapiens_faces.shape[:-1], 1, device=sapiens_faces.device, dtype=sapiens_faces.dtype)
                    sap_h = torch.cat([sapiens_faces, ones], dim=-1)  # (B, 68, 3)
                    sap_crop = torch.bmm(sap_h, M_o2c.transpose(1, 2))[..., :2]  # (B, 68, 2)
                    # Pad 68 -> 70 (extra 2 points are eye pupils, approximate from eye centers)
                    left_eye_center = sap_crop[:, 36:42, :].mean(dim=1, keepdim=True)
                    right_eye_center = sap_crop[:, 42:48, :].mean(dim=1, keepdim=True)
                    sap_crop_70 = torch.cat([sap_crop, left_eye_center, right_eye_center], dim=1)  # (B, 70, 2)
                    # Replace head_lmk_70 with Sapiens for all valid frames
                    for i in range(B):
                        if sap_valid[i]:
                            batch_flame_lmk['head_lmk_70'][i] = sap_crop_70[i]
                            batch_flame_lmk['head_lmk_valid'][i] = True
                    n_mp_invalid = (~batch_flame_lmk['head_lmk_valid_mp']).sum().item()
                    n_sap_valid = sap_valid.sum().item()
                    print(f"Sapiens face landmarks: {n_sap_valid}/{B} frames valid, {n_mp_invalid} had failed MediaPipe")

        # Run optimization on all frames
        optim_results, id_share_params_result = self.optimize(
            all_frame_keys, batch_flame_lmk, id_share_params_result, optim_cfg=optim_cfg,
            batch_id=0, batch_imgs=batch_face_imgs, interval=frame_interval
        )

        return optim_results, id_share_params_result


def data_to_device(data_dict, device='cuda'):
    """Recursively move data to device."""
    assert isinstance(data_dict, dict), 'Data must be a dictionary.'
    for key in data_dict:
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].to(device)
        elif isinstance(data_dict[key], np.ndarray):
            data_dict[key] = torch.tensor(data_dict[key], device=device)
        elif isinstance(data_dict[key], dict):
            data_dict[key] = data_to_device(data_dict[key], device=device)
        else:
            continue
    return data_dict
