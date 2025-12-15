"""
ehmx_refine_smplx.py - SMPL-X Refinement Pipeline
Refines SMPL-X body parameters from base tracking results.
"""
import os
import cv2
import glob
import torch
import importlib
import numpy as np
from tqdm.auto import tqdm
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from pytorch3d.renderer import PointLights

from .modules.ehm import EHM
from .modules.renderer.body_renderer import Renderer as BodyRenderer
from .modules.renderer.head_renderer import Renderer as HeadRenderer
from .modules.renderer.hand_renderer import Renderer as HandRenderer
from .utils.rprint import rlog as log
from .utils.graphics import GS_Camera
from .utils.draw import draw_landmarks
from .utils import rotation_converter as converter
from .losses import Landmark2DLoss, PoseLoss
from .modules.refiner.smplx_utils import smplx_joints_to_dwpose, smplx_to_dwpose

np.random.seed(0)


def expid2model(expr_dir):
    """Load VPoser model from experiment directory."""
    from configer import Configer
    if not os.path.exists(expr_dir):
        raise ValueError('Could not find the experiment directory: %s' % expr_dir)
    best_model_fname = sorted(glob.glob(os.path.join(expr_dir, 'snapshots', '*.pt')), key=os.path.getmtime)[-1]
    log(('Found Trained Model: %s' % best_model_fname))
    default_ps_fname = glob.glob(os.path.join(expr_dir, '*.ini'))[0]
    if not os.path.exists(default_ps_fname):
        raise ValueError('Could not find the appropriate vposer_settings: %s' % default_ps_fname)
    ps = Configer(default_ps_fname=default_ps_fname, work_dir=expr_dir, best_model_fname=best_model_fname)
    return ps, best_model_fname


def load_vposer(expr_dir, vp_model='snapshot'):
    """Load VPoser model for body pose prior."""
    ps, trained_model_fname = expid2model(expr_dir)
    if vp_model == 'snapshot':
        vposer_path = sorted(glob.glob(os.path.join(expr_dir, 'vposer_*.py')), key=os.path.getmtime)[-1]
        spec = importlib.util.spec_from_file_location('VPoser', vposer_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        vposer_pt = getattr(module, 'VPoser')(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)
    else:
        vposer_pt = vp_model(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)
    
    vposer_pt.load_state_dict(torch.load(trained_model_fname, map_location='cpu'))
    vposer_pt.eval()
    return vposer_pt, ps


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


class RefineSmplxPipeline(object):
    """Pipeline for refining SMPL-X parameters from base tracking results."""
    
    def __init__(self, cfg):
        """
        Initialize SMPL-X refinement pipeline.
        
        Args:
            cfg: DataPreparationConfig instance
        """
        self.cfg = cfg
        self.device = cfg.device
        self.body_image_size = cfg.body_hd_size
        self.head_image_size = cfg.head_crop_size
        self.body_focal_length = 1.0 / cfg.tanfov
        self.head_focal_length = 1.0 / cfg.tanfov
        self.hand_focal_length = 1.0 / cfg.tanfov
        
        # Initialize EHM model (includes SMPL-X, FLAME, MANO)
        self.ehm = EHM(cfg.flame_assets_dir, cfg.smplx_assets_dir, cfg.mano_assets_dir).to(self.device)
        
        # Initialize renderers
        self.body_renderer = BodyRenderer(cfg.smplx_assets_dir, self.body_image_size, 
                                         focal_length=self.body_focal_length).to(self.device)
        self.head_renderer = HeadRenderer(cfg.flame_assets_dir, self.head_image_size,
                                         focal_length=self.head_focal_length).to(self.device)
        self.hand_renderer = HandRenderer(cfg.mano_assets_dir, self.head_image_size,
                                         focal_length=self.hand_focal_length).to(self.device)
        
        # Initialize loss functions
        self.lmk2d_loss = Landmark2DLoss(
            self.ehm.smplx.lmk_203_left_indices,
            self.ehm.smplx.lmk_203_right_indices,
            self.ehm.smplx.lmk_203_front_indices,
            self.ehm.smplx.lmk_mp_indices,
            metric='l1'
        ).to(self.device)
        self.metric = torch.nn.L1Loss().to(self.device)
        self.pose_loss = PoseLoss().to(self.device)
        
        # Load VPoser if available
        self.use_vposer = os.path.exists(cfg.vposer_ckpt_dir)
        if self.use_vposer:
            self.vposer, _ = load_vposer(cfg.vposer_ckpt_dir, vp_model='snapshot')
            self.vposer = self.vposer.to(device=self.device)
            self.vposer.eval()
        
        # Get keypoint mappings
        self.kps_map, kps_w = smplx_to_dwpose()
        self.kps_w = torch.from_numpy(kps_w).unsqueeze(0).to(self.device)
        
        self.saving_root = None
    
    def build_cameras_kwargs(self, batch_size, focal_length):
        """Build camera parameters for rendering."""
        screen_size = torch.tensor([self.body_image_size, self.body_image_size],
                                   device=self.device).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': torch.zeros(batch_size, 2, device=self.device).float(),
            'focal_length': focal_length,
            'image_size': screen_size,
            'device': self.device,
        }
        return cameras_kwargs
    
    def transform_points3d(self, points3d: torch.Tensor, M: torch.Tensor):
        """Transform 3D points using affine transformation matrix."""
        R3d = torch.zeros_like(M)
        R3d[:, :2, :2] = M[:, :2, :2]
        scale = (M[:, 0, 0]**2 + M[:, 0, 1]**2)**0.5
        R3d[:, 2, 2] = scale
        
        trans = torch.zeros_like(M)[:, 0]
        trans[:, :2] = M[:, :2, 2]
        trans = trans.unsqueeze(1)
        return torch.bmm(points3d, R3d.mT) + trans
    
    def fix_mirror_issue(self, pts3d, image_size):
        """Fix mirror issue for hand rendering."""
        p = pts3d.clone()
        p[..., 1] = image_size - p[..., 1]
        p[..., 2] = -p[..., 2]
        return p
    
    def transform_hand_pts3d_to_image_coord(self, X, M, img_size=512, is_left=False):
        """Transform hand 3D points to image coordinates."""
        _X = self.fix_mirror_issue(X, img_size)
        if is_left:
            _X[..., 0] = img_size - 1 - _X[..., 0]
        _X = self.transform_points3d(_X, M.to(_X.device))
        return _X
    
    def transform_head_pts3d_to_image_coord(self, X, M):
        """Transform head 3D points to image coordinates."""
        _X = self.transform_points3d(X, M.to(X.device))
        return _X
    
    def prepare_ref_vertices(self, head_coeff_lst, head_crop_meta,
                           left_hand_coeff_lst, left_hand_crop_meta,
                           right_hand_coeff_lst, right_hand_crop_meta, body_crop_meta):
        """Prepare reference vertices from FLAME and MANO models."""
        img_size = self.head_image_size
        
        # Process head vertices
        head_ret_dict = self.ehm.flame(head_coeff_lst)
        render_rlt = self.head_renderer(
            head_ret_dict['vertices'],
            transform_matrix=head_coeff_lst['camera_RT_params'],
            landmarks={'joints': head_ret_dict['joints']},
            ret_image=False
        )
        
        ref_head_vertices = self.transform_head_pts3d_to_image_coord(
            render_rlt[0][:, np.unique(self.ehm.flame.head_index)],
            body_crop_meta['M_o2c-hd'] @ head_crop_meta['M_c2o']
        )
        ref_head_joints = self.transform_head_pts3d_to_image_coord(
            render_rlt[1]['joints'],
            body_crop_meta['M_o2c-hd'] @ head_crop_meta['M_c2o']
        )
        ref_head_vertices[..., 2] = (ref_head_vertices[..., 2] - ref_head_joints[:, 3:5, 2].mean(dim=1, keepdim=True))
        
        # Process left hand vertices
        hand_l_ret_dict = self.ehm.mano(left_hand_coeff_lst, pose_type='aa')
        render_rlt = self.hand_renderer(
            hand_l_ret_dict['vertices'],
            landmarks={'joints': hand_l_ret_dict['joints']},
            is_left=True,
            transform_matrix=left_hand_coeff_lst['camera_RT_params'],
            ret_image=False
        )
        ref_hand_l_vertices = self.transform_hand_pts3d_to_image_coord(
            render_rlt[0],
            body_crop_meta['M_o2c-hd'] @ left_hand_crop_meta['M_c2o'],
            img_size, True
        )
        ref_hand_l_joints = self.transform_hand_pts3d_to_image_coord(
            render_rlt[1]['joints'],
            body_crop_meta['M_o2c-hd'] @ left_hand_crop_meta['M_c2o'],
            img_size, True
        )
        ref_hand_l_vertices[..., 2] = ref_hand_l_vertices[..., 2] - ref_hand_l_joints[:, 0:1, 2]
        
        # Process right hand vertices
        hand_r_ret_dict = self.ehm.mano(right_hand_coeff_lst, pose_type='aa')
        render_rlt = self.hand_renderer(
            hand_r_ret_dict['vertices'],
            landmarks={'joints': hand_r_ret_dict['joints']},
            is_left=False,
            transform_matrix=right_hand_coeff_lst['camera_RT_params'],
            ret_image=False
        )
        ref_hand_r_vertices = self.transform_hand_pts3d_to_image_coord(
            render_rlt[0],
            body_crop_meta['M_o2c-hd'] @ right_hand_crop_meta['M_c2o'],
            img_size
        )
        ref_hand_r_joints = self.transform_hand_pts3d_to_image_coord(
            render_rlt[1]['joints'],
            body_crop_meta['M_o2c-hd'] @ right_hand_crop_meta['M_c2o'],
            img_size
        )
        ref_hand_r_vertices[..., 2] = ref_hand_r_vertices[..., 2] - ref_hand_r_joints[:, 0:1, 2]
        
        return ref_head_vertices, ref_hand_l_vertices, ref_hand_r_vertices, ref_hand_l_joints, ref_hand_r_joints
    
    def optimize(self, track_frames, batch_base, id_share_params, optim_cfg,
                 batch_id=0, batch_imgs=None, interval=1):
        """
        Optimize SMPL-X parameters for a batch of frames.
        
        Args:
            track_frames: List of frame keys
            batch_base: Dictionary with smplx_coeffs, flame_coeffs, and landmarks
            id_share_params: Shared identity parameters
            optim_cfg: Optimization configuration
            batch_id: Batch index for logging
            batch_imgs: Optional batch of images for visualization
            interval: Frame interval for temporal regularization
        
        Returns:
            Tuple of (optimized_results, updated_id_share_params)
        """
        steps = optim_cfg.get('steps', 201)
        share_id = optim_cfg.get('share_id', True)
        share_pose = optim_cfg.get('share_pose', False)
        optim_camera = optim_cfg.get('optim_camera', False)
        
        # Extract lambda weights from config
        lambda_3d_head = optim_cfg.get('lambda_3d_head', 10.0)
        lambda_3d_hand_l = optim_cfg.get('lambda_3d_hand_l', 10.0)
        lambda_3d_hand_r = optim_cfg.get('lambda_3d_hand_r', 10.0)
        lambda_3d_z = optim_cfg.get('lambda_3d_z', 0.1)
        lambda_2d_kpt = optim_cfg.get('lambda_2d_kpt', 10.0)
        lambda_2d_knee_feet = optim_cfg.get('lambda_2d_knee_feet', 100.0)
        lambda_smplx_init = optim_cfg.get('lambda_smplx_init', 0.1)
        lambda_prior = optim_cfg.get('lambda_prior', 1.0)
        lambda_smplx_shape_reg = optim_cfg.get('lambda_smplx_shape_reg', 0.1)
        lambda_mano_shape_reg = optim_cfg.get('lambda_mano_shape_reg', 0.1)
        lambda_smplx_pose = optim_cfg.get('lambda_smplx_pose', 1.0)
        lambda_smplx_pose_reg_base = optim_cfg.get('lambda_smplx_pose_reg_base', 1.0)
        lambda_smplx_leg_pose = optim_cfg.get('lambda_smplx_leg_pose', 1.0)
        lambda_smplx_freezed_pose = optim_cfg.get('lambda_smplx_freezed_pose', 1.0)
        lambda_smplx_hand_pose_reg = optim_cfg.get('lambda_smplx_hand_pose_reg', 0.1)
        lambda_scale_reg = optim_cfg.get('lambda_scale_reg', 100.0)
        lambda_joint_offset_reg = optim_cfg.get('lambda_joint_offset_reg', 1.0)
        lambda_mtn_body_pose = optim_cfg.get('lambda_mtn_body_pose', 200.0)
        lambda_mtn_rot6d = optim_cfg.get('lambda_mtn_rot6d', 100.0)
        lambda_mtn_trans = optim_cfg.get('lambda_mtn_trans', 100.0)
        lambda_mtn_trans_z = optim_cfg.get('lambda_mtn_trans_z', 100.0)
        lambda_mtn_vertices = optim_cfg.get('lambda_mtn_vertices', 1.0)
        
        batch_size = len(track_frames)
        batch_smplx = batch_base['smplx_coeffs']
        batch_flame = batch_base['flame_coeffs']
        gt_lmk_2d = batch_base['dwpose_rlt']
        batch_mano_left = batch_base['left_mano_coeffs']
        batch_mano_right = batch_base['right_mano_coeffs']
        head_lmk_valid = batch_base['head_lmk_valid']
        left_hand_valid = batch_base['left_hand_valid']
        right_hand_valid = batch_base['right_hand_valid']
        
        # Convert hand poses
        b, n = batch_mano_left["hand_pose"][:, 0].shape[:2]
        left_hand_pose = converter.batch_matrix2axis(batch_mano_left["hand_pose"][:, 0].flatten(0, 1)).reshape(b, n * 3)
        right_hand_pose = converter.batch_matrix2axis(batch_mano_right["hand_pose"][:, 0].flatten(0, 1)).reshape(b, 1, n, 3)
        left_hand_pose[:, 1::3] *= -1
        left_hand_pose[:, 2::3] *= -1
        batch_smplx["left_hand_pose"] = left_hand_pose.detach().clone().reshape(b, 1, n, 3)
        batch_smplx["right_hand_pose"] = right_hand_pose.detach().clone()
        
        # Squeeze batch dimensions
        batch_smplx = {kk: vv.squeeze() for kk, vv in batch_smplx.items()}
        batch_mano_left = {kk: vv.squeeze() for kk, vv in batch_mano_left.items()}
        batch_mano_right = {kk: vv.squeeze() for kk, vv in batch_mano_right.items()}
        if batch_size == 1:
            batch_smplx = {kk: vv[None] for kk, vv in batch_smplx.items()}
            batch_mano_left = {kk: vv[None] for kk, vv in batch_mano_left.items()}
            batch_mano_right = {kk: vv[None] for kk, vv in batch_mano_right.items()}
        
        body_lmk_score = gt_lmk_2d['scores']
        
        # Initialize identity-shared parameters
        assert share_id == True
        if share_id:
            head_scale = torch.tensor(id_share_params['head_scale'], device=self.device)
            hand_scale = torch.tensor(id_share_params['hand_scale'], device=self.device)
            joints_offset = torch.tensor(id_share_params['joints_offset'], device=self.device)
            g_smplx_shape = torch.tensor(id_share_params['smplx_shape'], device=self.device).float()
            g_flame_shape = torch.tensor(id_share_params['flame_shape'], device=self.device).float()
            left_hand_shape = torch.tensor(id_share_params['left_mano_shape'], device=self.device)
            right_hand_shape = torch.tensor(id_share_params['right_mano_shape'], device=self.device)
            
            batch_smplx['head_scale'] = head_scale.expand(batch_size, -1)
            batch_smplx['hand_scale'] = hand_scale.expand(batch_size, -1)
            batch_smplx['joints_offset'] = joints_offset.expand(batch_size, -1, -1)
            batch_flame['shape_params'] = g_flame_shape.expand(batch_size, -1)
            batch_smplx['shape'] = g_smplx_shape.expand(batch_size, -1)
        
        # Extract pose parameters
        camera_RT_params = batch_smplx['camera_RT_params']
        global_pose = batch_smplx['global_pose']
        body_pose = batch_smplx['body_pose']
        left_hand_pose = batch_smplx['left_hand_pose']
        right_hand_pose = batch_smplx['right_hand_pose']
        
        if share_pose:
            body_pose = body_pose.mean(dim=0, keepdim=True).float()
            global_pose = global_pose.mean(dim=0, keepdim=True).float()
            left_hand_pose = left_hand_pose.mean(dim=0, keepdim=True).float()
            right_hand_pose = right_hand_pose.mean(dim=0, keepdim=True).float()
            batch_smplx['body_pose'] = body_pose.expand(batch_size, -1, -1)
            batch_smplx['global_pose'] = global_pose.expand(batch_size, -1)
            batch_smplx['left_hand_pose'] = left_hand_pose.expand(batch_size, -1, -1)
            batch_smplx['right_hand_pose'] = right_hand_pose.expand(batch_size, -1, -1)
        else:
            body_pose = body_pose.view(batch_size, -1, 3).float()
            global_pose = global_pose.view(batch_size, -1).float()
            left_hand_pose = left_hand_pose.view(batch_size, -1, 3).float()
            right_hand_pose = right_hand_pose.view(batch_size, -1, 3).float()
            batch_smplx['body_pose'] = body_pose
            batch_smplx['global_pose'] = global_pose
            batch_smplx['left_hand_pose'] = left_hand_pose
            batch_smplx['right_hand_pose'] = right_hand_pose
        
        for key in batch_smplx:
            if isinstance(batch_smplx[key], torch.Tensor):
                batch_smplx[key] = batch_smplx[key].float()
        
        # Prepare reference vertices
        (ref_head_vertices, ref_hand_l_vertices, ref_hand_r_vertices,
         ref_hand_l_joints, ref_hand_r_joints) = self.prepare_ref_vertices(
            batch_flame, batch_base['head_crop'],
            batch_mano_left, batch_base['left_hand_crop'],
            batch_mano_right, batch_base['right_hand_crop'],
            batch_base['body_crop']
        )
        
        # Validate hand tracking
        left_wrist_kpt2d = gt_lmk_2d['keypoints'][:, 7, :2]
        left_elbow_kpt2d = gt_lmk_2d['keypoints'][:, 6, :2]
        right_wrist_kpt2d = gt_lmk_2d['keypoints'][:, 4, :2]
        right_elbow_kpt2d = gt_lmk_2d['keypoints'][:, 3, :2]
        
        left_wrist_elbow_dist = torch.norm(left_wrist_kpt2d - left_elbow_kpt2d, dim=1)
        right_wrist_elbow_dist = torch.norm(right_wrist_kpt2d - right_elbow_kpt2d, dim=1)
        
        if left_hand_valid.sum() > 0:
            left_hand_wrist_3d = ref_hand_l_vertices[:, self.ehm.mano.selected_vert_ids].mean(dim=1)[:, :2]
            left_hand_wrist_dist = torch.norm(left_hand_wrist_3d - left_wrist_kpt2d, dim=1)
            left_hand_valid = left_hand_valid & (left_hand_wrist_dist < left_wrist_elbow_dist)
        
        if right_hand_valid.sum() > 0:
            right_hand_wrist_3d = ref_hand_r_vertices[:, self.ehm.mano.selected_vert_ids].mean(dim=1)[:, :2]
            right_hand_wrist_dist = torch.norm(right_hand_wrist_3d - right_wrist_kpt2d, dim=1)
            right_hand_valid = right_hand_valid & (right_hand_wrist_dist < right_wrist_elbow_dist)
        
        # Learning rate decay for multi-batch
        _lr_decay = 1.0
        if batch_id > 0:
            _lr_decay = 0.0
        
        # Initialize reference info
        with torch.no_grad():
            smplx_init_pose = body_pose.clone()
            smplx_init_dict = self.ehm(batch_smplx, batch_flame, pose_type='aa')
            
            cameras_kwargs = self.build_cameras_kwargs(batch_size, focal_length=self.body_focal_length)
            cameras = GS_Camera(**cameras_kwargs).to(self.device)
            R, T = camera_RT_params.detach().split([3, 1], dim=-1)
            T = T.squeeze(-1)
            ref_proj_vertices = cameras.transform_points_screen(smplx_init_dict['vertices'], R=R, T=T)
            ref_proj_joints = cameras.transform_points_screen(smplx_init_dict['joints'], R=R, T=T)
            ret_body_ref = smplx_joints_to_dwpose(ref_proj_joints)[0]
        
        # Setup optimization parameters
        g_smplx_shape.requires_grad = True
        left_hand_shape.requires_grad = True
        right_hand_shape.requires_grad = True
        body_pose.requires_grad = True
        left_hand_pose.requires_grad = True
        right_hand_pose.requires_grad = True
        
        gl_T = camera_RT_params[:, :3, 3].detach().clone()
        gl_R = camera_RT_params[:, :3, :3].detach().clone()
        gl_R_6d = matrix_to_rotation_6d(gl_R).detach().clone()
        
        _lr_camera = 0.0
        if optim_camera:
            gl_T[:, 2] = (gl_T[:, 2].mean(dim=0)[None].expand(batch_size, -1).detach().clone())[:, 0]
            _lr_camera = 1.0
        
        gl_T.requires_grad = True
        gl_R.requires_grad = True
        gl_R_6d.requires_grad = True
        joints_offset.requires_grad = True
        g_flame_shape.requires_grad = True
        head_scale.requires_grad = True
        hand_scale.requires_grad = True
        
        # Define joint groups for regularization
        leg_body_joints = [4, 5, 7, 8, 10]
        freezed_body_joints = [1, 3, 2, 6, 9]
        
        # Setup optimizer
        opt_p = torch.optim.AdamW([
            {'params': [body_pose], 'lr': 1e-3},
            {'params': [g_smplx_shape], 'lr': 1e-4 * _lr_decay},
            {'params': [left_hand_shape], 'lr': 1e-4 * _lr_decay},
            {'params': [right_hand_shape], 'lr': 1e-4 * _lr_decay},
            {'params': [right_hand_pose], 'lr': 1e-5},
            {'params': [left_hand_pose], 'lr': 1e-5},
            {'params': [gl_T], 'lr': 5e-3 * _lr_camera},
            {'params': [gl_R], 'lr': 5e-4 * _lr_camera},
            {'params': [gl_R_6d], 'lr': 5e-4 * _lr_camera},
            {'params': [joints_offset], 'lr': 1e-5 * _lr_decay},
            {'params': [g_flame_shape], 'lr': 2e-5 * _lr_decay},
            {'params': [head_scale], 'lr': 1e-4 * _lr_decay},
            {'params': [hand_scale], 'lr': 1e-4 * _lr_decay},
        ])
        
        # Optimization loop
        t_bar = tqdm(range(steps), desc='Tuning SMPLX global params')
        for i_step in t_bar:
            batch_smplx['head_scale'] = head_scale.expand(batch_size, -1)
            batch_smplx['hand_scale'] = hand_scale.expand(batch_size, -1)
            batch_smplx['shape'] = g_smplx_shape.expand(batch_size, -1)
            batch_smplx['joints_offset'] = joints_offset.expand(batch_size, -1, -1)
            batch_mano_left['betas'] = left_hand_shape.expand(batch_size, -1)
            batch_mano_right['betas'] = right_hand_shape.expand(batch_size, -1)
            batch_flame['shape_params'] = g_flame_shape.expand(batch_size, -1)
            
            batch_smplx['body_pose'] = body_pose.expand(batch_size, -1, -1)
            batch_smplx['global_pose'] = global_pose.expand(batch_size, -1)
            batch_smplx['left_hand_pose'] = left_hand_pose.expand(batch_size, -1, -1)
            batch_smplx['right_hand_pose'] = right_hand_pose.expand(batch_size, -1, -1)
            
            if self.use_vposer:
                body_embedding_mean = self.vposer.encode(body_pose).mean
            else:
                body_embedding_mean = 0
            
            smplx_dict = self.ehm(batch_smplx, batch_flame, pose_type='aa')
            T = gl_T
            R = rotation_6d_to_matrix(gl_R_6d)
            T = T.squeeze(-1)
            proj_vertices = cameras.transform_points_screen(smplx_dict['vertices'], R=R, T=T)
            proj_joints = cameras.transform_points_screen(smplx_dict['joints'], R=R, T=T)
            proj_face_lmk_203 = cameras.transform_points_screen(smplx_dict['face_lmk_203'], R=R, T=T)
            
            ### 3D vertices loss
            loss_3d_hand_l, loss_3d_hand_r, loss_3d_head = 0.0, 0.0, 0.0
            
            pred_head_vertices = proj_vertices[:, self.ehm.smplx.smplx2flame_ind][:, self.ehm.head_index]
            pred_head_joint = proj_joints[:, 23:25].mean(dim=1, keepdim=True)
            pred_head_vertices[..., 2] = (pred_head_vertices[..., 2] - pred_head_joint[..., 2])
            if head_lmk_valid.sum() > 0:
                loss_3d_head = self.metric(pred_head_vertices[head_lmk_valid],
                                          ref_head_vertices[head_lmk_valid]) * lambda_3d_head
            
            pred_hand_l_vertices = proj_vertices[:, self.ehm.smplx.smplx2mano_ind['left_hand']]
            pred_hand_l_joint = proj_joints[:, 20:21, :]
            pred_hand_l_vertices[..., 2] = (pred_hand_l_vertices[..., 2] - pred_hand_l_joint[..., 2])
            if left_hand_valid.sum() > 0:
                loss_3d_hand_l = self.metric(
                    pred_hand_l_vertices[left_hand_valid][:, self.ehm.mano.selected_vert_ids],
                    ref_hand_l_vertices[left_hand_valid][:, self.ehm.mano.selected_vert_ids]
                ) * lambda_3d_hand_l
            
            pred_hand_r_vertices = proj_vertices[:, self.ehm.smplx.smplx2mano_ind['right_hand']]
            pred_hand_r_joint = proj_joints[:, 21:22, :]
            pred_hand_r_vertices[..., 2] = (pred_hand_r_vertices[..., 2] - pred_hand_r_joint[..., 2])
            if right_hand_valid.sum() > 0:
                loss_3d_hand_r = self.metric(
                    pred_hand_r_vertices[right_hand_valid][:, self.ehm.mano.selected_vert_ids],
                    ref_hand_r_vertices[right_hand_valid][:, self.ehm.mano.selected_vert_ids]
                ) * lambda_3d_hand_r
            
            loss_3d_z = (self.metric(proj_joints[..., 2], ref_proj_joints[..., 2]) +
                        self.metric(proj_vertices[..., 2], ref_proj_vertices[..., 2])) * lambda_3d_z
            
            loss_3d = (loss_3d_head + loss_3d_hand_l + loss_3d_hand_r + loss_3d_z)
            
            ### 2D landmark loss
            pred_kps3d = smplx_joints_to_dwpose(proj_joints)[0]
            
            # Define keypoint groups
            knee_feet_indices = [9, 10, 12, 13, 18, 19, 20, 21, 22, 23]
            face_kpt2d_indices = list(range(24, 92)) + [14, 15, 16, 17]
            lhand_kpt2d_indices = list(range(92, 113)) + [6, 7]
            rhand_kpt2d_indices = list(range(113, 134)) + [3, 4]
            all_kpt2d_indices = list(range(pred_kps3d.shape[1]))
            body_kpt2d_indices = [i for i in all_kpt2d_indices 
                                 if i not in knee_feet_indices and i not in face_kpt2d_indices 
                                 and i not in lhand_kpt2d_indices and i not in rhand_kpt2d_indices]
            
            # Body keypoints loss
            loss_2d_kpt = self.metric(
                pred_kps3d[:, body_kpt2d_indices, :2],
                gt_lmk_2d['keypoints'][:, body_kpt2d_indices, :2].float()
            ) * lambda_2d_kpt
            
            # Knee and feet loss
            loss_2d_knee_feet = self.metric(
                pred_kps3d[:, knee_feet_indices, :2],
                gt_lmk_2d['keypoints'][:, knee_feet_indices, :2].float()
            ) * lambda_2d_knee_feet
            
            # Face landmarks loss
            loss_2d_face = 0.0
            if head_lmk_valid.sum() > 0:
                loss_2d_face = self.metric(
                    pred_kps3d[head_lmk_valid][:, face_kpt2d_indices, :2],
                    gt_lmk_2d['keypoints'][head_lmk_valid][:, face_kpt2d_indices, :2].float()
                ) * lambda_2d_kpt
            if (~head_lmk_valid).sum() > 0:
                loss_2d_face += self.metric(
                    pred_kps3d[~head_lmk_valid][:, 15, :2],
                    ret_body_ref[~head_lmk_valid][:, 15, :2]
                ) * lambda_smplx_init
            
            # Left hand landmarks loss
            loss_2d_lhand = 0.0
            if left_hand_valid.sum() > 0:
                loss_2d_lhand = self.metric(
                    pred_kps3d[left_hand_valid][:, lhand_kpt2d_indices, :2],
                    gt_lmk_2d['keypoints'][left_hand_valid][:, lhand_kpt2d_indices, :2].float()
                ) * lambda_2d_kpt
            if (~left_hand_valid).sum() > 0:
                loss_2d_lhand += self.metric(
                    pred_kps3d[~left_hand_valid][:, 7, :2],
                    ret_body_ref[~left_hand_valid][:, 7, :2]
                ) * lambda_smplx_init
            
            # Right hand landmarks loss
            loss_2d_rhand = 0.0
            if right_hand_valid.sum() > 0:
                loss_2d_rhand = self.metric(
                    pred_kps3d[right_hand_valid][:, rhand_kpt2d_indices, :2],
                    gt_lmk_2d['keypoints'][right_hand_valid][:, rhand_kpt2d_indices, :2].float()
                ) * lambda_2d_kpt
            if (~right_hand_valid).sum() > 0:
                loss_2d_rhand += self.metric(
                    pred_kps3d[~right_hand_valid][:, 4, :2],
                    ret_body_ref[~right_hand_valid][:, 4, :2]
                ) * lambda_smplx_init
            
            loss_2d = loss_2d_kpt + loss_2d_knee_feet + loss_2d_face + loss_2d_lhand + loss_2d_rhand
            
            ### Prior losses
            loss_prior = torch.abs(body_embedding_mean.mean()) * lambda_prior + \
                        torch.mean(torch.square(g_smplx_shape)) * lambda_smplx_shape_reg + \
                        torch.mean(torch.square(left_hand_shape)) * lambda_mano_shape_reg + \
                        torch.mean(torch.square(right_hand_shape)) * lambda_mano_shape_reg
            
            loss_smplx_pose = self.pose_loss(body_pose, smplx_init_pose) * lambda_smplx_pose
            loss_smplx_leg_pose = torch.mean((body_pose[:, leg_body_joints]) ** 2) * lambda_smplx_leg_pose
            loss_smplx_pose_reg = torch.mean(body_pose[:, freezed_body_joints] ** 2) * lambda_smplx_freezed_pose
            loss_smplx_pose_reg += torch.mean(body_pose**2) * lambda_smplx_pose_reg_base
            loss_smplx_hand_pose_reg = (
                self.pose_loss(right_hand_pose, torch.zeros_like(right_hand_pose)) +
                self.pose_loss(left_hand_pose, torch.zeros_like(left_hand_pose))
            ) * lambda_smplx_hand_pose_reg
            loss_scale_reg = (((hand_scale - 1.0)**2).mean() + ((head_scale - 1.0)**2).mean()) * lambda_scale_reg
            loss_joint_offset_reg = (joints_offset**2).mean() * lambda_joint_offset_reg
            
            loss_prior = loss_prior + loss_smplx_pose + loss_smplx_pose_reg + loss_scale_reg + \
                        loss_joint_offset_reg + loss_smplx_leg_pose + loss_smplx_hand_pose_reg
            
            ### Motion regularization loss
            mtn_reg_loss = 0
            if body_pose.shape[0] > 1:
                mtn_reg_loss += self.metric(body_pose[1:], body_pose[:-1]) * lambda_mtn_body_pose / (interval * 1)
                mtn_reg_loss += self.metric(gl_R_6d[1:], gl_R_6d[:-1]) * lambda_mtn_rot6d / (interval * 1)
                mtn_reg_loss += self.metric(T[1:], T[:-1]) * lambda_mtn_trans / (interval * 1)
                mtn_reg_loss += self.metric(T[1:, 2], T[:-1, 2]) * lambda_mtn_trans_z / (interval * 1)
                mtn_reg_loss += self.metric(proj_vertices[1:], proj_vertices[:-1]) * lambda_mtn_vertices / (interval * 1)
            
            total_loss = loss_3d + loss_2d + loss_prior + mtn_reg_loss
            loss_line = f'total: {total_loss:.2f} | 3d: {loss_3d:.2f} | 2d: {loss_2d:.2f} | prior: {loss_prior:.2f} | mtn: {mtn_reg_loss:.2f}'
            loss_info = f'Batch: {batch_id:02d} | Iter: {i_step:03d} >> Loss: {loss_line}'
            t_bar.set_description(loss_info)
            
            opt_p.zero_grad()
            total_loss.backward()
            opt_p.step()
            
            # Visualization at iter 0 and last step
            if batch_imgs is not None and (i_step == 0 or i_step == steps - 1):
                self._visualize_smplx(batch_imgs, smplx_dict, gt_lmk_2d, pred_kps3d, proj_face_lmk_203,
                                    ref_hand_l_joints, ref_hand_r_joints, ref_head_vertices,
                                    ref_hand_l_vertices, ref_hand_r_vertices,
                                    head_lmk_valid, left_hand_valid, right_hand_valid,
                                    R, T, cameras, batch_id, i_step,
                                    body_kpt2d_indices, knee_feet_indices, face_kpt2d_indices,
                                    lhand_kpt2d_indices, rhand_kpt2d_indices)
        
        # Update camera parameters
        batch_smplx['camera_RT_params'][:, :3, 3] = gl_T.detach()
        batch_smplx['camera_RT_params'][:, :3, :3] = rotation_6d_to_matrix(gl_R_6d.detach())
        
        # Package optimized results
        optim_smplx_results = {}
        id_share_params['smplx_shape'] = g_smplx_shape.detach().float().cpu().numpy()
        id_share_params['head_scale'] = head_scale.detach().float().cpu().numpy()
        id_share_params['hand_scale'] = hand_scale.detach().float().cpu().numpy()
        id_share_params['joints_offset'] = joints_offset.detach().float().cpu().numpy()
        id_share_params['flame_shape'] = g_flame_shape.detach().float().cpu().numpy()
        
        for idx, name in enumerate(track_frames):
            optim_smplx_results[name] = {
                'exp': batch_smplx['exp'][idx].detach().float().cpu().numpy(),
                'global_pose': batch_smplx['global_pose'][idx].detach().float().cpu().numpy(),
                'body_pose': batch_smplx['body_pose'][idx].detach().float().cpu().numpy(),
                'body_cam': batch_smplx['body_cam'][idx].detach().float().cpu().numpy(),
                'camera_RT_params': batch_smplx['camera_RT_params'][idx].detach().float().cpu().numpy(),
                'left_hand_pose': batch_smplx['left_hand_pose'][idx].detach().float().cpu().numpy(),
                'right_hand_pose': batch_smplx['right_hand_pose'][idx].detach().float().cpu().numpy(),
            }
        
        return optim_smplx_results, id_share_params
    
    def _visualize_smplx(self, batch_imgs, smplx_dict, gt_lmk_2d, pred_kps3d, proj_face_lmk_203,
                        ref_hand_l_joints, ref_hand_r_joints, ref_head_vertices,
                        ref_hand_l_vertices, ref_hand_r_vertices,
                        head_lmk_valid, left_hand_valid, right_hand_valid,
                        R, T, cameras, batch_id, i_step,
                        body_kpt2d_indices, knee_feet_indices, face_kpt2d_indices,
                        lhand_kpt2d_indices, rhand_kpt2d_indices):
        """Visualize SMPL-X optimization results - save all frames in 2:1 grid layout."""
        if not self.saving_root:
            return
        
        save_path = os.path.join(self.saving_root, "visual_results")
        os.makedirs(save_path, exist_ok=True)
        
        n_imgs = len(batch_imgs)
        lights = PointLights(device=self.device, location=[[0.0, -1.0, -10.0]])
        
        # Visualize frames 
        indices = list(range(n_imgs))[::10]
        
        vis_imgs = []
        for im_idx in indices:
            _img = batch_imgs[im_idx].clone().cpu().numpy().transpose(1, 2, 0).astype(np.uint8).copy()
            _t_lmk_dwp = pred_kps3d[im_idx, :, :2]
            _landmark_dwp = gt_lmk_2d['keypoints'][im_idx, ...].detach().cpu().numpy()
            
            # Draw connection lines for body keypoints
            for kp_idx in body_kpt2d_indices + knee_feet_indices:
                pt1 = tuple(_landmark_dwp[kp_idx].astype(int))
                pt2 = tuple(_t_lmk_dwp[kp_idx].detach().cpu().numpy().astype(int))
                cv2.line(_img, pt1, pt2, (255, 255, 0), 1)
            
            # Draw face keypoints if valid
            if head_lmk_valid[im_idx]:
                for kp_idx in face_kpt2d_indices:
                    pt1 = tuple(_landmark_dwp[kp_idx].astype(int))
                    pt2 = tuple(_t_lmk_dwp[kp_idx].detach().cpu().numpy().astype(int))
                    cv2.line(_img, pt1, pt2, (255, 200, 0), 1)
            
            # Draw left hand keypoints if valid
            if left_hand_valid[im_idx]:
                for kp_idx in lhand_kpt2d_indices:
                    pt1 = tuple(_landmark_dwp[kp_idx].astype(int))
                    pt2 = tuple(_t_lmk_dwp[kp_idx].detach().cpu().numpy().astype(int))
                    cv2.line(_img, pt1, pt2, (0, 255, 255), 1)
            
            # Draw right hand keypoints if valid
            if right_hand_valid[im_idx]:
                for kp_idx in rhand_kpt2d_indices:
                    pt1 = tuple(_landmark_dwp[kp_idx].astype(int))
                    pt2 = tuple(_t_lmk_dwp[kp_idx].detach().cpu().numpy().astype(int))
                    cv2.line(_img, pt1, pt2, (255, 0, 255), 1)
            
            # Draw landmarks
            _img = draw_landmarks(_landmark_dwp, _img, color=(0, 255, 0), viz_index=False)
            _img = draw_landmarks(_t_lmk_dwp, _img, color=(255, 0, 0))
            _img = draw_landmarks(proj_face_lmk_203[im_idx, :, :2], _img, color=(0, 0, 255))
            
            # Render mesh
            t_camera = GS_Camera(**self.build_cameras_kwargs(1, self.body_focal_length),
                                R=R[None, im_idx], T=T[None, im_idx])
            mesh_img = self.body_renderer.render_mesh(smplx_dict['vertices'][None, im_idx, ...],
                                                     t_camera, lights=lights)
            mesh_img = (mesh_img[:, :3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1, 2, 0)
            mesh_img = cv2.addWeighted(_img, 0.3, mesh_img, 0.7, 0)
            
            # Draw hand details
            img_hand = batch_imgs[im_idx].clone().cpu().numpy().transpose(1, 2, 0).astype(np.uint8).copy()
            img_hand = draw_landmarks(ref_hand_l_joints[im_idx, :, :2], img_hand, color=(0, 0, 255))
            img_hand = draw_landmarks(ref_hand_r_joints[im_idx, :, :2], img_hand, color=(0, 0, 255))
            img_hand = draw_landmarks(_landmark_dwp[-42:, :], img_hand, color=(0, 255, 0))
            img_hand = draw_landmarks(_t_lmk_dwp[-42:, :], img_hand, color=(255, 0, 0))
            
            if head_lmk_valid[im_idx]:
                img_hand = draw_landmarks(ref_head_vertices[im_idx, :, :2], img_hand, color=(255, 0, 255), radius=1)
            if left_hand_valid[im_idx]:
                img_hand = draw_landmarks(
                    ref_hand_l_vertices[:, self.ehm.mano.selected_vert_ids][im_idx, :, :2],
                    img_hand, color=(255, 0, 255), radius=1
                )
            if right_hand_valid[im_idx]:
                img_hand = draw_landmarks(
                    ref_hand_r_vertices[:, self.ehm.mano.selected_vert_ids][im_idx, :, :2],
                    img_hand, color=(255, 0, 255), radius=1
                )
            
            _img = np.concatenate((_img, mesh_img, img_hand), axis=1)
            vis_imgs.append(_img)
        
        if len(vis_imgs) > 0:
            # Calculate 2:1 aspect ratio grid layout
            total_frames = len(vis_imgs)
            grid_rows = max(1, int(np.sqrt(total_frames / 2)))
            grid_cols = int(np.ceil(total_frames / grid_rows))
            
            # Adjust to maintain 2:1 ratio
            while grid_cols < 2 * grid_rows and grid_rows > 1:
                grid_rows -= 1
                grid_cols = int(np.ceil(total_frames / grid_rows))
            
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
            cv2.imwrite(os.path.join(save_path, f"vis_fit_smplx_bid-{batch_id}_stp-{i_step}.jpg"),
                       cv2.cvtColor(grid.copy(), cv2.COLOR_RGB2BGR))
    
    def generate_preview(self, frames_keys, base_results, refined_results, body_images, save_path, id_share_params=None):
        """
        Generate preview visualization grid with SMPL-X rendering.
        
        Args:
            frames_keys: List of frame keys
            base_results: Base tracking results
            refined_results: Refined tracking results with smplx_coeffs
            body_images: Dictionary of body image tensors (C, H, W)
            save_path: Path to save the preview image
            id_share_params: Optional identity shared parameters
        """
        print(f"  Generating preview grid with SMPL-X rendering...")
        
        # Collect all valid frames
        preview_images = []
        valid_keys = []
        
        for frame_key in frames_keys:
            if frame_key in refined_results and frame_key in body_images:
                valid_keys.append(frame_key)
        
        if len(valid_keys) == 0:
            print(f"  Warning: No valid frames found for preview")
            return
        
        # Calculate 2:1 aspect ratio grid layout
        total_frames = len(valid_keys)
        grid_rows = max(1, int(np.sqrt(total_frames / 2)))
        grid_cols = int(np.ceil(total_frames / grid_rows))
        
        # Adjust to maintain 2:1 ratio
        while grid_cols < 2 * grid_rows and grid_rows > 1:
            grid_rows -= 1
            grid_cols = int(np.ceil(total_frames / grid_rows))
        
        # Render sample frames for preview
        from .modules.refiner.smplx_utils import smplx_joints_to_dwpose
        from pytorch3d.renderer import PointLights
        
        # Select frames for preview (max 20 frames)
        sample_indices = np.linspace(0, len(valid_keys) - 1, min(20, len(valid_keys)), dtype=int)
        sample_keys = [valid_keys[i] for i in sample_indices]
        
        for frame_key in tqdm(sample_keys, desc="Rendering preview frames"):
            body_img = body_images[frame_key].numpy().transpose(1, 2, 0).astype(np.uint8).copy()
            
            # Draw DWPose keypoints if available
            if 'dwpose_rlt' in base_results[frame_key]:
                dwpose_kpts = base_results[frame_key]['dwpose_rlt']['keypoints']
                body_img = draw_landmarks(dwpose_kpts, body_img, color=(0, 255, 0), radius=2)
            
            preview_images.append(body_img)
        
        # Get image dimensions
        img_height, img_width = preview_images[0].shape[:2]
        
        # Recalculate grid for sampled frames
        total_frames = len(preview_images)
        grid_rows = max(1, int(np.sqrt(total_frames / 2)))
        grid_cols = int(np.ceil(total_frames / grid_rows))
        while grid_cols < 2 * grid_rows and grid_rows > 1:
            grid_rows -= 1
            grid_cols = int(np.ceil(total_frames / grid_rows))
        
        # Create blank grid
        grid_height = grid_rows * img_height
        grid_width = grid_cols * img_width
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Place images in grid
        for idx, img in enumerate(preview_images):
            row = idx // grid_cols
            col = idx % grid_cols
            
            y_start = row * img_height
            x_start = col * img_width
            
            grid[y_start:y_start + img_height, x_start:x_start + img_width] = img
        
        # Save grid
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  Saved preview grid ({grid_rows}x{grid_cols}, {total_frames} frames) to: {save_path}")
    
    def run(self, tracked_rlt, id_share_params_result, optim_cfg, batch_imgs_dict=None, frame_interval=1):
        """
        Run SMPL-X refinement on tracking results.
        
        Args:
            tracked_rlt: Dictionary of per-frame tracking results
            id_share_params_result: Shared identity parameters
            optim_cfg: Optimization configuration
            batch_imgs_dict: Optional dictionary mapping frame_key to images
            frame_interval: Frame interval for temporal regularization
        
        Returns:
            Tuple of (optimized_results, updated_id_share_params)
        """
        # Initialize id_share_params if needed
        if 'head_scale' not in id_share_params_result:
            id_share_params_result['head_scale'] = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        if 'hand_scale' not in id_share_params_result:
            id_share_params_result['hand_scale'] = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        if 'joints_offset' not in id_share_params_result:
            id_share_params_result['joints_offset'] = np.zeros((1, 55, 3), dtype=np.float32)
        
        # Optimize all frames together
        all_frame_keys = list(tracked_rlt.keys())
        
        # Prepare batch images if provided
        batch_body_imgs = None
        if batch_imgs_dict is not None:
            batch_body_imgs = [batch_imgs_dict[key] for key in all_frame_keys]
        
        # Collate all frames
        batch_data = [tracked_rlt[key] for key in all_frame_keys]
        batch_data = torch.utils.data.default_collate(batch_data)
        batch_data = data_to_device(batch_data, device=self.device)
        
        # Run optimization on all frames
        log(f"Running SMPL-X optimization on {len(all_frame_keys)} frames...")
        optim_results, id_share_params_result = self.optimize(
            all_frame_keys,
            batch_data,
            id_share_params_result,
            optim_cfg,
            batch_id=0,
            batch_imgs=batch_body_imgs,
            interval=frame_interval
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
