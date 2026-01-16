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
from .ehmx_track_base import split_frame_key

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


def make_pshuman_camera(original_RT, view_id, device='cuda'):
    """
    Calculate PSHuman camera RT parameters from original camera.
    
    Args:
        original_RT: Original camera RT parameters (3x4 or 4x4)
        view_id: PSHuman view identifier (e.g., 'pshuman_02', 'pshuman_03')
        device: Device for tensor operations
    
    Returns:
        Transformed camera RT parameters for PSHuman view
    """
    # Convert PyTorch 3D coordinate system to COLMAP coordinate system
    c2c_mat = torch.tensor([[-1, 0, 0, 0],
                            [ 0,-1, 0, 0],
                            [ 0, 0, 1, 0],
                            [ 0, 0, 0, 1]],
                           dtype=torch.float32, device=device)
    
    RT_mat = torch.eye(4, dtype=torch.float32, device=device)
    if original_RT.shape[0] == 3:
        RT_mat[:3, :4] = original_RT
    else:
        RT_mat = original_RT
    
    w2c_cam = torch.matmul(c2c_mat, RT_mat)
    c2w_cam = torch.linalg.inv(w2c_cam)
    
    # Apply view-specific transformation
    if 'color_02' in view_id or 'pshuman_02' in view_id:
        # Mode 4: -90° Y-rotation for left side view
        rot_y_neg90 = torch.tensor([
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=device)
        c2w_pshuman = torch.matmul(rot_y_neg90, c2w_cam)
    
    elif 'color_03' in view_id or 'pshuman_03' in view_id:
        """
        # 1. If color_03 of pshuman is directly loaded without flipping:
        # Solution A1 and A2 are equivalent.

        # [Solution A1]
        # It's contour-aligned back view w.r.t. the original front view.
        rot_y_180 = torch.tensor([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=device)
        mirror_x = torch.tensor([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=device)
        c2w_pshuman = torch.matmul(mirror_x, torch.matmul(rot_y_180, c2w_cam))

        [Solution A2]
        # Simple Z-axis flip for back view (if didn't flip while loading)
        flip_z = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=device)
        c2w_pshuman = torch.matmul(flip_z, torch.matmul(c2w_cam, mirror_x))
    
        # 2. If color_03 of pshuman is horizontally flipped to correct left/right:
        # Apply mirror_x on image coordinate system first, then do Z-flip.
        mirror_x = torch.tensor([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=device)

        flip_z = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=device)
        c2w_pshuman = torch.matmul(flip_z, torch.matmul(c2w_cam, mirror_x))
        """
        # mirror_x applied first for horizontal flip
        mirror_x = torch.tensor([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=device)

        # Z-axis flip for back view (if didn't flip while loading)
        flip_z = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=device)
        c2w_pshuman = torch.matmul(flip_z, torch.matmul(c2w_cam, mirror_x))
    
    else:
        # Unknown view, return original
        c2w_pshuman = c2w_cam
    
    # Convert back to PyTorch 3D coordinate system
    w2c_pshuman = torch.linalg.inv(c2w_pshuman)
    RT_pshuman_mat = torch.matmul(torch.linalg.inv(c2c_mat), w2c_pshuman)
    
    return RT_pshuman_mat[:3, :4]


def group_frames_by_pose(frame_keys):
    """
    Group frame indices by pose (same frame_id shares same pose).
    
    Args:
        frame_keys: List of frame keys (may include view_ids)
    
    Returns:
        Dictionary mapping frame_id to list of indices that share that pose
    """
    pose_groups = {}
    for idx, frame_key in enumerate(frame_keys):
        shot_id, frame_id, view_id = split_frame_key(frame_key)
        pose_key = f"{shot_id}/{frame_id}"
        if pose_key not in pose_groups:
            pose_groups[pose_key] = []
        pose_groups[pose_key].append(idx)
    
    return pose_groups


def group_frames_by_camera(frame_keys):
    """
    Group frame indices by camera key.
    
    Camera key rules:
    1. <shot_id>/<frame_id> => camera_key: <shot_id>/<frame_id>
    2. <shot_id>/<frame_id>/<normal view_id> => camera_key: <shot_id>/<view_id>
    3. <shot_id>/<frame_id>/<pshuman view_id> => derived from camera at <shot_id>/<frame_id>
    
    Args:
        frame_keys: List of frame keys (may include view_ids)
    
    Returns:
        Dictionary mapping camera_key to (list of indices, is_pshuman, source_camera_key)
    """
    camera_groups = {}
    for idx, frame_key in enumerate(frame_keys):
        shot_id, frame_id, view_id = split_frame_key(frame_key)
        
        if view_id is None:
            # Original frame: camera_key = shot_id/frame_id
            camera_key = f"{shot_id}/{frame_id}"
            is_pshuman = False
            source_camera_key = None
        elif 'pshuman' in view_id or 'color' in view_id:
            # PSHuman view: derived from shot_id/frame_id
            camera_key = f"{shot_id}/{frame_id}/{view_id}"
            is_pshuman = True
            source_camera_key = f"{shot_id}/{frame_id}"
        else:
            # Normal multi-view: camera_key = shot_id/view_id
            camera_key = f"{shot_id}/{view_id}"
            is_pshuman = False
            source_camera_key = None
        
        if camera_key not in camera_groups:
            camera_groups[camera_key] = ([], is_pshuman, source_camera_key)
        camera_groups[camera_key][0].append(idx)
    
    return camera_groups


def make_camera_R6d_T(unique_gl_T, unique_gl_R_6d, batch_to_camera_idx, unique_cameras, camera_to_idx, camera_is_pshuman, camera_source_key, batch_size, device='cuda'):
    """
    Expand camera parameters for batch based on camera types.
    For regular cameras, uses optimized parameters directly.
    For PSHuman cameras, derives from source camera using transformation.
    
    Args:
        unique_gl_T: Optimized translation parameters for unique cameras
        unique_gl_R_6d: Optimized 6D rotation parameters for unique cameras
        batch_to_camera_idx: Tensor mapping batch indices to camera indices
        unique_cameras: List of unique camera keys
        camera_to_idx: Mapping from camera_key to camera index
        camera_is_pshuman: Dict mapping camera_key to boolean (is PSHuman?)
        camera_source_key: Dict mapping camera_key to source camera_key (for PSHuman)
        batch_size: Number of frames in batch
        device: Device for tensor operations
    
    Returns:
        Tuple of (gl_T, gl_R_6d) tensors expanded to batch size
    """
    gl_T_list = []
    gl_R_6d_list = []
    
    for batch_idx in range(batch_size):
        camera_idx = batch_to_camera_idx[batch_idx].item()
        camera_key = unique_cameras[camera_idx]
        
        if camera_is_pshuman[camera_key]:
            # PSHuman camera: derive from source camera
            source_key = camera_source_key[camera_key]
            if source_key in camera_to_idx:
                source_camera_idx = camera_to_idx[source_key]
                
                # Build source camera RT from optimized params
                source_RT = torch.eye(4, device=device, dtype=torch.float32)
                source_RT[:3, 3] = unique_gl_T[source_camera_idx]
                source_RT[:3, :3] = rotation_6d_to_matrix(unique_gl_R_6d[source_camera_idx])
                
                # Extract view_id from camera_key (format: shot_id/frame_id/view_id)
                view_id = camera_key.split('/')[-1]
                
                # Calculate PSHuman camera
                pshuman_RT = make_pshuman_camera(source_RT[:3, :4], view_id, device)
                
                # Extract T and R_6d
                gl_T_list.append(pshuman_RT[:3, 3])
                gl_R_6d_list.append(matrix_to_rotation_6d(pshuman_RT[:3, :3]))
            else:
                # Fallback: use camera params directly
                gl_T_list.append(unique_gl_T[camera_idx])
                gl_R_6d_list.append(unique_gl_R_6d[camera_idx])
        else:
            # Regular camera: use optimized parameters directly
            gl_T_list.append(unique_gl_T[camera_idx])
            gl_R_6d_list.append(unique_gl_R_6d[camera_idx])
    
    return torch.stack(gl_T_list, dim=0), torch.stack(gl_R_6d_list, dim=0)


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
        lambda_mtn_hand_pose = optim_cfg.get('lambda_mtn_hand_pose', 0.0)
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
        
        # Multi-view support: Group frames by pose and camera
        pose_groups = group_frames_by_pose(track_frames)
        camera_groups = group_frames_by_camera(track_frames)
        
        # Create pose indices mapping: each unique pose gets an index
        unique_poses = sorted(pose_groups.keys())
        pose_to_idx = {pose_key: idx for idx, pose_key in enumerate(unique_poses)}
        n_unique_poses = len(unique_poses)
        
        # Create camera indices mapping: each unique camera gets an index
        unique_cameras = sorted(camera_groups.keys())
        camera_to_idx = {cam_key: idx for idx, cam_key in enumerate(unique_cameras)}
        n_unique_cameras = len(unique_cameras)
        
        # Extract camera metadata
        camera_is_pshuman = {cam_key: camera_groups[cam_key][1] for cam_key in unique_cameras}
        camera_source_key = {cam_key: camera_groups[cam_key][2] for cam_key in unique_cameras}
        
        # Build mapping from batch index to pose/camera indices
        batch_to_pose_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        batch_to_camera_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        for batch_idx, frame_key in enumerate(track_frames):
            shot_id, frame_id, view_id = split_frame_key(frame_key)
            pose_key = f"{shot_id}/{frame_id}"
            
            # Determine camera key using same logic as group_frames_by_camera
            if view_id is None:
                camera_key = f"{shot_id}/{frame_id}"
            elif 'pshuman' in view_id or 'color' in view_id:
                camera_key = f"{shot_id}/{frame_id}/{view_id}"
            else:
                camera_key = f"{shot_id}/{view_id}"
            
            batch_to_pose_idx[batch_idx] = pose_to_idx[pose_key]
            batch_to_camera_idx[batch_idx] = camera_to_idx[camera_key]
        
        # Initialize unique pose parameters
        if not share_pose:
            # Create unique pose tensors (one per unique frame_id)
            unique_body_pose = torch.zeros(n_unique_poses, body_pose.shape[1], 3, 
                                          dtype=torch.float32, device=self.device)
            unique_global_pose = torch.zeros(n_unique_poses, global_pose.shape[1],
                                            dtype=torch.float32, device=self.device)
            unique_left_hand_pose = torch.zeros(n_unique_poses, left_hand_pose.shape[1], 3,
                                               dtype=torch.float32, device=self.device)
            unique_right_hand_pose = torch.zeros(n_unique_poses, right_hand_pose.shape[1], 3,
                                                dtype=torch.float32, device=self.device)
            
            # Initialize from first occurrence of each pose
            for pose_key, batch_indices in pose_groups.items():
                pose_idx = pose_to_idx[pose_key]
                first_batch_idx = batch_indices[0]
                unique_body_pose[pose_idx] = body_pose[first_batch_idx].view(-1, 3).float()
                unique_global_pose[pose_idx] = global_pose[first_batch_idx].float()
                unique_left_hand_pose[pose_idx] = left_hand_pose[first_batch_idx].view(-1, 3).float()
                unique_right_hand_pose[pose_idx] = right_hand_pose[first_batch_idx].view(-1, 3).float()
            
            # Expand to batch size using mapping
            body_pose = unique_body_pose[batch_to_pose_idx]
            global_pose = unique_global_pose[batch_to_pose_idx]
            left_hand_pose = unique_left_hand_pose[batch_to_pose_idx]
            right_hand_pose = unique_right_hand_pose[batch_to_pose_idx]
        
        # Initialize unique camera parameters
        unique_camera_RT = torch.zeros(n_unique_cameras, 3, 4, dtype=torch.float32, device=self.device)
        
        # Initialize from first occurrence of each camera
        for camera_key in unique_cameras:
            batch_indices, is_pshuman, source_camera_key = camera_groups[camera_key]
            camera_idx = camera_to_idx[camera_key]
            first_batch_idx = batch_indices[0]
            
            if is_pshuman:
                # PSHuman camera: derive from source camera
                if source_camera_key in camera_to_idx:
                    # Get source camera index
                    source_batch_indices = camera_groups[source_camera_key][0]
                    source_batch_idx = source_batch_indices[0]
                    source_RT = camera_RT_params[source_batch_idx]
                    
                    # Extract view_id for transformation
                    _, _, view_id = split_frame_key(track_frames[first_batch_idx])
                    unique_camera_RT[camera_idx] = make_pshuman_camera(source_RT, view_id, self.device)
                else:
                    # Fallback: use current frame's RT
                    unique_camera_RT[camera_idx] = camera_RT_params[first_batch_idx]
            else:
                # Regular camera: use RT from first occurrence
                unique_camera_RT[camera_idx] = camera_RT_params[first_batch_idx]
        
        # Expand cameras to batch size using mapping
        camera_RT_params = unique_camera_RT[batch_to_camera_idx]
        
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
        
        # # Validate hand tracking
        # left_wrist_kpt2d = gt_lmk_2d['keypoints'][:, 7, :2]
        # left_elbow_kpt2d = gt_lmk_2d['keypoints'][:, 6, :2]
        # right_wrist_kpt2d = gt_lmk_2d['keypoints'][:, 4, :2]
        # right_elbow_kpt2d = gt_lmk_2d['keypoints'][:, 3, :2]
        
        # left_wrist_elbow_dist = torch.norm(left_wrist_kpt2d - left_elbow_kpt2d, dim=1)
        # right_wrist_elbow_dist = torch.norm(right_wrist_kpt2d - right_elbow_kpt2d, dim=1)
        
        # if left_hand_valid.sum() > 0:
        #     left_hand_wrist_3d = ref_hand_l_vertices[:, self.ehm.mano.selected_vert_ids].mean(dim=1)[:, :2]
        #     left_hand_wrist_dist = torch.norm(left_hand_wrist_3d - left_wrist_kpt2d, dim=1)
        #     left_hand_valid = left_hand_valid & (left_hand_wrist_dist < left_wrist_elbow_dist)
        
        # if right_hand_valid.sum() > 0:
        #     right_hand_wrist_3d = ref_hand_r_vertices[:, self.ehm.mano.selected_vert_ids].mean(dim=1)[:, :2]
        #     right_hand_wrist_dist = torch.norm(right_hand_wrist_3d - right_wrist_kpt2d, dim=1)
        #     right_hand_valid = right_hand_valid & (right_hand_wrist_dist < right_wrist_elbow_dist)
        
        # Identify pshuman frames for motion regularization anchoring
        is_pshuman_list = []
        for frame_key in track_frames:
            _, _, view_id = split_frame_key(frame_key)
            is_pshuman = (view_id is not None and 'pshuman' in view_id)
            is_pshuman_list.append(is_pshuman)
        is_pshuman_tensor = torch.tensor(is_pshuman_list, device=self.device).bool()

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
        
        # For multi-view: optimize unique poses
        if not share_pose:
            unique_body_pose.requires_grad = True
            unique_left_hand_pose.requires_grad = True
            unique_right_hand_pose.requires_grad = True
            unique_global_pose.requires_grad = True
        else:
            body_pose.requires_grad = True
            left_hand_pose.requires_grad = True
            right_hand_pose.requires_grad = True
        
        # For multi-view: optimize unique cameras
        unique_gl_T = unique_camera_RT[:, :3, 3].detach().clone()
        unique_gl_R_6d = matrix_to_rotation_6d(unique_camera_RT[:, :3, :3]).detach().clone()
        
        _lr_camera = 0.0
        if optim_camera:
            # For original cameras, can align Z
            for cam_idx, view_key in enumerate(unique_cameras):
                if view_key == "original":
                    # Align Z across all original cameras
                    orig_indices = [i for i, k in enumerate(unique_cameras) if k == "original"]
                    if len(orig_indices) > 0:
                        unique_gl_T[orig_indices, 2] = unique_gl_T[orig_indices, 2].mean()
            _lr_camera = 1.0
        
        unique_gl_T.requires_grad = True
        unique_gl_R_6d.requires_grad = True
        joints_offset.requires_grad = True
        g_flame_shape.requires_grad = True
        head_scale.requires_grad = True
        hand_scale.requires_grad = True
        
        # Define joint groups for regularization
        leg_body_joints = [4, 5, 7, 8, 10]
        freezed_body_joints = [1, 3, 2, 6, 9]
        
        # Setup optimizer - handle multi-view parameters
        opt_params = [
            {'params': [g_smplx_shape], 'lr': 1e-4 * _lr_decay},
            {'params': [left_hand_shape], 'lr': 1e-4 * _lr_decay},
            {'params': [right_hand_shape], 'lr': 1e-4 * _lr_decay},
            {'params': [unique_gl_T], 'lr': 5e-3 * _lr_camera},
            {'params': [unique_gl_R_6d], 'lr': 5e-4 * _lr_camera},
            {'params': [joints_offset], 'lr': 1e-5 * _lr_decay},
            {'params': [g_flame_shape], 'lr': 2e-5 * _lr_decay},
            {'params': [head_scale], 'lr': 1e-4 * _lr_decay},
            {'params': [hand_scale], 'lr': 1e-4 * _lr_decay},
        ]
        
        if not share_pose:
            opt_params.extend([
                {'params': [unique_body_pose], 'lr': 1e-3},
                {'params': [unique_left_hand_pose], 'lr': 1e-5},
                {'params': [unique_right_hand_pose], 'lr': 1e-5},
                {'params': [unique_global_pose], 'lr': 1e-3},
            ])
        else:
            opt_params.extend([
                {'params': [body_pose], 'lr': 1e-3},
                {'params': [left_hand_pose], 'lr': 1e-5},
                {'params': [right_hand_pose], 'lr': 1e-5},
            ])
        
        opt_p = torch.optim.AdamW(opt_params)
        
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
            
            # Expand unique poses to batch size using mapping
            if not share_pose:
                body_pose = unique_body_pose[batch_to_pose_idx]
                global_pose = unique_global_pose[batch_to_pose_idx]
                left_hand_pose = unique_left_hand_pose[batch_to_pose_idx]
                right_hand_pose = unique_right_hand_pose[batch_to_pose_idx]
            
            batch_smplx['body_pose'] = body_pose.expand(batch_size, -1, -1) if share_pose else body_pose
            batch_smplx['global_pose'] = global_pose.expand(batch_size, -1) if share_pose else global_pose
            batch_smplx['left_hand_pose'] = left_hand_pose.expand(batch_size, -1, -1) if share_pose else left_hand_pose
            batch_smplx['right_hand_pose'] = right_hand_pose.expand(batch_size, -1, -1) if share_pose else right_hand_pose
            
            # Expand camera parameters (regular directly, PSHuman derived)
            gl_T, gl_R_6d = make_camera_R6d_T(
                unique_gl_T, unique_gl_R_6d, batch_to_camera_idx,
                unique_cameras, camera_to_idx, camera_is_pshuman, camera_source_key,
                batch_size, self.device
            )
            
            if self.use_vposer:
                # VPoser expects unique poses
                vp_body_pose = unique_body_pose if not share_pose else body_pose
                body_embedding_mean = self.vposer.encode(vp_body_pose).mean
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
                # Helper for anchored motion loss
                def get_anchored_diff(tensor, is_anchor):
                    curr = tensor[1:]
                    prev = tensor[:-1]
                    return curr, prev
                    # curr_anchor = is_anchor[1:]
                    # prev_anchor = is_anchor[:-1]
                    
                    # view_shape = [-1] + [1] * (tensor.ndim - 1)
                    # curr_mask = curr_anchor.view(*view_shape)
                    # prev_mask = prev_anchor.view(*view_shape)
                    
                    # curr_term = torch.where(curr_mask, curr.detach(), curr)
                    # prev_term = torch.where(prev_mask, prev.detach(), prev)
                    # return curr_term, prev_term

                curr_bp, prev_bp = get_anchored_diff(body_pose, is_pshuman_tensor)
                mtn_reg_loss += self.metric(curr_bp, prev_bp) * lambda_mtn_body_pose / (interval * 1)
                
                curr_lhp, prev_lhp = get_anchored_diff(left_hand_pose, is_pshuman_tensor)
                mtn_reg_loss += self.metric(curr_lhp, prev_lhp) * lambda_mtn_hand_pose / (interval * 1)
                
                curr_rhp, prev_rhp = get_anchored_diff(right_hand_pose, is_pshuman_tensor)
                mtn_reg_loss += self.metric(curr_rhp, prev_rhp) * lambda_mtn_hand_pose / (interval * 1)
                
                curr_rot, prev_rot = get_anchored_diff(gl_R_6d, is_pshuman_tensor)
                mtn_reg_loss += self.metric(curr_rot, prev_rot) * lambda_mtn_rot6d / (interval * 1)
                
                curr_T, prev_T = get_anchored_diff(T, is_pshuman_tensor)
                mtn_reg_loss += self.metric(curr_T, prev_T) * lambda_mtn_trans / (interval * 1)
                
                curr_Tz, prev_Tz = get_anchored_diff(T[..., 2:3], is_pshuman_tensor)
                mtn_reg_loss += self.metric(curr_Tz, prev_Tz) * lambda_mtn_trans_z / (interval * 1)
                
                curr_v, prev_v = get_anchored_diff(proj_vertices, is_pshuman_tensor)
                mtn_reg_loss += self.metric(curr_v, prev_v) * lambda_mtn_vertices / (interval * 1)
            
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
        
        # Build final camera RT for each batch index
        final_gl_T, final_gl_R_6d = make_camera_R6d_T(
            unique_gl_T.detach(), unique_gl_R_6d.detach(), batch_to_camera_idx,
            unique_cameras, camera_to_idx, camera_is_pshuman, camera_source_key,
            batch_size, self.device
        )
        
        final_camera_RT_list = []
        for batch_idx in range(batch_size):
            final_RT = torch.eye(4, device=self.device, dtype=torch.float32)
            final_RT[:3, 3] = final_gl_T[batch_idx]
            final_RT[:3, :3] = rotation_6d_to_matrix(final_gl_R_6d[batch_idx])
            final_camera_RT_list.append(final_RT[:3, :4])
        
        final_camera_RT = torch.stack(final_camera_RT_list, dim=0)
        final_body_pose = unique_body_pose[batch_to_pose_idx] if not share_pose else body_pose
        final_global_pose = unique_global_pose[batch_to_pose_idx] if not share_pose else global_pose
        final_left_hand_pose = unique_left_hand_pose[batch_to_pose_idx] if not share_pose else left_hand_pose
        final_right_hand_pose = unique_right_hand_pose[batch_to_pose_idx] if not share_pose else right_hand_pose
        
        batch_smplx['camera_RT_params'] = final_camera_RT
        batch_smplx['body_pose'] = final_body_pose
        batch_smplx['global_pose'] = final_global_pose
        batch_smplx['left_hand_pose'] = final_left_hand_pose
        batch_smplx['right_hand_pose'] = final_right_hand_pose
        
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
                'global_pose': final_global_pose[idx].detach().float().cpu().numpy(),
                'body_pose': final_body_pose[idx].detach().float().cpu().numpy(),
                'body_cam': batch_smplx['body_cam'][idx].detach().float().cpu().numpy(),
                'camera_RT_params': final_camera_RT[idx].detach().float().cpu().numpy(),
                'left_hand_pose': final_left_hand_pose[idx].detach().float().cpu().numpy(),
                'right_hand_pose': final_right_hand_pose[idx].detach().float().cpu().numpy(),
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
        
        # Calculate 1:2 aspect ratio grid layout
        total_frames = len(valid_keys)
        grid_rows = max(1, int(np.sqrt(total_frames * 2)))
        grid_cols = int(np.ceil(total_frames / grid_rows))
        
        # Adjust to maintain 1:2 ratio
        while grid_rows < 2 * grid_cols and grid_cols > 1:
            grid_cols -= 1
            grid_rows = int(np.ceil(total_frames / grid_cols))
        
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
        grid_rows = max(1, int(np.sqrt(total_frames * 2)))
        grid_cols = int(np.ceil(total_frames / grid_rows))
        while grid_rows < 2 * grid_cols and grid_cols > 1:
            grid_cols -= 1
            grid_rows = int(np.ceil(total_frames / grid_cols))
        
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
