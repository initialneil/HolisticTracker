"""
Pixel3DMM FLAME Initializer for ehm-tracker.

Replaces TEASER with pixel3dmm's neural network + optimization-based FLAME fitting.
Produces TEASER-compatible output format for downstream compatibility.

Pipeline:
1. Load p3dmm network (DINO ViT → transformer → UV/normal prediction)
2. Run network inference on face crop → UV map + normals
3. Run optimization-based FLAME fitting using UV/normal predictions
4. Convert to TEASER-compatible output format
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from omegaconf import OmegaConf
from tqdm import tqdm

from . import env_config as env_paths
from . import nvdiffrast_util
from .utils_3d import rotation_6d_to_matrix, matrix_to_rotation_6d, euler_angles_to_matrix
from .losses import UVLoss


# Default config for FLAME model
_DEFAULT_FLAME_CONFIG = {
    'use_flame2023': False,
    'num_shape_params': 300,
    'num_exp_params': 100,
    'image_size': (512, 512),
    'size': 512,
}

# Default config for optimization
_DEFAULT_OPT_CONFIG = {
    'iters': 200,
    'lr_exp': 0.005,
    'lr_R': 0.005,
    'lr_t': 0.01,
    'lr_jaw': 0.005,
    'lr_neck': 0.005,
    'lr_eyes': 0.005,
    'lr_eyelids': 0.005,
    'lr_shape': 0.002,
    'lr_focal': 0.0001,
    'lr_pp': 0.0001,
    'w_uv': 2000.0,
    'w_normal': 1000.0,
    'w_exp': 0.05,
    'w_shape': 0.2,
    'w_jaw': 0.01,
    'w_neck': 0.01,
    'delta_uv': 0.00005,
    'dist_uv': 20,
    'delta_n': 0.15,
    'early_stopping_delta': 5.0,
}


class SimpleStruct:
    """Simple config struct."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _load_network(ckpt_path, device='cuda:0'):
    """Load pretrained p3dmm network from checkpoint."""
    import pytorch_lightning as L
    from .p3dmm_system import system as P3DMMSystem

    # Load checkpoint to extract config
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['hyper_parameters']['cfg']

    # Create system and load weights
    model = P3DMMSystem.load_from_checkpoint(ckpt_path, strict=False, weights_only=False,
                                              map_location=device)
    model.eval()
    model.to(device)
    return model.net, cfg


class Pixel3DMMInitializer:
    """
    Self-contained pixel3dmm FLAME initializer.

    Takes a face crop image and returns FLAME parameters in TEASER-compatible format.
    """

    def __init__(self, assets_dir, flame_assets_dir, network_ckpt_path, device='cuda:0',
                 opt_config=None):
        """
        Args:
            assets_dir: Path to pixel3dmm assets (UV coords, masks, head template, etc.)
            flame_assets_dir: Path to FLAME model files (generic_model.pkl, etc.)
            network_ckpt_path: Path to p3dmm UV network checkpoint
            device: CUDA device
            opt_config: Optional dict of optimization config overrides
        """
        self.device = device
        self.opt_cfg = {**_DEFAULT_OPT_CONFIG, **(opt_config or {})}

        # Initialize global env_paths config (used by losses, renderer, FLAME, etc.)
        env_paths.init_config(assets_dir, flame_assets_dir, network_ckpt_path, device)

        print("[Pixel3DMM] Loading network...")
        self.network, self.net_cfg = _load_network(network_ckpt_path, device)
        self.network.eval()

        print("[Pixel3DMM] Loading FLAME model...")
        flame_cfg = SimpleStruct(**_DEFAULT_FLAME_CONFIG)
        from .flame.FLAME import FLAME
        self.flame = FLAME(flame_cfg).to(device)

        print("[Pixel3DMM] Setting up renderer...")
        self._setup_renderer(assets_dir)

        print("[Pixel3DMM] Setting up UV loss...")
        self.uv_loss_fn = UVLoss(
            stricter_mask=False,
            delta_uv=self.opt_cfg['delta_uv'],
            dist_uv=self.opt_cfg['dist_uv'],
        )

        # Load mirror index for symmetry loss
        mirror_index_path = os.path.join(assets_dir, 'flame_mirror_index.npy')
        if os.path.exists(mirror_index_path):
            self.mirror_order = torch.from_numpy(np.load(mirror_index_path)).long().to(device)
        else:
            self.mirror_order = None

        # FLAME face mask for rendering
        flame_mask_path = os.path.join(flame_assets_dir, 'FLAME2020', 'FLAME_masks', 'FLAME_masks.pkl')
        if os.path.exists(flame_mask_path):
            import pickle
            with open(flame_mask_path, 'rb') as f:
                flame_mesh_mask = pickle.load(f, encoding='latin1')
            self.vertex_face_mask = torch.from_numpy(flame_mesh_mask['face']).long().to(device)
            self.flame.vertex_face_mask = self.vertex_face_mask

        # Identity rotation in 6D format
        self.I6D = matrix_to_rotation_6d(torch.eye(3)[None].to(device))

        print("[Pixel3DMM] Initializer ready.")

    def _setup_renderer(self, assets_dir):
        """Setup nvdiffrast renderer for FLAME mesh."""
        from .renderer_nvdiffrast import NVDRenderer
        head_template_path = os.path.join(assets_dir, 'head_template.obj')
        self.diff_renderer = NVDRenderer(
            image_size=512,
            obj_filename=head_template_path,
            uv_size=512,
        )
        self.diff_renderer.to(self.device)

    def _setup_camera(self):
        """Setup initial camera parameters (OpenGL convention)."""
        import dreifus
        from dreifus.matrix import Pose

        pose_mat = np.eye(4)
        pose_mat[2, 3] = -1
        cam_pose = Pose(pose_mat,
                       camera_coordinate_convention=dreifus.matrix.CameraCoordinateConvention.OPEN_CV)
        cam_pose = cam_pose.change_pose_type(dreifus.matrix.PoseType.CAM_2_WORLD)
        cam_pose.look_at(np.zeros(3), np.array([0, 1, 0]))
        cam_pose = cam_pose.change_pose_type(dreifus.matrix.PoseType.WORLD_2_CAM)

        cam_pose_c2w = cam_pose.change_pose_type(dreifus.matrix.PoseType.CAM_2_WORLD)
        cam_pose_nvd = cam_pose_c2w.copy()
        cam_pose_nvd = cam_pose_nvd.change_camera_coordinate_convention(
            new_camera_coordinate_convention=dreifus.matrix.CameraCoordinateConvention.OPEN_GL)
        cam_pose_nvd = cam_pose_nvd.change_pose_type(dreifus.matrix.PoseType.WORLD_2_CAM)

        cam_pose_nvd_tensor = torch.from_numpy(cam_pose_nvd.copy()).float().to(self.device)
        R_base = torch.from_numpy(cam_pose_nvd.get_rotation_matrix()).unsqueeze(0).to(self.device)
        t_base = torch.from_numpy(cam_pose_nvd.get_translation()).unsqueeze(0).to(self.device)

        return cam_pose_nvd_tensor, R_base, t_base

    @torch.no_grad()
    def _run_network(self, face_crop):
        """
        Run p3dmm network inference to get UV map and normals.

        Args:
            face_crop: (512, 512, 3) uint8 RGB image

        Returns:
            uv_map: (1, 2, 512, 512) tensor
            normal_map: (1, 3, 512, 512) tensor
            uv_mask: (1, 3, 512, 512) tensor
            normal_mask: (1, 3, 512, 512) tensor
        """
        # Prepare input: normalize to [0, 1], channels-last (B, N, H, W, C)
        img = torch.from_numpy(face_crop).float().to(self.device) / 255.0
        img = img.unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512, 3)

        # Network expects a dict batch with 'tar_rgb' key
        batch = {'tar_rgb': img}
        with torch.no_grad():
            pred, conf = self.network(batch)

        # Extract predictions
        uv_map = pred.get('uv_map', None)
        normal_map = pred.get('normals', None)

        # Generate masks (non-zero predictions are valid)
        if uv_map is not None:
            uv_map = uv_map.squeeze(1)  # remove view dim → (B, 2, H, W)
            uv_mask = (uv_map.abs().sum(dim=1, keepdim=True) > 0.01).float().expand_as(
                torch.zeros(1, 3, 512, 512, device=self.device))
        else:
            uv_mask = torch.ones(1, 3, 512, 512, device=self.device)

        if normal_map is not None:
            normal_map = normal_map.squeeze(1)  # remove view dim → (B, 3, H, W)
            normal_mask = (normal_map.abs().sum(dim=1, keepdim=True) > 0.01).float().expand_as(
                torch.zeros(1, 3, 512, 512, device=self.device))
        else:
            normal_mask = torch.ones(1, 3, 512, 512, device=self.device)

        return uv_map, normal_map, uv_mask, normal_mask

    def _run_network_with_mirror(self, face_crop):
        """Run network with mirror augmentation (average original + flipped)."""
        uv_map, normal_map, uv_mask, normal_mask = self._run_network(face_crop)

        # Mirror augmentation: flip horizontally, run again, average
        face_crop_flip = cv2.flip(face_crop, 1)
        uv_map_flip, normal_map_flip, _, _ = self._run_network(face_crop_flip)

        if uv_map is not None and uv_map_flip is not None:
            # Flip the flipped prediction back
            uv_map_flip = torch.flip(uv_map_flip, [-1])
            uv_map_flip[:, 0] = 1.0 - uv_map_flip[:, 0]  # flip U coordinate
            uv_map = (uv_map + uv_map_flip) / 2.0

        if normal_map is not None and normal_map_flip is not None:
            normal_map_flip = torch.flip(normal_map_flip, [-1])
            normal_map_flip[:, 0] = -normal_map_flip[:, 0]  # flip X normal
            normal_map = (normal_map + normal_map_flip) / 2.0

        return uv_map, normal_map, uv_mask, normal_mask

    def _optimize_flame(self, uv_map, normal_map, uv_mask, normal_mask, init_params=None, landmarks_2d=None):
        """
        Optimize FLAME parameters to match UV map and normal predictions.

        Args:
            init_params: Optional dict of initial parameters from previous frame (for video tracking).
                         Keys: shape, exp, jaw, neck, eyes, eyelids, R, t, focal_length, principal_point
            landmarks_2d: Optional (1, 68, 2) tensor of detected face landmarks in 512 pixel coords.
                         Used to constrain face position in the first half of optimization.

        Returns:
            Dict of optimized parameters (all on GPU as tensors)
        """
        cfg = self.opt_cfg
        size = 512
        bs = 1

        # Setup camera
        cam_pose_nvd, R_base, t_base = self._setup_camera()

        # Initialize FLAME parameters (from init_params if provided, else zeros)
        if init_params is not None:
            init_f = init_params['focal_length'].item() * size
            focal_length = nn.Parameter(init_params['focal_length'].clone())
            principal_point = nn.Parameter(init_params['principal_point'].clone())
            shape = nn.Parameter(init_params['shape'].clone())
            exp = nn.Parameter(init_params['exp'].clone())
            jaw = nn.Parameter(init_params['jaw'].clone())
            neck = nn.Parameter(init_params['neck'].clone())
            eyes = nn.Parameter(init_params['eyes'].clone())
            eyelids = nn.Parameter(init_params['eyelids'].clone())
            R = nn.Parameter(init_params['R'].clone())
            t = nn.Parameter(init_params['t'].clone())
        else:
            init_f = 2000 * size / 512
            focal_length = nn.Parameter(torch.tensor([[init_f / size]]).float().to(self.device))
            principal_point = nn.Parameter(torch.tensor([[0.0, 0.0]]).float().to(self.device))
            shape = nn.Parameter(torch.zeros(1, 300, device=self.device))
            exp = nn.Parameter(torch.zeros(1, 100, device=self.device))
            jaw = nn.Parameter(self.I6D.clone())
            neck = nn.Parameter(self.I6D.clone())
            eyes = nn.Parameter(torch.cat([self.I6D.clone()] * 2, dim=1))
            eyelids = nn.Parameter(torch.zeros(1, 2, device=self.device))
            R = nn.Parameter(matrix_to_rotation_6d(torch.eye(3)[None].to(self.device)))
            t = nn.Parameter(torch.zeros(1, 3, device=self.device))

        mica_shape = torch.zeros(1, 300, device=self.device)

        # Build MVP matrix
        intrinsics = torch.eye(3)[None].float().to(self.device)
        intrinsics[:, 0, 0] = focal_length.squeeze() * size
        intrinsics[:, 1, 1] = focal_length.squeeze() * size
        intrinsics[:, :2, 2] = size / 2 + 0.5

        proj = nvdiffrast_util.intrinsics2projection(intrinsics, znear=0.1, zfar=10,
                                                      width=size, height=size)
        w2c = torch.eye(4)[None].float().to(self.device)
        w2c[:, :3, :3] = R_base
        w2c[:, :3, 3] = t_base
        r_mvp = (proj @ cam_pose_nvd)[None, ...]

        # Setup optimizer
        opt_params = [
            {'params': [exp], 'lr': cfg['lr_exp'], 'name': 'exp'},
            {'params': [jaw], 'lr': cfg['lr_jaw'], 'name': 'jaw'},
            {'params': [neck], 'lr': cfg['lr_neck'], 'name': 'neck'},
            {'params': [eyes], 'lr': cfg['lr_eyes'], 'name': 'eyes'},
            {'params': [eyelids], 'lr': cfg['lr_eyelids'], 'name': 'eyelids'},
            {'params': [R], 'lr': cfg['lr_R'], 'name': 'R'},
            {'params': [t], 'lr': cfg['lr_t'], 'name': 't'},
            {'params': [shape], 'lr': cfg['lr_shape'], 'name': 'shape'},
            {'params': [focal_length], 'lr': cfg['lr_focal'], 'name': 'focal'},
            {'params': [principal_point], 'lr': cfg['lr_pp'], 'name': 'pp'},
        ]
        optimizer = torch.optim.Adam(opt_params)

        # Compute UV correspondences (once)
        if uv_map is not None:
            self.uv_loss_fn.compute_corresp(uv_map)

        # Optimization loop
        best_loss = float('inf')
        stagnant_window = 10
        past_k = np.array([100.0] * stagnant_window)
        prev_loss = 100.0

        iters = cfg['iters']
        iterator = tqdm(range(iters), desc='[Pixel3DMM] Fitting FLAME', leave=True, miniters=50)

        for p in iterator:
            optimizer.zero_grad()
            self.diff_renderer.reset()

            # Recompute MVP with current focal length
            intr = torch.eye(3)[None].float().to(self.device)
            intr[:, 0, 0] = focal_length.squeeze() * size
            intr[:, 1, 1] = focal_length.squeeze() * size
            intr[:, :2, 2] = size / 2 + 0.5 + principal_point * (size / 2 + 0.5)
            # flip x (consistent with tracker.py's use_hack=False in get_intrinsics)

            proj_mat = nvdiffrast_util.intrinsics2projection(intr, znear=0.1, zfar=5,
                                                              width=size, height=size)
            extr = w2c.clone()
            cur_mvp = torch.matmul(proj_mat, extr)

            # Forward FLAME
            R_base_inv = torch.inverse(R_base)
            vertices_can, lmk68, _, v_can_can, vertices_noneck = self.flame(
                cameras=R_base_inv.expand(bs, -1, -1),
                shape_params=shape,
                expression_params=exp,
                eye_pose_params=eyes,
                jaw_pose_params=jaw,
                neck_pose_params=neck,
                eyelid_params=eyelids,
                rot_params_lmk_shift=matrix_to_rotation_6d(torch.inverse(rotation_6d_to_matrix(R))),
            )

            # Apply head rotation + translation
            rot_mat = rotation_6d_to_matrix(R)
            vertices = torch.einsum('bny,bxy->bnx', vertices_can, rot_mat) + t.unsqueeze(1)
            vertices_noneck = torch.einsum('bny,bxy->bnx', vertices_noneck, rot_mat) + t.unsqueeze(1)

            # Project vertices to screen space
            proj_vertices = self._project_screen(vertices, focal_length, principal_point,
                                                  R_base, t_base, size)

            # Render mesh
            ops = self.diff_renderer(
                vertices, None, None, cur_mvp, R_base, t_base,
                verts_can=vertices_can,
                verts_noneck=vertices_noneck,
                verts_can_can=v_can_can,
                verts_depth=proj_vertices[:, :, 2:3],
            )

            losses = {}

            # UV loss
            if uv_map is not None and cfg['w_uv'] > 0:
                # Occlusion test
                grabbed_depth = ops['actual_rendered_depth'][:, 0,
                    torch.clamp(proj_vertices[:, :, 1].long(), 0, size - 1),
                    torch.clamp(proj_vertices[:, :, 0].long(), 0, size - 1),
                ][:, 0, :]
                is_visible = grabbed_depth < (proj_vertices[:, :, 2] + 1e-2)

                uv_loss = self.uv_loss_fn.compute_loss(
                    proj_vertices, is_visible_verts_idx=is_visible)
                losses['uv'] = uv_loss * cfg['w_uv']

                # Pixel-level UV loss
                gt_uv = uv_map[:, :2, :, :].permute(0, 2, 3, 1)
                uv_pixel_loss = ((gt_uv - ops['uv_images']) * uv_mask[:, 0:1, :, :].permute(0, 2, 3, 1)).abs().mean()
                losses['uv_pixel'] = uv_pixel_loss * cfg['w_uv']

            # Normal loss
            if normal_map is not None and cfg['w_normal'] > 0:
                from torchvision.transforms.functional import gaussian_blur
                dilated_eye_mask = 1 - (gaussian_blur(
                    ops['mask_images_eyes'], [15, 15], sigma=[15, 15]) > 0).float()
                pred_normals = ops['normal_images']
                rot_mat_r = rotation_6d_to_matrix(R)
                pred_normals_flame = torch.einsum('bxy,bxhw->byhw', rot_mat_r, pred_normals)

                l_map = normal_map - pred_normals_flame
                valid = ((l_map.abs().sum(dim=1) / 3) < cfg['delta_n']).unsqueeze(1)
                normal_loss_map = l_map * valid.float() * normal_mask * dilated_eye_mask
                losses['normal'] = normal_loss_map.abs().mean() * cfg['w_normal']

            # Regularizers
            right_eye, left_eye = eyes[:, :6], eyes[:, 6:]
            losses['reg_exp'] = torch.sum(exp ** 2, dim=-1).mean() * cfg['w_exp']
            losses['reg_jaw'] = torch.sum((self.I6D - jaw) ** 2, dim=-1).mean() * cfg['w_jaw']
            losses['reg_neck'] = torch.sum((self.I6D - neck) ** 2, dim=-1).mean() * cfg['w_neck']
            losses['reg_shape'] = torch.sum((shape - mica_shape) ** 2, dim=-1).mean() * cfg['w_shape']
            losses['reg_eye_sym'] = torch.sum((right_eye - left_eye) ** 2, dim=-1).mean() * 0.1
            losses['reg_eye_l'] = torch.sum((self.I6D - left_eye) ** 2, dim=-1).mean() * 0.01
            losses['reg_eye_r'] = torch.sum((self.I6D - right_eye) ** 2, dim=-1).mean() * 0.01
            losses['reg_pp'] = torch.sum(principal_point ** 2, dim=-1).mean()

            # Landmark loss (constrains face position, first half of optimization only)
            if landmarks_2d is not None and p <= iters // 2:
                lmk68_world = torch.einsum('bny,bxy->bnx', lmk68, rot_mat) + t.unsqueeze(1)
                proj_lmk68 = self._project_screen(lmk68_world, focal_length, principal_point,
                                                   R_base, t_base, size)
                lmk_diff = (proj_lmk68[..., :2] - landmarks_2d) / size
                losses['lmk68'] = (lmk_diff ** 2).mean() * 3000

            # Mirror symmetry loss
            if self.mirror_order is not None:
                verts_mirrored = v_can_can[:, self.mirror_order, :]
                verts_mirrored_sym = torch.zeros_like(verts_mirrored)
                verts_mirrored_sym[:, :, 0] = -verts_mirrored[:, :, 0]
                verts_mirrored_sym[:, :, 1:] = verts_mirrored[:, :, 1:]
                losses['reg_mirror'] = (verts_mirrored_sym - v_can_can).square().sum(-1).mean() * 5000

            all_loss = sum(losses.values())
            all_loss.backward()
            optimizer.step()

            loss_val = all_loss.item()
            iterator.set_description(f'[Pixel3DMM] Loss {loss_val:.4f}')

            # Early stopping
            if p > 0:
                past_k[p % stagnant_window] = abs(loss_val - prev_loss)
            prev_loss = loss_val

            if p > stagnant_window and np.mean(past_k) < cfg['early_stopping_delta']:
                print(f'[Pixel3DMM] Early stopping at iter {p}')
                break

        # Return optimized parameters
        return {
            'shape': shape.detach(),
            'exp': exp.detach(),
            'jaw': jaw.detach(),
            'neck': neck.detach(),
            'eyes': eyes.detach(),
            'eyelids': eyelids.detach(),
            'R': R.detach(),
            't': t.detach(),
            'focal_length': focal_length.detach(),
            'principal_point': principal_point.detach(),
        }

    def _project_screen(self, points3d, focal_length, principal_point, R_base, t_base, size=512):
        """Project 3D points to screen space (same as tracker.py's project_points_screen_space)."""
        # Build intrinsics
        intr = torch.eye(3)[None].float().to(self.device)
        intr[:, 0, 0] = focal_length.squeeze() * size
        intr[:, 1, 1] = focal_length.squeeze() * size
        intr[:, :2, 2] = size / 2 + 0.5 + principal_point * (size / 2 + 0.5)

        # Build extrinsics
        w2c = torch.eye(4)[None].float().to(self.device)
        w2c[:, :3, :3] = R_base
        w2c[:, :3, 3] = t_base

        B = points3d.shape[0]
        # Apply w2c
        pts_hom = torch.cat([points3d, torch.ones_like(points3d[..., :1])], dim=-1)
        pts_cam = torch.bmm(pts_hom, w2c.permute(0, 2, 1).expand(B, -1, -1))

        # Project (OpenGL convention: -z forward)
        pts_norm = pts_cam[..., :3] / -pts_cam[..., [2]]
        pts_screen = (-1) * torch.bmm(pts_norm, intr.permute(0, 2, 1).expand(B, -1, -1))[..., :2]
        pts_screen = torch.stack([size - 1 - pts_screen[..., 0], pts_screen[..., 1],
                                  pts_cam[..., 2]], dim=-1)
        return pts_screen

    def _convert_to_teaser_format(self, params):
        """
        Convert pixel3dmm optimized parameters to TEASER-compatible format.

        TEASER output format:
            pose_params: (1, 3) - axis-angle rotation
            cam: (1, 3) - weak perspective camera [s, tx, ty]
            shape_params: (1, 300) - FLAME shape
            expression_params: (1, 50) - expression (truncated from 100)
            eyelid_params: (1, 2) - eyelids
            jaw_params: (1, 3) - jaw rotation (axis-angle)
        """
        # Convert head rotation from 6D to axis-angle
        R_mat = rotation_6d_to_matrix(params['R'])  # (1, 3, 3)
        # axis-angle from rotation matrix
        pose_aa = self._rotation_matrix_to_axis_angle(R_mat[0])  # (3,)
        pose_params = pose_aa.unsqueeze(0).cpu().numpy()  # (1, 3)

        # Convert jaw from 6D to axis-angle (approximate via euler)
        jaw_mat = rotation_6d_to_matrix(params['jaw'])  # (1, 3, 3)
        jaw_aa = self._rotation_matrix_to_axis_angle(jaw_mat[0])
        jaw_params = jaw_aa.unsqueeze(0).cpu().numpy()  # (1, 3)

        # Construct weak-perspective camera approximation from perspective params
        # s ≈ 2 * focal / tz, tx ≈ proj_x, ty ≈ proj_y
        focal = params['focal_length'].item()
        t = params['t'].cpu().numpy()[0]  # (3,)
        s = 2.0 * focal / max(abs(t[2]), 0.1) if abs(t[2]) > 0.01 else 5.0
        cam = np.array([[s, t[0] * s, t[1] * s]], dtype=np.float32)

        # Truncate expression from 100 to 50 dims (TEASER uses 50)
        expression_100 = params['exp'].cpu().numpy()  # (1, 100)
        expression_50 = expression_100[:, :50]  # (1, 50)

        return {
            'pose_params': pose_params.astype(np.float32),
            'cam': cam.astype(np.float32),
            'shape_params': params['shape'].cpu().numpy().astype(np.float32),
            'expression_params': expression_50.astype(np.float32),
            'eyelid_params': params['eyelids'].cpu().numpy().astype(np.float32),
            'jaw_params': jaw_params.astype(np.float32),
        }

    def project_flame_to_crop(self, params):
        """
        Project FLAME vertices to 2D crop coordinates using pixel3dmm's camera.
        Returns (N, 2) numpy array of 2D vertex positions in the 512x512 crop space.
        """
        size = 512
        bs = 1

        cam_pose_nvd, R_base, t_base = self._setup_camera()

        # Forward FLAME with optimized params
        R_base_inv = torch.inverse(R_base)
        vertices_can, lmk68, _, v_can_can, vertices_noneck = self.flame(
            cameras=R_base_inv.expand(bs, -1, -1),
            shape_params=params['shape'],
            expression_params=params['exp'],
            eye_pose_params=params['eyes'],
            jaw_pose_params=params['jaw'],
            neck_pose_params=params['neck'],
            eyelid_params=params['eyelids'],
            rot_params_lmk_shift=matrix_to_rotation_6d(torch.inverse(rotation_6d_to_matrix(params['R']))),
        )

        # Apply head rotation + translation
        rot_mat = rotation_6d_to_matrix(params['R'])
        vertices = torch.einsum('bny,bxy->bnx', vertices_can, rot_mat) + params['t'].unsqueeze(1)

        # Project to screen
        proj = self._project_screen(vertices, params['focal_length'], params['principal_point'],
                                     R_base, t_base, size)
        return proj[0, :, :2].cpu().numpy()  # (N, 2)

    @staticmethod
    def _rotation_matrix_to_axis_angle(R):
        """Convert a 3x3 rotation matrix to axis-angle representation."""
        # Use Rodrigues formula inverse
        theta = torch.acos(torch.clamp((R.trace() - 1) / 2, -1.0, 1.0))
        if theta.abs() < 1e-6:
            return torch.zeros(3, device=R.device)
        k = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        k = k / (2 * torch.sin(theta))
        return k * theta

    def __call__(self, face_crop, use_mirror=True, landmarks_2d=None):
        """
        Run pixel3dmm FLAME initialization on a face crop image.

        Args:
            face_crop: (H, W, 3) uint8 RGB image (will be resized to 512x512)
            use_mirror: Whether to use mirror augmentation (default True)
            landmarks_2d: Optional (68, 2) numpy array of detected face landmarks in 512 pixel coords.
                         Used to constrain face position during optimization.

        Returns:
            dict with TEASER-compatible keys:
                pose_params, cam, shape_params, expression_params,
                eyelid_params, jaw_params
        """
        # Resize to 512x512 if needed
        if face_crop.shape[0] != 512 or face_crop.shape[1] != 512:
            face_crop = cv2.resize(face_crop, (512, 512))

        # Run network
        if use_mirror:
            uv_map, normal_map, uv_mask, normal_mask = self._run_network_with_mirror(face_crop)
        else:
            uv_map, normal_map, uv_mask, normal_mask = self._run_network(face_crop)

        # Prepare landmarks tensor if provided
        lmk_tensor = None
        if landmarks_2d is not None:
            lmk_tensor = torch.from_numpy(landmarks_2d[:68]).float().unsqueeze(0).to(self.device)  # (1, 68, 2)

        # Run optimization
        with torch.enable_grad():
            params = self._optimize_flame(uv_map, normal_map, uv_mask, normal_mask, landmarks_2d=lmk_tensor)

        # Convert to TEASER format
        result = self._convert_to_teaser_format(params)
        # Store projected 2D vertices in crop space for visualization
        result['_projected_verts_2d'] = self.project_flame_to_crop(params)
        return result

    def track_video(self, face_crops, use_mirror=True):
        """
        Run pixel3dmm on a sequence of face crops with temporal coherence.
        Each frame is initialized from the previous frame's optimized parameters.

        Args:
            face_crops: list of (H, W, 3) uint8 RGB images
            use_mirror: Whether to use mirror augmentation

        Returns:
            list of dicts with TEASER-compatible keys
        """
        results = []
        prev_params = None

        for i, face_crop in enumerate(face_crops):
            if face_crop.shape[0] != 512 or face_crop.shape[1] != 512:
                face_crop = cv2.resize(face_crop, (512, 512))

            if use_mirror:
                uv_map, normal_map, uv_mask, normal_mask = self._run_network_with_mirror(face_crop)
            else:
                uv_map, normal_map, uv_mask, normal_mask = self._run_network(face_crop)

            with torch.enable_grad():
                params = self._optimize_flame(uv_map, normal_map, uv_mask, normal_mask,
                                              init_params=prev_params)

            prev_params = {k: v.detach().clone() for k, v in params.items()}

            result = self._convert_to_teaser_format(params)
            result['_projected_verts_2d'] = self.project_flame_to_crop(params)
            results.append(result)

        return results
