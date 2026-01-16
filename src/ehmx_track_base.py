"""
TrackBasePipeline - Simplified tracking pipeline without LMDB
"""
import os
import cv2
import torch
import numpy as np
from PIL import Image as PILImage

from .utils.io import load_config
from .utils.helper import load_onnx_model, instantiate_from_config, image2tensor
from .utils.crop import crop_image, parse_bbox_from_landmark_lite, crop_image_by_bbox, _transform_pts
from .utils.landmark_runner import LandmarkRunner
from .modules.smplx.utils import orginaze_body_pose
from .modules.renderer.util import cam2persp_cam_fov, cam2persp_cam_fov_body


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


def split_frame_key(frame_key):
    """Split frame key into shot_id, frame_id, and optional view_id."""
    splits = frame_key.split('/')
    if len(splits) == 2:
        shot_id, frame_id, view_id = *splits, None
    elif len(splits) == 3:
        shot_id, frame_id, view_id = splits
    else:
        raise ValueError(f"Invalid frame_key format: {frame_key}")
    return shot_id, frame_id, view_id


def load_frame_image(images_dir, video_name, frame_key, pshuman_dir=None):
    shot_id, frame_id, view_id = split_frame_key(frame_key)
    
    if view_id is None:
        img_path = os.path.join(images_dir, video_name, shot_id, f"{frame_id}.jpg")
    elif 'pshuman' in view_id:
        img_path = os.path.join(pshuman_dir, video_name, shot_id, frame_id, f"color_{view_id.split('_')[-1]}.jpg")
    else:
        img_path = os.path.join(images_dir, video_name, shot_id, frame_id, f"{view_id:02}.jpg")
        
    if not os.path.exists(img_path):
        return None
        
    img = load_image(img_path)
    if view_id is not None and 'pshuman' in view_id:
        if '03' in view_id:
            img = cv2.flip(img, 1)  # Horizontal flip for left view
    return img


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


class TrackBasePipeline:
    """
    Simplified tracking pipeline for processing frames without LMDB.
    Only uses no_body_crop mode (processes images directly without body cropping).
    """
    
    def __init__(self, config):
        """
        Initialize the tracking pipeline.
        
        Args:
            config: Configuration object with paths to models and parameters
        """
        self.cfg = config
        self.device = config.device
        
        # Initialize detection and encoding models
        print("Initializing models...")
        self.dwpose_detector = instantiate_from_config(load_config(self.cfg.dwpose_cfg_path))
        self.dwpose_detector.warmup()
        
        self.pixie_encoder = instantiate_from_config(load_config(self.cfg.pixie_cfg_path))
        self.pixie_encoder.to(self.device)
        
        self.matte = instantiate_from_config(load_config(self.cfg.matting_cfg_path))
        self.matte.to(self.device)
        
        self.lmk70_detector = load_onnx_model(load_config(self.cfg.kp70_cfg_path))
        self.mp_detector = instantiate_from_config(load_config(self.cfg.mp_cfg_path))
        self.mp_detector.warmup()
        
        self.teaser_encoder = load_onnx_model(load_config(self.cfg.teaser_cfg_path))
        self.hamer_encoder = load_onnx_model(load_config(self.cfg.hamer_cfg_path))
        
        self.landmark_runner = LandmarkRunner(
            ckpt_path=self.cfg.kp203_path, 
            onnx_provider=self.device
        )
        self.landmark_runner.warmup()
        
        print("Models initialized successfully")
    
    def cvt_cam_to_persp_cam_fov(self, wcam):
        """Convert camera parameters to perspective camera with FOV."""
        R, T = cam2persp_cam_fov(wcam, tanfov=self.cfg.tanfov)
        return torch.cat((R, T[..., None]), axis=-1)
    
    def cvt_cam_to_persp_cam_fov_body(self, wcam):
        """Convert body camera parameters to perspective camera with FOV."""
        R, T = cam2persp_cam_fov_body(wcam, tanfov=self.cfg.tanfov)
        return torch.cat((R, T[..., None]), axis=-1)
    
    def track_base(self, img_rgb, no_body_crop=True, crop_scale=None,
                   skip_face=False, skip_hands=False):
        """
        Perform base tracking on a single frame.
        
        Args:
            img_rgb: Input RGB image (numpy array), should have matte already applied if needed
            no_body_crop: Whether to skip body cropping (default: True)
            crop_scale: Scale factor for cropping (default: None, uses 1.1 if no_body_crop is False)
            
        Returns:
            Tuple of (ret_images, base_results, mean_shape_results)
            - ret_images: Dictionary containing processed images
            - base_results: Dictionary containing tracking results
            - mean_shape_results: Dictionary containing shape parameters
            Returns (None, None, None) if detection fails
        """
        ret_images = {}
        base_results = {}
        mean_shape_results = {}
        
        ret_images['ori_image'] = img_rgb
        
        # Detect body pose and keypoints
        det_info, det_raw_info = self.dwpose_detector(img_rgb)
        
        # Handle no_body_crop mode
        if no_body_crop:
            # Use full image as bbox
            h, w = img_rgb.shape[:2]
            det_info['bbox'] = np.array([0, 0, w, h], dtype=np.float32)
            crop_scale = 1.0
        else:
            # Check for valid detection
            if det_info['bbox'] is None:
                print("          Missing bbox")
                return None, None, None
            # Set crop_scale if not provided
            if crop_scale is None:
                crop_scale = 1.1
        
        # Crop body region
        crop_info_hd = crop_image_by_bbox(
            img_rgb, det_info['bbox'], 
            dsize=self.cfg.body_hd_size, 
            scale=crop_scale
        )
        crop_info = crop_image_by_bbox(
            img_rgb, det_info['bbox'], 
            dsize=self.cfg.body_crop_size, 
            scale=crop_scale
        )
        
        base_results['body_crop'] = {
            'M_o2c': crop_info['M_o2c'], 
            'M_c2o': crop_info['M_c2o'],
            'M_o2c-hd': crop_info_hd['M_o2c'], 
            'M_c2o-hd': crop_info_hd['M_c2o']
        }
        base_results['dwpose_raw'] = det_raw_info
        base_results['dwpose_rlt'] = {
            'keypoints': _transform_pts(det_raw_info['keypoints'], crop_info_hd['M_o2c']),
            'scores': det_raw_info['scores'],
            'faces': det_info['faces'],
            'hands': det_info['hands']
        }
        
        # Check hand validity
        if not skip_hands:
            if base_results['dwpose_rlt']['scores'][-42:-21].mean() >= self.cfg.check_hand_score:
                base_results['left_hand_valid'] = True
            else:
                base_results['left_hand_valid'] = False
            if base_results['dwpose_rlt']['scores'][-21:].mean() >= self.cfg.check_hand_score:
                base_results['right_hand_valid'] = True
            else:
                base_results['right_hand_valid'] = False
        else:
            base_results['left_hand_valid'] = False
            base_results['right_hand_valid'] = False
        
        # Check if hands are crossing
        if ((base_results['dwpose_rlt']['keypoints'][-42:-21] - 
             base_results['dwpose_rlt']['keypoints'][-21:])**2).mean() < self.cfg.check_hand_dist:
            base_results['left_hand_valid'] = False
            base_results['right_hand_valid'] = False
        
        # Process body image
        img_crop = crop_info['img_crop']
        img_hd = crop_info_hd['img_crop']
        ret_images['body_image'] = img_hd
        
        img_crop = image2tensor(img_crop).to(self.device).unsqueeze(0)
        img_hd = image2tensor(img_hd).to(self.device).unsqueeze(0)
        
        # For simplicity, use original image without matting (matte should be pre-applied)
        img_crop = torch.nn.functional.interpolate(img_hd, (224, 224))
        
        # Extract body parameters
        coeff_param = self.pixie_encoder(img_crop, img_hd)['body']
        coeff_param = orginaze_body_pose(coeff_param)
        coeff_param['camera_RT_params'] = self.cvt_cam_to_persp_cam_fov_body(coeff_param['body_cam'])
        coeff_param = {k: v.cpu().numpy() for k, v in coeff_param.items()}
        base_results['smplx_coeffs'] = coeff_param
        mean_shape_results['smplx_shape'] = coeff_param['shape']
        
        # Process head
        crop_info = crop_image(img_rgb, det_info['faces'], dsize=self.cfg.head_crop_size, scale=1.75)
        base_results['head_crop'] = {'M_o2c': crop_info['M_o2c'], 'M_c2o': crop_info['M_c2o']}
        ret_images['head_image'] = crop_info['img_crop']
        
        # Detect landmarks
        lmk203 = self.landmark_runner.run(crop_info['img_crop'])['pts']
        t_img = crop_info['img_crop'].transpose((2, 0, 1)).astype(np.float32)
        lmk70 = self.lmk70_detector.run(t_img[None] / 255.)['pts'] * 2
        lmk_mp = self.mp_detector.run(crop_info['img_crop'])['pts']
        
        if skip_face or lmk203 is None or lmk_mp is None or lmk70 is None:
            base_results.update({
                'head_lmk_203': np.zeros((203, 2)),
                'head_lmk_70': np.zeros((70, 2)),
                'head_lmk_mp': np.zeros((478, 2)),
                'head_lmk_valid': False
            })
        else:
            if len(lmk203.shape) == 3:
                lmk203 = lmk203[0]
            if len(lmk70.shape) == 3:
                lmk70 = lmk70[0]
            if len(lmk_mp.shape) == 3:
                lmk_mp = lmk_mp[0]
            
            base_results.update({
                'head_lmk_203': lmk203,
                'head_lmk_70': lmk70,
                'head_lmk_mp': lmk_mp,
                'head_lmk_valid': True
            })
        
        # Extract FLAME parameters
        cropped_image = cv2.resize(
            crop_info['img_crop'], 
            (self.cfg.teaser_input_size, self.cfg.teaser_input_size)
        )
        cropped_image = np.transpose(cropped_image, (2, 0, 1))[None, ...] / 255.0
        coeff_param = self.teaser_encoder(cropped_image.astype(np.float32))
        coeff_param['camera_RT_params'] = self.cvt_cam_to_persp_cam_fov(
            torch.from_numpy(coeff_param['cam'])
        ).numpy()
        base_results['flame_coeffs'] = coeff_param
        mean_shape_results['flame_shape'] = coeff_param['shape_params']
        
        # Process hands
        all_hands_kps = det_info['hands']
        hand_kps_l = all_hands_kps[0]
        hand_kps_r = all_hands_kps[1]
        
        # Left hand
        crop_info = crop_image_by_bbox(
            img_rgb,
            parse_bbox_from_landmark_lite(hand_kps_l)['bbox'],
            lmk=hand_kps_l,
            scale=1.3,
            dsize=self.cfg.hand_crop_size
        )
        ret_images['left_hand_image'] = crop_info['img_crop']
        
        is_left = True
        t_img = cv2.flip(crop_info['img_crop'], 1) if is_left else crop_info['img_crop']
        cropped_image = np.transpose(t_img, (2, 0, 1))[None, ...] / 255.0
        coeff_param = self.hamer_encoder(cropped_image.astype(np.float32))
        coeff_param['camera_RT_params'] = self.cvt_cam_to_persp_cam_fov(
            torch.from_numpy(coeff_param['pred_cam'])
        ).numpy()
        base_results['left_mano_coeffs'] = coeff_param
        base_results['left_hand_crop'] = {'M_o2c': crop_info['M_o2c'], 'M_c2o': crop_info['M_c2o']}
        mean_shape_results['left_mano_shape'] = coeff_param['betas']
        
        # Right hand
        crop_info = crop_image_by_bbox(
            img_rgb,
            parse_bbox_from_landmark_lite(hand_kps_r)['bbox'],
            lmk=hand_kps_r,
            scale=1.3,
            dsize=self.cfg.hand_crop_size
        )
        ret_images['right_hand_image'] = crop_info['img_crop']
        cropped_image = np.transpose(crop_info['img_crop'], (2, 0, 1))[None, ...] / 255.0
        coeff_param = self.hamer_encoder(cropped_image.astype(np.float32))
        coeff_param['camera_RT_params'] = self.cvt_cam_to_persp_cam_fov(
            torch.from_numpy(coeff_param['pred_cam'])
        ).numpy()
        base_results['right_mano_coeffs'] = coeff_param
        base_results['right_hand_crop'] = {'M_o2c': crop_info['M_o2c'], 'M_c2o': crop_info['M_c2o']}
        mean_shape_results['right_mano_shape'] = coeff_param['betas']
        
        return ret_images, base_results, mean_shape_results
    
    def crop_face_and_hands(self, img_rgb, base_results):
        """
        Crop face and hand regions from original image using saved transformation matrices.
        
        Args:
            img_rgb: Original RGB image
            base_results: Base tracking results containing crop transformation matrices
            
        Returns:
            Dictionary with 'head_image', 'left_hand_image', 'right_hand_image'
        """
        ret_images = {}
        
        # Crop head/face region
        if 'head_crop' in base_results and 'M_o2c' in base_results['head_crop']:
            M_o2c = base_results['head_crop']['M_o2c']
            # Apply transformation to get cropped face
            from .utils.crop import _transform_img
            head_img = _transform_img(img_rgb, M_o2c, dsize=self.cfg.head_crop_size)
            ret_images['head_image'] = head_img
        else:
            ret_images['head_image'] = None
        
        # Crop left hand region
        if 'left_hand_crop' in base_results and 'M_o2c' in base_results['left_hand_crop']:
            M_o2c = base_results['left_hand_crop']['M_o2c']
            # Apply transformation to get cropped left hand
            from .utils.crop import _transform_img
            left_hand_img = _transform_img(img_rgb, M_o2c, dsize=self.cfg.hand_crop_size)
            # Flip left hand (as done in track_base)
            left_hand_img = cv2.flip(left_hand_img, 1)
            ret_images['left_hand_image'] = left_hand_img
        else:
            ret_images['left_hand_image'] = None
        
        # Crop right hand region
        if 'right_hand_crop' in base_results and 'M_o2c' in base_results['right_hand_crop']:
            M_o2c = base_results['right_hand_crop']['M_o2c']
            # Apply transformation to get cropped right hand
            from .utils.crop import _transform_img
            right_hand_img = _transform_img(img_rgb, M_o2c, dsize=self.cfg.hand_crop_size)
            ret_images['right_hand_image'] = right_hand_img
        else:
            ret_images['right_hand_image'] = None
        
        return ret_images
    
    def prepare_ref_vertices(self, base_results, img_size=512):
        """
        Compute dense FLAME and MANO vertices from saved coefficients.
        Matches prepare_ref_vertices in ehm_refiner.py.
        
        Args:
            base_results: Base tracking results containing coefficients and crop info
            img_size: Image size for coordinate transformation (default 512)
            
        Returns:
            Tuple of (ref_head_vertices, ref_hand_l_vertices, ref_hand_r_vertices)
            All vertices are in body_hd (body crop high-res) coordinates, with shape (1, N, 3)
        """
        from .modules.flame import FLAME
        from .modules.mano import MANO
        from .modules.renderer.head_renderer import Renderer as HeadRenderer
        from .modules.renderer.hand_renderer import Renderer as HandRenderer
        
        # Initialize models if needed
        if not hasattr(self, 'flame'):
            flame_assets_dir = os.path.join(os.path.dirname(__file__), '../assets/FLAME')
            self.flame = FLAME(flame_assets_dir).to(self.device)
            self.head_renderer = HeadRenderer(flame_assets_dir, img_size, focal_length=1.0/12).to(self.device)
        
        if not hasattr(self, 'mano'):
            mano_assets_dir = os.path.join(os.path.dirname(__file__), '../assets/MANO')
            self.mano = MANO(mano_assets_dir).to(self.device)
            self.hand_renderer = HandRenderer(mano_assets_dir, img_size, focal_length=1.0/12).to(self.device)
        
        # Process head/face
        flame_coeffs = {}
        for k, v in base_results['flame_coeffs'].items():
            if isinstance(v, np.ndarray):
                v_tensor = torch.from_numpy(v).float().to(self.device)
                # Ensure first dimension is batch size (add if missing)
                if v_tensor.ndim == 1:
                    v_tensor = v_tensor.unsqueeze(0)
                elif v_tensor.ndim == 2 and v_tensor.shape[0] != 1:
                    # If it's (N, 3) but first dim is not 1, add batch dim
                    v_tensor = v_tensor.unsqueeze(0)
                flame_coeffs[k] = v_tensor
            else:
                flame_coeffs[k] = v
        
        head_ret_dict = self.flame(flame_coeffs)
        render_rlt = self.head_renderer(
            head_ret_dict['vertices'], 
            transform_matrix=flame_coeffs['camera_RT_params'],
            landmarks={'joints': head_ret_dict['joints']}, 
            ret_image=False
        )
        
        # Transform head vertices to body_hd coordinates
        M_transform = torch.from_numpy(
            base_results['body_crop']['M_o2c-hd'] @ base_results['head_crop']['M_c2o']
        ).float().to(self.device).unsqueeze(0)
        
        ref_head_vertices = self._transform_head_pts3d_to_image_coord(
            render_rlt[0][:, np.unique(self.flame.head_index)], M_transform
        )
        ref_head_joints = self._transform_head_pts3d_to_image_coord(
            render_rlt[1]['joints'], M_transform
        )
        ref_head_vertices[..., 2] = (ref_head_vertices[..., 2] - ref_head_joints[:, 3:5, 2].mean(dim=1, keepdim=True))
        
        # Process left hand
        from .utils import rotation_converter as converter
        
        left_mano_coeffs = {}
        for k, v in base_results['left_mano_coeffs'].items():
            if isinstance(v, np.ndarray):
                v_tensor = torch.from_numpy(v).float().to(self.device)
                # Convert rotation matrices to axis-angle
                if k == 'global_orient':
                    # (1, 1, 3, 3) -> flatten to (1, 3, 3) -> convert to axis-angle (1, 3)
                    b, n = v_tensor.shape[:2]
                    v_tensor = converter.batch_matrix2axis(v_tensor.flatten(0, 1)).reshape(b, n, 3)
                elif k == 'hand_pose':
                    # (1, 15, 3, 3) -> flatten to (15, 3, 3) -> convert to axis-angle (1, 45)
                    b, n = v_tensor.shape[:2]
                    v_tensor = converter.batch_matrix2axis(v_tensor.flatten(0, 1)).reshape(b, n, 3)
                else:
                    # For other params like betas, camera_RT_params
                    if v_tensor.ndim == 1:
                        v_tensor = v_tensor.unsqueeze(0)
                    elif v_tensor.ndim == 2 and v_tensor.shape[0] != 1:
                        v_tensor = v_tensor.unsqueeze(0)
                left_mano_coeffs[k] = v_tensor
            else:
                left_mano_coeffs[k] = v
        
        hand_l_ret_dict = self.mano(left_mano_coeffs, pose_type='aa')
        render_rlt = self.hand_renderer(
            hand_l_ret_dict['vertices'],
            landmarks={'joints': hand_l_ret_dict['joints']},
            is_left=True,
            transform_matrix=left_mano_coeffs['camera_RT_params'],
            ret_image=False
        )
        
        # Transform left hand vertices to body_hd coordinates
        M_transform = torch.from_numpy(
            base_results['body_crop']['M_o2c-hd'] @ base_results['left_hand_crop']['M_c2o']
        ).float().to(self.device).unsqueeze(0)
        
        ref_hand_l_vertices = self._transform_hand_pts3d_to_image_coord(
            render_rlt[0], M_transform, img_size, True
        )
        ref_hand_l_joints = self._transform_hand_pts3d_to_image_coord(
            render_rlt[1]['joints'], M_transform, img_size, True
        )
        ref_hand_l_vertices[..., 2] = ref_hand_l_vertices[..., 2] - ref_hand_l_joints[:, 0:1, 2]
        
        # Process right hand
        right_mano_coeffs = {}
        for k, v in base_results['right_mano_coeffs'].items():
            if isinstance(v, np.ndarray):
                v_tensor = torch.from_numpy(v).float().to(self.device)
                # Convert rotation matrices to axis-angle
                if k == 'global_orient':
                    # (1, 1, 3, 3) -> flatten to (1, 3, 3) -> convert to axis-angle (1, 3)
                    b, n = v_tensor.shape[:2]
                    v_tensor = converter.batch_matrix2axis(v_tensor.flatten(0, 1)).reshape(b, n, 3)
                elif k == 'hand_pose':
                    # (1, 15, 3, 3) -> flatten to (15, 3, 3) -> convert to axis-angle (1, 45)
                    b, n = v_tensor.shape[:2]
                    v_tensor = converter.batch_matrix2axis(v_tensor.flatten(0, 1)).reshape(b, n, 3)
                else:
                    # For other params like betas, camera_RT_params
                    if v_tensor.ndim == 1:
                        v_tensor = v_tensor.unsqueeze(0)
                    elif v_tensor.ndim == 2 and v_tensor.shape[0] != 1:
                        v_tensor = v_tensor.unsqueeze(0)
                right_mano_coeffs[k] = v_tensor
            else:
                right_mano_coeffs[k] = v
        
        hand_r_ret_dict = self.mano(right_mano_coeffs, pose_type='aa')
        render_rlt = self.hand_renderer(
            hand_r_ret_dict['vertices'],
            landmarks={'joints': hand_r_ret_dict['joints']},
            is_left=False,
            transform_matrix=right_mano_coeffs['camera_RT_params'],
            ret_image=False
        )
        
        # Transform right hand vertices to body_hd coordinates
        M_transform = torch.from_numpy(
            base_results['body_crop']['M_o2c-hd'] @ base_results['right_hand_crop']['M_c2o']
        ).float().to(self.device).unsqueeze(0)
        
        ref_hand_r_vertices = self._transform_hand_pts3d_to_image_coord(
            render_rlt[0], M_transform, img_size
        )
        ref_hand_r_joints = self._transform_hand_pts3d_to_image_coord(
            render_rlt[1]['joints'], M_transform, img_size
        )
        ref_hand_r_vertices[..., 2] = ref_hand_r_vertices[..., 2] - ref_hand_r_joints[:, 0:1, 2]
        
        return ref_head_vertices, ref_hand_l_vertices, ref_hand_r_vertices
    
    def _transform_points3d(self, points3d, M):
        """Transform 3D points using transformation matrix."""
        R3d = torch.zeros_like(M)
        R3d[:, :2, :2] = M[:, :2, :2]
        scale = (M[:, 0, 0]**2 + M[:, 0, 1]**2)**0.5
        R3d[:, 2, 2] = scale
        
        trans = torch.zeros_like(M)[:, 0]
        trans[:, :2] = M[:, :2, 2]
        trans = trans.unsqueeze(1)
        return torch.bmm(points3d, R3d.mT) + trans
    
    def _fix_mirror_issue(self, pts3d, image_size):
        """Fix mirror issue for hand vertices."""
        p = pts3d.clone()
        p[..., 1] = image_size - p[..., 1]
        p[..., 2] = -p[..., 2]
        return p
    
    def _transform_hand_pts3d_to_image_coord(self, X, M, img_size=512, is_left=False):
        """Transform hand 3D points to image coordinates."""
        _X = self._fix_mirror_issue(X, img_size)
        if is_left:
            _X[..., 0] = img_size - 1 - _X[..., 0]
        _X = self._transform_points3d(_X, M.to(_X.device))
        return _X
    
    def _transform_head_pts3d_to_image_coord(self, X, M):
        """Transform head 3D points to image coordinates."""
        _X = self._transform_points3d(X, M.to(X.device))
        return _X
    
    def reconstruct_cropped_images(self, img_rgb, base_results):
        """
        Reconstruct cropped body, face and hand images from original image using base_results.
        
        Args:
            img_rgb: Original RGB image
            base_results: Base tracking results for this frame
            
        Returns:
            Dictionary with cropped images
        """
        from .utils.crop import _transform_img
        
        ret_images = {
            'ori_image': img_rgb,
        }
        
        # Crop body region using body_crop transformation
        if 'body_crop' in base_results and 'M_o2c-hd' in base_results['body_crop']:
            M_o2c_hd = base_results['body_crop']['M_o2c-hd']
            body_img = _transform_img(img_rgb, M_o2c_hd, dsize=self.cfg.body_hd_size)
            ret_images['body_image'] = body_img
        else:
            ret_images['body_image'] = img_rgb
        
        # Crop face and hands using the saved transformation matrices
        cropped = self.crop_face_and_hands(img_rgb, base_results)
        ret_images.update(cropped)
        
        return ret_images
    
    def create_visualization_grid(self, ret_images_dict, frames_keys, base_results=None, grid_cols=None):
        """
        Create visualization grid with body, face, and hands.
        
        Layout per frame:
        - Top: 128x128 body image with dwpose keypoints
        - Middle: 128x128 face image with FLAME dense vertices
        - Bottom: 64x64 left hand + 64x64 right hand with MANO dense vertices
        
        Total size per frame column: 128 wide x (128 + 128 + 64) = 128 x 320
        
        Grid layout ratio: 2:1 (width:height) across frames
        
        Args:
            ret_images_dict: Dictionary mapping frame_key to ret_images dict
            frames_keys: List of frame keys in order
            base_results: Optional base tracking results for vertex drawing
            grid_cols: Number of columns (auto-calculated if None based on 2:1 ratio)
        
        Returns:
            numpy array of the complete grid image
        """
        from .utils.draw import draw_landmarks
        
        num_frames = len(frames_keys)
        if num_frames == 0:
            return np.zeros((320, 128, 3), dtype=np.uint8)
        
        # Calculate grid dimensions for 2:1 width:height ratio
        if grid_cols is None:
            grid_cols = int(np.ceil(np.sqrt(num_frames * 5)))
        
        grid_rows = int(np.ceil(num_frames / grid_cols))
        
        # Cell dimensions
        cell_width = 128
        body_height = 128
        face_height = 128
        hand_height = 64
        cell_height = body_height + face_height + hand_height  # 320
        
        # Create blank grid
        grid_height = grid_rows * cell_height
        grid_width = grid_cols * cell_width
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Font settings for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_thickness = 1
        text_color = (255, 255, 255)
        
        for idx, frame_key in enumerate(frames_keys):
            row = idx // grid_cols
            col = idx % grid_cols
            
            # Calculate position in grid
            x_start = col * cell_width
            y_start = row * cell_height
            
            ret_images = ret_images_dict.get(frame_key)
            
            if ret_images is None:
                # Draw frame key on black cell
                text_pos = (x_start + 5, y_start + 15)
                cv2.putText(grid, frame_key, text_pos, font, font_scale, text_color, font_thickness)
                continue
            
            # Get frame base results
            frame_base_results = base_results.get(frame_key) if base_results else None
            
            # Body image (128x128) with dwpose keypoints
            body_img = ret_images.get('body_image')
            if body_img is not None:
                body_resized = cv2.resize(body_img, (cell_width, body_height))
                
                # Draw dwpose keypoints on body image
                if frame_base_results and 'dwpose_rlt' in frame_base_results:
                    dwpose_rlt = frame_base_results['dwpose_rlt']
                    if 'keypoints' in dwpose_rlt:
                        # Scale keypoints to body_resized size
                        scale_x = cell_width / body_img.shape[1]
                        scale_y = body_height / body_img.shape[0]
                        body_kps = dwpose_rlt['keypoints'].copy()
                        body_kps[:, 0] *= scale_x
                        body_kps[:, 1] *= scale_y
                        body_resized = draw_landmarks(body_kps, body_resized, color=(0, 255, 0), radius=1)
                
                grid[y_start:y_start+body_height, x_start:x_start+cell_width] = body_resized
            
            # Compute dense vertices once for this frame
            ref_head_vertices = None
            ref_hand_l_vertices = None
            ref_hand_r_vertices = None
            
            if frame_base_results:
                try:
                    # Check if all required data exists
                    has_flame = 'flame_coeffs' in frame_base_results and 'head_crop' in frame_base_results
                    has_left_hand = 'left_mano_coeffs' in frame_base_results and 'left_hand_crop' in frame_base_results
                    has_right_hand = 'right_mano_coeffs' in frame_base_results and 'right_hand_crop' in frame_base_results
                    
                    if has_flame or has_left_hand or has_right_hand:
                        ref_head_vertices, ref_hand_l_vertices, ref_hand_r_vertices = self.prepare_ref_vertices(frame_base_results)
                except Exception as e:
                    print(f"Warning: Failed to compute vertices for {frame_key}: {e}")
                    pass
            
            # Face image (128x128) with FLAME dense vertices
            face_img = ret_images.get('head_image')
            if face_img is not None:
                face_resized = cv2.resize(face_img, (cell_width, face_height))
                
                # Draw FLAME vertices if available
                if ref_head_vertices is not None and frame_base_results:
                    try:
                        # ref_head_vertices are already in body_hd coordinates
                        # Just need to scale them to the resized face image dimensions
                        head_verts_body = ref_head_vertices[0, :, :2].cpu().numpy()  # (N, 2) in body_hd coords
                        
                        # Scale from body_hd size to resized face dimensions
                        body_hd_size = self.cfg.body_hd_size
                        scale_x = cell_width / body_hd_size
                        scale_y = face_height / body_hd_size
                        head_verts_display = head_verts_body.copy()
                        head_verts_display[:, 0] *= scale_x
                        head_verts_display[:, 1] *= scale_y
                        
                        face_resized = draw_landmarks(head_verts_display, face_resized, color=(255, 0, 255), radius=1)
                    except Exception as e:
                        pass
                
                y_face = y_start + body_height
                grid[y_face:y_face+face_height, x_start:x_start+cell_width] = face_resized
            
            # Hands (64x64 each) with MANO dense vertices
            left_hand_img = ret_images.get('left_hand_image')
            right_hand_img = ret_images.get('right_hand_image')
            
            y_hands = y_start + body_height + face_height
            hand_width = 64
            
            if left_hand_img is not None:
                left_resized = cv2.resize(left_hand_img, (hand_width, hand_height))
                
                # Draw left MANO vertices if available
                if ref_hand_l_vertices is not None and frame_base_results:
                    try:
                        # ref_hand_l_vertices are already in body_hd coordinates
                        # Just need to scale them to the resized hand image dimensions
                        hand_verts_body = ref_hand_l_vertices[0, :, :2].cpu().numpy()  # (N, 2) in body_hd coords
                        
                        # Scale from body_hd size to resized hand dimensions
                        body_hd_size = self.cfg.body_hd_size
                        scale_x = hand_width / body_hd_size
                        scale_y = hand_height / body_hd_size
                        hand_verts_display = hand_verts_body.copy()
                        hand_verts_display[:, 0] *= scale_x
                        hand_verts_display[:, 1] *= scale_y
                        
                        left_resized = draw_landmarks(hand_verts_display, left_resized, color=(255, 0, 255), radius=1)
                    except Exception as e:
                        pass
                
                grid[y_hands:y_hands+hand_height, x_start:x_start+hand_width] = left_resized
            
            if right_hand_img is not None:
                right_resized = cv2.resize(right_hand_img, (hand_width, hand_height))
                x_right = x_start + hand_width
                
                # Draw right MANO vertices if available
                if ref_hand_r_vertices is not None and frame_base_results:
                    try:
                        # ref_hand_r_vertices are already in body_hd coordinates
                        # Just need to scale them to the resized hand image dimensions
                        hand_verts_body = ref_hand_r_vertices[0, :, :2].cpu().numpy()  # (N, 2) in body_hd coords
                        
                        # Scale from body_hd size to resized hand dimensions
                        body_hd_size = self.cfg.body_hd_size
                        scale_x = hand_width / body_hd_size
                        scale_y = hand_height / body_hd_size
                        hand_verts_display = hand_verts_body.copy()
                        hand_verts_display[:, 0] *= scale_x
                        hand_verts_display[:, 1] *= scale_y
                        
                        right_resized = draw_landmarks(hand_verts_display, right_resized, color=(255, 0, 255), radius=1)
                    except Exception as e:
                        pass
                
                grid[y_hands:y_hands+hand_height, x_right:x_right+hand_width] = right_resized
            
            # Draw frame key text on the body image area
            text_pos = (x_start + 5, y_start + 15)
            (text_width, text_height), _ = cv2.getTextSize(frame_key, font, font_scale, font_thickness)
            cv2.rectangle(grid, (x_start + 2, y_start + 5), 
                         (x_start + text_width + 8, y_start + text_height + 10),
                         (0, 0, 0), -1)
            cv2.putText(grid, frame_key, text_pos, font, font_scale, text_color, font_thickness)
        
        return grid
