import os
import cv2
import torch
import imageio
import numpy as np
import os.path as osp
import math,shutil,json,torchvision
from pytorch3d.renderer import PointLights
from tqdm.auto import tqdm
from PIL import Image as PILImage
import shutil
from .utils.graphics import GS_Camera
from .utils.lmdb import LMDBEngine
from .utils.rprint import rlog as log
from .utils.video import images2video
from .utils.bbox_utils import crop_image_from_bbox
from .modules.dwpose import inference_detector
from .utils.landmark_runner import LandmarkRunner
from .modules.smplx.utils import orginaze_body_pose
from .configs.argument_config import ArgumentConfig
from .modules.renderer.util import weak_cam2persp_cam,cam2persp_cam_fov,cam2persp_cam_fov_body
from .modules.refiner.flame_refiner import FlameOptimizer
from .modules.refiner.ehm_refiner import EhmOptimizer
from .configs.data_prepare_config import DataPreparationConfig
from .utils.io import load_config, write_dict_pkl, load_dict_pkl
from .utils.crop import crop_image, parse_bbox_from_landmark_lite, crop_image_by_bbox, _transform_pts
from .utils.helper import load_onnx_model, instantiate_from_config, image2tensor, image2tensor, get_machine_info
from .modules.refiner.smplx_utils import smplx_joints_to_dwpose


def split_frame_key(frm_key):
    *segment_id, fn = frm_key.split('_')
    return '_'.join(segment_id), fn


class DataPreparePipeline(object):
    def __init__(self, data_prepare_cfg: DataPreparationConfig):
        self.cfg = data_prepare_cfg
        self.device = self.cfg.device

        self.dwpose_detector = instantiate_from_config(load_config(self.cfg.dwpose_cfg_path))
        self.dwpose_detector.warmup()
        self.pixie_encoder = instantiate_from_config(load_config(self.cfg.pixie_cfg_path))
        self.pixie_encoder.to(self.device)
        self.matte = instantiate_from_config(load_config(self.cfg.matting_cfg_path))
        self.matte.to(self.device)

        self.lmk70_detector = load_onnx_model(load_config(self.cfg.kp70_cfg_path))
        self.mp_detector   = instantiate_from_config(load_config(self.cfg.mp_cfg_path))
        self.mp_detector.warmup()
        self.teaser_encoder = load_onnx_model(load_config(self.cfg.teaser_cfg_path))
        self.hamer_encoder = load_onnx_model(load_config(self.cfg.hamer_cfg_path))
        self.landmark_runner = LandmarkRunner(ckpt_path=self.cfg.kp203_path, onnx_provider=self.device)
        self.landmark_runner.warmup()

        self.flame_opt = FlameOptimizer(self.cfg.flame_assets_dir, device=self.device, image_size=self.cfg.head_crop_size,tanfov=self.cfg.tanfov)
        self.ehm_opt   = EhmOptimizer(self.cfg.flame_assets_dir, self.cfg.smplx_assets_dir, self.cfg.mano_assets_dir, 
                                        device=self.device, body_image_size=self.cfg.body_hd_size, head_image_size=self.cfg.head_crop_size,tanfov=self.cfg.tanfov,
                                        vposer_ckpt=self.cfg.vposer_ckpt_dir)
        
        self.flame = self.flame_opt.flame
        self.ehm_opt.ehm.flame = self.flame                      
        self.ehm_opt.head_renderer = self.flame_opt.renderer 
        self.ehm = self.ehm_opt.ehm
        self.head_renderer = self.flame_opt.renderer
        self.body_renderer = self.ehm_opt.body_renderer

    def get_video_name(self, video_fp, using_last_k=1):
        aa_names  = []
        sub_names = video_fp.split(os.sep)
        for ii, xx in enumerate(sub_names):
            if ii >= len(sub_names) - using_last_k:
                aa_names.append(xx)
        if aa_names[0].startswith('__'): aa_names[0] = aa_names[0][2:]
        if aa_names[-1].endswith('.mp4'):
            aa_names[-1] = aa_names[-1].split('.')[0]
        return '__'.join(aa_names)

    def get_union_human_box_in_video(self, video_fp,frame_interval=1):
        reader = imageio.get_reader(video_fp)
        num_frames = reader.count_frames()
        union_bbox = []
        for idx in tqdm(range(0,num_frames,frame_interval),desc='Getting unioned bbox ...'):
            img_rgb=reader.get_data(idx)
            try:
                bbox = inference_detector(self.dwpose_detector.pose_estimation.session_det, img_rgb)[0]
                union_bbox.append(bbox)
            except Exception as e:
                pass
        
        if len(union_bbox) == 0:
            return None
        uu = np.array(union_bbox)
        ul, ut, ur, ub = uu[:, 0].min(), uu[:, 1].min(), uu[:, 2].max(), uu[:, 3].max()
        ucx, ucy = (ul + ur) / 2, (ut + ub) / 2
        usize = max(ur - ul, ub - ut)
        union_box = [ucx - usize / 2, ucy - usize / 2, ucx + usize / 2, ucy + usize / 2]
        return union_box

    def cvt_weak_cam_to_persp_cam(self, wcam):
        R, T = weak_cam2persp_cam(wcam, focal_length=self.cfg.focal_length, z_dist=self.cfg.z_dist)
        return torch.cat((R, T[..., None]), axis=-1)
    def cvt_cam_to_persp_cam_fov(self, wcam):
        R, T = cam2persp_cam_fov(wcam, tanfov=self.cfg.tanfov)
        return torch.cat((R, T[..., None]), axis=-1)
    def cvt_cam_to_persp_cam_fov_body(self, wcam):
        R, T = cam2persp_cam_fov_body(wcam, tanfov=self.cfg.tanfov)
        return torch.cat((R, T[..., None]), axis=-1)
    
    def extract_frames(self, video_fp, out_lmdb_dir, frame_interval, num_frames):
        """
        Extract frames from video and save to LMDB.
        
        Args:
            video_fp: Path to video file
            out_lmdb_dir: Directory for LMDB database
            frame_interval: Interval between frames to extract
            num_frames: Total number of frames in video
            
        Returns:
            Tuple of (all_frames, frames_keys)
        """
        log(f"Extracting frames from video: {video_fp}")
        reader = imageio.get_reader(video_fp)
        lmdb_engine = LMDBEngine(out_lmdb_dir, write=True)
        
        all_frames = []
        frames_keys = []
        
        img_idx = 0
        for idx in tqdm(range(0, num_frames, frame_interval), 
                       desc='Extracting frames from video...', 
                       total=num_frames//frame_interval):
            img_rgb = reader.get_data(idx)
            b_name = f'frame_{img_idx:06d}'
            
            # Save original image to LMDB
            lmdb_engine.dump(f'{b_name}/ori_image', 
                           payload=image2tensor(img_rgb, norm=False), 
                           type='image')
            
            all_frames.append(img_rgb)
            frames_keys.append(b_name)
            img_idx += 1
        
        reader.close()
        lmdb_engine.close()
        
        log(f"Extracted {len(all_frames)} frames")
        return all_frames, frames_keys

    def _load_image_from_disk(self, video_name, frame_key, extra_info, to_tensor=False):
        frames_root = extra_info['frames_root']
        frames_ext = extra_info.get('frames_ext', '.png')
        seg_id, fn = split_frame_key(frame_key)
        img = np.array(PILImage.open(osp.join(frames_root, video_name, seg_id, f'{fn}{frames_ext}')))

        matte_root = extra_info.get('matte_root', None)
        if matte_root is not None:
            matte = np.array(PILImage.open(osp.join(matte_root, video_name, seg_id, f'{fn}.png')))
            matte = matte[..., None].repeat(3, -1)
        else:
            matte = None

        wbbox_info = extra_info.get('wbbox_info')
        if wbbox_info is not None:
            if video_name in wbbox_info:
                wbbox_info = wbbox_info[video_name]
            wbbox_xyxy = wbbox_info.get(frame_key)
            if wbbox_xyxy is not None:
                wbbox_xyxy = np.array(wbbox_xyxy, dtype=np.float32).squeeze(0)
                img, _ = crop_image_from_bbox(img, wbbox_xyxy, return_pad_mask=True)
                if matte is not None:
                    matte, pad_mask = crop_image_from_bbox(matte, wbbox_xyxy, return_pad_mask=True)

        if to_tensor:
            _image = torchvision.transforms.ToTensor()(img)
            if matte is not None:
                _mask = torchvision.transforms.ToTensor()(matte)
                return _image, _mask
            else:
                return _image, None
        else:
            return img, matte

    def _load_image_from_lmdb(self, frame_key: str, lmdb_engine: LMDBEngine):
        if lmdb_engine.exists(f'{frame_key}/body_image'):
            img_tensor = lmdb_engine[f'{frame_key}/body_image']
        elif lmdb_engine.exists(f'{frame_key}/ori_image'):
            img_tensor = lmdb_engine[f'{frame_key}/ori_image']
        else:
            return None
        
        img_rgb = img_tensor.numpy().transpose(1, 2, 0).astype(np.uint8)
        return img_rgb

    def load_frames(self, video_name, frames_keys,
                    extra_info=None, lmdb_dir=None):
        if lmdb_dir is not None and osp.exists(lmdb_dir):
            lmdb_engine = LMDBEngine(lmdb_dir, write=False)
        else:
            lmdb_engine = None

        all_frames = []
        for frame_key in tqdm(frames_keys, desc='Loading frames from disk...'):
            try:
                img, matte = None, None
                if lmdb_engine is not None:
                    img = self._load_image_from_lmdb(frame_key, lmdb_engine)

                if img is None:
                    img, matte = self._load_image_from_disk(video_name, frame_key, extra_info,
                                                            to_tensor=False)
                all_frames.append((img, matte))
            except Exception as e:
                log(f"Warning: Could not load frame {frame_key}: {e}")
        return all_frames
    
    def load_frames_from_disk(self, video_name, frames_keys, extra_info):
        """
        Load frames from disk using extra info.
        
        Args:
            frames_keys: List of frame keys to load
            extra_info: Extra info dict containing paths
        Returns:
            List of loaded frames (numpy arrays)
        """
        log(f"Loading frames from disk using extra info")
        all_frames = []
        for frame_key in tqdm(frames_keys, desc='Loading frames from disk...'):
            try:
                img, matte = self._load_image_from_disk(video_name, frame_key, extra_info,
                                                        to_tensor=False)
                all_frames.append((img, matte))
            except Exception as e:
                log(f"Warning: Could not load frame {frame_key}: {e}")
        return all_frames

    def load_frames_lmdb(self, lmdb_dir, frames_keys):
        """
        Load frames from existing LMDB database.
        
        Args:
            lmdb_dir: Directory with LMDB database
            frames_keys: List of frame keys to load
            
        Returns:
            List of loaded frames (numpy arrays)
        """
        log(f"Loading frames from LMDB: {lmdb_dir}")
        lmdb_engine = LMDBEngine(lmdb_dir, write=False)
        
        all_frames = []
        for frame_key in tqdm(frames_keys, desc='Loading frames from LMDB...'):
            try:
                # Try to load from body_image first (processed), fallback to ori_image
                try:
                    img_tensor = lmdb_engine[f'{frame_key}/body_image']
                except:
                    img_tensor = lmdb_engine[f'{frame_key}/ori_image']
                
                img_rgb = img_tensor.numpy().transpose(1, 2, 0).astype(np.uint8)
                all_frames.append(img_rgb)
            except Exception as e:
                log(f"Warning: Could not load frame {frame_key}: {e}")
        
        lmdb_engine.close()
        log(f"Loaded {len(all_frames)} frames")
        return all_frames
    
    def base_track(self, all_frames, frames_keys, out_lmdb_dir, base_track_fp, 
                   id_share_params_fp, skipped_flag, 
                   check_hand=True, 
                   no_body_crop=False,
                   save_images_to_lmdb=True):
        """
        Perform base tracking on frames and save results.
        
        Args:
            all_frames: List of frames (numpy arrays)
            frames_keys: List of frame keys corresponding to frames
            out_lmdb_dir: Directory for LMDB database
            base_track_fp: Path to save base tracking results
            id_share_params_fp: Path to save identity-shared parameters
            skipped_flag: Path to skipped flag file
            check_hand: Whether to check hand validity
            save_images_to_lmdb: Whether to save processed images to LMDB
            
        Returns:
            Tuple of (base_results, id_share_params_results, success)
        """
        log(f"Performing base tracking on {len(all_frames)} frames...")
        
        # Cropping
        if no_body_crop:
            crop_scale = 1.0
            img_rgb = all_frames[0]
            if isinstance(img_rgb, tuple):
                img_rgb = img_rgb[0]
            union_box = [0, 0, img_rgb.shape[1], img_rgb.shape[0]]
        else:
            crop_scale = 1.1
            # Compute union bounding box
            union_bbox = []
            for img_rgb in tqdm(all_frames, desc='Computing union bbox...'):
                try:
                    if isinstance(img_rgb, tuple):
                        img_rgb = img_rgb[0]
                    bbox = inference_detector(self.dwpose_detector.pose_estimation.session_det, img_rgb)[0]
                    union_bbox.append(bbox)
                except Exception as e:
                    pass
            
            if len(union_bbox) == 0:
                union_box = None
            else:
                uu = np.array(union_bbox)
                ul, ut, ur, ub = uu[:, 0].min(), uu[:, 1].min(), uu[:, 2].max(), uu[:, 3].max()
                ucx, ucy = (ul + ur) / 2, (ut + ub) / 2
                usize = max(ur - ul, ub - ut)
                union_box = [ucx - usize / 2, ucy - usize / 2, ucx + usize / 2, ucy + usize / 2]
        
        # Open LMDB for writing processed images
        lmdb_engine = LMDBEngine(out_lmdb_dir, write=save_images_to_lmdb) if save_images_to_lmdb else None
        
        base_results = {}
        id_share_params_results = {}
        
        with torch.no_grad():
            last_results = None
            valid_idx = 0
            
            for idx, (img_rgb, b_name) in enumerate(tqdm(zip(all_frames, frames_keys), 
                                                         desc='Processing base tracking...', 
                                                         total=len(all_frames))):
                if isinstance(img_rgb, tuple):
                    img_rgb, matte_rgb = img_rgb
                else:
                    matte_rgb = None

                ret_images, ret_results, shape_results = self.track_base(
                    img_rgb, union_box, 
                    last_results=last_results, 
                    crop_scale=crop_scale,
                    matte_rgb=matte_rgb)
                last_results = ret_results
                
                if ret_results is None:
                    log(f"Skipping frame {idx} ({b_name}) due to incomplete facial landmark extraction")
                    continue
                
                if check_hand:
                    if not ret_results['left_hand_valid'] or not ret_results['right_hand_valid']:
                        log(f"Skipping frame {idx} ({b_name}) due to missing left or right hand")
                        continue
                
                # Save processed images to LMDB if requested
                if save_images_to_lmdb and lmdb_engine is not None:
                    for k, v in ret_images.items():
                        if k == 'ori_image':  # Skip ori_image as it's already saved
                            continue
                        if len(v.shape) == 2:
                            v = v[:, :, None]
                        lmdb_engine.dump(f'{b_name}/{k}', payload=image2tensor(v, norm=False), type='image')
                
                # Remove per-frame shape parameters
                del ret_results['flame_coeffs']['shape_params']
                del ret_results['smplx_coeffs']['shape']
                base_results[b_name] = ret_results
                
                # Collect shape parameters for averaging
                for k, v in shape_results.items():
                    if k not in id_share_params_results:
                        id_share_params_results[k] = []
                    id_share_params_results[k].append(v)
                
                valid_idx += 1
        
        if lmdb_engine:
            lmdb_engine.close()
        
        # Check minimum frames requirement
        if valid_idx < self.cfg.min_frames:
            log(f"Insufficient valid frames (min required: {self.cfg.min_frames}, found: {valid_idx})")
            with open(skipped_flag, 'w') as f:
                f.write(f"Insufficient valid frames (min required: {self.cfg.min_frames}, found: {valid_idx})")
            return None, None, False
        
        # Average shape parameters
        for k, v in id_share_params_results.items():
            id_share_params_results[k] = np.array(v).mean(0)
        
        # Save results
        write_dict_pkl(id_share_params_fp, id_share_params_results)
        write_dict_pkl(base_track_fp, base_results)
        
        # Visualize if LMDB was used
        if save_images_to_lmdb:
            lmdb_engine = LMDBEngine(out_lmdb_dir, write=False)
            lmdb_engine.random_visualize(osp.join(out_lmdb_dir, 'visualize.jpg'))
            lmdb_engine.close()
        
        log(f'Base tracking completed with {valid_idx} valid frames')
        return base_results, id_share_params_results, True
    
    def track_base(self, img_rgb, union_box=None, last_results=None, 
                   crop_scale=1.1, matte_rgb=None):
        ret_images = {}
        base_results = {}
        mean_shape_results = {}

        ret_images[f'ori_image'] = img_rgb
        det_info, det_raw_info = self.dwpose_detector(img_rgb)
        if union_box is not None: det_info['bbox'] = union_box
        if det_info['bbox'] is None: 
            print("          Missing box")
            return None, None, None
        crop_info_hd = crop_image_by_bbox(img_rgb, det_info['bbox'], dsize=self.cfg.body_hd_size, scale=crop_scale)
        crop_info  = crop_image_by_bbox(img_rgb, det_info['bbox'],   dsize=self.cfg.body_crop_size, scale=crop_scale) 
        base_results['body_crop'] = {'M_o2c': crop_info['M_o2c'], 'M_c2o': crop_info['M_c2o'], 
                                     'M_o2c-hd': crop_info_hd['M_o2c'], 'M_c2o-hd': crop_info_hd['M_c2o']}
        base_results['dwpose_raw'] = det_raw_info
        base_results['dwpose_rlt'] = {'keypoints': _transform_pts(det_raw_info['keypoints'], crop_info_hd['M_o2c']), 
                                      'scores': det_raw_info['scores'], 'faces': det_info['faces'], 'hands': det_info['hands']}

        # check hand confidence and fingers cross
        base_results['left_hand_valid'] = base_results['dwpose_rlt']['scores'][-42:-21].mean() >= self.cfg.check_hand_score
        base_results['right_hand_valid'] = base_results['dwpose_rlt']['scores'][-21:].mean() >= self.cfg.check_hand_score
        if ((base_results['dwpose_rlt']['keypoints'][-42:-21] - 
             base_results['dwpose_rlt']['keypoints'][-21:])**2).mean() < self.cfg.check_hand_dist:
            base_results['left_hand_valid'] = False
            base_results['right_hand_valid'] = False

        img_crop = crop_info['img_crop']
        img_hd   = crop_info_hd['img_crop']
        ret_images['body_image'] = img_hd
        
        img_crop = image2tensor(img_crop).to(self.device).unsqueeze(0)
        img_hd = image2tensor(img_hd).to(self.device).unsqueeze(0)

        # matting related
        if matte_rgb is None:
            t_matting = self.matte(img_hd.contiguous(), 'alpha')
        else:
            matte_hd = crop_image_by_bbox(matte_rgb, det_info['bbox'], dsize=self.cfg.body_hd_size, scale=crop_scale)['img_crop']
            t_matting = image2tensor(matte_hd)[0].to(self.device)
        ret_images['body_mask'] = (np.clip(t_matting.cpu().numpy(), 0, 1) * 255).round().astype(np.uint8)
        predict = t_matting.expand(3, -1, -1)
        matting_image = img_hd.clone()[0]
        background_rgb = 1.0
        background_rgb = matting_image.new_ones(matting_image.shape) * background_rgb
        matting_image = matting_image * predict + (1-predict) * background_rgb
        ret_images['body_matting'] =  np.transpose(np.clip(matting_image.cpu().numpy(), 0, 1) * 255,(1,2,0)).round().astype(np.uint8)
        img_hd = matting_image.unsqueeze(0)
        img_crop = torch.nn.functional.interpolate(img_hd, (224, 224))
        
        # body related
        coeff_param = self.pixie_encoder(img_crop, img_hd)['body']
        coeff_param = orginaze_body_pose(coeff_param)
        coeff_param['camera_RT_params'] = self.cvt_cam_to_persp_cam_fov_body(coeff_param['body_cam'])
        coeff_param = {k: v.cpu().numpy() for k, v in coeff_param.items()}
        base_results['smplx_coeffs'] = coeff_param
        mean_shape_results['smplx_shape'] = coeff_param['shape']

        # head related
        crop_info  = crop_image(img_rgb, det_info['faces'], dsize=self.cfg.head_crop_size, scale=1.75) 
        base_results['head_crop'] = {'M_o2c': crop_info['M_o2c'], 'M_c2o': crop_info['M_c2o']}
        ret_images['head_image'] = crop_info['img_crop']
        
        lmk203 = self.landmark_runner.run(crop_info['img_crop'])['pts']
        t_img = crop_info['img_crop'].transpose((2, 0, 1)).astype(np.float32)
        lmk70 = self.lmk70_detector.run(t_img[None]/255.)['pts'] * 2
        lmk_mp = self.mp_detector.run(crop_info['img_crop'])['pts']
        
        if lmk203 is None or lmk_mp is None or lmk70 is None:
            # if last_results is not None:
            #     base_results.update({'head_lmk_203': last_results['head_lmk_203'], 
            #                          'head_lmk_70':  last_results['head_lmk_70'], 
            #                          'head_lmk_mp':  last_results['head_lmk_mp']})
            # else:
            #     return None, None, None
            base_results.update({'head_lmk_203': np.zeros((203,2)), 
                                 'head_lmk_70':  np.zeros((70,2)), 
                                 'head_lmk_mp':  np.zeros((478,2)),
                                 'head_lmk_valid': False})
        else:
            if len(lmk203.shape) == 3: lmk203 = lmk203[0]
            if len(lmk70.shape) == 3: lmk70 = lmk70[0]
            if len(lmk_mp.shape) == 3: lmk_mp = lmk_mp[0]

            base_results.update({'head_lmk_203': lmk203, 'head_lmk_70': lmk70, 'head_lmk_mp': lmk_mp, 'head_lmk_valid': True})

        cropped_image = cv2.resize(crop_info['img_crop'], (self.cfg.teaser_input_size, self.cfg.teaser_input_size))
        cropped_image = np.transpose(cropped_image, (2,0,1))[None, ...] / 255.0
        coeff_param = self.teaser_encoder(cropped_image.astype(np.float32))
        coeff_param['camera_RT_params'] = self.cvt_cam_to_persp_cam_fov(torch.from_numpy(coeff_param['cam'])).numpy()
        base_results.update({'flame_coeffs': coeff_param})
        mean_shape_results['flame_shape'] = coeff_param['shape_params']

        # hand related
        all_hands_kps = det_info['hands']
        hand_kps_l = all_hands_kps[0]
        hand_kps_r = all_hands_kps[1]
        crop_info  = crop_image_by_bbox(img_rgb, parse_bbox_from_landmark_lite(hand_kps_l)['bbox'], 
                                        lmk=hand_kps_l, scale=1.3,
                                        dsize=self.cfg.hand_crop_size) 
        ret_images['left_hand_image'] = crop_info['img_crop']
        is_left = True
        t_img = cv2.flip(crop_info['img_crop'], 1) if is_left else crop_info['img_crop']
        cropped_image = np.transpose(t_img, (2,0,1))[None, ...] / 255.0
        coeff_param = self.hamer_encoder(cropped_image.astype(np.float32))
        coeff_param['camera_RT_params'] = self.cvt_cam_to_persp_cam_fov(torch.from_numpy(coeff_param['pred_cam'])).numpy()
        base_results.update({'left_mano_coeffs': coeff_param})
        base_results.update({'left_hand_crop': {'M_o2c': crop_info['M_o2c'], 'M_c2o': crop_info['M_c2o']}})
        mean_shape_results['left_mano_shape'] = coeff_param['betas']

        crop_info  = crop_image_by_bbox(img_rgb, parse_bbox_from_landmark_lite(hand_kps_r)['bbox'], 
                                        lmk=hand_kps_r, scale=1.3,
                                        dsize=self.cfg.hand_crop_size) 
        ret_images['right_hand_image'] = crop_info['img_crop']
        cropped_image = np.transpose(crop_info['img_crop'], (2,0,1))[None, ...] / 255.0
        coeff_param = self.hamer_encoder(cropped_image.astype(np.float32))
        coeff_param['camera_RT_params'] = self.cvt_cam_to_persp_cam_fov(torch.from_numpy(coeff_param['pred_cam'])).numpy()
        base_results.update({'right_mano_coeffs': coeff_param})
        base_results.update({'right_hand_crop': {'M_o2c': crop_info['M_o2c'], 'M_c2o': crop_info['M_c2o']}})
        mean_shape_results['right_mano_shape'] = coeff_param['betas']
        return ret_images, base_results, mean_shape_results

    def execute(self, args: ArgumentConfig):
        out_dir = args.output_dir
        log(str(get_machine_info()))
        self.args=args
        for video_idx, video_fp in enumerate(args.source_dir):
            video_name = self.get_video_name(video_fp, 1)
            saving_root = osp.join(out_dir, video_name)
            out_video_fp=video_fp
            out_lmdb_dir = osp.join(saving_root, 'img_lmdb')
            id_share_params_fp = osp.join(saving_root, 'id_share_params.pkl')
            base_track_fp = osp.join(saving_root, 'base_tracking.pkl')
            skipped_flag = osp.join(saving_root, f"skipped.txt")
            optim_track_fp_flame = osp.join(saving_root, 'optim_tracking_flame.pkl')
            optim_track_fp_smplx = osp.join(saving_root, 'optim_tracking_ehm.pkl')
            videos_info_path = osp.join(saving_root, 'videos_info.json')
            os.makedirs(saving_root,exist_ok=True)
            os.makedirs(out_lmdb_dir,exist_ok=True)
            if os.path.exists(skipped_flag) :
                log("Exist skipping flag, Skipping!")
                continue
            if os.path.exists(optim_track_fp_smplx):
                log("Tracking results already esist !")
                if not args.save_vis_video: continue
            
            try:
                reader = imageio.get_reader(out_video_fp)
                num_frames = reader.count_frames()
                frame_interval=1
                if self.cfg.tracking_with_interval:
                    frame_interval=self.cfg.default_frame_interval
                    if num_frames//frame_interval>self.cfg.max_frames:
                        frame_interval=int(math.ceil(num_frames/self.cfg.max_frames))
                    elif num_frames//frame_interval<self.cfg.min_frames:
                        frame_interval=int(num_frames//self.cfg.min_frames)
                    if frame_interval<1:
                        log(f"[{video_idx:04d}/{len(args.source_dir)}] Skipping video file: {out_video_fp} because it has fewer than {self.cfg.min_frames} frames")
                        continue
                
                if not (self.cfg.check_skip_extraction and osp.exists(base_track_fp)):
                    log(f"[{video_idx:04d}/{len(args.source_dir)}] Processing video file: {out_video_fp}")
                    
                    # Extract frames from video
                    all_frames, frames_keys = self.extract_frames(
                        out_video_fp, out_lmdb_dir, frame_interval, num_frames
                    )
                    
                    # Perform base tracking
                    base_results, id_share_params_results, success = self.base_track(
                        all_frames, frames_keys, out_lmdb_dir, base_track_fp, 
                        id_share_params_fp, skipped_flag, 
                        check_hand=not args.not_check_hand, 
                        no_body_crop=False,
                        save_images_to_lmdb=True,
                    )
                    
                    if not success:
                        continue
                    
                    log(f'Prepare OK: ...{video_fp[-20:]} ==> {saving_root}')
                else:
                    base_results = load_dict_pkl(base_track_fp)
                    id_share_params_results= load_dict_pkl(id_share_params_fp)

                if args.save_images:
                    lmdb_engine = LMDBEngine(out_lmdb_dir, write=True)
                    image_save_dir = osp.join(saving_root, 'images')
                    mask_save_dir = osp.join(saving_root, 'masks')
                    if os.path.exists(image_save_dir): shutil.rmtree(image_save_dir)
                    if os.path.exists(mask_save_dir): shutil.rmtree(mask_save_dir)
                    os.makedirs(mask_save_dir, exist_ok=True)
                    os.makedirs(image_save_dir, exist_ok=True)
                    image_keys = sorted(list(base_results.keys()))
                    for key in image_keys:
                        body_image=lmdb_engine[f'{key}/body_image']/255.0
                        body_mask=lmdb_engine[f'{key}/body_mask']/255.0
                        body_image=torch.cat([body_image,body_mask.mean(dim=0,keepdim=True)],dim=0)
                        torchvision.utils.save_image(body_image, osp.join(image_save_dir, f"{key.split('_')[-1]}.png"))
                        torchvision.utils.save_image(body_mask, osp.join(mask_save_dir, f"{key.split('_')[-1]}.png"))
                    lmdb_engine.close()

                optimized_result = base_results
                if os.path.exists(optim_track_fp_flame):
                    optimized_result = load_dict_pkl(optim_track_fp_flame)
                    
                else:
                    lmdb_engine = LMDBEngine(out_lmdb_dir, write=True)
                    if self.cfg.fit_flame:
                        self.flame_opt.saving_root=saving_root
                        log(f"[{video_idx:04d}/{len(args.source_dir)}] Refining head parameters: {video_name}")
                        opt_flame_coeff,id_share_params_results = self.flame_opt.run(optimized_result,id_share_params_results,lmdb_engine,frame_interval)
                        for key in base_results.keys():
                            base_results[key]['flame_coeffs'] = opt_flame_coeff[key]
                        write_dict_pkl(optim_track_fp_flame, base_results)
                        optimized_result=base_results
                    lmdb_engine.close()
                
                if os.path.exists(optim_track_fp_smplx):
                    optimized_result = load_dict_pkl(optim_track_fp_smplx)
                else:
                    lmdb_engine = LMDBEngine(out_lmdb_dir, write=True)
                    if self.cfg.fit_ehm:
                        self.ehm_opt.saving_root=saving_root
                        log(f"[{video_idx:04d}/{len(args.source_dir)}] Refining ehm-smplx parameters: {video_name}")
                        opt_smplx_coeff,id_share_params_results = self.ehm_opt.run(optimized_result,id_share_params_results,lmdb_engine,frame_interval)
                        for key in base_results.keys():
                            optimized_result[key]['smplx_coeffs'] = opt_smplx_coeff[key]
                            del optimized_result[key]['left_mano_coeffs']['betas']
                            del optimized_result[key]['right_mano_coeffs']['betas']
                        write_dict_pkl(optim_track_fp_smplx, optimized_result)
                        write_dict_pkl(id_share_params_fp, id_share_params_results)
                    lmdb_engine.close()
                
                frames_key=list(optimized_result.keys())
                videos_info={video_name:{"frames_num":len(frames_key),"frames_keys":frames_key}}
                with open(videos_info_path, 'w', encoding='utf-8') as json_file:
                    json.dump(videos_info, json_file, ensure_ascii=False, indent=4)
                if args.save_vis_video:
                    track_video_fp = osp.join(saving_root, 'viz_tracking.mp4')
                    if not os.path.exists(track_video_fp):
                        lmdb_engine = LMDBEngine(out_lmdb_dir, write=True)
                        all_images = []
                        device = self.cfg.device
                        cameras_kwargs = self.ehm_opt.build_cameras_kwargs(1,self.ehm_opt.body_focal_length)
                    
                        with torch.no_grad():
                            lights=PointLights(device=self.device, location=[[0.0, -1.0, -100.0]])
                            for idx, image_key in tqdm(enumerate(optimized_result.keys()), desc='Saving visualized results', total=len(optimized_result)):
                                t_flame_coeffs,t_smplx_coeffs,t_left_mano_coeffs,t_right_mano_coeffs=self.convert_traking_params(optimized_result,id_share_params_results,image_key,device)
                                ret_body = self.ehm_opt.ehm(t_smplx_coeffs, t_flame_coeffs)
                                xx=ret_body['vertices']
                                camera_RT_params=torch.tensor(optimized_result[image_key]['smplx_coeffs']['camera_RT_params']).to(device)
                                R, T = camera_RT_params.split([3, 1], dim=-1)
                                T = T.squeeze(-1)
                                R,T=R[None],T[None]
                                cameras = GS_Camera(R=R,T=T,**cameras_kwargs).to(device)
                                proj_joints   = cameras.transform_points_screen(ret_body['joints'], R=R, T=T)
                                pred_kps2d = smplx_joints_to_dwpose(proj_joints)[0][..., :2]
                                gt_lmk_2d=optimized_result[image_key]['dwpose_rlt']['keypoints']
                                rendered_img = self.body_renderer.render_mesh(xx,cameras,lights=lights,smplx2flame_ind=self.ehm_opt.ehm.smplx.smplx2flame_ind)
                                t_img  = (rendered_img[:,:3].cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1,2,0)
                                if args.save_visual_render:
                                    os.makedirs(osp.join(saving_root, 'mesh_rendered'), exist_ok=True)
                                    rendered_img=rendered_img.cpu().numpy().clip(0, 255).astype(np.uint8)[0].transpose(1,2,0)
                                    img_ = PILImage.fromarray(rendered_img)
                                    img_.save(osp.join(saving_root, 'mesh_rendered',f"{str(idx).zfill(5)}.png"))
                                img_inp = lmdb_engine[f'{image_key}/body_image'].numpy().transpose(1,2,0)
                                img_bld = cv2.addWeighted(img_inp, 0.5, t_img, 0.5, 1)
                                img_ret = cv2.hconcat([img_inp, t_img, img_bld])
                                # Draw predicted keypoints in red
                                for kp in pred_kps2d[0].cpu().numpy():
                                    x, y = int(kp[0]), int(kp[1])
                                    cv2.circle(img_bld, (x,y), 3, (0,0,255), -1)
                                    
                                # Draw ground truth keypoints in green 
                                for kp in gt_lmk_2d:
                                    x, y = int(kp[0]), int(kp[1])
                                    cv2.circle(img_bld, (x,y), 3, (0,255,0), -1)
                                all_images.append(img_ret)
                        lmdb_engine.random_visualize(osp.join(out_lmdb_dir, 'visualize.jpg'))
                        lmdb_engine.close()
                        images2video(all_images, track_video_fp, fps=30)
                    log(f'Tacking results is saved {video_name[-20:]} ==> {track_video_fp}')
                    
            except Exception as e:
                log(f"Skipping {out_video_fp} due to error {e}")
                import traceback
                log(f"Traceback:{traceback.format_exc()}")
                skipped_flag = osp.join(saving_root, f"skipped.txt")
                if os.path.isdir(skipped_flag):
                    try:
                        shutil.rmtree(skipped_flag)
                        print(f"Folder {skipped_flag} has been deleted.")
                    except Exception as e:
                        print(f"Failed to delete folder {skipped_flag}: {e}")
                try:
                    with open(skipped_flag, 'w') as f:
                        f.write(f"Skipping {out_video_fp} due to error:\n {e}")
                        f.write(f"\nTraceback:\n{traceback.format_exc()}")
                except:
                    pass
                
                try: 
                    lmdb_engine.close()
                except:
                    pass
                continue
            
    def convert_traking_params(self,optimized_result,id_share_params_results,image_key,device):
        t_flame_coeffs = {k:torch.from_numpy(v)[None].to(device) for k, v in optimized_result[image_key]['flame_coeffs'].items()}
        t_smplx_coeffs = {k:torch.from_numpy(v)[None].to(device) for k, v in optimized_result[image_key]['smplx_coeffs'].items()}
        t_left_mano_coeffs = {k:torch.from_numpy(v).to(device) for k, v in optimized_result[image_key]['left_mano_coeffs'].items()}
        t_right_mano_coeffs = {k:torch.from_numpy(v).to(device) for k, v in optimized_result[image_key]['right_mano_coeffs'].items()}
        t_smplx_coeffs["shape"],t_flame_coeffs["shape_params"]=torch.from_numpy(id_share_params_results["smplx_shape"]).to(device),torch.from_numpy(id_share_params_results["flame_shape"]).to(device)
        t_smplx_coeffs["joints_offset"]=torch.from_numpy(id_share_params_results["joints_offset"]).to(device)
        t_smplx_coeffs["head_scale"]=torch.from_numpy(id_share_params_results["head_scale"]).to(device)
        t_smplx_coeffs["hand_scale"]=torch.from_numpy(id_share_params_results["hand_scale"]).to(device)
        t_left_mano_coeffs["betas"],t_right_mano_coeffs["betas"]=torch.from_numpy(id_share_params_results["left_mano_shape"]).to(device),torch.from_numpy(id_share_params_results["right_mano_shape"]).to(device)
        return t_flame_coeffs,t_smplx_coeffs,t_left_mano_coeffs,t_right_mano_coeffs

    def refine(self, args: ArgumentConfig, optim_cfg):
        """
        Refine processed data by re-optimizing parameters.
        
        Assumes:
        - Frames are already processed and stored in LMDB
        - pkl files (base_tracking.pkl, optim_tracking_*.pkl) are available
        - videos_info.json contains frame metadata
        - May re-extract intermediate files (keypoints) if needed
        
        Args:
            args: ArgumentConfig with output_dir and source_dir
        """
        out_dir = args.output_dir
        log(str(get_machine_info()))
        self.args = args
        
        for video_idx, video_dir in enumerate(args.source_dir):
            video_root = osp.dirname(video_dir)
            video_name = osp.basename(osp.normpath(video_dir))
            saving_root = osp.join(out_dir, video_name)
            
            # Define file paths
            base_id_share_params_fp = osp.join(video_dir, 'base_id_share_params.pkl')
            id_share_params_fp = osp.join(video_dir, 'id_share_params.pkl')
            optim_track_fp_flame = osp.join(video_dir, 'optim_tracking_flame.pkl')
            optim_track_fp_smplx = osp.join(video_dir, 'optim_tracking_ehm.pkl')
            videos_info_path = osp.join(video_dir, 'videos_info.json')
            extra_info_path = osp.join(video_dir, 'extra_info.json')
            base_track_fp = osp.join(saving_root, 'base_tracking.pkl')
            refine_id_share_params_fp = osp.join(saving_root, 'id_share_params.pkl')
            refine_track_fp_flame = osp.join(saving_root, 'optim_tracking_flame.pkl')
            refine_track_fp_smplx = osp.join(saving_root, 'optim_tracking_ehm.pkl')
            out_lmdb_dir = osp.join(saving_root, 'img_lmdb')
            
            # # Check if base data exists
            # if not os.path.exists(base_track_fp):
            #     log(f"[{video_idx:04d}/{len(args.source_dir)}] Skipping {video_name}: base_tracking.pkl not found")
            #     continue
            
            # if not os.path.exists(out_lmdb_dir):
            #     log(f"[{video_idx:04d}/{len(args.source_dir)}] Skipping {video_name}: LMDB database not found")
            #     continue
            
            # Load video metadata
            with open(videos_info_path, 'r', encoding='utf-8') as json_file:
                videos_info = json.load(json_file)
                video_data = videos_info.get(video_name, {})
                frames_keys = video_data['frames_keys']

            # extra info for frame and mask loading
            if os.path.exists(extra_info_path):
                with open(extra_info_path, 'r') as f:
                    extra_info = json.load(f)
            else:
                extra_info = {
                    'frames_root': video_root,
                    'frames_ext': '.jpg',
                }
            
            if len(frames_keys) == 0:
                log(f"Error: No frame keys found in videos_info.json")
                continue

            try:
                # Load existing data
                log(f"[{video_idx:04d}/{len(args.source_dir)}] Refining: {video_name}")
                base_results = load_dict_pkl(base_track_fp) if osp.exists(base_track_fp) else None
                base_id_share_params_results = load_dict_pkl(base_id_share_params_fp) if os.path.exists(base_id_share_params_fp) else {}

                # Redo base track if missing - re-process
                if base_results is None:
                    log(f"Base tracking not found, re-processing...")
                    
                    # Load frames from LMDB
                    all_frames = self.load_frames(video_name, frames_keys, 
                                                  extra_info=extra_info, lmdb_dir=out_lmdb_dir)
                    
                    if len(all_frames) == 0:
                        log(f"Error: No frames could be loaded from LMDB")
                        continue
                    
                    # Perform base tracking on loaded frames
                    skipped_flag = osp.join(video_dir, "skipped.txt")
                    no_body_crop = optim_cfg.get('no_body_crop', False)
                    base_results, base_id_share_params_results, success = self.base_track(
                        all_frames, frames_keys, out_lmdb_dir, base_track_fp,
                        base_id_share_params_fp, skipped_flag,
                        check_hand=not args.not_check_hand,
                        no_body_crop=no_body_crop,
                        save_images_to_lmdb=True  # Save newly processed images
                    )
                    
                    if not success:
                        log(f"Failed to re-process base tracking for {video_name}")
                        continue
                    
                    log(f"Successfully re-processed base tracking for {video_name}")
                    
                # Try to load optimized results, fallback to base results
                id_share_params_results = load_dict_pkl(id_share_params_fp) if os.path.exists(id_share_params_fp) else {}

                if os.path.exists(optim_track_fp_smplx):
                    optimized_result = load_dict_pkl(optim_track_fp_smplx)
                elif os.path.exists(optim_track_fp_flame):
                    optimized_result = load_dict_pkl(optim_track_fp_flame)
                else:
                    optimized_result = base_results

                reset_to_base_track = optim_cfg.get('reset_to_base_track', False)
                if reset_to_base_track:
                    log(f"Resetting to base tracking results for refinement")
                    id_share_params_results = base_id_share_params_results
                    optimized_result = base_results
                else:
                    log(f"Using existing optimized results for refinement")
                    # Update missing keys from base results
                    for key in base_id_share_params_results:
                        if key not in id_share_params_results:
                            id_share_params_results[key] = base_id_share_params_results[key]

                    for frame_key in optimized_result.keys():
                        for key in base_results[frame_key]:
                            if key not in optimized_result[frame_key]:
                                optimized_result[frame_key][key] = base_results[frame_key][key]
                            elif isinstance(base_results[frame_key][key], dict):
                                for sub_key in base_results[frame_key][key]:
                                    if sub_key not in optimized_result[frame_key][key]:
                                        optimized_result[frame_key][key][sub_key] = base_results[frame_key][key][sub_key]

                # Detect frame interval from frame naming
                frame_interval = 1
                if len(frames_keys) > 1:
                    frame_nums = [int(k.split('_')[-1]) for k in frames_keys]
                    if len(set(np.diff(frame_nums))) == 1:
                        frame_interval = np.diff(frame_nums)[0]

                # Re-optimize FLAME parameters with stricter criteria
                if self.cfg.fit_flame and not os.path.exists(refine_track_fp_flame):
                    log(f"[{video_idx:04d}/{len(args.source_dir)}] Re-refining FLAME parameters: {video_name}")
                    lmdb_engine = LMDBEngine(out_lmdb_dir, write=False)
                    self.flame_opt.saving_root = saving_root
                    
                    try:
                        opt_flame_coeff, id_share_params_results = self.flame_opt.run(
                            optimized_result, id_share_params_results, 
                            optim_cfg.optim_flame,
                            lmdb_engine, frame_interval,
                        )
                        
                        # Update results with refined FLAME coefficients
                        refined_result = {k: v.copy() for k, v in optimized_result.items()}
                        for key in refined_result.keys():
                            if key in opt_flame_coeff:
                                refined_result[key]['flame_coeffs'] = opt_flame_coeff[key]
                        
                        write_dict_pkl(refine_track_fp_flame, refined_result)
                        optimized_result = refined_result
                        log(f"FLAME refinement completed for {video_name}")
                    except Exception as e:
                        log(f"Warning: FLAME refinement failed for {video_name}: {e}")
                    finally:
                        lmdb_engine.close()
                else:
                    # load previous refined results if exist
                    if os.path.exists(refine_track_fp_flame):
                        optimized_result = load_dict_pkl(refine_track_fp_flame)
                
                # Re-optimize SMPL-X parameters with stricter criteria
                if self.cfg.fit_ehm and not os.path.exists(refine_track_fp_smplx):
                    log(f"[{video_idx:04d}/{len(args.source_dir)}] Re-refining SMPL-X parameters: {video_name}")
                    lmdb_engine = LMDBEngine(out_lmdb_dir, write=False)
                    self.ehm_opt.saving_root = saving_root
                    
                    try:
                        opt_smplx_coeff, id_share_params_results = self.ehm_opt.run(
                            optimized_result, id_share_params_results, 
                            optim_cfg.optim_ehm,
                            lmdb_engine, frame_interval,
                        )
                        
                        # Update results with refined SMPL-X coefficients
                        refined_result = {k: v.copy() for k, v in optimized_result.items()}
                        for key in refined_result.keys():
                            if key in opt_smplx_coeff:
                                refined_result[key]['smplx_coeffs'] = opt_smplx_coeff[key]
                                # Clean up mano betas if present
                                if 'left_mano_coeffs' in refined_result[key] and 'betas' in refined_result[key]['left_mano_coeffs']:
                                    del refined_result[key]['left_mano_coeffs']['betas']
                                if 'right_mano_coeffs' in refined_result[key] and 'betas' in refined_result[key]['right_mano_coeffs']:
                                    del refined_result[key]['right_mano_coeffs']['betas']
                        
                        shutil.copyfile(videos_info_path, osp.join(saving_root, 'videos_info.json'))
                        if osp.exists(extra_info_path):
                            shutil.copyfile(extra_info_path, osp.join(saving_root, 'extra_info.json'))
                        write_dict_pkl(refine_id_share_params_fp, id_share_params_results)
                        write_dict_pkl(refine_track_fp_smplx, refined_result)
                        optimized_result = refined_result
                        log(f"SMPL-X refinement completed for {video_name}")
                    except Exception as e:
                        log(f"Warning: SMPL-X refinement failed for {video_name}: {e}")
                    finally:
                        lmdb_engine.close()
                
                # Generate refined visualization if requested
                if args.save_vis_video:
                    refine_video_fp = osp.join(saving_root, 'viz_refine_tracking.mp4')
                    
                    if not os.path.exists(refine_video_fp):
                        lmdb_engine = LMDBEngine(out_lmdb_dir, write=True)
                        all_images = []
                        device = self.cfg.device
                        cameras_kwargs = self.ehm_opt.build_cameras_kwargs(1, self.ehm_opt.body_focal_length)
                        
                        with torch.no_grad():
                            lights = PointLights(device=self.device, location=[[0.0, -1.0, -100.0]])
                            for idx, image_key in tqdm(
                                enumerate(optimized_result.keys()),
                                desc='Generating refined visualization',
                                total=len(optimized_result)
                            ):
                                t_flame_coeffs, t_smplx_coeffs, t_left_mano_coeffs, t_right_mano_coeffs = \
                                    self.convert_traking_params(optimized_result, id_share_params_results, image_key, device)
                                
                                ret_body = self.ehm_opt.ehm(t_smplx_coeffs, t_flame_coeffs)
                                xx = ret_body['vertices']
                                
                                camera_RT_params = torch.tensor(optimized_result[image_key]['smplx_coeffs']['camera_RT_params']).to(device)
                                R, T = camera_RT_params.split([3, 1], dim=-1)
                                T = T.squeeze(-1)
                                R, T = R[None], T[None]
                                
                                cameras = GS_Camera(R=R, T=T, **cameras_kwargs).to(device)
                                proj_joints = cameras.transform_points_screen(ret_body['joints'], R=R, T=T)
                                pred_kps2d = smplx_joints_to_dwpose(proj_joints)[0][..., :2]
                                gt_lmk_2d = optimized_result[image_key]['dwpose_rlt']['keypoints']
                                
                                rendered_img = self.body_renderer.render_mesh(
                                    xx, cameras, lights=lights,
                                    smplx2flame_ind=self.ehm_opt.ehm.smplx.smplx2flame_ind
                                )
                                t_img = (rendered_img[:, :3].cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1, 2, 0)
                                
                                img_inp = lmdb_engine[f'{image_key}/body_image'].numpy().transpose(1, 2, 0)
                                img_bld = cv2.addWeighted(img_inp, 0.5, t_img, 0.5, 1)
                                img_ret = cv2.hconcat([img_inp, t_img, img_bld])
                                
                                # Draw predicted keypoints in red
                                for kp in pred_kps2d[0].cpu().numpy():
                                    x, y = int(kp[0]), int(kp[1])
                                    cv2.circle(img_bld, (x, y), 3, (0, 0, 255), -1)
                                
                                # Draw ground truth keypoints in green
                                for kp in gt_lmk_2d:
                                    x, y = int(kp[0]), int(kp[1])
                                    cv2.circle(img_bld, (x, y), 3, (0, 255, 0), -1)
                                
                                all_images.append(img_ret)
                        
                        lmdb_engine.close()
                        images2video(all_images, refine_video_fp, fps=30)
                        log(f"Refined visualization saved: {video_name} ==> {refine_video_fp}")
                
                log(f"Refinement completed for {video_name}")
                
            except Exception as e:
                log(f"Error during refinement of {video_name}: {e}")
                import traceback
                log(f"Traceback: {traceback.format_exc()}")
                continue
    
    def del_extra_params_values(optim_frame_params):
        del optim_frame_params['body_crop']
        del optim_frame_params['dwpose_raw']
        del optim_frame_params['dwpose_rlt']
        del optim_frame_params['head_crop']
        del optim_frame_params['head_lmk_203']
        del optim_frame_params['head_lmk_70']
        del optim_frame_params['head_lmk_mp']
        del optim_frame_params['left_hand_crop']
        del optim_frame_params['right_hand_crop']
        return optim_frame_params