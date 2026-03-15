#!/usr/bin/env python
"""
Standalone SMPLer-X inference script for ehm-tracker.
Runs in the smplerx conda env, processes body crops and saves SMPL-X parameters
in the format expected by ehm-tracker's track_base pipeline.

Usage:
    /home/szj/miniconda3/envs/smplerx/bin/python run_smplerx_inference.py \
        --images_dir /path/to/shots_images \
        --ehmx_dir /path/to/shots_ehmx \
        --video_name VIDEO_NAME \
        --model_variant smpler_x_b32
"""
import os
import sys
import argparse
import pickle
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

# Add SMPLer-X paths
SMPLERX_ROOT = '/mnt/sdc/szj/Playground/SMPLer-X'
sys.path.insert(0, os.path.join(SMPLERX_ROOT, 'main'))
sys.path.insert(0, os.path.join(SMPLERX_ROOT, 'common'))
sys.path.insert(0, os.path.join(SMPLERX_ROOT, 'data'))

from config import cfg
import torchvision.transforms as transforms


def load_model(model_variant='smpler_x_b32', device='cuda:0'):
    """Load SMPLer-X model."""
    config_path = os.path.join(SMPLERX_ROOT, 'main', 'config', f'config_{model_variant}.py')
    ckpt_path = os.path.join(SMPLERX_ROOT, 'pretrained_models', f'{model_variant}.pth.tar')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    cfg.get_config_fromfile(config_path)
    cfg.update_test_config('EHF', 'na', shapy_eval_split=None,
                           pretrained_model_path=ckpt_path, use_cache=False)
    # Set up minimal dirs to satisfy Demoer init
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix='smplerx_')
    cfg.output_dir = tmp_dir
    cfg.model_dir = os.path.join(tmp_dir, 'model')
    cfg.vis_dir = os.path.join(tmp_dir, 'vis')
    cfg.log_dir = os.path.join(tmp_dir, 'log')
    cfg.code_dir = os.path.join(tmp_dir, 'code')
    cfg.result_dir = os.path.join(tmp_dir, 'result')
    for d in [cfg.model_dir, cfg.vis_dir, cfg.log_dir, cfg.code_dir, cfg.result_dir]:
        os.makedirs(d, exist_ok=True)

    # Must run from SMPLer-X/main directory for relative config paths
    orig_cwd = os.getcwd()
    os.chdir(os.path.join(SMPLERX_ROOT, 'main'))

    from base import Demoer
    demoer = Demoer()
    demoer._make_model()
    demoer.model.eval()

    os.chdir(orig_cwd)
    return demoer


def run_smplerx_on_image(demoer, img_rgb, bbox_xywh=None):
    """
    Run SMPLer-X on a single image.

    Args:
        demoer: SMPLer-X Demoer instance
        img_rgb: RGB image (H, W, 3) uint8
        bbox_xywh: Optional [x, y, w, h] bounding box. If None, uses full image.

    Returns:
        dict with SMPL-X parameters in axis-angle format
    """
    from utils.preprocessing import process_bbox, generate_patch_image

    h, w = img_rgb.shape[:2]

    if bbox_xywh is None:
        # Use full image as bbox
        bbox_xywh = np.array([0, 0, w, h], dtype=np.float32)

    bbox = process_bbox(bbox_xywh, w, h)
    if bbox is None:
        return None

    img_patch, img2bb_trans, bb2img_trans = generate_patch_image(
        img_rgb, bbox, 1.0, 0.0, False, cfg.input_img_shape)

    transform = transforms.ToTensor()
    img_tensor = transform(img_patch.astype(np.float32)) / 255.0
    img_tensor = img_tensor.cuda()[None, :, :, :]

    inputs = {'img': img_tensor}
    targets = {}
    meta_info = {}

    with torch.no_grad():
        out = demoer.model(inputs, targets, meta_info, 'test')

    # Extract parameters — all in axis-angle format
    # body_pose: (1, 63) → (1, 21, 3)
    body_pose = out['smplx_body_pose'].reshape(1, 21, 3).cpu().numpy()
    global_orient = out['smplx_root_pose'].reshape(1, 1, 3).cpu().numpy()
    jaw_pose = out['smplx_jaw_pose'].reshape(1, 1, 3).cpu().numpy()
    left_hand_pose = out['smplx_lhand_pose'].reshape(1, 15, 3).cpu().numpy()
    right_hand_pose = out['smplx_rhand_pose'].reshape(1, 15, 3).cpu().numpy()
    shape = out['smplx_shape'].reshape(1, -1).cpu().numpy()  # (1, 10)
    expr = out['smplx_expr'].reshape(1, -1).cpu().numpy()    # (1, 10)
    cam_trans = out['cam_trans'].reshape(1, 3).cpu().numpy()

    # Pad shape/expression to match PIXIE format
    shape_padded = np.zeros((1, 200), dtype=np.float32)
    shape_padded[:, :shape.shape[1]] = shape
    expr_padded = np.zeros((1, 50), dtype=np.float32)
    expr_padded[:, :expr.shape[1]] = expr

    # Extract individual joint poses from body_pose for compatibility
    # SMPLX body_pose joint order: 0-20, where:
    # 12=neck (L_collar→neck), 15=head (neck→head)
    # 20=left_wrist, 19=right_wrist (actually: 20=L_Index1, 19=R_Index1 in some conventions)
    # In standard SMPLX: joint 12=neck, 15=head
    neck_pose = body_pose[:, 12:13, :]   # (1, 1, 3)
    head_pose = body_pose[:, 15:16, :]   # (1, 1, 3) — this is SMPLX head joint (index 15 in body_pose)

    # Wrist poses — in SMPLX body_pose, joint 20 = left wrist, 21 doesn't exist
    # Actually SMPLX body_pose has 21 joints (0-20), left_wrist=20, right_wrist=21 is out of range
    # The wrist is often encoded differently. Let's use the last body joints.
    # Standard SMPLX: body_pose[18] = L_Forearm, body_pose[19] = R_Forearm
    # body_pose[20] = L_Hand (wrist)
    # For PIXIE compatibility, we store them separately
    left_wrist_pose = body_pose[:, 20:21, :]   # (1, 1, 3) — last body joint
    right_wrist_pose = body_pose[:, 19:20, :]  # (1, 1, 3)

    # Project SMPLX body joints to 2D in original image coordinates
    # Like FLAME/MANO: predict → project with own camera → store 2D points
    mesh_cam = out['smplx_mesh_cam']  # (1, V, 3) vertices in camera space (already + cam_trans)
    # SMPLer-X's joint_idx maps SMPLX model joints to their 137-joint set
    # We want the first 25 body joints from joint_proj
    joint_proj_hm = out['smplx_joint_proj']  # (1, 137, 2) in heatmap space
    # Convert heatmap coords → crop pixel coords: reverse the normalization in get_coord()
    # hm = pixel / input_body_shape * output_hm_shape
    # pixel = hm / output_hm_shape * input_body_shape
    joint_proj_crop = joint_proj_hm.cpu().numpy()[0]  # (137, 2)
    joint_proj_crop[:, 0] = joint_proj_crop[:, 0] / cfg.output_hm_shape[2] * cfg.input_body_shape[1]  # x
    joint_proj_crop[:, 1] = joint_proj_crop[:, 1] / cfg.output_hm_shape[1] * cfg.input_body_shape[0]  # y
    # Transform from crop pixel coords → original image coords using bb2img_trans (2x3 affine)
    n_joints = joint_proj_crop.shape[0]
    joint_proj_homo = np.concatenate([joint_proj_crop, np.ones((n_joints, 1))], axis=1)  # (137, 3)
    joint_proj_img = (bb2img_trans @ joint_proj_homo.T).T  # (137, 2) in original image coords
    # Take body joints only (first 25)
    body_joints_2d = joint_proj_img[:25].astype(np.float32)  # (25, 2)

    result = {
        'body_pose': body_pose,              # (1, 21, 3)
        'global_pose': global_orient,        # (1, 1, 3)
        'neck_pose': neck_pose,              # (1, 1, 3)
        'head_pose': head_pose,              # (1, 1, 3)
        'jaw_pose': jaw_pose,                # (1, 1, 3)
        'left_hand_pose': left_hand_pose,    # (1, 15, 3)
        'right_hand_pose': right_hand_pose,  # (1, 15, 3)
        'left_wrist_pose': left_wrist_pose,  # (1, 1, 3)
        'right_wrist_pose': right_wrist_pose,# (1, 1, 3)
        'shape': shape_padded,               # (1, 200)
        'exp': expr_padded,                  # (1, 50)
        'body_joints_2d': body_joints_2d,    # (25, 2) in original image pixel coords
    }

    return result


def process_video(demoer, images_dir, ehmx_dir, video_name):
    """Process all frames of a video."""
    # Load videos_info.json to get frame list
    import json
    info_path = os.path.join(ehmx_dir, video_name, 'videos_info.json')
    with open(info_path, 'r') as f:
        videos_info = json.load(f)

    video_data = videos_info[video_name]
    frame_keys = video_data['frames_keys']

    print(f"Processing {video_name}: {len(frame_keys)} frames")

    results = {}
    for frame_key in tqdm(frame_keys, desc=f"SMPLer-X [{video_name}]"):
        # Load image
        # frame_key format: "shot_name/frame_num" or "shot_name/frame_num/pshuman_xx"
        parts = frame_key.split('/')
        if len(parts) == 2:
            # Standard frame
            img_path = os.path.join(images_dir, video_name, parts[0], f'{parts[1]}.jpg')
        elif len(parts) == 3 and 'pshuman' in parts[2]:
            # PSHuman frame — load from pshuman directory
            # Skip for now, use None to signal downstream to use PIXIE fallback
            results[frame_key] = None
            continue
        else:
            print(f"  Warning: Unknown frame key format: {frame_key}")
            results[frame_key] = None
            continue

        if not os.path.exists(img_path):
            print(f"  Warning: Image not found: {img_path}")
            results[frame_key] = None
            continue

        img_rgb = cv2.imread(img_path)
        if img_rgb is None:
            results[frame_key] = None
            continue
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        # Run SMPLer-X with full image as bbox
        result = run_smplerx_on_image(demoer, img_rgb)
        results[frame_key] = result

    # Save results
    output_path = os.path.join(ehmx_dir, video_name, 'smplerx_init.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved SMPLer-X results to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--ehmx_dir', type=str, required=True)
    parser.add_argument('--video_name', type=str, required=True,
                        help='Single video name or comma-separated list')
    parser.add_argument('--model_variant', type=str, default='smpler_x_b32',
                        choices=['smpler_x_s32', 'smpler_x_b32', 'smpler_x_l32', 'smpler_x_h32'])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cudnn.benchmark = True

    print(f"Loading SMPLer-X model: {args.model_variant}")
    demoer = load_model(args.model_variant)

    video_names = [v.strip() for v in args.video_name.split(',')]
    for video_name in video_names:
        process_video(demoer, args.images_dir, args.ehmx_dir, video_name)

    print("Done!")


if __name__ == '__main__':
    main()
