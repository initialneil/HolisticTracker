import os
import os.path as osp
import shutil
from pathlib import Path
import argparse
import cv2
import torch
import numpy as np
from gfava.datasets.mixed_dataset import MixedDataset, collate_fn, YoutubeGestureDatasetAiOS
from gfava.utils.omegaconf_utils import load_from_config
from gfava.utils.lmdb import LMDBEngine
from gfava.utils import smplx_utils
from gfava.utils.ply_utils import save_points_to_ply
from omegaconf import OmegaConf
from tqdm import tqdm
import pickle
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Test Dataset')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--video_id', type=str, required=True)
    parser.add_argument('--segment_id', type=str, default='body')
    args, extras = parser.parse_known_args()
    
    # root to the dataset
    return args, extras

def main():
    args, extras = parse_args()
    # set_seed(42)

    # dataset
    dataset = YoutubeGestureDatasetAiOS({
        'data_root': args.data_root,
    }, mode='train')

    video_id = args.video_id
    video_data = dataset.video_data[video_id]
    segment_id = 'body'
    data_list = [d for d in video_data['data_list'] if d['segment_id'] == segment_id]
    print(f'Video {video_id}, segment {segment_id}, {len(data_list)} frames.')

    out_dir = osp.join(args.out_dir, f'{video_id}.{segment_id}')
    os.makedirs(out_dir, exist_ok=True)

    # dict_keys(['smplx_shape', 'flame_shape', 'left_mano_shape', 'right_mano_shape', 'head_scale', 'hand_scale', 'joints_offset'])
    first_item = dataset.load_frame(video_id, segment_id, data_list[0]['fn'])
    smplx_shape = first_item['smplx_params']['betas'][:1].detach().cpu().numpy()
    smplx_shape = np.concatenate([smplx_shape, np.zeros((1, 200 - smplx_shape.shape[1]))], axis=-1)  # pad to 200
    joints_offset = np.zeros((1, 55, 3))
    tracked_data_id_share = {
        'smplx_shape': smplx_shape,
        'joints_offset': joints_offset,
    }
    tracked_data_id_share_path = os.path.join(out_dir, 'id_share_params.pkl')
    with open(tracked_data_id_share_path, 'wb') as f:
        pickle.dump(tracked_data_id_share, f)
    print(f'Saved id share params to {tracked_data_id_share_path}')

    # smplx_coeffs: dict_keys(['exp', 'global_pose', 'body_pose', 'body_cam', 'camera_RT_params', 'left_hand_pose', 'right_hand_pose'])
    # flame_coeffs: dict_keys(['expression_params', 'jaw_params', 'pose_params', 'neck_pose_params', 'eye_pose_params', 'eyelid_params', 'camera_RT_params', 'cam'])
    tracked_data = {}
    videos_info = {
        video_id: {
            'frames_num': len(data_list),
            'frames_keys': [],
        }
    }

    lmdb_engine = LMDBEngine(os.path.join(out_dir, 'img_lmdb'), write=True)

    SMPLX_PATH = 'data/body_models/smplx'
    smplx_cfg = {
        'type': 'smplx',
        'gender': 'neutral_2020',
        'num_expression_coeffs': 50,
        'num_betas': 50,
        'use_pca': False,
        'flat_hand_mean': True,
        'use_face_contour': True
    }
    smplx = smplx_utils.create_smplx_model(model_path=SMPLX_PATH, **smplx_cfg)

    for item in tqdm(data_list):
        fn = item['fn']
        frm_data = dataset.load_frame(video_id, segment_id, fn)

        smplx_params = frm_data['smplx_params']
        transl = smplx_params.pop('transl')
        smplx_params = smplx_utils.fit_smplx_flat_hand_mean(smplx, smplx_params, 
                                                            from_flat_hand_mean=False, to_flat_hand_mean=True)
        """
        smplx_out = smplx_utils.infer_smplx_full(smplx, smplx_params)
        vertices = smplx_out.vertices
        save_points_to_ply(osp.join(out_dir, f'{fn}_smplx.ply'), vertices[0].detach().cpu().numpy())
        """

        exp = smplx_params['expression'].squeeze(0).detach().cpu().numpy()
        exp = np.concatenate([exp, np.zeros(50 - exp.shape[0])])  # pad to 50
        global_pose = smplx_params['global_orient'].squeeze(0).detach().cpu().numpy()
        body_pose = smplx_params['body_pose'].squeeze(0).detach().cpu().numpy()
        left_hand_pose = smplx_params['left_hand_pose'].squeeze(0).detach().cpu().numpy().reshape(-1, 3)
        right_hand_pose = smplx_params['right_hand_pose'].squeeze(0).detach().cpu().numpy().reshape(-1, 3)
        jaw_pose = smplx_params['jaw_pose'].squeeze(0).detach().cpu().numpy()
        leye_pose = smplx_params['leye_pose'].squeeze(0).detach().cpu().numpy()
        reye_pose = smplx_params['reye_pose'].squeeze(0).detach().cpu().numpy()

        image = frm_data['img_cv_full']
        K = frm_data['cam_int_full']
        height, width = image.shape[:2]
        invtanfov = K[0, 0] / (width * 0.5)

        
        cam_R, cam_t = frm_data['cam_R'].detach(), frm_data['cam_t'].detach()
        cam_c = -cam_R.T @ cam_t
        cam_c -= transl.squeeze(0)
        cam_t = -cam_R @ cam_c


        # to PyTorch3D coordinate
        w2c_cam = torch.tensor([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], dtype=torch.float32).to(cam_R.device)
        w2c_cam[:3, :3] = cam_R
        w2c_cam[:3, 3] = cam_t.squeeze()
        
        c2c_mat=torch.tensor([[-1, 0, 0, 0],
                            [ 0,-1, 0, 0],
                            [ 0, 0, 1, 0],
                            [ 0, 0, 0, 1],
                            ],dtype=torch.float32)
        w2c_cam=torch.matmul(torch.linalg.inv(c2c_mat).to(w2c_cam.device), w2c_cam)
        camera_RT_params = w2c_cam[:3, :4].detach().cpu().numpy()


        tracked_data[fn] = {
            'smplx_coeffs': {
                'exp': exp,
                'global_pose': global_pose,
                'body_pose': body_pose,
                'left_hand_pose': left_hand_pose,
                'right_hand_pose': right_hand_pose,
                'jaw_pose': jaw_pose,
                'leye_pose': leye_pose,
                'reye_pose': reye_pose,
                'camera_RT_params': camera_RT_params,
                'K': K,
                'width': torch.tensor([width]),
                'height': torch.tensor([height]),
                'invtanfov': torch.tensor([invtanfov]),
            },
            # 'flame_coeffs': {
            #     'expression_params': exp,
            #     'jaw_params': jaw_params,
            # }
        }

        videos_info[video_id]['frames_keys'].append(fn)

        body_image = torch.from_numpy(image[..., :3]).permute(2, 0, 1)  # to CHW
        body_mask = torch.from_numpy(image[..., 3:]).repeat(1, 1, 3).permute(2, 0, 1)  # to CHW
        lmdb_engine.dump(f"{fn}/body_image", body_image)
        lmdb_engine.dump(f"{fn}/body_mask", body_mask)

    tracked_data_path = os.path.join(out_dir, 'optim_tracking_ehm.pkl')
    with open(tracked_data_path, 'wb') as f:
        pickle.dump(tracked_data, f)
    print(f'Saved tracked data to {tracked_data_path}')

    videos_info_path = os.path.join(out_dir, 'videos_info.json')
    with open(videos_info_path, 'w') as f:
        json.dump(videos_info, f, indent=4)

    lmdb_engine.close()
    pass

if __name__ == "__main__":
    main()

