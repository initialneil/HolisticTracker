import os
import cv2
import glob
import torch
import importlib
import numpy as np
from tqdm.auto import tqdm
from pytorch3d.transforms import matrix_to_rotation_6d
from pytorch3d.transforms import matrix_to_rotation_6d,rotation_6d_to_matrix
from pytorch3d.renderer import PointLights
from src.utils.draw import draw_landmarks
from .smplx_utils import smplx_joints_to_dwpose, smplx_to_dwpose
from ...utils.rprint import rlog as log
from ...utils.helper import build_minibatch
from ...utils.io import load_config
from ...utils import rotation_converter as converter
from ...utils.graphics import GS_Camera
from ...losses import Landmark2DLoss, PoseLoss
from ...modules.renderer.body_renderer import Renderer as BodyRenderer
from ...modules.renderer.head_renderer import Renderer as HeadRenderer
from ...modules.renderer.hand_renderer import Renderer as HandRenderer
from ...modules.ehm import EHM
np.random.seed(0)

def expid2model(expr_dir):
    from configer import Configer
    if not os.path.exists(expr_dir): raise ValueError('Could not find the experiment directory: %s' % expr_dir)
    best_model_fname = sorted(glob.glob(os.path.join(expr_dir, 'snapshots', '*.pt')), key=os.path.getmtime)[-1]
    log(('Found Trained Model: %s' % best_model_fname))
    default_ps_fname = glob.glob(os.path.join(expr_dir,'*.ini'))[0]
    if not os.path.exists(
        default_ps_fname): raise ValueError('Could not find the appropriate vposer_settings: %s' % default_ps_fname)
    ps = Configer(default_ps_fname=default_ps_fname, work_dir=expr_dir, best_model_fname=best_model_fname)
    return ps, best_model_fname


def load_vposer(expr_dir, vp_model='snapshot'):
    '''
    :param expr_dir:
    :param vp_model: either 'snapshot' to use the experiment folder's code or a VPoser imported module, e.g.
    from human_body_prior.train.vposer_smpl import VPoser, then pass VPoser to this function
    :param if True will load the model definition used for training, and not the one in current repository
    :return:
    '''
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


class EhmOptimizer(object):
    def __init__(self, flame_assets_dir, smplx_assets_dir, mano_assets_dir, 
                 device='cuda:0', body_image_size=1024, head_image_size=512, 
                 tanfov=1.0/12, vposer_ckpt='', bbone_cfg_fp=''):
        """
        Optimize the body pose and shape parameters using two stage optimization.
        Stage 1: Optimize the body pose and shape parameters for better head & hand alignment.
        Stage 2: Optimize the body pose parameters for better shoulder & elbow alignment.
        """
        self.smplx_assets_dir = smplx_assets_dir
        self.device = device
        self.use_vposer = os.path.exists(vposer_ckpt)
        self.body_image_size = body_image_size
        self.head_image_size = head_image_size
        #self.focal_length = focal_length
        self.body_focal_length,self.head_focal_length,self.hand_focal_length=1.0/tanfov,1.0/tanfov,1.0/tanfov
        #self.ehm = EHM(flame_assets_dir, smplx_assets_dir, mano_assets_dir).to(self.device)
        self.ehm = EHM(flame_assets_dir, smplx_assets_dir,mano_assets_dir).to(self.device)
        self.body_renderer = BodyRenderer(smplx_assets_dir, body_image_size, focal_length=self.body_focal_length).to(self.device)
        self.head_renderer = HeadRenderer(flame_assets_dir, head_image_size, focal_length=self.head_focal_length).to(self.device)
        self.hand_renderer = HandRenderer(mano_assets_dir,  head_image_size, focal_length=self.hand_focal_length).to(self.device)
        
        #bbone_cfg = load_config(bbone_cfg_fp)
        self.lmk2d_loss = Landmark2DLoss(self.ehm.smplx.lmk_203_left_indices,
                                         self.ehm.smplx.lmk_203_right_indices,
                                         self.ehm.smplx.lmk_203_front_indices,
                                         self.ehm.smplx.lmk_mp_indices, metric='l1').to(self.device)
        self.metric = torch.nn.L1Loss().to(self.device)
        self.pose_loss = PoseLoss().to(self.device)
        
        if self.use_vposer:
            self.vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
            self.vposer = self.vposer.to(device=device)
            self.vposer.eval()

        self.check_pose = True
        self.kps_map, kps_w = smplx_to_dwpose()
        self.kps_w = torch.from_numpy(kps_w).unsqueeze(0).to(self.device)

    def transform_points3d(self, points3d:torch.Tensor, M:torch.Tensor):
        R3d = torch.zeros_like(M)
        R3d[:, :2, :2] = M[:, :2, :2]
        scale = (M[:, 0, 0]**2 + M[:, 0, 1]**2)**0.5
        R3d[:, 2, 2] = scale

        trans = torch.zeros_like(M)[:, 0]
        trans[:, :2] = M[:, :2, 2]
        trans = trans.unsqueeze(1)
        return torch.bmm(points3d, R3d.mT) + trans

    def transform_hand_pts3d_to_image_coord(self, X, M, img_size=512, is_left=False):
        # deal with mirror issue
        _X = self.fix_mirror_issue(X, img_size)

        if is_left: _X[..., 0] = img_size - 1 - _X[..., 0]
        _X = self.transform_points3d(_X, M.to(_X.device))

        return _X

    def transform_head_pts3d_to_image_coord(self, X, M):
        _X = self.transform_points3d(X, M.to(X.device))
        return _X

    def prepare_ref_vertices(self, head_coeff_lst, head_crop_meta,
                             left_hand_coeff_lst, left_hand_crop_meta, 
                             right_hand_coeff_lst, right_hand_crop_meta, body_crop_meta):
        img_size = self.head_image_size
        head_ret_dict = self.ehm.flame(head_coeff_lst)
        render_rlt = self.head_renderer(head_ret_dict['vertices'], transform_matrix=head_coeff_lst['camera_RT_params'], 
                                        landmarks={'joints': head_ret_dict['joints']}, ret_image=False)
        
        ref_head_vertices = self.transform_head_pts3d_to_image_coord(
            render_rlt[0][:, np.unique(self.ehm.flame.head_index)], body_crop_meta['M_o2c-hd']@head_crop_meta['M_c2o'])
        ref_head_joints = self.transform_head_pts3d_to_image_coord(
            render_rlt[1]['joints'], body_crop_meta['M_o2c-hd']@head_crop_meta['M_c2o'])
        ref_head_vertices[..., 2] = (ref_head_vertices[..., 2] - ref_head_joints[:, 3:5, 2].mean(dim=1, keepdim=True))
        
        hand_l_ret_dict = self.ehm.mano(left_hand_coeff_lst, pose_type='aa')
        render_rlt = self.hand_renderer(hand_l_ret_dict['vertices'], landmarks={'joints': hand_l_ret_dict['joints']}, is_left=True,
                                         transform_matrix=left_hand_coeff_lst['camera_RT_params'], ret_image=False)
        ref_hand_l_vertices = self.transform_hand_pts3d_to_image_coord(
            render_rlt[0], body_crop_meta['M_o2c-hd']@left_hand_crop_meta['M_c2o'], img_size, True)
        ref_hand_l_joints = self.transform_hand_pts3d_to_image_coord(
            render_rlt[1]['joints'], body_crop_meta['M_o2c-hd']@left_hand_crop_meta['M_c2o'], img_size, True)
        ref_hand_l_vertices[..., 2] = ref_hand_l_vertices[..., 2] - ref_hand_l_joints[:, 0:1, 2]

        hand_r_ret_dict = self.ehm.mano(right_hand_coeff_lst, pose_type='aa')
        render_rlt = self.hand_renderer(hand_r_ret_dict['vertices'], landmarks={'joints': hand_r_ret_dict['joints']}, is_left=False,
                                         transform_matrix=right_hand_coeff_lst['camera_RT_params'], ret_image=False)
        ref_hand_r_vertices = self.transform_hand_pts3d_to_image_coord(
            render_rlt[0], body_crop_meta['M_o2c-hd']@right_hand_crop_meta['M_c2o'], img_size)
        ref_hand_r_joints = self.transform_hand_pts3d_to_image_coord(
            render_rlt[1]['joints'], body_crop_meta['M_o2c-hd']@right_hand_crop_meta['M_c2o'], img_size)
        ref_hand_r_vertices[..., 2] = ref_hand_r_vertices[..., 2] - ref_hand_r_joints[:, 0:1, 2]

        return ref_head_vertices, ref_hand_l_vertices, ref_hand_r_vertices,ref_hand_l_joints, ref_hand_r_joints

    def build_cameras_kwargs(self, batch_size,focal_length):
        screen_size = torch.tensor([self.body_image_size, self.body_image_size], device=self.device).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': torch.zeros(batch_size, 2, device=self.device).float(), 
            'focal_length': focal_length, 
            'image_size': screen_size, 'device': self.device,
        }
        return cameras_kwargs

    def fix_mirror_issue(self, pts3d, image_size):
        p = pts3d.clone()
        p[..., 1] = image_size - p[..., 1]
        p[..., 2] = -p[..., 2]
        return p

    def optimize(self, track_frames, batch_base, id_share_parms, optim_cfg,
                 batch_id=0, batch_imgs=None, interval=1):
        steps = optim_cfg.get('steps', 1001)
        share_id = optim_cfg.get('share_id', True)
        share_pose = optim_cfg.get('share_pose', False)
        optim_camera = optim_cfg.get('optim_camera', False)
        
        # Extract lambda weights from config
        lambda_3d_head = optim_cfg.get('lambda_3d_head', 5.0)
        lambda_3d_hand_l = optim_cfg.get('lambda_3d_hand_l', 0.25)
        lambda_3d_hand_r = optim_cfg.get('lambda_3d_hand_r', 0.25)
        lambda_3d_z = optim_cfg.get('lambda_3d_z', 2.5)
        lambda_2d_kpt = optim_cfg.get('lambda_2d_kpt', 0.2)
        lambda_2d_knee_feet = optim_cfg.get('lambda_2d_knee_feet', 2.0)
        lambda_smplx_init = optim_cfg.get('lambda_smplx_init', 1.0)
        lambda_prior = optim_cfg.get('lambda_prior', 10.0)
        lambda_smplx_shape_reg = optim_cfg.get('lambda_smplx_shape_reg', 0.1)
        lambda_mano_shape_reg = optim_cfg.get('lambda_mano_shape_reg', 0.1)
        lambda_smplx_pose = optim_cfg.get('lambda_smplx_pose', 0.1)
        lambda_smplx_pose_reg_base = optim_cfg.get('lambda_smplx_pose_reg_base', 100.0)
        lambda_smplx_leg_pose = optim_cfg.get('lambda_smplx_leg_pose', 400.0)
        lambda_smplx_freezed_pose = optim_cfg.get('lambda_smplx_freezed_pose', 400.0)
        lambda_smplx_hand_pose_reg = optim_cfg.get('lambda_smplx_hand_pose_reg', 0.1)
        lambda_scale_reg = optim_cfg.get('lambda_scale_reg', 100.0)
        lambda_joint_offset_reg = optim_cfg.get('lambda_joint_offset_reg', 10000.0)
        lambda_mtn_body_pose = optim_cfg.get('lambda_mtn_body_pose', 200.0)
        lambda_mtn_rot6d = optim_cfg.get('lambda_mtn_rot6d', 100.0)
        lambda_mtn_trans = optim_cfg.get('lambda_mtn_trans', 100.0)
        lambda_mtn_trans_z = optim_cfg.get('lambda_mtn_trans_z', 100.0)
        lambda_mtn_vertices = optim_cfg.get('lambda_mtn_vertices', 1.0)

        batch_size = len(track_frames)
        """
        img = batch_imgs[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        lmk_2d = gt_lmk_2d['keypoints'][0]
        canvas = draw_landmarks(lmk_2d, img, color=(0, 255, 0), viz_index=False)
        """

        batch_smplx, batch_flame, gt_lmk_2d = batch_base['smplx_coeffs'], batch_base['flame_coeffs'], batch_base['dwpose_rlt']
        batch_mano_left, batch_mano_right = batch_base['left_mano_coeffs'], batch_base['right_mano_coeffs']
        head_lmk_valid = batch_base['head_lmk_valid']
        left_hand_valid = batch_base['left_hand_valid']
        right_hand_valid = batch_base['right_hand_valid']

        b, n = batch_mano_left["hand_pose"][:,0].shape[:2]
        left_hand_pose=converter.batch_matrix2axis(batch_mano_left["hand_pose"][:,0].flatten(0,1)).reshape(b, n*3)#roma.mappings.rotmat_to_rotvec(batch_mano_left["hand_pose"][:,0,...])[:,None,...]
        right_hand_pose=converter.batch_matrix2axis(batch_mano_right["hand_pose"][:,0].flatten(0,1)).reshape(b,1, n, 3)#roma.mappings.rotmat_to_rotvec(batch_mano_right["hand_pose"][:,0,...])[:,None,...] 
        left_hand_pose[:,1::3]*=-1
        left_hand_pose[:,2::3]*=-1
        batch_smplx["left_hand_pose"]=left_hand_pose.detach().clone().reshape(b,1, n, 3)
        batch_smplx["right_hand_pose"]=right_hand_pose.detach().clone()
        
        batch_smplx  = {kk: vv.squeeze() for kk, vv in batch_smplx.items()}
        batch_mano_left  = {kk: vv.squeeze() for kk, vv in batch_mano_left.items()}
        batch_mano_right = {kk: vv.squeeze() for kk, vv in batch_mano_right.items()}
        if batch_size==1:
            batch_smplx  = {kk: vv[None] for kk, vv in batch_smplx.items()}
            batch_mano_left  = {kk: vv[None] for kk, vv in batch_mano_left.items()}
            batch_mano_right = {kk: vv[None] for kk, vv in batch_mano_right.items()}
        body_lmk_score = gt_lmk_2d['scores']
        assert share_id==True
        if share_id:
            head_scale=torch.tensor(id_share_parms['head_scale'],device=self.device)
            hand_scale=torch.tensor(id_share_parms['hand_scale'],device=self.device)
            joints_offset=torch.tensor(id_share_parms['joints_offset'],device=self.device)
            g_smplx_shape = torch.tensor(id_share_parms['smplx_shape'],device=self.device).float()
            g_flame_shape = torch.tensor(id_share_parms['flame_shape'],device=self.device).float()
            left_hand_shape,right_hand_shape=torch.tensor(id_share_parms['left_mano_shape'],device=self.device),torch.tensor(id_share_parms['right_mano_shape'],device=self.device)
            batch_smplx['head_scale']=head_scale.expand(batch_size, -1)
            batch_smplx['hand_scale']=hand_scale.expand(batch_size, -1)
            batch_smplx['joints_offset']=joints_offset.expand(batch_size, -1,-1)
            batch_flame['shape_params']=g_flame_shape.expand(batch_size, -1)
            batch_smplx['shape']=g_smplx_shape.expand(batch_size, -1)

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
            
        (ref_head_vertices, ref_hand_l_vertices, ref_hand_r_vertices,
         ref_hand_l_joints, ref_hand_r_joints) = self.prepare_ref_vertices(
             batch_flame, batch_base['head_crop'],
             batch_mano_left, batch_base['left_hand_crop'],
             batch_mano_right, batch_base['right_hand_crop'],
             batch_base['body_crop'])


        # Validate hand tracking by comparing hand vertices to wrist keypoints
        # Wrist indices: 7 (left wrist), 4 (right wrist)
        # Elbow indices: 6 (left elbow), 3 (right elbow)
        left_wrist_kpt2d = gt_lmk_2d['keypoints'][:, 7, :2]
        left_elbow_kpt2d = gt_lmk_2d['keypoints'][:, 6, :2]
        right_wrist_kpt2d = gt_lmk_2d['keypoints'][:, 4, :2]
        right_elbow_kpt2d = gt_lmk_2d['keypoints'][:, 3, :2]
        
        # Calculate average wrist-to-elbow distances as threshold
        left_wrist_elbow_dist = torch.norm(left_wrist_kpt2d - left_elbow_kpt2d, dim=1)
        right_wrist_elbow_dist = torch.norm(right_wrist_kpt2d - right_elbow_kpt2d, dim=1)
        
        # Check left hand validity
        if left_hand_valid.sum() > 0:
            left_hand_wrist_3d = ref_hand_l_vertices[:, self.ehm.mano.selected_vert_ids].mean(dim=1)[:, :2]  # Average of selected vertices
            left_hand_wrist_dist = torch.norm(left_hand_wrist_3d - left_wrist_kpt2d, dim=1)
            left_hand_valid = left_hand_valid & (left_hand_wrist_dist < left_wrist_elbow_dist)
        
        # Check right hand validity
        if right_hand_valid.sum() > 0:
            right_hand_wrist_3d = ref_hand_r_vertices[:, self.ehm.mano.selected_vert_ids].mean(dim=1)[:, :2]  # Average of selected vertices
            right_hand_wrist_dist = torch.norm(right_hand_wrist_3d - right_wrist_kpt2d, dim=1)
            right_hand_valid = right_hand_valid & (right_hand_wrist_dist < right_wrist_elbow_dist)


        weight, weight_keep = get_2d_keypoints_weight(body_lmk_score)
        
        _lr_decay=1.0
        if batch_id>0:
            _lr_decay=0.0
        
        # init ref info
        with torch.no_grad():
            smplx_init_pose = body_pose.clone()
            smplx_init_dict = self.ehm(batch_smplx, batch_flame, {'left_hand': batch_mano_left, 'right_hand': batch_mano_right}, pose_type='aa')
            
            cameras_kwargs = self.build_cameras_kwargs(batch_size,focal_length=self.body_focal_length)
            cameras = GS_Camera(**cameras_kwargs).to(self.device)
            R, T = camera_RT_params.detach().split([3, 1], dim=-1)
            T = T.squeeze(-1)
            ref_proj_vertices = cameras.transform_points_screen(smplx_init_dict['vertices'], R=R, T=T)
            ref_proj_joints   = cameras.transform_points_screen(smplx_init_dict['joints'], R=R, T=T)
            ret_body_ref = smplx_joints_to_dwpose(ref_proj_joints)[0]
            
            # Extract reference projections for hands and face from initial pose
            ref_proj_lhand = ref_proj_joints[:, 20:21, :2]  # Left wrist from joints
            ref_proj_rhand = ref_proj_joints[:, 21:22, :2]  # Right wrist from joints
            ref_proj_face = ref_proj_joints[:, 15:16, :2]   # Head joint from joints

        # optimizer related
        # global_pose.requires_grad = True
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
            gl_T[:,2]=(gl_T[:,2].mean(dim=0)[None].expand(batch_size,-1).detach().clone())[:,0]
            _lr_camera = 1.0

        gl_T.requires_grad,gl_R.requires_grad = True,True
        gl_R_6d.requires_grad=True
        joints_offset.requires_grad=True
        g_flame_shape.requires_grad=True
        head_scale.requires_grad = True
        hand_scale.requires_grad = True
        leg_body_joints =[4,5,7,8,10]
        freezed_body_joints=[1,3,2,6,9]
        opt_p = torch.optim.AdamW([
            {'params': [body_pose], 'lr': 1e-3},
            {'params': [g_smplx_shape],  'lr': 1e-4 * _lr_decay},
            {'params': [left_hand_shape],  'lr': 1e-4 * _lr_decay},
            {'params': [right_hand_shape],  'lr': 1e-4 * _lr_decay},
            {'params': [right_hand_pose],  'lr': 1e-5},
            {'params': [left_hand_pose],  'lr': 1e-5},
            {'params': [gl_T], 'lr': 5e-3 * _lr_camera},
            {'params': [gl_R], 'lr': 5e-4 * _lr_camera},
            {'params': [gl_R_6d], 'lr': 5e-4 * _lr_camera},
            {'params': [joints_offset], 'lr': 1e-5 * _lr_decay},
            {'params':[g_flame_shape], 'lr': 2e-5 * _lr_decay},
            {"params":[head_scale],"lr":1e-4 * _lr_decay},
            {"params":[hand_scale],"lr":1e-4 * _lr_decay},
        ])

        t_bar = tqdm(range(steps), desc='Start tuning SMPLX global params [Warmup]')
        for i_step in t_bar:
            batch_smplx['head_scale'] = head_scale.expand(batch_size, -1)
            batch_smplx['hand_scale'] = hand_scale.expand(batch_size, -1)

            batch_smplx['shape'] = g_smplx_shape.expand(batch_size, -1)
            batch_smplx['joints_offset'] = joints_offset.expand(batch_size, -1, -1)
            batch_mano_left['betas'], batch_mano_right['betas'] = left_hand_shape.expand(batch_size, -1), right_hand_shape.expand(batch_size, -1)
            batch_flame['shape_params'] = g_flame_shape.expand(batch_size, -1)

            batch_smplx['body_pose'] = body_pose.expand(batch_size, -1, -1)
            batch_smplx['global_pose'] = global_pose.expand(batch_size, -1)
            batch_smplx['left_hand_pose'] = left_hand_pose.expand(batch_size, -1, -1)
            batch_smplx['right_hand_pose'] = right_hand_pose.expand(batch_size, -1, -1)
            
            if self.use_vposer:
                body_embedding_mean = self.vposer.encode(body_pose).mean
            else:
                body_embedding_mean = 0
            smplx_dict = self.ehm(batch_smplx, batch_flame, pose_type='aa',)
            T=gl_T
            R=rotation_6d_to_matrix(gl_R_6d)
            T = T.squeeze(-1)
            proj_vertices = cameras.transform_points_screen(smplx_dict['vertices'], R=R, T=T)
            proj_joints   = cameras.transform_points_screen(smplx_dict['joints'], R=R, T=T)
            proj_face_lmk_203=cameras.transform_points_screen(smplx_dict['face_lmk_203'], R=R, T=T)
            loss_3d = 0
            loss_3d_hand_l,loss_3d_hand_r,loss_3d_head=.0,.0,.0
            
            ### 3D vertices loss
            pred_head_vertices = proj_vertices[:, self.ehm.smplx.smplx2flame_ind][:, self.ehm.head_index]
            pred_head_joint    = proj_joints[:, 23:25].mean(dim=1, keepdim=True)
            pred_head_vertices[..., 2] = (pred_head_vertices[..., 2] - pred_head_joint[..., 2])
            if head_lmk_valid.sum() > 0:
                loss_3d_head = self.metric(pred_head_vertices[head_lmk_valid], 
                                        ref_head_vertices[head_lmk_valid]) * lambda_3d_head

            pred_hand_l_vertices = proj_vertices[:, self.ehm.smplx.smplx2mano_ind['left_hand']]
            pred_hand_l_joint    = proj_joints[:, 20:21, :]
            pred_hand_l_vertices[..., 2] = (pred_hand_l_vertices[..., 2] - pred_hand_l_joint[..., 2])
            if left_hand_valid.sum() > 0:
                loss_3d_hand_l = self.metric(
                    pred_hand_l_vertices[left_hand_valid][:, self.ehm.mano.selected_vert_ids], 
                    ref_hand_l_vertices[left_hand_valid][:, self.ehm.mano.selected_vert_ids]
                ) * lambda_3d_hand_l

            pred_hand_r_vertices = proj_vertices[:, self.ehm.smplx.smplx2mano_ind['right_hand']]
            pred_hand_r_joint    = proj_joints[:, 21:22, :]
            pred_hand_r_vertices[..., 2] = (pred_hand_r_vertices[..., 2] - pred_hand_r_joint[..., 2])
            if right_hand_valid.sum() > 0:
                loss_3d_hand_r = self.metric(pred_hand_r_vertices[right_hand_valid][:, self.ehm.mano.selected_vert_ids], 
                                             ref_hand_r_vertices[right_hand_valid][:, self.ehm.mano.selected_vert_ids]) * lambda_3d_hand_r

            loss_3d_z = (self.metric(proj_joints[..., 2], ref_proj_joints[..., 2]) + \
                         self.metric(proj_vertices[..., 2], ref_proj_vertices[..., 2])) * lambda_3d_z
            
            loss_3d = (loss_3d_head + loss_3d_hand_l + loss_3d_hand_r + loss_3d_z) 
            
            ### 2D landmark loss
            pred_kps3d = smplx_joints_to_dwpose(proj_joints)[0]
            
            # Define keypoint groups
            # Knee indices: 9 (right knee), 12 (left knee)
            # Ankle indices: 10 (right ankle), 13 (left ankle)  
            # Foot indices: 18 (left big toe), 19 (left small toe), 20 (left heel), 21 (right big toe), 22 (right small toe), 23 (right heel)
            knee_feet_indices = [9, 10, 12, 13, 18, 19, 20, 21, 22, 23]
            
            # Face indices: 14-17 (body face points: left eye, right eye, left ear, right ear) + 24-91 (face contour + face landmarks)
            face_kpt2d_indices = list(range(24, 92))
            face_kpt2d_indices += [14, 15, 16, 17]
            
            # Hand indices: 92-112 (left hand), 113-133 (right hand)
            # Left elbow and wrist: 6 (left elbow), 7 (left wrist)
            # Right elbow and wrist: 3 (right elbow), 4 (right wrist)
            lhand_kpt2d_indices = list(range(92, 113)) + [6, 7]
            rhand_kpt2d_indices = list(range(113, 134)) + [3, 4]
            
            # Body keypoint indices: all indices excluding knee/feet, face, and hands
            all_kpt2d_indices = list(range(pred_kps3d.shape[1]))
            body_kpt2d_indices = [i for i in all_kpt2d_indices if i not in knee_feet_indices and i not in face_kpt2d_indices and i not in lhand_kpt2d_indices and i not in rhand_kpt2d_indices]
            
            # Body keypoints loss (excluding knee/feet and face)
            loss_2d_kpt = self.metric(
                pred_kps3d[:, body_kpt2d_indices, :2], 
                gt_lmk_2d['keypoints'][:, body_kpt2d_indices, :2].float()
            ) * lambda_2d_kpt
            
            # Knee and feet loss with higher weight
            loss_2d_knee_feet = self.metric(
                pred_kps3d[:, knee_feet_indices, :2], 
                gt_lmk_2d['keypoints'][:, knee_feet_indices, :2].float()
            ) * lambda_2d_knee_feet
            
            # Face landmarks loss (valid: use 2D keypoints, invalid: use init pose projection)
            loss_2d_face = 0.0
            if head_lmk_valid.sum() > 0:
                loss_2d_face = self.metric(
                    pred_kps3d[head_lmk_valid][:, face_kpt2d_indices, :2], 
                    gt_lmk_2d['keypoints'][head_lmk_valid][:, face_kpt2d_indices, :2].float()
                ) * lambda_2d_kpt
            if (~head_lmk_valid).sum() > 0:
                # For invalid faces, regularize towards initial pose projection
                loss_2d_face += self.metric(
                    pred_kps3d[~head_lmk_valid][:, 15, :2],  # Head joint
                    ret_body_ref[~head_lmk_valid][:, 15, :2]
                ) * lambda_smplx_init
            
            # Left hand landmarks loss (valid: use 2D keypoints, invalid: use init pose projection)
            loss_2d_lhand = 0.0
            if left_hand_valid.sum() > 0:
                loss_2d_lhand = self.metric(
                    pred_kps3d[left_hand_valid][:, lhand_kpt2d_indices, :2], 
                    gt_lmk_2d['keypoints'][left_hand_valid][:, lhand_kpt2d_indices, :2].float()
                ) * lambda_2d_kpt
            if (~left_hand_valid).sum() > 0:
                # For invalid left hands, regularize towards initial pose projection
                loss_2d_lhand += self.metric(
                    pred_kps3d[~left_hand_valid][:, 7, :2],  # Left wrist
                    ret_body_ref[~left_hand_valid][:, 7, :2]
                ) * lambda_smplx_init
            
            # Right hand landmarks loss (valid: use 2D keypoints, invalid: use init pose projection)
            loss_2d_rhand = 0.0
            if right_hand_valid.sum() > 0:
                loss_2d_rhand = self.metric(
                    pred_kps3d[right_hand_valid][:, rhand_kpt2d_indices, :2], 
                    gt_lmk_2d['keypoints'][right_hand_valid][:, rhand_kpt2d_indices, :2].float()
                ) * lambda_2d_kpt
            if (~right_hand_valid).sum() > 0:
                # For invalid right hands, regularize towards initial pose projection
                loss_2d_rhand += self.metric(
                    pred_kps3d[~right_hand_valid][:, 4, :2],  # Right wrist
                    ret_body_ref[~right_hand_valid][:, 4, :2]
                ) * lambda_smplx_init
            
            loss_2d = loss_2d_kpt + loss_2d_knee_feet + loss_2d_face + loss_2d_lhand + loss_2d_rhand

            ### 3D joint loss and their prior loss
            loss_prior = torch.abs(body_embedding_mean.mean()) * lambda_prior + \
                torch.mean(torch.square(g_smplx_shape)) * lambda_smplx_shape_reg + \
                torch.mean(torch.square(left_hand_shape)) * lambda_mano_shape_reg + \
                torch.mean(torch.square(right_hand_shape)) * lambda_mano_shape_reg
            loss_smplx_pose = self.pose_loss(body_pose, smplx_init_pose) * lambda_smplx_pose

            loss_smplx_leg_pose = torch.mean((body_pose[:, leg_body_joints]) ** 2) * lambda_smplx_leg_pose
            loss_smplx_pose_reg = torch.mean(body_pose[:, freezed_body_joints] ** 2) * lambda_smplx_freezed_pose
            loss_smplx_pose_reg += torch.mean(body_pose**2) * lambda_smplx_pose_reg_base
            loss_smplx_hand_pose_reg = (
                self.pose_loss(right_hand_pose, torch.zeros_like(right_hand_pose)) + \
                self.pose_loss(left_hand_pose, torch.zeros_like(left_hand_pose))
            ) * lambda_smplx_hand_pose_reg
            loss_scale_reg = (((hand_scale-1.0)**2).mean() + ((head_scale-1.0)**2).mean()) * lambda_scale_reg
            
            loss_joint_offset_reg = (joints_offset**2).mean() * lambda_joint_offset_reg
            
            loss_prior = loss_prior + loss_smplx_pose + loss_smplx_pose_reg + loss_scale_reg + loss_joint_offset_reg + loss_smplx_leg_pose + loss_smplx_hand_pose_reg

            ### motion regularization loss
            mtn_reg_loss = 0
            if body_pose.shape[0] > 1:
                mtn_reg_loss += self.metric(body_pose[1:], body_pose[:-1]) * lambda_mtn_body_pose / (interval * 1)
                mtn_reg_loss += self.metric(gl_R_6d[1:], gl_R_6d[:-1]) * lambda_mtn_rot6d / (interval * 1)
                mtn_reg_loss += self.metric(T[1:], T[:-1]) * lambda_mtn_trans / (interval * 1)
                mtn_reg_loss += self.metric(T[1:,2], T[:-1,2]) * lambda_mtn_trans_z / (interval * 1)
                mtn_reg_loss += self.metric(proj_vertices[1:], proj_vertices[:-1]) * lambda_mtn_vertices / (interval * 1)
            
            total_loss = loss_3d + loss_2d + loss_prior + mtn_reg_loss
            loss_line = f'total: {total_loss:.2f} | 3d: {loss_3d:.2f} | 2d: {loss_2d:.2f} | prior: {loss_prior:.2f} | mtn_reg: {mtn_reg_loss:.2f} '
            loss_info = f'Batch: {batch_id:02d} | Iter: {i_step:03d} >> Loss: {loss_line}'
            t_bar.set_description(loss_info)

            opt_p.zero_grad()
            total_loss.backward()
            opt_p.step()
            
            if batch_imgs is not None and i_step%(steps-1)==0:
                with torch.no_grad():
                    n_imgs = len(batch_imgs)
                    lights=PointLights(device=self.device, location=[[0.0, -1.0, -10.0]])
                    full_indices = np.linspace(0, n_imgs - 1, min(5, n_imgs), dtype=int)
                    save_path = os.path.join(self.saving_root, "visual_results")
                    os.makedirs(save_path, exist_ok=True)
                    
                    # Process images in groups of 10
                    for group_start in range(0, len(full_indices), 10):
                        group_end = min(group_start + 10, n_imgs)
                        img_indices = full_indices[group_start:group_end]
                        vis_imgs = []
                        for im_idx in img_indices:
                            _img=batch_imgs[im_idx].clone().numpy().transpose(1,2,0)
                            _t_lmk_dwp=pred_kps3d[im_idx, :,:2]
                            _landmark_dwp=gt_lmk_2d['keypoints'][im_idx, ...].detach().cpu().numpy()
                            
                            # Draw lines connecting corresponding landmarks
                            # Draw body keypoints (always)
                            for kp_idx in body_kpt2d_indices + knee_feet_indices:
                                pt1 = tuple(_landmark_dwp[kp_idx].astype(int))
                                pt2 = tuple(_t_lmk_dwp[kp_idx].detach().cpu().numpy().astype(int))
                                cv2.line(_img, pt1, pt2, (255, 255, 0), 1)  # Yellow lines
                                cv2.putText(_img, str(kp_idx), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                            
                            # Draw face keypoints (only if face is valid)
                            if head_lmk_valid[im_idx]:
                                for kp_idx in face_kpt2d_indices:
                                    pt1 = tuple(_landmark_dwp[kp_idx].astype(int))
                                    pt2 = tuple(_t_lmk_dwp[kp_idx].detach().cpu().numpy().astype(int))
                                    cv2.line(_img, pt1, pt2, (255, 200, 0), 1)  # Orange lines
                                    cv2.putText(_img, str(kp_idx), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 0), 1)
                            
                            # Draw left hand keypoints (only if left hand is valid)
                            if left_hand_valid[im_idx]:
                                for kp_idx in lhand_kpt2d_indices:
                                    pt1 = tuple(_landmark_dwp[kp_idx].astype(int))
                                    pt2 = tuple(_t_lmk_dwp[kp_idx].detach().cpu().numpy().astype(int))
                                    cv2.line(_img, pt1, pt2, (0, 255, 255), 1)  # Cyan lines
                                    cv2.putText(_img, str(kp_idx), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                            
                            # Draw right hand keypoints (only if right hand is valid)
                            if right_hand_valid[im_idx]:
                                for kp_idx in rhand_kpt2d_indices:
                                    pt1 = tuple(_landmark_dwp[kp_idx].astype(int))
                                    pt2 = tuple(_t_lmk_dwp[kp_idx].detach().cpu().numpy().astype(int))
                                    cv2.line(_img, pt1, pt2, (255, 0, 255), 1)  # Magenta lines
                                    cv2.putText(_img, str(kp_idx), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)

                            
                            _img=draw_landmarks(_landmark_dwp,_img,color=(0, 255, 0),viz_index=False)
                            _img=draw_landmarks(_t_lmk_dwp,_img,color=(255, 0, 0))
                            _img=draw_landmarks(proj_face_lmk_203[im_idx,:,:2],_img,color=(0, 0, 255))
                            t_camera=GS_Camera(**self.build_cameras_kwargs(1,self.body_focal_length),R=R[None,im_idx],T=T[None,im_idx])
                            mesh_img=self.body_renderer.render_mesh(smplx_dict['vertices'][None,im_idx,...],t_camera,lights=lights) 
                            mesh_img  = (mesh_img[:,:3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1,2,0)

                            mesh_img=cv2.addWeighted(_img,0.3,mesh_img,0.7,0)
                            img_hand=batch_imgs[im_idx].clone().numpy().transpose(1,2,0)
                            img_hand = draw_landmarks(ref_hand_l_joints[im_idx,:,:2], img_hand, color=(0, 0, 255))#left 3d prior
                            img_hand = draw_landmarks(ref_hand_r_joints[im_idx,:,:2], img_hand, color=(0, 0, 255)) #right 3d prior
                            img_hand = draw_landmarks(_landmark_dwp[-42:,:],img_hand,color=(0, 255, 0))#green 2d prior
                            img_hand = draw_landmarks(_t_lmk_dwp[-42:,:],img_hand,color=(255, 0, 0))
                            if head_lmk_valid[im_idx]:
                                img_hand=draw_landmarks(ref_head_vertices[im_idx,:,:2],img_hand,color=(255, 0, 255),radius=1)
                            if left_hand_valid[im_idx]:
                                img_hand=draw_landmarks(ref_hand_l_vertices[:, self.ehm.mano.selected_vert_ids][im_idx,:,:2],img_hand,color=(255, 0, 255),radius=1)
                            if right_hand_valid[im_idx]:
                                img_hand=draw_landmarks(ref_hand_r_vertices[:, self.ehm.mano.selected_vert_ids][im_idx,:,:2],img_hand,color=(255, 0, 255),radius=1)
                            _img = np.concatenate((_img,mesh_img, img_hand), axis=1)
                            vis_imgs.append(_img)

                            if len(vis_imgs) > 0:
                                vis_img = np.concatenate(vis_imgs, axis=0)
                                cv2.imwrite(os.path.join(save_path,f"vis_fit_smplx_bid-{batch_id}_stp-{i_step}_grp-{group_start:04d}.png"), 
                                            cv2.cvtColor(vis_img.copy(), cv2.COLOR_RGB2BGR))

        batch_smplx['camera_RT_params'][:, :3, 3] = gl_T.detach()
        # batch_smplx['camera_RT_params'][:, :3, :3] = gl_R.detach()
        batch_smplx['camera_RT_params'][:, :3, :3] = rotation_6d_to_matrix(gl_R_6d.detach())

        optim_smplx_results = {}
        optim_mano_left_results={}
        optim_mano_right_results={}
        optim_flame_results={}

        id_share_parms['smplx_shape'] = g_smplx_shape.detach().float().cpu().numpy()
        id_share_parms['head_scale'] = head_scale.detach().float().cpu().numpy()
        id_share_parms['hand_scale'] = hand_scale.detach().float().cpu().numpy()
        id_share_parms['joints_offset'] = joints_offset.detach().float().cpu().numpy()
        id_share_parms['flame_shape'] = g_flame_shape.detach().float().cpu().numpy()

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

        return optim_smplx_results,id_share_parms

    def run(self, tracked_rlt, id_share_rlt, optim_cfg, lmdb_engine=None, interval=1):
        mini_batch_size = optim_cfg.get('mini_batch_size', 1024)
        share_id = optim_cfg.get('share_id', True)
        if mini_batch_size > 0:
            mini_batchs = build_minibatch(list(tracked_rlt.keys()), share_id=share_id, batch_size=mini_batch_size)
        else:
            mini_batchs = [list(tracked_rlt.keys())]
        optim_results = {}
        id_share_rlt['head_scale']=np.array([[1.0,1.0,1.0]],dtype=np.float32)
        id_share_rlt['hand_scale']=np.array([[1.0,1.0,1.0]],dtype=np.float32)
        id_share_rlt['joints_offset']=np.zeros((1,55,3),dtype=np.float32)
        for batch_id, mini_batch in enumerate(mini_batchs):
            mini_batch_body_imgs=[lmdb_engine[f'{key}/body_image'] for key in mini_batch] if lmdb_engine is not None else None
            mini_batch_flame_lmk = [tracked_rlt[key] for key in mini_batch]
            mini_batch_flame_lmk = torch.utils.data.default_collate(mini_batch_flame_lmk)
            mini_batch_flame_lmk = data_to_device(mini_batch_flame_lmk, device=self.device)
            optim_result, id_share_rlt = self.optimize(
                mini_batch, mini_batch_flame_lmk, id_share_rlt, optim_cfg, 
                batch_id=batch_id, batch_imgs=mini_batch_body_imgs, interval=interval
            )
            
            optim_results.update(optim_result)
        return optim_results,id_share_rlt


def data_to_device(data_dict, device='cuda'):
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

def get_2d_keypoints_weight(body_lmk_score):
    weight = body_lmk_score.clone()
    weight[weight <= 0.3] *= 0
    weight[weight > 0.3]  *= 2

    weight[:,-42:]=(body_lmk_score[:,-42:]>0.7)*weight[:,-42:]
    kps_w = torch.from_numpy(smplx_to_dwpose()[1]).unsqueeze(0).to(body_lmk_score.device)
    weight = weight * kps_w
    weight[:,[9,10,12,13,18,19,20,21,22,23]]=(body_lmk_score[:,[9,10,12,13,18,19,20,21,22,23]]>0.87)*weight[:,[9,10,12,13,18,19,20,21,22,23]]
    weight = weight.unsqueeze(-1).float()
    weight_keep = torch.zeros_like(weight)
    weight_keep[weight < 0.2] = 10
    return weight, weight_keep

