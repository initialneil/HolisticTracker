# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
import numpy as np
import torch
import torch.nn as nn

def to_tensor(tensor, dtype=torch.float32):
    if torch.Tensor == type(tensor):
        return tensor.clone().detach()
    else:
        return torch.tensor(tensor, dtype)


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


def max_grad_change(grad_arr):
    return grad_arr.abs().max()


class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)


class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist


def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))


def smplx_to_dwpose():
    body_mapping = np.array([68, 12, 17, 19, 21, 16, 18, 20, 2, 5,
                             8, 1, 4, 7, 24, 23, 107, 122],
                         dtype=np.int32)
    mapping = [body_mapping]

    lfoot_mapping = np.array([124, 132, 10], dtype=np.int32)
    rfoot_mapping = np.array([135, 143, 11], dtype=np.int32)
    
    mapping += [lfoot_mapping, rfoot_mapping]

    face_contour_mapping = np.arange(106, 106 + 17, dtype=np.int32)
    face_mapping = np.arange(55, 55 + 51, dtype=np.int32)
    mapping += [face_contour_mapping, face_mapping]

    lhand_mapping = np.array([20, 37, 38, 39, 133, 
                                  25, 26, 27, 128, 
                                  28, 29, 30, 129, 
                                  34, 35, 36, 131,
                                  31, 32, 33, 130], dtype=np.int32)
    rhand_mapping = np.array([21, 52, 53, 54, 144, 
                                  40, 41, 42, 139,
                                  43, 44, 45, 140, 
                                  49, 50, 51, 142, 
                                  46, 47, 48, 141], dtype=np.int32)

    mapping += [lhand_mapping, rhand_mapping]

    weights = np.ones([134])
    # weights_0  = np.array([0, 1, 14, 15, 16 ,17], dtype=np.int32)
    weights_0  = np.array([0, 14, 15, 16 ,17], dtype=np.int32)
    weights_5  = np.arange(24, 41, dtype=np.int32)
    weights_10 = np.arange(92, 134, dtype=np.int32)
    # weights_20 = np.array([3, 4, 6, 7], dtype=np.int32)
    weights_20 = np.array([3, 4, 6, 7,9,10,12,13,18,19,20,21,22,23], dtype=np.int32)
    weights_35 = np.array([2,5], dtype=np.int32)
    weights_50 = np.array([3, 6], dtype=np.int32)
    weights[weights_0] = 0
    weights[weights_5] = 5
    weights[weights_10] = 10
    weights[weights_20] = 20
    weights[weights_35] = 50
    weights[weights_50] = 100

    return np.concatenate(mapping), weights


def smplx_joints_to_dwpose(joints3d):
    body_mapping = np.array([68, 12, 17, 19, 21, 16, 18, 20, 2, 5,
                             8, 1, 4, 7, 24, 23, 107, 122],
                         dtype=np.int32)
    mapping = [body_mapping]

    lfoot_mapping = np.array([124, 132, 10], dtype=np.int32)
    rfoot_mapping = np.array([135, 143, 11], dtype=np.int32)
    
    mapping += [lfoot_mapping, rfoot_mapping]

    face_contour_mapping = np.arange(106, 106 + 17, dtype=np.int32)
    face_mapping = np.arange(55, 55 + 51, dtype=np.int32)
    mapping += [face_contour_mapping, face_mapping]

    lhand_mapping = np.array([20, 37, 38, 39, 133, 
                                  25, 26, 27, 128, 
                                  28, 29, 30, 129, 
                                  34, 35, 36, 131,
                                  31, 32, 33, 130], dtype=np.int32)
    rhand_mapping = np.array([21, 52, 53, 54, 144, 
                                  40, 41, 42, 139,
                                  43, 44, 45, 140, 
                                  49, 50, 51, 142, 
                                  46, 47, 48, 141], dtype=np.int32)

    mapping += [lhand_mapping, rhand_mapping]
    weights = np.ones([134])
    weights_0  = np.array([0, 1, 14, 15, 16 ,17], dtype=np.int32)
    weights_5  = np.arange(24, 41, dtype=np.int32)
    weights_10 = np.arange(92, 134, dtype=np.int32)
    weights_20 = np.array([3, 4, 6, 7], dtype=np.int32)
    weights_50 = np.array([3, 6], dtype=np.int32)
    weights[weights_0] = 0
    weights[weights_5] = 5
    weights[weights_10] = 10
    weights[weights_20] = 20
    weights[weights_50] = 100

    mapping = np.concatenate(mapping)
    ret_kps3d = joints3d[:, mapping]
    ret_kps3d[:, 1, :2] = (ret_kps3d[:, 2, :2] + ret_kps3d[:, 5, :2]) / 2

    return ret_kps3d, weights


def _sapiens_mapping():
    """Build COCO Wholebody 133-keypoint mapping from EHM 145 joints.

    Returns the index array (133,) where mapping[i] is the EHM joint index
    for COCO Wholebody keypoint i.
    """
    # Body (COCO 0-16)
    body_mapping = np.array([
        68,   # 0:  nose (face inner landmark)
        23,   # 1:  left_eye (SMPLX left_eye joint)
        24,   # 2:  right_eye (SMPLX right_eye joint)
        122,  # 3:  left_ear (face contour endpoint)
        107,  # 4:  right_ear (face contour endpoint)
        16,   # 5:  left_shoulder
        17,   # 6:  right_shoulder
        18,   # 7:  left_elbow
        19,   # 8:  right_elbow
        20,   # 9:  left_wrist
        21,   # 10: right_wrist
        1,    # 11: left_hip
        2,    # 12: right_hip
        4,    # 13: left_knee
        5,    # 14: right_knee
        7,    # 15: left_ankle
        8,    # 16: right_ankle
    ], dtype=np.int32)

    # Feet (COCO 17-22)
    feet_mapping = np.array([
        124,  # 17: left_bigtoe
        132,  # 18: left_smalltoe
        10,   # 19: left_heel (SMPLX left_foot)
        135,  # 20: right_bigtoe
        143,  # 21: right_smalltoe
        11,   # 22: right_heel (SMPLX right_foot)
    ], dtype=np.int32)

    # Face contour (COCO 23-39): 17 points, EHM 106-122
    face_contour_mapping = np.arange(106, 123, dtype=np.int32)

    # Face inner (COCO 40-90): 51 points, EHM 55-105
    face_inner_mapping = np.arange(55, 106, dtype=np.int32)

    # Left hand (COCO 91-111): 21 keypoints in openpose hand order
    lhand_mapping = np.array([
        20,   # 91:  wrist
        37, 38, 39, 133,    # thumb
        25, 26, 27, 128,    # index
        28, 29, 30, 129,    # middle
        34, 35, 36, 131,    # ring
        31, 32, 33, 130,    # pinky
    ], dtype=np.int32)

    # Right hand (COCO 112-132): 21 keypoints
    rhand_mapping = np.array([
        21,   # 112: wrist
        52, 53, 54, 144,    # thumb
        40, 41, 42, 139,    # index
        43, 44, 45, 140,    # middle
        49, 50, 51, 142,    # ring
        46, 47, 48, 141,    # pinky
    ], dtype=np.int32)

    return np.concatenate([body_mapping, feet_mapping, face_contour_mapping,
                           face_inner_mapping, lhand_mapping, rhand_mapping])


def _sapiens_weights():
    """Build per-keypoint optimization weights for COCO Wholebody 133 format."""
    weights = np.ones([133])
    weights[np.array([0, 1, 2, 3, 4])] = 0       # nose, eyes, ears
    weights[np.arange(23, 40)] = 5                  # face contour
    weights[np.arange(91, 133)] = 10                # hands
    weights[np.array([7, 8, 9, 10, 13, 14, 15, 16,
                      17, 18, 19, 20, 21, 22])] = 20  # limbs + feet
    weights[np.array([5, 6])] = 50                  # shoulders
    weights[np.array([7, 8])] = 100                 # elbows (highest)
    return weights


def smplx_to_sapiens():
    """Return (mapping[133], weights[133]) for COCO Wholebody 133-keypoint format.

    ``mapping[i]`` is the EHM-145 joint index for COCO Wholebody keypoint *i*.
    No synthetic joints are needed (unlike DWPose neck).
    """
    return _sapiens_mapping(), _sapiens_weights()


def smplx_joints_to_sapiens(joints3d):
    """Project EHM joints to COCO Wholebody 133-keypoint ordering.

    Args:
        joints3d: (B, 145, 3) EHM joint tensor

    Returns:
        ret_kps3d: (B, 133, 3) re-ordered keypoints
        weights:   (133,) per-keypoint weights
    """
    mapping = _sapiens_mapping()
    weights = _sapiens_weights()
    ret_kps3d = joints3d[:, mapping]
    return ret_kps3d, weights