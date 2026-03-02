# COCO Wholebody 133-keypoint definitions for Sapiens pose estimation
# Reference: https://github.com/open-mmlab/mmpose/blob/main/mmpose/datasets/datasets/wholebody/

import numpy as np

NUM_KEYPOINTS = 133

# Index ranges for body parts
BODY_INDICES = list(range(0, 17))       # 17 body keypoints
FEET_INDICES = list(range(17, 23))       # 6 feet keypoints
FACE_INDICES = list(range(23, 91))       # 68 face keypoints (17 contour + 51 inner)
FACE_CONTOUR_INDICES = list(range(23, 40))  # 17 face contour
FACE_INNER_INDICES = list(range(40, 91))    # 51 face inner landmarks
LHAND_INDICES = list(range(91, 112))    # 21 left hand keypoints
RHAND_INDICES = list(range(112, 133))   # 21 right hand keypoints

COCO_WHOLEBODY_KEYPOINT_NAMES = [
    # Body (0-16)
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    # Feet (17-22)
    'left_bigtoe', 'left_smalltoe', 'left_heel',
    'right_bigtoe', 'right_smalltoe', 'right_heel',
    # Face contour (23-39)
    'right_contour_1', 'right_contour_2', 'right_contour_3', 'right_contour_4',
    'right_contour_5', 'right_contour_6', 'right_contour_7', 'right_contour_8',
    'contour_middle',
    'left_contour_8', 'left_contour_7', 'left_contour_6', 'left_contour_5',
    'left_contour_4', 'left_contour_3', 'left_contour_2', 'left_contour_1',
    # Face inner (40-90): eyebrows, nose, eyes, mouth
    'right_eyebrow_1', 'right_eyebrow_2', 'right_eyebrow_3',
    'right_eyebrow_4', 'right_eyebrow_5',
    'left_eyebrow_5', 'left_eyebrow_4', 'left_eyebrow_3',
    'left_eyebrow_2', 'left_eyebrow_1',
    'nosebridge_1', 'nosebridge_2', 'nosebridge_3', 'nosebridge_4',
    'right_nose_2', 'right_nose_1', 'nose_middle', 'left_nose_1', 'left_nose_2',
    'right_eye_1', 'right_eye_2', 'right_eye_3',
    'right_eye_4', 'right_eye_5', 'right_eye_6',
    'left_eye_4', 'left_eye_3', 'left_eye_2',
    'left_eye_1', 'left_eye_6', 'left_eye_5',
    'right_mouth_1', 'right_mouth_2', 'right_mouth_3',
    'mouth_top', 'left_mouth_3', 'left_mouth_2', 'left_mouth_1',
    'left_mouth_5', 'left_mouth_4', 'mouth_bottom',
    'right_mouth_4', 'right_mouth_5',
    'right_lip_1', 'right_lip_2', 'lip_top', 'left_lip_2', 'left_lip_1',
    'left_lip_3', 'lip_bottom', 'right_lip_3',
    # Left hand (91-111)
    'left_hand_root',
    'left_thumb_1', 'left_thumb_2', 'left_thumb_3', 'left_thumb',
    'left_index_1', 'left_index_2', 'left_index_3', 'left_index',
    'left_middle_1', 'left_middle_2', 'left_middle_3', 'left_middle',
    'left_ring_1', 'left_ring_2', 'left_ring_3', 'left_ring',
    'left_pinky_1', 'left_pinky_2', 'left_pinky_3', 'left_pinky',
    # Right hand (112-132)
    'right_hand_root',
    'right_thumb_1', 'right_thumb_2', 'right_thumb_3', 'right_thumb',
    'right_index_1', 'right_index_2', 'right_index_3', 'right_index',
    'right_middle_1', 'right_middle_2', 'right_middle_3', 'right_middle',
    'right_ring_1', 'right_ring_2', 'right_ring_3', 'right_ring',
    'right_pinky_1', 'right_pinky_2', 'right_pinky_3', 'right_pinky',
]

# Skeleton connectivity for visualization (body only)
BODY_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # head
    (5, 6),                                 # shoulders
    (5, 7), (7, 9),                         # left arm
    (6, 8), (8, 10),                        # right arm
    (5, 11), (6, 12),                       # torso
    (11, 12),                               # hips
    (11, 13), (13, 15),                     # left leg
    (12, 14), (14, 16),                     # right leg
    (15, 17), (15, 18), (15, 19),           # left foot
    (16, 20), (16, 21), (16, 22),           # right foot
]

# Hand skeleton connectivity (relative to hand root at 0)
HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
]

# Keypoint colors (BGR) for visualization
BODY_COLORS = [
    (0, 255, 0)] * 17 + [(0, 200, 200)] * 6  # body green, feet yellow-ish

FACE_COLORS = [(255, 255, 255)] * 68  # white

HAND_COLORS = [(0, 128, 255)] * 21 + [(255, 128, 0)] * 21  # left orange, right blue
