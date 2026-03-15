# Sapiens Pose Detector
# Wraps Meta's Sapiens 1B wholebody pose model with yolox person detection
# Outputs 133 COCO Wholebody keypoints with per-keypoint confidence scores

import os
import numpy as np
import cv2
import torch
import onnxruntime as ort

from .pose_utils import udp_decode, top_down_affine_transform, get_udp_warp_matrix
from .constants import (
    NUM_KEYPOINTS, BODY_INDICES, FEET_INDICES, FACE_INDICES,
    LHAND_INDICES, RHAND_INDICES
)
from ...utils.timer import Timer
from ...utils.rprint import rlog as log


class SapiensDetector:
    """Sapiens wholebody pose estimator with yolox person detection.
    
    Produces 133 COCO Wholebody keypoints with confidence scores.
    Uses yolox for person detection and Sapiens TorchScript model for pose.
    
    Args:
        ckpt_path: Path to Sapiens TorchScript .pt2 checkpoint
        yolox_ckpt_dir: Directory containing yolox_l.onnx
        input_shape: Model input size as [height, width], default [1024, 768]
        num_keypoints: Number of keypoints (133 for wholebody)
        heatmap_scale: Downscale factor from input to heatmap, default 4
        device: CUDA device string
    """
    
    # Normalization constants for Sapiens (BGR order reversed to RGB)
    MEAN = np.array([123.5, 116.5, 103.5], dtype=np.float32)
    STD = np.array([58.5, 57.0, 57.5], dtype=np.float32)
    
    def __init__(self, ckpt_path, yolox_ckpt_dir, 
                 input_shape=None, num_keypoints=133,
                 heatmap_scale=4, device='cuda:0'):
        if input_shape is None:
            input_shape = [1024, 768]
        
        self.ckpt_path = ckpt_path
        self.num_keypoints = num_keypoints
        self.heatmap_scale = heatmap_scale
        self.device = device
        self.input_shape = tuple(input_shape)  # (H, W)
        self.timer = Timer()
        
        # Load yolox person detector (ONNX)
        yolox_path = os.path.join(yolox_ckpt_dir, 'yolox_l.onnx')
        providers = ['CUDAExecutionProvider'] if 'cuda' in device else ['CPUExecutionProvider']
        self.yolox_session = ort.InferenceSession(
            path_or_bytes=yolox_path, providers=providers)
        
        # Load Sapiens pose model (TorchScript)
        print(f"  Loading Sapiens model from: {ckpt_path}")
        use_torchscript = '_torchscript' in ckpt_path or ckpt_path.endswith('.pt2')
        if use_torchscript:
            self.pose_model = torch.jit.load(ckpt_path, map_location=device)
            self.dtype = torch.float32
        else:
            self.pose_model = torch.export.load(ckpt_path).module()
            self.pose_model = self.pose_model.to(torch.bfloat16).to(device)
            self.dtype = torch.bfloat16
        
        self.pose_model.eval()
        print(f"  Sapiens model loaded (input: {self.input_shape}, "
              f"keypoints: {self.num_keypoints}, device: {device})")
    
    def warmup(self):
        """Run warmup inference on a dummy image."""
        self.timer.tic()
        self(np.zeros((512, 512, 3), dtype=np.uint8))
        elapse = self.timer.toc()
        log(f'SapiensDetector warmup time: {elapse:.3f}s')
    
    def _detect_person(self, oriImg):
        """Detect person bounding boxes using yolox.
        
        Returns:
            np.ndarray: Person bboxes as [[x1,y1,x2,y2], ...] or empty array
        """
        from ..dwpose.onnxdet import inference_detector
        return inference_detector(self.yolox_session, oriImg)
    
    def _get_largest_bbox(self, bboxes):
        """Select the largest bbox by area."""
        if len(bboxes) == 0:
            return None
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        return bboxes[np.argmax(areas)]
    
    def _preprocess(self, img, bbox):
        """Preprocess image for Sapiens inference.
        
        Args:
            img: RGB image (H, W, 3), np.uint8
            bbox: Person bbox [x1, y1, x2, y2]
            
        Returns:
            tuple: (tensor, center, scale) where tensor is (1, 3, H, W)
        """
        # Affine transform to crop person region and resize to input_shape
        img_warped, center, scale = top_down_affine_transform(
            img.copy(), bbox, self.input_shape, padding=1.25)
        
        # Convert to float, BGR reorder, normalize
        img_t = img_warped.astype(np.float32)
        img_t = img_t[:, :, ::-1].copy()  # RGB -> BGR
        img_t = (img_t - self.MEAN) / self.STD
        
        # To tensor: (H, W, 3) -> (1, 3, H, W) 
        img_t = torch.from_numpy(img_t.transpose(2, 0, 1)).unsqueeze(0)
        img_t = img_t.to(dtype=self.dtype, device=self.device)
        
        return img_t, center, scale
    
    def _inference(self, img_tensor):
        """Run Sapiens model inference.
        
        Args:
            img_tensor: (1, 3, H, W) tensor
            
        Returns:
            np.ndarray: Heatmaps (K, Hh, Hw)
        """
        with torch.no_grad():
            if self.dtype == torch.float32:
                heatmaps = self.pose_model(img_tensor)
            else:
                with torch.autocast(device_type='cuda', dtype=self.dtype):
                    heatmaps = self.pose_model(img_tensor)
        
        return heatmaps[0].cpu().float().numpy()  # (K, Hh, Hw)
    
    def _decode_and_transform(self, heatmaps, center, scale):
        """Decode heatmaps to keypoints and transform to original image coords.
        
        Args:
            heatmaps: (K, Hh, Hw) numpy array
            center: Bbox center from affine transform
            scale: Bbox scale from affine transform
            
        Returns:
            tuple: (keypoints (133, 2), scores (133,))
        """
        input_h, input_w = self.input_shape
        hm_h = input_h // self.heatmap_scale
        hm_w = input_w // self.heatmap_scale
        
        keypoints, scores = udp_decode(
            heatmaps,
            input_size=(input_w, input_h),
            heatmap_size=(hm_w, hm_h),
        )
        # keypoints: (1, K, 2), scores: (1, K)
        keypoints = keypoints[0]  # (K, 2)
        scores = scores[0]        # (K,)
        
        # Transform keypoints back to original image coordinates
        # keypoints are in input_shape space, need to map back via inverse of affine
        # Original formula: keypoints_orig = (keypoints / input_size) * scale + center - 0.5 * scale
        keypoints = (keypoints / np.array([input_w, input_h])) * scale + center - 0.5 * scale
        
        return keypoints, scores

    # COCO Wholebody 133 left-right swap pairs: (left_idx, right_idx)
    _LR_PAIRS = (
        # Body: eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
        [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)] +
        # Feet: bigtoe, smalltoe, heel
        [(17, 20), (18, 21), (19, 22)] +
        # Face contour: 23-39 symmetric around center (31)
        [(23+i, 39-i) for i in range(8)] +
        # Face eyebrows: right 40-44 ↔ left 45-49
        [(40+i, 49-i) for i in range(5)] +
        # Face nose horizontal: 55↔58, 56↔57
        [(55, 58), (56, 57)] +
        # Face eyes: right 59-64 ↔ left 65-70
        [(59, 65), (60, 70), (61, 69), (62, 68), (63, 67), (64, 66)] +
        # Face mouth outer: 71↔77, 72↔76, 73↔75, 78↔82, 79↔81
        [(71, 77), (72, 76), (73, 75), (78, 82), (79, 81)] +
        # Face mouth inner: 83↔87, 84↔86, 88↔90
        [(83, 87), (84, 86), (88, 90)] +
        # Hands: left 91-111 ↔ right 112-132
        [(91+i, 112+i) for i in range(21)]
    )

    def _swap_lr(self, keypoints, scores):
        """Swap left/right keypoint pairs. Coordinates stay the same, indices swap."""
        kps = keypoints.copy()
        sc = scores.copy()
        for l, r in self._LR_PAIRS:
            kps[l], kps[r] = keypoints[r].copy(), keypoints[l].copy()
            sc[l], sc[r] = scores[r], scores[l]
        return kps, sc

    def __call__(self, oriImg, swap_lr=False):
        """Run Sapiens pose detection on an image.

        Args:
            oriImg: RGB image (H, W, 3), np.uint8
            swap_lr: If True, swap left/right keypoint assignments after detection.
                     Use for horizontally-flipped pshuman back-view images where
                     Sapiens detects good keypoints but L/R semantics are reversed.

        Returns:
            tuple: (det_info, raw_info)
            - det_info: dict with 'bodies', 'hands', 'faces', 'feet', 'bbox'
              (hands shape: (2, 21, 2), faces shape: (68, 2), feet shape: (6, 2))
            - raw_info: dict with 'keypoints' (133, 2), 'scores' (133,), 'bbox' (4,)
        """
        H, W = oriImg.shape[:2]

        # Detect person (always on the input image for best quality)
        bboxes = self._detect_person(oriImg)

        if len(bboxes) == 0:
            # Fallback: use full image as bbox
            bbox = np.array([0, 0, W, H], dtype=np.float32)
        else:
            bbox = self._get_largest_bbox(bboxes)

        # Preprocess
        img_tensor, center, scale = self._preprocess(oriImg, bbox)

        # Inference
        heatmaps = self._inference(img_tensor)

        # Decode
        keypoints, scores = self._decode_and_transform(heatmaps, center, scale)

        # For swap_lr: swap left/right keypoint pairs (coordinates stay the same)
        if swap_lr:
            keypoints, scores = self._swap_lr(keypoints, scores)
        
        # Build raw_info (133 keypoints in original image coords)
        raw_info = {
            'keypoints': keypoints.astype(np.float32),  # (133, 2)
            'scores': scores.astype(np.float32),          # (133,)
            'bbox': bbox.astype(np.float32),              # (4,)
        }
        
        # Build det_info with body parts split out
        # Apply visibility threshold for det_info components
        vis_threshold = 0.3
        
        # Face keypoints (indices 23-90) in original pixel coords
        face_kps = keypoints[FACE_INDICES].copy()  # (68, 2)
        face_scores = scores[FACE_INDICES]
        face_kps[face_scores < vis_threshold] = -1
        
        # Hand keypoints: (2, 21, 2) [left_hand, right_hand] in original pixel coords
        lhand_kps = keypoints[LHAND_INDICES].copy()  # (21, 2)
        rhand_kps = keypoints[RHAND_INDICES].copy()  # (21, 2)
        lhand_scores = scores[LHAND_INDICES]
        rhand_scores = scores[RHAND_INDICES]
        lhand_kps[lhand_scores < vis_threshold] = -1
        rhand_kps[rhand_scores < vis_threshold] = -1
        hands = np.stack([lhand_kps, rhand_kps], axis=0)  # (2, 21, 2)
        
        # Feet keypoints (indices 17-22)
        feet_kps = keypoints[FEET_INDICES].copy()  # (6, 2)
        
        det_info = {
            'faces': face_kps.astype(np.float32),
            'hands': hands.astype(np.float32),
            'feet': feet_kps.astype(np.float32),
            'bbox': bbox.astype(np.float32),
        }
        
        return det_info, raw_info
