# Heatmap decoding utilities for Sapiens pose estimation
# Ported from Meta's Sapiens codebase with UDP (Unbiased Data Processing) decoding

import math
import numpy as np
import cv2


def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate heatmap distribution with Gaussian.

    Args:
        heatmaps (np.ndarray[K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape

    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps


def get_heatmap_maximum(heatmaps: np.ndarray):
    """Get maximum response location and value from heatmaps.

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps, np.ndarray), 'heatmaps should be numpy.ndarray'
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, f'Invalid shape {heatmaps.shape}'

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.] = -1

    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)

    return locs, vals


def refine_keypoints_dark_udp(keypoints: np.ndarray, heatmaps: np.ndarray,
                              blur_kernel_size: int) -> np.ndarray:
    """Refine keypoint predictions using DARK-UDP sub-pixel refinement.

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)
        blur_kernel_size (int): The Gaussian blur kernel size

    Returns:
        np.ndarray: Refined keypoint coordinates in shape (N, K, D)
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    # modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50., heatmaps)
    np.log(heatmaps, heatmaps)

    heatmaps_pad = np.pad(
        heatmaps, ((0, 0), (1, 1), (1, 1)), mode='edge').flatten()

    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum('imn,ink->imk', hessian,
                                  derivative).squeeze()

    return keypoints


def udp_decode(heatmaps, input_size, heatmap_size, blur_kernel_size=11):
    """UDP decoding for keypoint location refinement.

    Args:
        heatmaps (np.ndarray[K, H, W]): model predicted heatmaps.
        input_size (tuple): input image size (W, H) or (size, size)
        heatmap_size (tuple): heatmap size (W, H)
        blur_kernel_size (int): Gaussian kernel size for modulation.

    Returns:
        tuple: (keypoints, scores) where keypoints shape is (1, K, 2) and scores is (1, K)
    """
    keypoints, scores = get_heatmap_maximum(heatmaps)
    # unsqueeze the instance dimension for single-instance results
    keypoints = keypoints[None]
    scores = scores[None]
    keypoints = refine_keypoints_dark_udp(
        keypoints, heatmaps, blur_kernel_size=blur_kernel_size)

    W, H = heatmap_size
    keypoints = (keypoints / [W - 1, H - 1]) * input_size
    return keypoints, scores


def get_udp_warp_matrix(center, scale, rot, output_size):
    """Calculate the affine transformation matrix under unbiased constraint.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (tuple): Size ([w, h]) of the output image.

    Returns:
        np.ndarray: A 2x3 transformation matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    rot_rad = np.deg2rad(rot)
    warp_mat = np.zeros((2, 3), dtype=np.float32)
    scale_x = (output_size[0] - 1) / scale[0]
    scale_y = (output_size[1] - 1) / scale[1]
    warp_mat[0, 0] = math.cos(rot_rad) * scale_x
    warp_mat[0, 1] = -math.sin(rot_rad) * scale_x
    warp_mat[0, 2] = scale_x * (-0.5 * center[0] * 2 * math.cos(rot_rad) +
                                0.5 * center[1] * 2 * math.sin(rot_rad) +
                                0.5 * scale[0])
    warp_mat[1, 0] = math.sin(rot_rad) * scale_y
    warp_mat[1, 1] = math.cos(rot_rad) * scale_y
    warp_mat[1, 2] = scale_y * (-0.5 * center[0] * 2 * math.sin(rot_rad) -
                                0.5 * center[1] * 2 * math.cos(rot_rad) +
                                0.5 * scale[1])
    return warp_mat


def top_down_affine_transform(img, bbox, input_shape, padding=1.25):
    """Crop and resize image around bbox with aspect-ratio-preserving padding.

    Args:
        img (np.ndarray): Image to transform (H, W, 3).
        bbox (np.ndarray): Bounding box [x1, y1, x2, y2].
        input_shape (tuple): Target size (height, width).
        padding (float): Padding factor for bbox expansion.

    Returns:
        tuple: (transformed_img, center, scale)
    """
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    target_h, target_w = input_shape
    aspect_ratio = target_w / target_h

    # reshape bbox to fixed aspect ratio
    box_w, box_h = np.hsplit(scale, [1])
    scale = np.where(box_w > box_h * aspect_ratio,
                     np.hstack([box_w, box_w / aspect_ratio]),
                     np.hstack([box_h * aspect_ratio, box_h]))

    rot = 0.
    warp_mat = get_udp_warp_matrix(
        center, scale, rot, output_size=(target_w, target_h))

    img_warped = cv2.warpAffine(
        img, warp_mat, (target_w, target_h), flags=cv2.INTER_LINEAR)

    return img_warped, center, scale
