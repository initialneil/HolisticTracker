"""Check which videos have significant face crop drift due to bbox smoothing.

Re-computes raw Sapiens face bboxes from stored keypoints, applies the old
smoothing logic, and measures how much smoothing shifts each crop. Videos
with large shifts need rerunning with smoothing disabled.
"""
import pickle
import numpy as np
import os
import sys
from multiprocessing import Pool
from collections import OrderedDict


HEAD_KP_INDICES = [0, 1, 2, 3, 4]


def compute_raw_face_bbox(keypoints, scores):
    """Compute face bbox from raw Sapiens keypoints."""
    head_kps = keypoints[HEAD_KP_INDICES]
    head_scores = scores[HEAD_KP_INDICES]
    valid = head_scores >= 0.3
    valid_kps = head_kps[valid]

    if len(valid_kps) < 2:
        return None

    center = valid_kps.mean(axis=0)
    ear_valid = head_scores[3] >= 0.3 and head_scores[4] >= 0.3
    if ear_valid:
        head_width = np.linalg.norm(head_kps[3] - head_kps[4])
    else:
        head_width = max(
            valid_kps[:, 0].max() - valid_kps[:, 0].min(),
            valid_kps[:, 1].max() - valid_kps[:, 1].min()
        )
    head_size = max(head_width * 1.3, 50)
    return np.array([
        center[0] - head_size / 2,
        center[1] - head_size / 2,
        center[0] + head_size / 2,
        center[1] + head_size / 2,
    ], dtype=np.float32)


def smooth_bboxes(bboxes, alpha=0.7):
    """Reproduce the old EMA smoothing."""
    smoothed = []
    prev = None
    for bbox in bboxes:
        if bbox is None:
            smoothed.append(prev)  # carry forward
            continue
        if prev is None:
            prev = bbox.copy()
        else:
            prev = alpha * prev + (1 - alpha) * bbox
        smoothed.append(prev.copy())
    return smoothed


def bbox_center(bbox):
    if bbox is None:
        return None
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def bbox_size(bbox):
    if bbox is None:
        return 0
    return bbox[2] - bbox[0]


def check_video(args):
    """Check a single video for crop drift."""
    video_dir, video_id = args
    pkl_path = os.path.join(video_dir, 'base_tracking.pkl')
    if not os.path.exists(pkl_path):
        return video_id, 0, 0, 0, []

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception:
        return video_id, 0, 0, 0, []

    # Get frame keys in order
    frame_keys = sorted(data.keys())
    if not frame_keys:
        return video_id, 0, 0, 0, []

    # Compute raw face bboxes for all frames
    raw_bboxes = []
    for fk in frame_keys:
        dw = data[fk].get('dwpose_raw')
        if dw is None:
            raw_bboxes.append(None)
            continue
        raw_bboxes.append(compute_raw_face_bbox(dw['keypoints'], dw['scores']))

    # Group by shot (to reproduce per-shot smoothing)
    shot_groups = OrderedDict()
    for i, fk in enumerate(frame_keys):
        shot_id = fk.split('/')[0] if '/' in fk else ''
        if shot_id not in shot_groups:
            shot_groups[shot_id] = []
        shot_groups[shot_id].append(i)

    # Apply old smoothing per shot
    smoothed = [None] * len(frame_keys)
    for shot_id, indices in shot_groups.items():
        shot_raw = [raw_bboxes[i] for i in indices]
        shot_smoothed = smooth_bboxes(shot_raw, alpha=0.7)
        for j, i in enumerate(indices):
            smoothed[i] = shot_smoothed[j]

    # Compare raw vs smoothed
    drifts = []
    bad_frames = []
    for i, fk in enumerate(frame_keys):
        raw = raw_bboxes[i]
        sm = smoothed[i]
        if raw is None or sm is None:
            continue

        raw_c = bbox_center(raw)
        sm_c = bbox_center(sm)
        size = bbox_size(raw)
        if size < 1:
            continue

        drift = np.linalg.norm(raw_c - sm_c) / size
        drifts.append(drift)

        if drift > 0.15:
            bad_frames.append((fk, float(drift)))

    if not drifts:
        return video_id, 0, 0, 0, []

    max_drift = float(max(drifts))
    num_bad = len(bad_frames)

    return video_id, max_drift, num_bad, len(drifts), bad_frames[:10]


def main():
    ehmx_dir = sys.argv[1] if len(sys.argv) > 1 else '/home/szj/Datasets/TEDWB1k/shots_ehmx'
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    threshold_bad_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 3  # at least 3 bad frames

    video_ids = sorted([
        d for d in os.listdir(ehmx_dir)
        if os.path.isdir(os.path.join(ehmx_dir, d))
        and os.path.exists(os.path.join(ehmx_dir, d, 'base_tracking.pkl'))
    ])
    print(f"Checking {len(video_ids)} videos with {num_workers} workers...")

    tasks = [(os.path.join(ehmx_dir, vid), vid) for vid in video_ids]

    results = []
    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(check_video, tasks)):
            results.append(result)
            if (i + 1) % 200 == 0:
                print(f"  checked {i+1}/{len(video_ids)}")

    # Sort by number of bad frames descending
    results.sort(key=lambda x: x[2], reverse=True)

    rerun = []
    print(f"\n{'Video':<25} {'Bad/Total':>12} {'Max Drift':>10}  Worst Frames")
    print("-" * 90)
    for vid, max_drift, num_bad, total, bad_frames in results:
        if total == 0:
            continue
        if num_bad >= threshold_bad_frames:
            rerun.append(vid)
            worst = ', '.join(f'{f[0]}({f[1]:.2f})' for f in bad_frames[:3])
            print(f"  {vid:<23} {num_bad:>4}/{total:<6} {max_drift:>9.2f}  {worst}")

    print(f"\n=== Summary ===")
    print(f"Total videos checked: {len(video_ids)}")
    print(f"Videos needing rerun (>={threshold_bad_frames} frames with >15% drift): {len(rerun)}")

    # Stats on all videos
    all_bad = [r[2] for r in results if r[3] > 0]
    print(f"Distribution: 0 bad={sum(1 for b in all_bad if b==0)}, "
          f"1-2 bad={sum(1 for b in all_bad if 1<=b<=2)}, "
          f"3-5 bad={sum(1 for b in all_bad if 3<=b<=5)}, "
          f"6+ bad={sum(1 for b in all_bad if b>=6)}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video_list_rerun_crop.txt')
    with open(out_path, 'w') as f:
        for vid in sorted(rerun):
            f.write(vid + '\n')
    print(f"Rerun list saved to: {out_path}")


if __name__ == '__main__':
    main()
