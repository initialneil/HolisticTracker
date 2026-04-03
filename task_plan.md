# TEDWB1k Dataset Preparation — Task Plan

## Goal
Prepare the TEDWB1k video dataset for HolisticAvatar training using the HolisticTracker (ehm-tracker) pipeline with PIXIE body initialization and Sapiens keypoints.

## Dataset Overview
- **Source**: ~1490 TED talk videos, pre-split into shots with extracted frames and background mattes
- **Location**: `/home/szj/Datasets/TEDWB1k/`
  - `shots_images/` — extracted video frames per shot
  - `shots_rmbg2/` — background mattes (RMBG v2)
  - `shots_ehmx/` — tracking output (EHM: SMPL-X + FLAME)
  - `test.txt` — 70 test video identities

## Pipeline Stages

### Phase 1: Initial Pipeline Run (PIXIE + Sapiens) `[complete]`
- Ran 3-stage pipeline: `track_base` → `flame` → `smplx`
- Used `infer_ehmx_parallel.py` with `--body_estimator_type pixie --body_landmark_type sapiens`
- 8 GPUs, 2 workers per GPU (16 parallel)
- **Result**: 1431/1490 videos processed (59 had no index files)
- **Duration**: ~5 days

### Phase 2: Fix Bbox Smoothing Bug `[complete]`
- **Problem**: Temporal EMA smoothing on face/hand bounding boxes caused misalignment at shot boundaries and during fast motion. Smoothed bbox drifted from actual Sapiens keypoint positions.
- **Root cause**: `smooth_bboxes()` with alpha=0.7 applied across frames within a shot; first frame of each shot inherited stale momentum from prior frames.
- **Fix**: Completely disabled bbox smoothing — use raw Sapiens keypoint bboxes directly per-frame. Sapiens detection is accurate enough per-frame.
- **File**: `src/ehmx_track_base.py` — replaced smoothing block with direct assignment

### Phase 3: ONNX CUDA Fix `[complete]`
- **Problem**: All ONNX models (hamer, lmk70, teaser, yolox) silently fell back to CPU because `libcublasLt.so.12` was not on LD_LIBRARY_PATH. This made `track_base` CPU-bound (~1000% CPU per worker).
- **Root cause**: `onnxruntime-gpu 1.24.1` needs CUDA 12 libs; they exist at `site-packages/nvidia/cublas/lib/` but aren't on LD_LIBRARY_PATH by default.
- **Fix**: Added LD_LIBRARY_PATH injection in `infer_ehmx_parallel.py` subprocess env to include nvidia cublas/cudnn lib paths. Also downgraded to `onnxruntime-gpu 1.17.1` (CUDA 11 compatible).
- **File**: `infer_ehmx_parallel.py` — added nvidia lib path discovery in `run_video_pipeline()`

### Phase 4: Full Rerun (All 1431 Videos) `[complete]`
- Reran all 1431 videos with smoothing disabled + ONNX on GPU
- Used GPUs 0,1,2,4,6 (5 workers), reserving 3,5,7 for training
- **Duration**: ~6 days total (interrupted by system reboots, GPU contention with training)
- Progress tracking via `video_list_rerun_remaining.txt` (timestamp-based completion detection)

### Phase 5: Dataset Merge `[complete]`
- Ran `merge_ehmx_dataset.py` to combine per-video tracking into single files
- **Splits** (from `test.txt` with 70 identities):
  - Train: 1,341 videos → 203,618 frames
  - Val: 20 videos → 646 frames (limited to 2 shots/video)
  - Test: 70 videos → 2,880 frames (limited to 2 shots/video)
- **Output files** (in `/home/szj/Datasets/TEDWB1k/shots_ehmx/`):
  - `optim_tracking_ehm.pkl` — 6.85GB, all per-frame SMPL-X + FLAME params
  - `id_share_params.pkl` — per-video identity params (shape, scale, offsets)
  - `videos_info.json` — frame listings per video
  - `dataset_frames.json` — train/valid/test frame splits
  - `extra_info.json` — `frames_root` and `matte_root` paths
- **Key fix**: `extra_info.json` uses `frames_root`/`matte_root` keys matching `TrackedData` loader expectations

### Phase 6: right_hand_center Bug `[identified, impact assessed]`
- **Bug**: `EHM.py` line 27: `self.register_buffer('right_hand_center', left_hand_center)` — copies left hand center to right hand buffer
- **Impact**: Affects hand center computation in EHM forward pass, but tracking optimization may compensate. Dataset already generated with this bug — rerunning would require another full pipeline pass.
- **Status**: Bug fixed in code, but existing tracking data was generated with the bug present.

## Output Data Format

Per-frame tracking data (`optim_tracking_ehm.pkl`):
```
{video_id: {frame_key: {
    'smplx_coeffs': {
        'global_pose': (3,),        # axis-angle
        'body_pose': (21, 3),       # axis-angle per joint
        'left_hand_pose': (15, 3),
        'right_hand_pose': (15, 3),
        'exp': (50,),               # expression
        'body_cam': (3,),
        'camera_RT_params': (3, 4),
    },
    'flame_coeffs': {
        'pose_params': (3,),
        'jaw_params': (3,),
        'neck_pose_params': (3,),   # all zeros (not optimized)
        'eye_pose_params': (6,),    # non-zero, optimized
        'eyelid_params': (2,),
        'expression_params': (50,),
        'cam': (3,),
        'camera_RT_params': (3, 4),
    },
    'body_crop': {'M_o2c': (3,3), 'M_c2o': (3,3), ...},
    'head_crop': {'M_o2c': (3,3), 'M_c2o': (3,3)},
    'left_hand_crop': ...,
    'right_hand_crop': ...,
    'body_lmk_rlt': {'keypoints': (133,2), 'scores': (133,)},
    'dwpose_raw': {'keypoints': (133,2), 'scores': (133,), 'bbox': (4,)},
    'head_lmk_203': ..., 'head_lmk_70': ..., 'head_lmk_mp': ...,
    'left_mano_coeffs': ..., 'right_mano_coeffs': ...,
}}}
```

Per-video identity data (`id_share_params.pkl`):
```
{video_id: {
    'smplx_shape': (1, 200),
    'flame_shape': (1, 300),
    'left_mano_shape': (1, 10),
    'right_mano_shape': (1, 10),
    'head_scale': (1, 3),
    'hand_scale': (1, 3),
    'joints_offset': (1, 55, 3),
}}
```

## Git History
All changes committed and pushed to `https://github.com/initialneil/HolisticTracker` (main branch).

## Errors Encountered
| Error | Resolution |
|-------|------------|
| CUDA OOM with 4 workers/GPU | Reduced to 2 workers/GPU |
| 3 workers/GPU crashed machine (CPU RAM 125GB) | Reduced to 2 workers/GPU (~96GB total) |
| Disk full (1.9TB) killed all training | Cleaned old checkpoints, reduced keep_last_n to 2 |
| ONNX CUDA provider silent fallback to CPU | Added LD_LIBRARY_PATH for nvidia libs |
| NCCL timeout during DDP checkpoint saves | Switched training to single GPU per run |
| Bbox smoothing caused face crop misalignment | Disabled smoothing entirely |
| `extra_info.json` key mismatch | Changed `images_dir`→`frames_root`, `mattes_dir`→`matte_root` |
