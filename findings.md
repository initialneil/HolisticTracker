# TEDWB1k Dataset Preparation — Findings

## Hardware Constraints (8× RTX 3090, 125GB RAM)
- Max 2 workers/GPU for full pipeline (each ~7GB CPU RAM + ~15GB GPU)
- 3 workers/GPU exceeds 125GB CPU RAM → OOM kill
- ONNX on CPU saturates all cores with 5+ workers — GPU utilization drops to 0%
- DDP checkpoint saves can trigger NCCL timeout under tight RAM

## Pipeline Timing (per video, ~150 frames avg)
- track_base: ~5 min (Sapiens 1B + PIXIE + face/hand crop + visualization)
- flame: ~15 min (FLAME optimization, 2-stage)
- smplx: ~15 min (SMPL-X optimization, 2-stage)
- Total: ~35-40 min/video

## Bbox Smoothing Analysis
- Old approach: EMA smoothing (alpha=0.7) per-shot on face/hand bboxes
- Problem: at shot boundaries, first frames inherit momentum from prior shot's bbox position
- Sapiens keypoint detection is accurate enough per-frame — smoothing unnecessary
- `check_crop_drift.py` can retroactively detect affected videos by comparing raw vs smoothed

## ONNX Runtime CUDA
- `onnxruntime-gpu 1.24.1` requires CUDA 12 libs but pt2.7 env has CUDA 11.8
- Libs exist at `site-packages/nvidia/cublas/lib/` but not on LD_LIBRARY_PATH
- Silent fallback to CPU with warning in stderr — easy to miss
- Fix: either set LD_LIBRARY_PATH or use onnxruntime-gpu 1.17.1 (CUDA 11 native)

## Data Format Notes
- `eye_pose_params` (6,) in flame_coeffs: non-zero, range [-0.54, 0.53]
- `neck_pose_params` (3,) in flame_coeffs: all zeros (not optimized by pipeline)
- `smplx_coeffs` has no eye_pose — eyes only in flame_coeffs
- `right_hand_center` bug in EHM.py: copies left hand center — affects all processed data

## Disk Usage
- Merged tracking pkl: 6.85 GB
- Each checkpoint (training): ~16 GB
- 3 training runs × 2 checkpoints = ~96 GB
- Total disk needed: ~200 GB for outputs + checkpoints
