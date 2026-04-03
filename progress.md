# TEDWB1k Dataset Preparation — Progress Log

## Session 1: Initial Pipeline Run (Mar 15-21, 2026)
- Tested 4 videos with PIXIE vs SMPLer-X — PIXIE results acceptable
- Added `--workers_per_gpu` arg to `infer_ehmx_parallel.py`
- Ran full pipeline on all ~1490 videos: 8 GPUs × 2 workers
- Handled SSL errors, CUDA OOM during model loading burst
- Result: 1431/1490 processed, 59 skipped (no index files)

## Session 2: Dataset Splits + Merge (Mar 22, 2026)
- Expanded `test.txt` from 22 to 70 video identities
- Rewrote `merge_ehmx_dataset.py` with test_list support
- Merge result: 1341 train / 20 val / 70 test
- Discovered `extra_info.json` key mismatch (images_dir vs frames_root)

## Session 3: Bbox Smoothing Fix (Mar 23-24, 2026)
- Investigated face misalignment in `b28brIs1OmM` — smoothing caused drift at shot boundaries
- Fixed: disabled all bbox smoothing, use raw Sapiens keypoints directly
- Verified fix on single video — much better results
- Started full rerun on GPUs 0,1,2,4,6

## Session 4: Full Rerun + ONNX Fix (Mar 25-31, 2026)
- Rerun interrupted by multiple system reboots
- Discovered ONNX models all falling back to CPU (libcublasLt.so.12 missing)
- Fixed: added LD_LIBRARY_PATH in subprocess env, downgraded onnxruntime-gpu to 1.17.1
- Completed all 1431 videos
- Final merge: 203,618 train / 646 val / 2,880 test frames
- Verified eye_pose_params are non-zero in merged data
- All changes committed and pushed to GitHub

## Key Files Modified
- `infer_ehmx_parallel.py` — workers_per_gpu, ONNX CUDA libs, better error reporting
- `infer_ehmx_track_base.py` — refactored for cleaner arg handling
- `src/ehmx_track_base.py` — disabled bbox smoothing
- `merge_ehmx_dataset.py` — test_list splits, frames_root/matte_root keys
- `check_crop_drift.py` — new: detect videos affected by smoothing drift
