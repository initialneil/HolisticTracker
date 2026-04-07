---
license: cc-by-nc-4.0
language:
- en
pretty_name: TEDWB1k
size_categories:
- 1K<n<10K
task_categories:
- other
tags:
- 3d-human
- smpl-x
- flame
- avatar
- gaussian-splatting
- video
- motion-capture
- ted
configs:
- config_name: subjects
  data_files:
  - split: train
    path: metadata/subjects_train.parquet
  - split: train_subset_x1
    path: metadata/subjects_train_subset_x1.parquet
  - split: train_subset_x12
    path: metadata/subjects_train_subset_x12.parquet
  - split: train_val
    path: metadata/subjects_train_val.parquet
  - split: test
    path: metadata/subjects_test.parquet
---

# TEDWB1k

**1,431 TED-talk speaker videos with per-frame SMPL-X + FLAME tracking, ready for 3D human / avatar research.**

TEDWB1k is a tracked subset of TED talks built with the [HolisticTracker](https://github.com/initialneil/HolisticTracker) (`ehm-tracker`) pipeline. Every subject is shot-segmented, background-matted, and fitted with whole-body SMPL-X (body + hands), FLAME (face + jaw + eyes), and per-shot whole-image keypoints. It is the dataset used to train the [HolisticAvatar](https://github.com/initialneil/HolisticAvatar) feed-forward Gaussian avatar model.

> **License caveat:** TED talks on ted.com are CC-BY-NC-ND. This dataset is released under **CC-BY-NC-4.0** — non-commercial research only. If you use it, you agree not to redistribute or use it commercially.

## At a glance

| Split | Subjects | Approx download | Notes |
|---|---:|---:|---|
| `train_subset_x1` | 1 | ~80 MB | tiny single-subject overfit (⊂ `train`) |
| `train_subset_x12` | 12 | ~1 GB | 12-subject overfit (⊂ `train`) |
| `train_val` | 20 | ~2 GB | monitored during training (⊂ `train`) |
| `test` | 70 | ~10 GB | identity-disjoint evaluation set |
| `train` | 1,361 | ~190 GB | full training pool |
| **total** | **1,431** | ~200 GB | |

`train` (1,361) and `test` (70) are identity-disjoint and together cover all 1,431 subjects. `train_subset_x1`, `train_subset_x12`, and `train_val` are all subsets of `train` — `train_val` are the 20 subjects whose frames the original training run reserved for validation monitoring (see `dataset_frames.json`); the small overfit subsets are intended for debugging.

The HF Dataset Viewer above renders one row per subject with a thumbnail of the final tracked SMPL-X overlay (`track_smplx.jpg`) and the per-subject frame and shot counts. Switch between the 5 split tabs to browse each subset.

## Quick start

```bash
pip install huggingface_hub

# Smallest possible test (1 subject):
python load_tedwb1k.py --split train_subset_x1 --out ./tedwb1k_x1

# 12-subject overfit set:
python load_tedwb1k.py --split train_subset_x12 --out ./tedwb1k_x12

# 20-subject training-monitor set (subset of train):
python load_tedwb1k.py --split train_val --out ./tedwb1k_train_val

# 70-subject test set:
python load_tedwb1k.py --split test --out ./tedwb1k_test

# Full training pool (1361 subjects):
python load_tedwb1k.py --split train --out ./tedwb1k_train
```

`load_tedwb1k.py` is included in this repo (or grab it from `ehm-tracker/release/load_tedwb1k.py`). It downloads only the matching subjects, merges per-subject tracking pickles into the format HolisticAvatar's `TrackedData` expects, extracts frames + mattes, and writes a fresh `extra_info.json` with absolute paths to the user's local data dir.

After it finishes, point your training config at `--out`:

```yaml
DATASET:
  data_path: ./tedwb1k_test
```

…and you can train / fine-tune HolisticAvatar with **zero code changes** to that codebase.

## Repository layout

```
TEDWB1k/
├── README.md                          this file
├── train.txt                          1,361 subject ids (full training pool)
├── train_subset_x1.txt                1 subject id (single-subject overfit, ⊂ train)
├── train_subset_x12.txt               12 subject ids (small overfit, ⊂ train)
├── train_val.txt                      20 subject ids (training monitor, ⊂ train)
├── test.txt                           70 subject ids (identity-disjoint evaluation)
├── dataset_frames.json                frame-level train/valid/test split used by HolisticAvatar
├── metadata/
│   ├── subjects_train.parquet            per-split subject tables w/ embedded thumbnails (HF Viewer)
│   ├── subjects_train_subset_x1.parquet
│   ├── subjects_train_subset_x12.parquet
│   ├── subjects_train_val.parquet
│   ├── subjects_test.parquet
│   ├── subjects.csv                      all rows in one CSV (programmatic use)
│   ├── skipped.txt                       (empty for the public release)
│   └── previews/<id>.jpg                 512 px thumbnails of track_smplx.jpg
└── subjects/<video_id>/
    ├── tracking/
    │   ├── optim_tracking_ehm.pkl     per-frame SMPL-X + FLAME parameters
    │   ├── id_share_params.pkl        per-video shape / scale / joint offsets
    │   └── videos_info.json           frame-key listing for this video
    ├── track_smplx.jpg                final SMPL-X overlay grid (~13 MB)
    ├── track_flame.jpg                intermediate FLAME overlay grid
    ├── track_base.jpg                 stage-1 PIXIE+Sapiens overlay grid
    ├── frames.tar                     per-shot RGB JPGs + audio.wav
    └── mattes.tar                     per-shot RMBG-v2 alpha mattes
```

Each `frames.tar` unpacks to:
```
<shot_id>/000000.jpg, 000001.jpg, ..., 0000NN.jpg
<shot_id>/audio.wav
<shot_id>.mp4               (the original shot clip, included)
```
where `<shot_id>` is `NNNNNN_NNNNNN` encoding `start_frame_end_frame` — the inclusive
keyframe indices of the shot inside the source TED talk, sampled at **0.5 fps** (one
keyframe every 2 seconds). For example `000015_000019` is keyframes 15..19 (5 frames,
covering seconds 30..38 in the source video). The JPGs inside the directory are
indexed locally per shot starting at `000000.jpg`.

## Tracking format

Per-frame data inside `optim_tracking_ehm.pkl` (after the loader merge, keyed by `{video_id: {frame_key: ...}}`):

```python
{
    'smplx_coeffs': {
        'global_pose':       (3,),       # axis-angle
        'body_pose':         (21, 3),    # axis-angle per joint
        'left_hand_pose':    (15, 3),
        'right_hand_pose':   (15, 3),
        'exp':               (50,),      # SMPL-X expression
        'body_cam':          (3,),
        'camera_RT_params':  (3, 4),
    },
    'flame_coeffs': {
        'pose_params':       (3,),
        'jaw_params':        (3,),
        'neck_pose_params':  (3,),       # all zero (not optimized)
        'eye_pose_params':   (6,),       # optimized
        'eyelid_params':     (2,),
        'expression_params': (50,),
        'cam':               (3,),
        'camera_RT_params':  (3, 4),
    },
    'body_crop':       {'M_o2c': (3,3), 'M_c2o': (3,3), ...},
    'head_crop':       {'M_o2c': (3,3), 'M_c2o': (3,3)},
    'left_hand_crop':  {...},
    'right_hand_crop': {...},
    'body_lmk_rlt':    {'keypoints': (133,2), 'scores': (133,)},
    'dwpose_raw':      {'keypoints': (133,2), 'scores': (133,), 'bbox': (4,)},
    'head_lmk_203':    {...},
    'head_lmk_70':     {...},
    'head_lmk_mp':     {...},
    'left_mano_coeffs':  {...},
    'right_mano_coeffs': {...},
}
```

Per-video identity data inside `id_share_params.pkl` (keyed by `{video_id: ...}`):

```python
{
    'smplx_shape':      (1, 200),
    'flame_shape':      (1, 300),
    'left_mano_shape':  (1, 10),
    'right_mano_shape': (1, 10),
    'head_scale':       (1, 3),
    'hand_scale':       (1, 3),
    'joints_offset':    (1, 55, 3),
}
```

## Pipeline

Tracking was produced by `ehm-tracker` (a fork of LHM_Track) in three stages:

1. **`track_base`** — per-frame perception:
   - **PIXIE** for SMPL-X body initialization
   - **Sapiens 1B** for 133 whole-body keypoints
   - **HaMeR** for per-hand MANO regression on left/right hand crops
   - **MediaPipe** FaceMesh for 478-point face landmarks
   - additional 70- and 203-point face landmark models for face fitting
   - face / hand crop computation from the keypoints
2. **`flame`** — 2-stage FLAME optimization for face, jaw, expression, eyes, eyelids.
3. **`smplx`** — 2-stage whole-body SMPL-X optimization (body, hands, expression) consistent with the FLAME face fit.

Each stage produces a sanity-check overlay grid (`track_base.jpg`, `track_flame.jpg`, `track_smplx.jpg`) that you can browse via the HF Dataset Viewer thumbnail column or in the per-subject directory.

## Known issues / caveats

Please read these before training — they affect what is and isn't reliable in the data.

- **`neck_pose_params` is all zero.** Not optimized by the pipeline; relying on neck rotation from FLAME will give you a static neck.
- **Eyes only live in `flame_coeffs`.** `smplx_coeffs` has no `eye_pose` field — the `flame_coeffs.eye_pose_params` is the source of truth. They are non-zero (range roughly `[-0.54, 0.53]`).
- **Per-subject pickles are flat.** If you skip the loader and read `subjects/<id>/tracking/optim_tracking_ehm.pkl` directly, the top-level keys are frame keys (e.g. `'000015_000019/000000'`), NOT video ids. The loader wraps them under `{video_id: ...}` so the merged file matches the format HolisticAvatar's `dataset/data_loader.py::TrackedData` expects.
- **`dataset_frames.json` train/valid/test split is shot-limited.** During the original training run we limited val and test to the first 2 shots of each video to keep evaluation fast. The per-subject `videos_info.json` retains every shot, so the per-subject `optim_tracking_ehm.pkl` has all frames — only the merged `dataset_frames.json` is restricted.
- **No videos / no source mp4s.** We do not redistribute the original TED video files. Each `frames.tar` does include the per-shot mp4 clip alongside the JPGs as a convenience.
- **Audio.** Per-shot `audio.wav` files are included inside `frames.tar` for future audio-driven work (talking-head, lip-sync, etc.). They are NOT used by HolisticAvatar.
- **The original frames are stored as JPG** at the source resolution from `yt-dlp` of the TED talks. We did not re-encode.

## Citation

If you use TEDWB1k, please cite the HolisticAvatar paper and the HolisticTracker repository:

```bibtex
@misc{tedwb1k2026,
  title  = {TEDWB1k: 1.4K TED Talks with Whole-Body SMPL-X + FLAME Tracking},
  author = {neil},
  year   = {2026},
  url    = {https://huggingface.co/datasets/initialneil/TEDWB1k},
}
```

And acknowledge the source content (TED, talks under CC-BY-NC-ND) in any derivative work.

## License

[**CC-BY-NC-4.0**](https://creativecommons.org/licenses/by-nc/4.0/). Non-commercial research use only. Attribution required.

The tracking parameters, JPG frames, and mattes are all derived works of TED talk videos that are themselves CC-BY-NC-ND on ted.com. This dataset is therefore restricted to non-commercial use.

## Links

- Tracking pipeline: <https://github.com/initialneil/HolisticTracker>
- HolisticAvatar (downstream model): <https://github.com/initialneil/HolisticAvatar>
- HF dataset: <https://huggingface.co/datasets/initialneil/TEDWB1k>
