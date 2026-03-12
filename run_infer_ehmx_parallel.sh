#!/bin/bash

# Configuration — TEDWB1k shots_images dataset
IMAGES_DIR="/home/szj/Datasets/TEDWB1k/shots_images"
MATTES_DIR="/home/szj/Datasets/TEDWB1k/shots_rmbg2"
PSHUMAN_DIR="/home/szj/Datasets/TEDWB1k/shots_pshuman"
EHMX_DIR="/home/szj/Datasets/TEDWB1k/shots_ehmx"

# GPU configuration — 8 GPUs
GPUS="0,1,2,3,4,5,6,7"

# Index mode: "video_name" or "video_name_pshuman"
INDEX_MODE="video_name_pshuman"

# Body landmark type: sapiens (default) or dwpose
BODY_LANDMARK_TYPE="sapiens"

# Overwrite flag (uncomment to enable specific steps or all)
# OVERWRITE="--overwrite all"
# OVERWRITE="--overwrite track_base"

# Python interpreter (pt2.7 env with pixel3dmm, sapiens, nvdiffrast)
PYTHON="/home/szj/miniconda3/envs/pt2.7/bin/python"

# Run the parallel inference
cd "$(dirname "$0")"
$PYTHON infer_ehmx_parallel.py \
  --images_dir "$IMAGES_DIR" \
  --mattes_dir "$MATTES_DIR" \
  --pshuman_dir "$PSHUMAN_DIR" \
  --ehmx_dir "$EHMX_DIR" \
  --distribute "$GPUS" \
  --index_mode "$INDEX_MODE" \
  --body_landmark_type "$BODY_LANDMARK_TYPE" \
  $OVERWRITE
