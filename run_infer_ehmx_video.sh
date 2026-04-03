#!/bin/bash
# Run 3-stage EHM tracking on a single video.
# Usage: CUDA_VISIBLE_DEVICES=0 bash run_infer_ehmx_video.sh

export PYTHONPATH="."
cd "$(dirname "$0")"

PYTHON="/home/szj/miniconda3/envs/pt2.7/bin/python"

IMAGES_DIR="/home/szj/Datasets/TEDWB1k/shots_images"
MATTES_DIR="/home/szj/Datasets/TEDWB1k/shots_rmbg2"
EHMX_DIR="/home/szj/Datasets/TEDWB1k/shots_ehmx"
INDEX_JSON="/home/szj/Datasets/TEDWB1k/shots_images/b28brIs1OmM/b28brIs1OmM.json"
CONFIG="src/configs/optim_configs/ehm_fbody_video.yaml"

# Stage 1: track_base
$PYTHON infer_ehmx_track_base.py \
    --images_dir $IMAGES_DIR --index_json $INDEX_JSON \
    --ehmx_dir $EHMX_DIR --mattes_dir $MATTES_DIR \
    --body_landmark_type sapiens --body_estimator_type pixie \
    --overwrite

# Stage 2: flame refinement
$PYTHON infer_ehmx_flame.py \
    --images_dir $IMAGES_DIR --index_json $INDEX_JSON \
    --ehmx_dir $EHMX_DIR --mattes_dir $MATTES_DIR \
    --config $CONFIG --overwrite

# Stage 3: smplx optimization
$PYTHON infer_ehmx_smplx.py \
    --images_dir $IMAGES_DIR --index_json $INDEX_JSON \
    --ehmx_dir $EHMX_DIR --mattes_dir $MATTES_DIR \
    --config $CONFIG --body_landmark_type sapiens \
    --overwrite
