#!/bin/bash

# Configuration
IMAGES_DIR="/home/szj/err_empty_syn/ytb/EHM-X/shots_images"
MATTES_DIR="/home/szj/err_empty_syn/ytb/EHM-X/shots_rmbg2"
PSHUMAN_DIR="/home/szj/err_empty_syn/ytb/EHM-X/shots_pshuman"
EHMX_DIR="/home/szj/err_empty_syn/ytb/EHM-X/shots_ehmx"

# GPU configuration
GPUS="0,1,2,3"

# Index mode: "video_name" or "video_name_pshuman"
INDEX_MODE="video_name_pshuman"

# Overwrite flag (uncomment to enable)
# OVERWRITE="--overwrite"

# Run the parallel inference
python infer_ehmx_parallel.py \
  --images_dir "$IMAGES_DIR" \
  --mattes_dir "$MATTES_DIR" \
  --pshuman_dir "$PSHUMAN_DIR" \
  --ehmx_dir "$EHMX_DIR" \
  --distribute "$GPUS" \
  --index_mode "$INDEX_MODE" \
  $OVERWRITE
