#!/bin/bash
# Rerun track_flame + track_smplx on 8 videos using 4 GPUs (round-robin)
set -e
cd "$(dirname "$0")"

IMAGES_DIR="/home/szj/Datasets/TEDWB1k/shots_images"
MATTES_DIR="/home/szj/Datasets/TEDWB1k/shots_rmbg2"
PSHUMAN_DIR="/home/szj/Datasets/TEDWB1k/shots_pshuman"
EHMX_DIR="/home/szj/Datasets/TEDWB1k/shots_ehmx"
CONFIG="src/configs/optim_configs/ehm_fbody_ytb.yaml"
PYTHON="/home/szj/miniconda3/envs/pt2.7/bin/python"

VIDEOS=(7MHOk7qVhYs b9jb9UjCpik BQZKs75RMqM Fivy99RtMfM FPhHHtn8On8 iE9HMudybyc mWA2uL8zXPI V84b-WIlNA0)
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

run_video() {
    local VIDEO=$1
    local GPU=$2
    local INDEX_JSON="$EHMX_DIR/$VIDEO/videos_info.json"
    local COMMON="--images_dir $IMAGES_DIR --mattes_dir $MATTES_DIR --pshuman_dir $PSHUMAN_DIR --ehmx_dir $EHMX_DIR --index_json $INDEX_JSON --config $CONFIG --overwrite"

    echo "[GPU $GPU] [$VIDEO] flame"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON infer_ehmx_flame.py $COMMON 2>&1 | tail -3
    echo "[GPU $GPU] [$VIDEO] smplx"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON infer_ehmx_smplx.py $COMMON 2>&1 | tail -3
    echo "[GPU $GPU] [$VIDEO] DONE"
}

# Launch all videos round-robin across GPUs
PIDS=()
for i in "${!VIDEOS[@]}"; do
    GPU_IDX=$((i % NUM_GPUS))
    GPU=${GPUS[$GPU_IDX]}
    run_video "${VIDEOS[$i]}" "$GPU" &
    PIDS+=($!)
done

echo "Launched ${#VIDEOS[@]} flame+smplx jobs on ${NUM_GPUS} GPUs. Waiting..."

# Wait for all
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo "All ${#VIDEOS[@]} flame+smplx finished."
