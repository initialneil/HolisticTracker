#!/bin/bash
set -e
cd "$(dirname "$0")"

IMAGES_DIR="/tmp/tedwb_4"
EHMX_DIR="/home/szj/Datasets/TEDWB1k/shots_ehmx"
MATTES_DIR="/home/szj/Datasets/TEDWB1k/shots_rmbg2"
CONFIG="src/configs/optim_configs/ehm_fbody_ytb.yaml"
PYTHON="/home/szj/miniconda3/envs/pt2.7/bin/python"

VIDEOS=(05jJodDVJRQ 08ZWROqoTZo 0bRocfcPhHU 0d6iSvF1UmA)
GPUS=(0 1 2 3)

run_video() {
    local VIDEO=$1
    local GPU=$2
    local JSON="$IMAGES_DIR/$VIDEO/${VIDEO}.json"
    local COMMON="--images_dir $IMAGES_DIR --ehmx_dir $EHMX_DIR --mattes_dir $MATTES_DIR --index_json $JSON"

    echo "[GPU $GPU] [$VIDEO] === track_base ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON infer_ehmx_track_base.py $COMMON \
        --body_landmark_type sapiens --body_estimator_type pixie

    echo "[GPU $GPU] [$VIDEO] === flame ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON infer_ehmx_flame.py $COMMON --config $CONFIG

    echo "[GPU $GPU] [$VIDEO] === smplx ==="
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON infer_ehmx_smplx.py $COMMON --config $CONFIG

    echo "[GPU $GPU] [$VIDEO] === DONE ==="
}

# Launch 4 videos on 4 GPUs in parallel
for i in "${!VIDEOS[@]}"; do
    run_video "${VIDEOS[$i]}" "${GPUS[$i]}" &
done

echo "Launched ${#VIDEOS[@]} videos. Waiting..."
wait
echo "All done!"
