#!/bin/bash
# Run ehm-tracker pipeline on remaining videos using all 8 GPUs
# Two tmux sessions, each with 8 workers (one per GPU) = 16 total workers, 2 per GPU

set -e
cd "$(dirname "$0")"

IMAGES_DIR="/home/szj/Datasets/TEDWB1k/shots_images"
MATTES_DIR="/home/szj/Datasets/TEDWB1k/shots_rmbg2"
EHMX_DIR="/home/szj/Datasets/TEDWB1k/shots_ehmx"
CONFIG="src/configs/optim_configs/ehm_fbody_ytb.yaml"
PYTHON="/home/szj/miniconda3/envs/pt2.7/bin/python"
GPUS="0,1,2,3,4,5,6,7"

LIST1="video_list_part1.txt"
LIST2="video_list_part2.txt"

CMD1="cd /home/szj/Playground/ehm-tracker && $PYTHON infer_ehmx_parallel.py \
  --images_dir $IMAGES_DIR \
  --mattes_dir $MATTES_DIR \
  --ehmx_dir $EHMX_DIR \
  --config $CONFIG \
  --distribute $GPUS \
  --index_mode video_name \
  --body_landmark_type sapiens \
  --body_estimator_type pixie \
  --video_list $LIST1 \
  2>&1 | tee /tmp/run_all_part1.log"

CMD2="cd /home/szj/Playground/ehm-tracker && $PYTHON infer_ehmx_parallel.py \
  --images_dir $IMAGES_DIR \
  --mattes_dir $MATTES_DIR \
  --ehmx_dir $EHMX_DIR \
  --config $CONFIG \
  --distribute $GPUS \
  --index_mode video_name \
  --body_landmark_type sapiens \
  --body_estimator_type pixie \
  --video_list $LIST2 \
  2>&1 | tee /tmp/run_all_part2.log"

tmux new-session -d -s run_part1 "$CMD1"
tmux new-session -d -s run_part2 "$CMD2"

echo "Launched 2 tmux sessions (run_part1, run_part2)"
echo "  Each: 8 GPUs (0-7), 8 workers per session = 16 total workers"
echo "  Part 1: $(wc -l < $LIST1) videos"
echo "  Part 2: $(wc -l < $LIST2) videos"
echo ""
echo "Monitor: tmux attach -t run_part1"
echo "Logs:    tail -f /tmp/run_all_part1.log"
echo "         tail -f /tmp/run_all_part2.log"
