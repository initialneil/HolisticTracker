#!/bin/bash

# Extract keyframes at 0.5 fps
python data_scripts/ytb1_extract_0.5fps.py \
    --videos_dir "/home/szj/err_empty_syn/ytb/videos" \
    --keyframes_dir "/home/szj/err_empty_syn/ytb/EHM-X/frames_0.5fps"

# Check foot visibility in keyframes
python data_scripts/ytb3_keyframe_check_feet.py \
    --keyframes_dir "/home/szj/err_empty_syn/ytb/EHM-X/frames_0.5fps" \
    --filtered_dir "/home/szj/err_empty_syn/ytb/EHM-X/frames_0.5fps_check_feet" \
    --distribute_gpus 1,2,3

# Check shot color uniformity
python data_scripts/ytb4_shots_check_color.py \
    --keyframes_dir "/home/szj/err_empty_syn/ytb/EHM-X/frames_0.5fps" \
    --filtered_dir "/home/szj/err_empty_syn/ytb/EHM-X/frames_0.5fps_check_feet" \
    --shots_dir "/home/szj/err_empty_syn/ytb/EHM-X/frames_0.5fps_shots"



# # Run RMBG 2.0
# CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
#     data_scripts/ytb6_rmbg2.py \
#     --images_dir "/home/szj/err_empty_syn/ytb/EHM-X/shots_full_fps_images" \
#     --mattes_dir "/home/szj/err_empty_syn/ytb/EHM-X/shots_full_fps_rmbg2" \
#     --distribute_gpus 1,2,3
