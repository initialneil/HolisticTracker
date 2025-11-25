# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
#     accelerate launch --main_process_port 29070 \
#     refine_video.py \
#     --in_root "/home/szj/err_empty_syn/scan/dataset_ehm_v0/" \
#     --output_dir "/home/szj/err_empty_syn/scan/dataset_ehm_v1/" \
#     --optim_cfg "src/configs/optim_configs/ehm_fbody_scan.yaml" \
#     --not_check_hand


# CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     accelerate launch --main_process_port 29070 \
#     refine_video.py \
#     --in_root "/home/szj/err_empty_syn/ytb/bbox_square_x1.1/dataset_ehm_v1" \
#     --output_dir "/home/szj/err_empty_syn/ytb/bbox_square_x1.1/dataset_ehm_v2.1" \
#     --optim_cfg "src/configs/optim_configs/ehm_fbody_ytb.yaml" \
#     --not_check_hand \

# CUDA_VISIBLE_DEVICES=3 \
#     accelerate launch --main_process_port 29071 \
#     refine_video.py \
#     --in_root "/home/szj/err_empty_syn/ytb/bbox_square_x1.1/dataset_ehm_v1" \
#     --output_dir "/home/szj/err_empty_syn/ytb/bbox_square_x1.1/dataset_ehm_v2.1" \
#     --optim_cfg "src/configs/optim_configs/ehm_fbody_ytb.yaml" \
#     --not_check_hand \
#     --reversed_order \

CUDA_VISIBLE_DEVICES=1 \
    accelerate launch --main_process_port 29070 \
    refine_video.py \
    --in_root "/home/szj/err_empty_syn/fashion/train_10fps" \
    --output_dir "/home/szj/err_empty_syn/fashion/train_10fps_v2" \
    --optim_cfg "src/configs/optim_configs/ehm_fbody_ytb.yaml" \
    --not_check_hand \

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     accelerate launch --main_process_port 29070 \
#     refine_video.py \
#     --in_root "/home/szj/err_empty_syn/fashion/test_30fps" \
#     --output_dir "/home/szj/err_empty_syn/fashion/test_30fps_v2" \
#     --optim_cfg "src/configs/optim_configs/ehm_fbody_ytb.yaml" \
#     --not_check_hand \

