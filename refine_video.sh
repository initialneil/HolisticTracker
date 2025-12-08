# # 14 threads for 7 gpus
# python refine_video.py \
#     --in_root "/home/szj/err_empty_syn/scan/dataset_ehm_v0/" \
#     --output_dir "/home/szj/err_empty_syn/scan/dataset_ehm_v1/" \
#     --optim_cfg "src/configs/optim_configs/ehm_fbody_scan.yaml" \
#     --not_check_hand \
#     -n 14 -v 1,2,3,4,5,6,7


# # 8 threads for 4 gpus
# python refine_video.py \
#     --in_root "/home/szj/err_empty_syn/ytb/bbox_square_x1.1/dataset_ehm_v1" \
#     --output_dir "/home/szj/err_empty_syn/ytb/bbox_square_x1.1/dataset_ehm_v2.1" \
#     --optim_cfg "src/configs/optim_configs/ehm_fbody_ytb.yaml" \
#     --not_check_hand \
#     -n 8 -v 0,1,2,3

# # Single GPU with reversed order
# python refine_video.py \
#     --in_root "/home/szj/err_empty_syn/ytb/bbox_square_x1.1/dataset_ehm_v1" \
#     --output_dir "/home/szj/err_empty_syn/ytb/bbox_square_x1.1/dataset_ehm_v2.1" \
#     --optim_cfg "src/configs/optim_configs/ehm_fbody_ytb.yaml" \
#     --not_check_hand \
#     --reversed_order \
#     -n 1 -v 3

# 8 threads for 4 gpus
python refine_video.py \
    --in_root "/home/szj/err_empty_syn/fashion/train_10fps" \
    --output_dir "/home/szj/err_empty_syn/fashion/train_10fps_v2" \
    --optim_cfg "src/configs/optim_configs/ehm_fbody_ytb.yaml" \
    --not_check_hand \
    -n 8 -v 0,1,2,3

# 8 threads for 4 gpus
python refine_video.py \
    --in_root "/home/szj/err_empty_syn/fashion/test_30fps" \
    --output_dir "/home/szj/err_empty_syn/fashion/test_30fps_v2" \
    --optim_cfg "src/configs/optim_configs/ehm_fbody_ytb.yaml" \
    --not_check_hand \
    -n 8 -v 0,1,2,3

