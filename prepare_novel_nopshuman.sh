CUDA_VISIBLE_DEVICES=0 python infer_ehmx_parallel.py \
    --images_dir "/home/szj/err_empty_syn/ytb/EHM-X/eval_images" \
    --mattes_dir "/home/szj/err_empty_syn/ytb/EHM-X/eval_rmbg2" \
    --index_mode "video_name_nopshuman_frame" \
    --pshuman_dir "/home/szj/err_empty_syn/ytb/EHM-X/eval_pshuman" \
    --ehmx_dir "/home/szj/err_empty_syn/ytb/EHM-X/eval_ehmx_frame_nopshuman" \
    --config "src/configs/optim_configs/ehm_fbody_frame.yaml" \
    --overwrite smplx

python data_scripts/ytb9_make_test_json.py \
    --ehmx_dir "/home/szj/err_empty_syn/ytb/EHM-X/eval_ehmx_frame_nopshuman" \
    --prefix "eval_"
    # "--anno_dir", "/home/szj/err_empty_syn/ytb/EHM-X/frames_0.5fps_anno",


