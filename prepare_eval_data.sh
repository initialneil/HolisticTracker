python infer_ehmx_parallel.py \
    --images_dir "/home/szj/err_empty_syn/ytb/EHM-X/eval_images" \
    --mattes_dir "/home/szj/err_empty_syn/ytb/EHM-X/eval_rmbg2" \
    --index_mode "video_name_pshuman" \
    --pshuman_dir "/home/szj/err_empty_syn/ytb/EHM-X/eval_pshuman" \
    --ehmx_dir "/home/szj/err_empty_syn/ytb/EHM-X/eval_ehmx_novel" \
    --config "src/configs/optim_configs/ehm_fbody_video.yaml" \
    --overwrite smplx \
    --distribute 0,1,2,3

python data_scripts/ytb9_make_test_json.py \
    --ehmx_dir "/home/szj/err_empty_syn/ytb/EHM-X/eval_ehmx_novel" \
    --prefix "eval_"

    # "--anno_dir", "/home/szj/err_empty_syn/ytb/EHM-X/frames_0.5fps_anno",


