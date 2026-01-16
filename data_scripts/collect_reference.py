import os
import os.path as osp
import json

data_root = '/home/szj/err_empty_syn/ytb/EHM-X/eval_images/'
video_list = [d for d in os.listdir(data_root) if osp.isdir(osp.join(data_root, d))]

selected_reference = {}
for video_name in video_list:
    video_dir = osp.join(data_root, video_name)
    info_path = osp.join(video_dir, f'{video_name}.json')
    if not osp.exists(info_path):
        print(f"Info file not found for {video_name}, skipping.")
        continue

    with open(info_path, 'r') as f:
        info_data = json.load(f)
    if 'selected_reference' in info_data[video_name]:
        selected_reference[video_name] = info_data[video_name]['selected_reference']
        print(f"Collected reference for {video_name}: {info_data[video_name]['selected_reference']}")

with open('/home/szj/err_empty_syn/ytb/EHM-X/selected_reference.json', 'w') as f:
    json.dump(selected_reference, f, indent=4)
