import os
import os.path as osp
import sys
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))
from pathlib import Path
import tyro
import pickle
import json
import numpy as np
from dataclasses import dataclass, field
from tqdm import tqdm


@dataclass
class Args:
    dataset_dir: str = '/home/szj/Datasets/TEDWB1k/shots_ehmx'
    test_list: str = '/home/szj/Datasets/TEDWB1k/test.txt'
    frames_root: str = '/home/szj/Datasets/TEDWB1k/shots_images'
    matte_root: str = '/home/szj/Datasets/TEDWB1k/shots_rmbg2'
    num_val: int = 20
    max_shots_val_test: int = 2  # limit shots per video in val/test splits (0 = no limit)


if __name__ == '__main__':
    args = tyro.cli(Args)
    dataset_dir = args.dataset_dir

    # Load test list
    test_set = set()
    if osp.exists(args.test_list):
        with open(args.test_list, 'r') as f:
            test_set = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(test_set)} test videos from {args.test_list}")
    else:
        print(f"Warning: test list not found at {args.test_list}")

    # Find all videos with complete tracking
    video_list = [d for d in os.listdir(dataset_dir) if (
        osp.isfile(osp.join(dataset_dir, d, 'videos_info.json')) and
        osp.isfile(osp.join(dataset_dir, d, 'optim_tracking_ehm.pkl')) and
        osp.isfile(osp.join(dataset_dir, d, 'id_share_params.pkl'))
    )]
    video_list.sort()

    # Split: test from file, then val from remaining, rest is train
    test_split = [v for v in video_list if v in test_set]
    remaining = [v for v in video_list if v not in test_set]
    val_split = remaining[-args.num_val:]
    train_split = remaining[:-args.num_val]

    print(f"Total {len(video_list)} videos")
    print(f"Train: {len(train_split)} videos")
    print(f"Val:   {len(val_split)} videos")
    print(f"Test:  {len(test_split)} videos")

    full_videos_info = {}
    full_dataset_frames = {
        'train': [],
        'valid': [],
        'test': []
    }
    full_extra_info = {
        'wbbox_info': {},
        'frames_root': args.frames_root,
        'matte_root': args.matte_root,
    }
    full_tracking_data = {}
    full_id_share_data = {}

    train_set = set(train_split)
    val_set = set(val_split)
    test_split_set = set(test_split)

    for video_id in tqdm(video_list, desc="Processing videos"):
        video_dir = osp.join(dataset_dir, video_id)
        videos_info_path = osp.join(video_dir, 'videos_info.json')
        extra_info_path = osp.join(video_dir, 'extra_info.json')
        tracking_path = osp.join(video_dir, 'optim_tracking_ehm.pkl')
        id_share_path = osp.join(video_dir, 'id_share_params.pkl')

        if not osp.exists(extra_info_path):
            print(f"Warning: {extra_info_path} does not exist. Skipping {video_id}.")
            continue

        with open(videos_info_path, 'r') as fp:
            videos_info = json.load(fp)
        with open(extra_info_path, 'r') as fp:
            extra_info = json.load(fp)
        with open(tracking_path, 'rb') as fp:
            tracking_data = pickle.load(fp)
        with open(id_share_path, 'rb') as fp:
            id_share_data = pickle.load(fp)

        full_videos_info.update(videos_info)
        full_tracking_data[video_id] = tracking_data
        full_id_share_data[video_id] = id_share_data

        if 'wbbox_info' in extra_info:
            wbbox_info = extra_info.pop('wbbox_info')
            full_extra_info['wbbox_info'][video_id] = wbbox_info

        whole_frames_keys = [f'{video_id}/{frame_id}' for frame_id in videos_info[video_id]['frames_keys']]

        if video_id in train_set:
            full_dataset_frames['train'].extend(whole_frames_keys)
        elif video_id in val_set:
            full_dataset_frames['valid'].extend(whole_frames_keys)
        elif video_id in test_split_set:
            full_dataset_frames['test'].extend(whole_frames_keys)

    print(f"\nAll videos processed!")

    # Limit shots per video in val/test splits
    if args.max_shots_val_test > 0:
        from collections import defaultdict
        for split in ['valid', 'test']:
            frames = full_dataset_frames[split]
            vid_shots = defaultdict(lambda: defaultdict(list))
            for f in frames:
                parts = f.split('/')
                vid, shot = parts[0], parts[1]
                vid_shots[vid][shot].append(f)
            new_frames = []
            for vid in sorted(vid_shots.keys()):
                shots = sorted(vid_shots[vid].keys())[:args.max_shots_val_test]
                for shot in shots:
                    new_frames.extend(vid_shots[vid][shot])
            print(f"  {split}: {len(frames)} -> {len(new_frames)} frames (limited to {args.max_shots_val_test} shots/video)")
            full_dataset_frames[split] = new_frames

    print(f"Train frames: {len(full_dataset_frames['train'])}")
    print(f"Val frames:   {len(full_dataset_frames['valid'])}")
    print(f"Test frames:  {len(full_dataset_frames['test'])}")

    with open(osp.join(dataset_dir, 'videos_info.json'), 'w') as fp:
        json.dump(full_videos_info, fp, indent=4)
    with open(osp.join(dataset_dir, 'dataset_frames.json'), 'w') as fp:
        json.dump(full_dataset_frames, fp, indent=4)
    with open(osp.join(dataset_dir, 'extra_info.json'), 'w') as fp:
        json.dump(full_extra_info, fp, indent=4)
    with open(osp.join(dataset_dir, 'optim_tracking_ehm.pkl'), 'wb') as fp:
        pickle.dump(full_tracking_data, fp)
    with open(osp.join(dataset_dir, 'id_share_params.pkl'), 'wb') as fp:
        pickle.dump(full_id_share_data, fp)

    print(f"\nSaved merged dataset to {dataset_dir}")
