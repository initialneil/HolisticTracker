#!/usr/bin/env python3
"""
ytb8_make_dataset.py - Aggregate tracking results into a dataset
Merges optim_tracking_ehm.pkl and id_share_params.pkl from multiple videos.
Generates optim_tracking_ehm.pkl, id_share_params.pkl, videos_info.json, and dataset_frames.json.
"""
import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm


def load_dict_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_dict_pkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def find_index_files(images_dir, index_mode):
    """Find video index files based on mode."""
    images_path = Path(images_dir)
    index_files = []
    
    if index_mode == 'video_name':
        pattern = '*/*.json'
        for json_file in images_path.glob(pattern):
            video_name = json_file.parent.name
            expected_name = f"{video_name}.json"
            if json_file.name == expected_name:
                index_files.append((video_name, str(json_file)))
    
    elif index_mode == 'video_name_pshuman':
        pattern = '*/*_pshuman.json'
        for json_file in images_path.glob(pattern):
            video_name = json_file.parent.name
            expected_name = f"{video_name}_pshuman.json"
            if json_file.name == expected_name:
                index_files.append((video_name, str(json_file)))
    
    return sorted(index_files)


def make_dataset(args):
    # Create output directory
    os.makedirs(args.data_root, exist_ok=True)
    
    # Output file paths
    out_tracking_path = os.path.join(args.data_root, 'optim_tracking_ehm.pkl')
    out_id_params_path = os.path.join(args.data_root, 'id_share_params.pkl')
    out_videos_info_path = os.path.join(args.data_root, 'videos_info.json')
    out_dataset_frames_path = os.path.join(args.data_root, 'dataset_frames.json')
    
    # Find videos to process
    print(f"Searching for videos in {args.images_dir} with mode {args.index_mode}...")
    index_files = find_index_files(args.images_dir, args.index_mode)
    print(f"Found {len(index_files)} videos.")
    
    # Load test list for splitting
    test_videos = set()
    test_list_path = os.path.join(args.data_root, args.test_list)
    if os.path.exists(test_list_path):
        print(f"Loading test list from {test_list_path}...")
        with open(test_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    test_videos.add(line)
        print(f"Found {len(test_videos)} test videos.")
    else:
        print(f"Test list {test_list_path} not found. All videos will be in train split.")
    
    # Containers for merged data
    all_optim_track = {}
    all_id_params = {}
    videos_info = {}
    
    # Process each video
    for video_name, _ in tqdm(index_files, desc="Processing videos"):
        # Paths for this video's results
        video_ehm_dir = os.path.join(args.ehmx_dir, video_name)
        track_pkl = os.path.join(video_ehm_dir, 'optim_tracking_ehm.pkl')
        id_pkl = os.path.join(video_ehm_dir, 'id_share_params.pkl')
        
        if not os.path.exists(track_pkl):
            # print(f"Warning: {track_pkl} does not exist. Skipping {video_name}.")
            continue
            
        try:
            # Load tracking results and ID params
            track_data = load_dict_pkl(track_pkl)
            id_data = load_dict_pkl(id_pkl)
            
            # Filter frames based on annotations if provided
            valid_frames = list(track_data.keys())
            
            if args.anno_dir:
                anno_path = os.path.join(args.anno_dir, f"{video_name}.json")
                if os.path.exists(anno_path):
                    try:
                        with open(anno_path, 'r') as f:
                            anno_data = json.load(f)
                        
                        wrong_list = anno_data.get('wrong', [])
                        if wrong_list:
                            # Skip frame if its key contains any string from the wrong list
                            # or if it exactly matches (depending on annotation format)
                            # Assuming substring match based on "contains key" requirement
                            original_count = len(valid_frames)
                            valid_frames = [
                                k for k in valid_frames 
                                if not any(w in k for w in wrong_list)
                            ]
                            skipped = original_count - len(valid_frames)
                            if skipped > 0:
                                pass
                                # print(f"  Skipped {skipped} frames for {video_name} based on annotation.")
                    except Exception as e:
                        print(f"Error reading annotation for {video_name}: {e}")
            
            if not valid_frames:
                print(f"No valid frames for {video_name}. Skipping.")
                continue
            
            if len(valid_frames) < args.min_frames:
                print(f"Video {video_name} has only {len(valid_frames)} frames (min {args.min_frames}). Skipping.")
                continue
            
            # Store data
            # Filter track_data to only include valid frames
            all_optim_track[video_name] = {k: track_data[k] for k in valid_frames}
            all_id_params[video_name] = id_data
            
            # Update videos info
            videos_info[video_name] = {
                "frames_num": len(valid_frames),
                "frames_keys": sorted(valid_frames)
            }
            
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            continue
    
    # Save merged data
    print(f"Saving merged tracking data to {out_tracking_path}...")
    write_dict_pkl(out_tracking_path, all_optim_track)
    
    print(f"Saving merged ID params to {out_id_params_path}...")
    write_dict_pkl(out_id_params_path, all_id_params)
    
    print(f"Saving videos info to {out_videos_info_path}...")
    with open(out_videos_info_path, 'w') as f:
        json.dump(videos_info, f, indent=4)
    
    # Generate train/test split
    dataset_frames = {'train': [], 'test': []}
    
    for video_name, info in videos_info.items():
        frames = [f"{video_name}/{k}" for k in info['frames_keys']] # Format: video/frame_key
        # Actually, usually dataset_frames just lists frame keys if they are unique globally, 
        # or video_name/frame_key. 
        # In build_lmdb_dataset.py: split_train_valid returns dict with 'train': [frame_keys...], 'test': ...
        # But wait, track_data keys are already frame keys (e.g. "shot/frame").
        # If we just list them, we lose video context if keys aren't unique across videos.
        # However, usually keys are unique or we use video/frame structure.
        # Let's check build_lmdb_dataset.py's split_train_valid.
        # It seems it just puts video_names into train/valid lists? 
        # No, "dataset_frames=split_train_valid(videos_info,num_valid=1)".
        # Let's assume we want a list of all valid frame identifiers.
        # Since all_optim_track is nested by video_name, maybe we just need the list of keys?
        # But standard datasets often use a flat list of "video_name/frame_key" or just "frame_key" if unique.
        # Given the structure `all_optim_track[video_name][frame_key]`, the dataset loader will likely need
        # to know which video a frame belongs to.
        # Let's store "video_name/frame_key" to be safe and explicit.
        
        # Re-reading build_lmdb_dataset.py:
        # It dumps to LMDB with key `video_id/frame_id/body_image`.
        # It saves `all_optim_track_fp_smplx[video_id][frame_id]`.
        # So the hierarchy is preserved.
        # `dataset_frames.json` usually lists the keys to iterate over.
        # If I look at `split_dataset.py` (imported in build_lmdb_dataset), I could know.
        # But I don't have it.
        # I will format as `video_name/frame_key` which is standard for this type of hierarchy.
        
        # Wait, `frames_keys` in `videos_info` are just the keys inside the pkl.
        # If I prepend video_name, it becomes `video_name/shot_id/frame_id`.
        
        frame_paths = [f"{video_name}/{k}" for k in info['frames_keys']]
        
        if video_name in test_videos:
            dataset_frames['test'].extend(frame_paths)
        else:
            dataset_frames['train'].extend(frame_paths)
    
    print(f"Saving dataset split to {out_dataset_frames_path}...")
    print(f"  Train frames: {len(dataset_frames['train'])}")
    print(f"  Test frames: {len(dataset_frames['test'])}")
    
    with open(out_dataset_frames_path, 'w') as f:
        json.dump(dataset_frames, f, indent=4)
    
    # Calculate stats
    total_videos = len(index_files)
    used_videos = len(videos_info)
    skipped_videos = total_videos - used_videos
    
    total_used_frames = 0
    total_used_shots = 0
    
    for info in videos_info.values():
        frames_count = info['frames_num']
        total_used_frames += frames_count
        
        # Try to determine shots from keys
        keys = info['frames_keys']
        if keys and '/' in keys[0]:
            # Assume format shot_id/frame_id
            shots = set(k.rsplit('/', 1)[0] for k in keys)
            total_used_shots += len(shots)
        else:
            # Assume 1 shot per video if no hierarchy in keys
            total_used_shots += 1
            
    avg_frames_per_shot = total_used_frames / total_used_shots if total_used_shots > 0 else 0
    avg_frames_per_video = total_used_frames / used_videos if used_videos > 0 else 0
    
    print("-" * 40)
    print("Statistics:")
    print(f"Total videos found: {total_videos}")
    print(f"Used videos: {used_videos}")
    print(f"Skipped videos: {skipped_videos}")
    print(f"Total used shots: {total_used_shots}")
    print(f"Total used frames: {total_used_frames}")
    print(f"Avg frames per shot: {avg_frames_per_shot:.2f}")
    print(f"Avg frames per video: {avg_frames_per_video:.2f}")
    print("-" * 40)

    print("Dataset creation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate EHM-X tracking results into a dataset")
    
    parser.add_argument('--data_root', required=True, 
                        help="Output root directory for dataset files")
    parser.add_argument('--images_dir', required=True, 
                        help="Root directory containing video subdirectories")
    parser.add_argument('--ehmx_dir', required=True, 
                        help="Root directory containing EHM-X tracking results (pkl files)")
    parser.add_argument('--anno_dir', default=None, 
                        help="Directory containing annotation JSONs to filter frames")
    parser.add_argument('--min_frames', type=int, default=10, 
                        help="Minimum number of frames required to include a video")
    parser.add_argument('--index_mode', default='video_name', 
                        choices=['video_name', 'video_name_pshuman'],
                        help="Mode to find video index files")
    parser.add_argument('--test_list', default='test.txt', 
                        help="Filename of test video list in data_root")
    
    args = parser.parse_args()
    make_dataset(args)
