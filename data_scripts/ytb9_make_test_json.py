

import os
import json
import argparse
import pickle


def load_dict_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def make_video_json(ehmx_video_dir, extra_info, anno_dir=None):
    ehmx_video_dir = ehmx_video_dir.rstrip('/')
    video_name = os.path.basename(ehmx_video_dir)
    
    print(f"Processing video: {video_name}")

    # Write extra_info
    extra_info_path = os.path.join(ehmx_video_dir, 'extra_info.json')
    with open(extra_info_path, 'w') as f:
        json.dump(extra_info, f, indent=4)
    print(f"Written extra_info to {extra_info_path}")

    # Load tracking results
    track_pkl = os.path.join(ehmx_video_dir, 'optim_tracking_ehm.pkl')
    if not os.path.exists(track_pkl):
        print(f"Error: {track_pkl} does not exist.")
        return

    track_data = load_dict_pkl(track_pkl)
    valid_frames = sorted(list(track_data.keys()))

    # Filter frames based on annotations
    if anno_dir:
        anno_path = os.path.join(anno_dir, f"{video_name}.json")
        if os.path.exists(anno_path):
            print(f"Loading annotation from {anno_path}")
            try:
                with open(anno_path, 'r') as f:
                    anno_data = json.load(f)
                
                wrong_list = anno_data.get('wrong', [])
                if wrong_list:
                    original_count = len(valid_frames)
                    valid_frames = [
                        k for k in valid_frames 
                        if not any(w in k for w in wrong_list)
                    ]
                    print(f"Filtered {original_count - len(valid_frames)} frames based on annotation.")
            except Exception as e:
                print(f"Error reading annotation for {video_name}: {e}")
        else:
            print(f"Annotation file {anno_path} not found.")

    # Construct videos_info
    videos_info = {
        video_name: {
            "frames_num": len(valid_frames),
            "frames_keys": valid_frames
        }
    }

    videos_info_path = os.path.join(ehmx_video_dir, 'videos_info.json')
    with open(videos_info_path, 'w') as f:
        json.dump(videos_info, f, indent=4)
    print(f"Written videos_info to {videos_info_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate test json files for videos in a directory")
    parser.add_argument('--ehmx_dir', required=True, help="Directory containing subdirectories of tracking results")
    parser.add_argument('--anno_dir', default=None, help="Directory containing annotation JSONs")
    parser.add_argument('--prefix', default="shots_", help="Prefix for data directories (default: shots_)")
    args = parser.parse_args()

    ehmx_dir = args.ehmx_dir.rstrip('/')
    if not os.path.isdir(ehmx_dir):
        print(f"Error: {ehmx_dir} is not a directory.")
        return

    root_dir = os.path.dirname(ehmx_dir)
    prefix = args.prefix
    
    extra_info = {
        "frames_root": os.path.join(root_dir, f"{prefix}images"),
        "matte_root": os.path.join(root_dir, f"{prefix}rmbg2"),
        "pshuman_root": os.path.join(root_dir, f"{prefix}pshuman")
    }
    
    print(f"Using extra_info: {json.dumps(extra_info, indent=4)}")

    for item in os.listdir(ehmx_dir):
        sub_dir = os.path.join(ehmx_dir, item)
        if os.path.isdir(sub_dir):
            track_pkl = os.path.join(sub_dir, 'optim_tracking_ehm.pkl')
            if os.path.exists(track_pkl):
                make_video_json(sub_dir, extra_info, args.anno_dir)

if __name__ == "__main__":
    main()

