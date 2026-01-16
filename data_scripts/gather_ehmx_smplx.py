import os
import argparse
import shutil
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Gather SMPLX visualization images.")
    parser.add_argument('--ehmx_dir', required=True, help="Root directory containing video subfolders")
    parser.add_argument('--gather_dir', required=True, help="Destination directory for gathered images")
    args = parser.parse_args()

    if not os.path.exists(args.gather_dir):
        os.makedirs(args.gather_dir, exist_ok=True)

    # List subfolders in ehmx_dir
    if not os.path.exists(args.ehmx_dir):
        print(f"Error: Source directory {args.ehmx_dir} does not exist.")
        return

    video_list = [d for d in os.listdir(args.ehmx_dir) if os.path.isdir(os.path.join(args.ehmx_dir, d))]
    video_list.sort()

    print(f"Found {len(video_list)} potential video folders in {args.ehmx_dir}")

    count = 0
    for video_name in tqdm(video_list, desc="Gathering images"):
        src_path = os.path.join(args.ehmx_dir, video_name, 'visual_results', 'vis_fit_smplx_bid-0_stp-500.jpg')
        dst_path = os.path.join(args.gather_dir, f"{video_name}.jpg")

        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)
            count += 1
    
    print(f"Successfully gathered {count} images to {args.gather_dir}")

if __name__ == "__main__":
    main()
