
import os
import shutil

input_root = '/home/szj/err_empty_syn/ytb/EHM-X/shots_full_fps_images/'
output_root = '/home/szj/err_empty_syn/ytb/EHM-X/shots_full_fps_videos/'

print(f"Scanning {input_root}...")
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.endswith('.mp4'):
            src_path = os.path.join(root, file)
            # Get relative path to preserve structure
            rel_path = os.path.relpath(src_path, input_root)
            dst_path = os.path.join(output_root, rel_path)
            
            # Create destination directory
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            print(f"Copying {file} to {dst_path}")
            shutil.copy2(src_path, dst_path)


