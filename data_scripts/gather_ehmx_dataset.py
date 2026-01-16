import os
import os.path as osp
import json
import math
import cv2
import numpy as np
from tqdm import tqdm

index_json = '/home/szj/err_empty_syn/ytb/EHM-X/dataset_frames.json'
images_dir = '/home/szj/err_empty_syn/ytb/EHM-X/shots_images/'
mattes_dir = '/home/szj/err_empty_syn/ytb/EHM-X/shots_rmbg2/'
with_matte = True
N, W, H = 30, 10, 3
EXT = '.png'
SIZE = 256

with open(index_json, 'r') as fp:
    index_data = json.load(fp)

all_frames_keys = index_data['train'] + index_data['test']
all_frames_keys = [key for key in all_frames_keys if 'pshuman' not in key]
all_frames_keys = [key for key in all_frames_keys if 'Scene' not in key]

video_frames = {}
for whole_key in all_frames_keys:
    video_name, shot_id, frame_id = whole_key.split('/')
    if video_name not in video_frames:
        video_frames[video_name] = []
    frame_key = f"{shot_id}/{frame_id}"
    video_frames[video_name].append(frame_key)

# Sort frames for each video
for v in video_frames:
    video_frames[v].sort()

video_names = list(video_frames.keys())
print(f"Found {len(video_names)} videos.")

video_names = video_names[:N]
out_dir = f'/home/szj/err_empty_syn/ytb/EHM-X/tmp/visulize_dataset/{W}x{H}'
os.makedirs(out_dir, exist_ok=True)

# Calculate grid layout for ALL videos
N = len(video_names)
rows = max(1, int(math.sqrt(N / (W/H))))
cols = int(math.ceil(N / rows))
while rows * cols < N:
    cols += 1

print(f"Grid layout: {rows} rows x {cols} cols")
canvas_h = rows * SIZE
canvas_w = cols * SIZE

# Generate 50 frames
for t in tqdm(range(50), desc="Generating preview frames"):
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    
    for i, video_name in enumerate(video_names):
        frame_list = video_frames[video_name]
        if not frame_list:
            continue
            
        # Pick frame t, looping if video is shorter
        frame_idx = t % len(frame_list)
        frame_key = frame_list[frame_idx]
        
        img_path = f'{images_dir}/{video_name}/{frame_key}.jpg'
        matte_path = f'{mattes_dir}/{video_name}/{frame_key}.png'
        
        img = None
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            
            if with_matte:
                matte = cv2.imread(matte_path, cv2.IMREAD_GRAYSCALE)
                if matte is not None:
                    if matte.shape != img.shape[:2]:
                        matte = cv2.resize(matte, (img.shape[1], img.shape[0]))
                    
                    # Apply matte
                    # img = img.astype(np.float32) * (matte[:, :, None].astype(np.float32) / 255.0)
                    img = np.concatenate([img, matte[:, :, None]], axis=-1)
                    img = img.astype(np.uint8)
        
        if img is None:
            # print(f"Warning: Failed to load {img_path}")
            img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (SIZE, SIZE))
            
        # Place in grid
        r = i // cols
        c = i % cols
        canvas[r*SIZE:(r+1)*SIZE, c*SIZE:(c+1)*SIZE] = img
        
    save_path = os.path.join(out_dir, f'preview_{t:03d}{EXT}')
    cv2.imwrite(save_path, canvas)

print(f"Saved 50 preview frames to {out_dir}")
