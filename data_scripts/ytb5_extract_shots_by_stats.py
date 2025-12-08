#!/usr/bin/env python3
"""
Extract images from video shots at calculated target FPS.
Reads shot metadata from JSON and extracts frames using cv2.
"""

import os
import sys
import json
import subprocess
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm

# Add EHM-Tracker to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.general_utils import parallel_foreach

# Import FPS calculation function
sys.path.insert(0, os.path.dirname(__file__))
from ytb4_shots_stats import calculate_video_extraction_fps


def load_videos_list(videos_json, split):
    """
    Load video list from videos_info.json based on split.
    
    Args:
        videos_json: Path to videos_info.json
        split: 'train' or 'test'
    
    Returns:
        List of video names
    """
    with open(videos_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    videos_list = data.get(split, [])
    return videos_list


def get_video_fps(video_path):
    """
    Get the FPS of a video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        FPS as float
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def frame_key_to_timestamp(frame_key, shots_fps):
    """
    Convert frame key (int string) to timestamp in seconds.
    
    Args:
        frame_key: String frame index (e.g., "000042")
        shots_fps: FPS used when extracting keyframes
    
    Returns:
        Timestamp in seconds (float)
    """
    frame_num = int(frame_key)
    timestamp = frame_num / shots_fps
    return timestamp


def calculate_square_bbox(bbox, scale=1.1):
    """
    Calculate square bbox from a regular bbox and expand by scale.
    
    Args:
        bbox: [x0, y0, x1, y1] bounding box
        scale: Scale factor to expand the bbox (default 1.1 = 10% expansion)
    
    Returns:
        Square bbox [x0, y0, x1, y1] centered on original bbox
    """
    x0, y0, x1, y1 = bbox
    
    # Calculate center and size
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = x1 - x0
    h = y1 - y0
    
    # Make it square (use the larger dimension) and expand by scale
    size = max(w, h) * scale
    half_size = size / 2
    
    # Calculate square bbox
    square_x0 = cx - half_size
    square_y0 = cy - half_size
    square_x1 = cx + half_size
    square_y1 = cy + half_size
    
    return [square_x0, square_y0, square_x1, square_y1]


def crop_image_with_bbox(image, bbox):
    """
    Crop image with bbox, clamping to image boundaries and padding if needed.
    
    Args:
        image: Input image (numpy array)
        bbox: [x0, y0, x1, y1] bounding box (can be float)
    
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    x0, y0, x1, y1 = [int(b) for b in bbox]
    
    # Calculate padding needed
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)
    
    # Clamp to image boundaries
    x0_img = max(0, min(x0, w))
    y0_img = max(0, min(y0, h))
    x1_img = max(0, min(x1, w))
    y1_img = max(0, min(y1, h))
    
    # Crop
    if x1_img <= x0_img or y1_img <= y0_img:
        # Invalid crop region, return original
        return image
    
    cropped = image[y0_img:y1_img, x0_img:x1_img]
    
    # Add padding if needed
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        import numpy as np
        cropped = np.pad(cropped, 
                        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                        mode='constant', constant_values=0)
    
    return cropped


def convert_shot_key(shot_key, shots_fps, extract_fps):
    """
    Convert shot_key from shots_fps to extract_fps frame numbers.
    
    Args:
        shot_key: Original shot key in format "start_end" (e.g., "000010_000050")
        shots_fps: FPS used when extracting keyframes
        extract_fps: Target FPS for extraction
    
    Returns:
        New shot key in extract_fps frame numbers
    """
    parts = shot_key.split('_')
    if len(parts) != 2:
        return shot_key
    
    start_frame_num = int(parts[0])
    end_frame_num = int(parts[1])
    
    # Convert to timestamps then to extract_fps frame numbers
    start_time = start_frame_num / shots_fps
    end_time = end_frame_num / shots_fps
    
    new_start = int(start_time * extract_fps)
    new_end = int(end_time * extract_fps)
    
    return f"{new_start:06d}_{new_end:06d}"


def create_preview_video(shot_output_dir, shot_key, extract_fps):
    """
    Create a preview MP4 video from extracted frames.
    
    Args:
        shot_output_dir: Directory containing extracted frames
        shot_key: Shot identifier for output filename
        extract_fps: Original extraction FPS (None = use 10fps for preview)
    
    Returns:
        True if successful, False otherwise
    """
    # Determine preview FPS (at least 10fps)
    if extract_fps is None:
        preview_fps = 10
    else:
        preview_fps = max(10, extract_fps)
    
    # Input pattern and output path
    input_pattern = str(shot_output_dir / "%06d.jpg")
    output_video = shot_output_dir.parent / f"{shot_key}.mp4"
    audio_file = shot_output_dir / "audio.wav"
    
    # Check if we should include audio (extract_fps >= preview_fps and audio exists)
    include_audio = (extract_fps is not None and extract_fps >= preview_fps and audio_file.exists())
    
    # Build ffmpeg command for preview video
    cmd = ['ffmpeg', '-framerate', str(preview_fps), '-i', input_pattern]
    
    # Add audio input if appropriate
    if include_audio:
        cmd.extend(['-i', str(audio_file)])
    
    # Video encoding options
    cmd.extend([
        # '-vf', 'scale=iw/2:ih/2',             # Half resolution
        '-c:v', 'libx264',                     # H.264 codec
        '-preset', 'medium',                   # Encoding preset
        '-crf', '23',                          # Quality (lower = better)
        '-pix_fmt', 'yuv420p',                # Pixel format for compatibility
    ])
    
    # Audio encoding options if audio is included
    if include_audio:
        cmd.extend([
            '-c:a', 'aac',                     # AAC audio codec
            '-b:a', '128k',                    # Audio bitrate
            '-shortest',                       # Match shortest stream duration
        ])
    
    # Output options
    cmd.extend([
        str(output_video),                     # Output file
        '-y',                                  # Overwrite without asking
        '-loglevel', 'error'                   # Only show errors
    ])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"    Warning: Failed to create preview video: {e.stderr.decode()}")
        return False


def extract_shot_frames(video_path, shot_key, start_frame, end_frame, shots_fps, extract_fps, output_dir, video_fps=None, crop_bbox=None, crop_size=None):
    """
    Extract frames from a video shot using cv2.
    
    Args:
        video_path: Path to source video file
        shot_key: Shot identifier (e.g., "000010_000050") - will be used as output directory name
        start_frame: Start frame key (string)
        end_frame: End frame key (string)
        shots_fps: FPS used for keyframe extraction
        extract_fps: Target FPS for extraction (None = use video's real FPS)
        output_dir: Output directory for frames
        video_fps: Video's real FPS (used when extract_fps is None)
        crop_bbox: Optional square bbox to crop frames [x0, y0, x1, y1]
        crop_size: Target size for resizing cropped images (e.g., 1024)
    
    Returns:
        Number of frames extracted
    """

    # Convert frame keys to timestamps
    start_time = frame_key_to_timestamp(start_frame, extract_fps)
    end_time = frame_key_to_timestamp(end_frame, extract_fps)
    duration = end_time - start_time
    
    # Create output directory
    shot_output_dir = output_dir / shot_key
    shot_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video with cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: Cannot open video {video_path}")
        return 0
    
    # Get video properties
    video_fps_actual = cap.get(cv2.CAP_PROP_FPS)
    
    # Determine extraction FPS
    if extract_fps is None:
        target_fps = video_fps_actual
        frame_interval = 1
    else:
        target_fps = extract_fps
        frame_interval = max(1, int(video_fps_actual / target_fps))
    
    # Calculate start and end frame numbers in video
    start_frame_num = int(start_time * video_fps_actual)
    end_frame_num = int(end_time * video_fps_actual)
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_num)
    
    # Extract frames
    saved_count = 0
    current_frame_num = start_frame_num
    
    while current_frame_num <= end_frame_num:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at specified interval
        if (current_frame_num - start_frame_num) % frame_interval == 0:
            # Crop if bbox provided
            if crop_bbox is not None:
                frame = crop_image_with_bbox(frame, crop_bbox)
            
            # Resize to target size if specified
            if crop_size is not None and crop_bbox is not None:
                frame = cv2.resize(frame, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
            
            # Save frame
            output_path = shot_output_dir / f"{saved_count:06d}.jpg"
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
        
        current_frame_num += 1
    
    cap.release()
    
    if saved_count == 0:
        print(f"  Warning: No frames extracted for {shot_key}")
        return 0
    
    # Extract audio segment
    audio_output = shot_output_dir / "audio.wav"
    audio_cmd = [
        'ffmpeg',
        '-ss', str(start_time),           # Start time
        '-t', str(duration),               # Duration
        '-i', str(video_path),             # Input video
        '-vn',                             # No video
        '-acodec', 'pcm_s16le',           # PCM 16-bit WAV codec
        '-ar', '44100',                    # Sample rate 44.1kHz
        '-ac', '2',                        # Stereo
        str(audio_output),                 # Output audio file
        '-y',                              # Overwrite without asking
        '-loglevel', 'error'               # Only show errors
    ]
    
    try:
        subprocess.run(audio_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        # Audio extraction failure is not critical, just warn
        pass
    
    # Count extracted frames
    extracted_frames = list(shot_output_dir.glob('*.jpg'))
    return len(extracted_frames)


def process_video(func_args):
    """
    Process one video and extract frames for all shots.
    Calculates optimal extraction FPS based on target_frames and min_frames.
    
    Args:
        func_args: Dictionary with:
            - video_name: Video name (without extension)
            - videos_dir: Directory containing source videos
            - shots_dir: Directory containing shot JSON files
            - images_dir: Output directory for extracted images
            - shots_fps: FPS used when extracting keyframes
            - target_frames: Target frame count per video
            - min_frames: Minimum frame count threshold
            - crop_shot_bbox: Target size for cropping and resizing (e.g., 1024), None to disable
    
    Returns:
        Number of shots processed
    """
    video_name = func_args['video_name']
    videos_dir = func_args['videos_dir']
    shots_dir = func_args['shots_dir']
    images_dir = func_args['images_dir']
    shots_fps = func_args['shots_fps']
    target_frames = func_args['target_frames']
    min_frames = func_args['min_frames']
    crop_shot_bbox = func_args.get('crop_shot_bbox', None)
    # Input paths
    shot_json = Path(shots_dir) / f"{video_name}.json"
    
    # Find video file by searching for files starting with video_name
    videos_dir_path = Path(videos_dir)
    matching_videos = [f for f in videos_dir_path.glob('*.mp4') if f.stem.startswith(video_name)]
    
    if len(matching_videos) == 0:
        print(f"Error: No video found starting with {video_name}")
        return 0
    elif len(matching_videos) > 1:
        print(f"Warning: Multiple videos found starting with {video_name}, using first: {matching_videos[0].name}")
    
    video_path = matching_videos[0]
    
    if not shot_json.exists():
        print(f"Error: Shot JSON not found: {shot_json}")
        return 0
    
    # Load shot metadata
    with open(shot_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if video_name not in data:
        print(f"Error: Video {video_name} not found in JSON")
        return 0
    
    video_data = data[video_name]
    shot_keys = video_data.get('shot_keys', [])
    
    if len(shot_keys) == 0:
        print(f"Warning: No shots found for {video_name}")
        return 0
    
    # Calculate optimal extraction FPS
    fps_result = calculate_video_extraction_fps(video_data, target_frames, min_frames)
    
    if fps_result['skip']:
        print(f"Skipping {video_name}: No FPS option gives at least {min_frames} frames")
        return 0
    
    extract_fps = fps_result['best_fps']
    estimated_frames = fps_result['estimated_frames']
    
    # Create output directory for this video
    output_dir = Path(images_dir) / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip if json exist
    output_json = output_dir / f"{video_name}.json"
    if output_json.exists():
        print(f"  Skipping {video_name}, output JSON already exists: {output_json}")
        return 0
    
    # Get video's real FPS
    try:
        video_fps = get_video_fps(video_path)
        print(f"\nProcessing {video_name}: {len(shot_keys)} shot(s), extract FPS: {extract_fps}, estimated frames: {estimated_frames:.0f}")
    except Exception as e:
        print(f"\nWarning: Could not get video FPS for {video_name}: {e}")
        print(f"Processing {video_name}: {len(shot_keys)} shot(s), extract FPS: {extract_fps}")
        video_fps = None
    
    shots_processed = 0
    total_frames = 0
    
    # Metadata for output JSON
    extracted_shot_keys = []
    all_frames_keys = []
    shot_bboxes = {}  # Store union bbox for each shot
    
    # Determine effective FPS for conversion
    effective_extract_fps = extract_fps if extract_fps is not None else video_fps
    
    # Get frames_keys with bbox info from the original shot JSON
    original_frames_keys = video_data.get('frames_keys', {})
    
    for shot_key in shot_keys:
        # Parse shot_key to get start and end frame
        parts = shot_key.split('_')
        if len(parts) != 2:
            print(f"  Warning: Invalid shot_key format: {shot_key}")
            continue
        
        start_frame, end_frame = parts
        
        # Skip single-frame shots
        if start_frame == end_frame:
            print(f"  Skipping single-frame shot: {shot_key}")
            continue
        
        # Convert shot_key to extract_fps frame numbers
        if effective_extract_fps is not None and effective_extract_fps != shots_fps:
            converted_shot_key = convert_shot_key(shot_key, shots_fps, effective_extract_fps)
        else:
            converted_shot_key = shot_key

        # Update start_frame and end_frame from converted_shot_key
        conv_parts = converted_shot_key.split('_')
        start_frame = int(conv_parts[0])
        end_frame = int(conv_parts[1])
        
        # Calculate union bbox first (will be used both for metadata and cropping)
        union_bbox = None
        for orig_frame_key in original_frames_keys:
            frame_info = original_frames_keys[orig_frame_key]
            # Check if this frame belongs to the current shot
            orig_frame_num = int(orig_frame_key)
            shot_start = int(parts[0])  # Original shot start
            shot_end = int(parts[1])    # Original shot end
            
            if shot_start <= orig_frame_num <= shot_end:
                bbox = frame_info.get('bbox_xyxy')
                if bbox is not None and len(bbox) == 4:
                    if union_bbox is None:
                        union_bbox = list(bbox)
                    else:
                        # Union: min of x0,y0 and max of x1,y1
                        union_bbox[0] = min(union_bbox[0], bbox[0])
                        union_bbox[1] = min(union_bbox[1], bbox[1])
                        union_bbox[2] = max(union_bbox[2], bbox[2])
                        union_bbox[3] = max(union_bbox[3], bbox[3])
        
        # Calculate square bbox if cropping is enabled
        crop_bbox = None
        if crop_shot_bbox is not None and union_bbox is not None:
            crop_bbox = calculate_square_bbox(union_bbox, scale=1.1)
        
        # Extract frames using converted shot key
        num_frames = extract_shot_frames(
            video_path, converted_shot_key, start_frame, end_frame,
            shots_fps, extract_fps, output_dir, video_fps, crop_bbox, crop_shot_bbox
        )
        
        if num_frames > 0:
            print(f"  Extracted {shot_key} -> {converted_shot_key}: {num_frames} frames")
            
            # Store the union bbox for this shot
            if union_bbox is not None:
                shot_bboxes[converted_shot_key] = union_bbox
            
            # Collect metadata
            extracted_shot_keys.append(converted_shot_key)
            shot_output_dir = output_dir / converted_shot_key
            for img in sorted(shot_output_dir.glob('*.jpg')):
                frame_key = img.stem
                all_frames_keys.append(f"{converted_shot_key}/{frame_key}")
            
            # Create preview video
            if create_preview_video(shot_output_dir, converted_shot_key, effective_extract_fps):
                print(f"    Created preview video: {converted_shot_key}.mp4")
            
            shots_processed += 1
            total_frames += num_frames
        else:
            print(f"  Warning: No frames extracted for {shot_key}")
    
    # Save metadata JSON
    metadata = {
        video_name: {
            "frames_num": total_frames,
            "frames_keys": all_frames_keys,
            "shot_keys": extracted_shot_keys,
            "shot_bbox_xyxy": shot_bboxes,
            "fps": effective_extract_fps
        }
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"  Total: {shots_processed} shot(s), {total_frames} frames")
    print(f"  Saved metadata to {output_json}")
    return shots_processed


def main():
    parser = argparse.ArgumentParser(description='Extract frames from video shots at calculated target FPS')
    parser.add_argument('--videos_dir', type=str, required=True,
                        help='Directory containing source MP4 videos')
    parser.add_argument('--shots_dir', type=str, required=True,
                        help='Directory containing shot JSON files')
    parser.add_argument('--shots_fps', type=float, required=True,
                        help='FPS used when extracting keyframes')
    parser.add_argument('--target_frames', type=int, default=150,
                        help='Target frame count per video (default: 150)')
    parser.add_argument('--min_frames', type=int, default=100,
                        help='Minimum frame count threshold (default: 100)')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Output directory for extracted images')
    parser.add_argument('--crop_shot_bbox', type=int, default=1024,
                        help='Target size for cropping and resizing frames (default: 1024). Set to 0 to disable cropping.')
    
    args = parser.parse_args()
    
    # Setup
    videos_dir = Path(args.videos_dir)
    shots_dir = Path(args.shots_dir)
    images_dir = Path(args.images_dir)
    
    if not videos_dir.exists():
        raise ValueError(f"Videos directory not found: {videos_dir}")
    if not shots_dir.exists():
        raise ValueError(f"Shots directory not found: {shots_dir}")
    
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all videos from shots_dir
    json_files = sorted(list(shots_dir.glob('*.json')))
    videos_list = []
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Each JSON file should have one video
        for video_name in data.keys():
            videos_list.append(video_name)
    
    if len(videos_list) == 0:
        raise ValueError(f"No videos found in {shots_dir}")
    
    print(f"Found {len(videos_list)} videos in shots directory")
    print(f"Shots FPS: {args.shots_fps}")
    print(f"Target frames: {args.target_frames}")
    print(f"Min frames: {args.min_frames}")
    print(f"Crop and resize: {'Enabled (' + str(args.crop_shot_bbox) + 'x' + str(args.crop_shot_bbox) + ')' if args.crop_shot_bbox > 0 else 'Disabled'}")
    
    # Prepare function arguments for parallel processing
    crop_size = args.crop_shot_bbox if args.crop_shot_bbox > 0 else None
    func_args_list = []
    for video_name in videos_list:
        func_args_list.append({
            'video_name': video_name,
            'videos_dir': videos_dir,
            'shots_dir': shots_dir,
            'images_dir': images_dir,
            'shots_fps': args.shots_fps,
            'target_frames': args.target_frames,
            'min_frames': args.min_frames,
            'crop_shot_bbox': crop_size
        })
    
    # Process videos in parallel
    print(f"\nProcessing videos in parallel...")
    results = parallel_foreach(process_video, func_args_list, max_workers=8)
    
    # Calculate total shots
    total_shots = sum(results)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total videos processed: {len(videos_list)}")
    print(f"Total shots extracted: {total_shots}")
    print(f"Output directory: {images_dir}")


if __name__ == '__main__':
    main()
