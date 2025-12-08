#!/usr/bin/env python3
"""
Print statistics about shot JSON files.
"""

import os
import sys
import json
from pathlib import Path
import argparse


def collect_stats(shots_dir):
    """
    Collect statistics from shot JSON files.
    
    Args:
        shots_dir: Directory containing shot JSON files
    
    Returns:
        Dictionary with statistics
    """
    shots_dir = Path(shots_dir)
    
    # Collect all JSON files
    json_files = sorted(list(shots_dir.glob('*.json')))
    
    if len(json_files) == 0:
        raise ValueError(f"No JSON files found in {shots_dir}")
    
    # Statistics
    total_videos = 0
    frames_per_video = []
    shots_per_video = []
    
    # Process each JSON file
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Each JSON file should have one video
        for video_name, video_data in data.items():
            total_videos += 1
            
            # Get number of frames
            frames_keys = video_data.get('frames_keys', {})
            num_frames = len(frames_keys)
            frames_per_video.append(num_frames)
            
            # Get number of shots
            shot_keys = video_data.get('shot_keys', [])
            num_shots = len(shot_keys)
            shots_per_video.append(num_shots)
    
    return {
        'total_videos': total_videos,
        'frames_per_video': frames_per_video,
        'shots_per_video': shots_per_video
    }


def calculate_video_extraction_fps(video_data, target_frames=150, min_frames=100, fps_options=None):
    """
    Calculate optimal extraction FPS for a single video.
    Choose FPS option that gives frames > min_frames and closest to target_frames.
    
    Args:
        video_data: Dictionary with shot_keys and frames_keys from JSON
        target_frames: Target frames per video (default: 150)
        min_frames: Minimum frames threshold (default: 100)
        fps_options: List of FPS options to choose from (default: [0.5, 1, 2, 5, 10])
    
    Returns:
        Dictionary with:
        - 'best_fps': Selected FPS (None if should skip)
        - 'estimated_frames': Estimated frame count at best_fps
        - 'total_duration_seconds': Total duration of multi-frame shots
        - 'num_multi_frame_shots': Number of multi-frame shots
        - 'skip': Boolean indicating if video should be skipped
    """
    if fps_options is None:
        fps_options = [0.5, 1, 2, 5, 10]
    
    shot_keys = video_data.get('shot_keys', [])
    
    # Filter non-single frame shots
    multi_frame_shots = []
    for shot_key in shot_keys:
        # shot_key format: "000001_000002" (start_end)
        parts = shot_key.split('_')
        if len(parts) == 2:
            start_frame = int(parts[0])
            end_frame = int(parts[1])
            # Duration in keyframes (2 seconds between keyframes)
            duration_keyframes = end_frame - start_frame + 1
            if duration_keyframes > 1:
                multi_frame_shots.append({
                    'shot_key': shot_key,
                    'duration_keyframes': duration_keyframes
                })
    
    # Calculate total duration in seconds (2 seconds between consecutive keyframes)
    total_duration_seconds = 0
    for shot_info in multi_frame_shots:
        # duration = (num_keyframes - 1) * 2 seconds
        total_duration_seconds += (shot_info['duration_keyframes'] - 1) * 2
    
    # Find FPS option that gives frames > min_frames and closest to target_frames
    best_fps = None
    best_diff = float('inf')
    
    for fps in fps_options:
        estimated_frames = total_duration_seconds * fps
        
        # Must be above min_frames
        if estimated_frames < min_frames:
            continue
        
        # Find closest to target_frames
        diff = abs(estimated_frames - target_frames)
        if diff < best_diff:
            best_diff = diff
            best_fps = fps
    
    # Determine if should skip
    skip = (best_fps is None)
    estimated_frames_at_best_fps = 0 if skip else total_duration_seconds * best_fps
    
    return {
        'best_fps': best_fps,
        'estimated_frames': estimated_frames_at_best_fps,
        'total_duration_seconds': total_duration_seconds,
        'num_multi_frame_shots': len(multi_frame_shots),
        'skip': skip
    }


def analyze_extraction_fps(shots_dir, target_frames=150, min_frames=100):
    """
    Analyze optimal extraction FPS for each video.
    Choose the least FPS option that gives more than target_frames.
    
    Args:
        shots_dir: Directory containing shot JSON files
        target_frames: Target minimum frames per video (default: 150)
        min_frames: Minimum frames to not skip video (default: 100)
    
    Returns:
        Dictionary with FPS analysis
    """
    shots_dir = Path(shots_dir)
    json_files = sorted(list(shots_dir.glob('*.json')))
    
    fps_options = [0.5, 1, 2, 5, 10]
    fps_counts = {fps: 0 for fps in fps_options}
    video_fps_decisions = []
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for video_name, video_data in data.items():
            # Calculate FPS for this video
            result = calculate_video_extraction_fps(video_data, target_frames, min_frames, fps_options)
            
            # Skip if no valid FPS found
            if result['skip']:
                continue
            
            best_fps = result['best_fps']
            fps_counts[best_fps] += 1
            video_fps_decisions.append({
                'video_name': video_name,
                'best_fps': best_fps,
                'num_multi_frame_shots': result['num_multi_frame_shots'],
                'total_duration_seconds': result['total_duration_seconds'],
                'estimated_frames': result['estimated_frames']
            })
    
    return {
        'fps_counts': fps_counts,
        'video_fps_decisions': video_fps_decisions
    }


def print_stats(stats):
    """
    Print statistics in a formatted way.
    
    Args:
        stats: Dictionary with statistics from collect_stats
    """
    total_videos = stats['total_videos']
    frames_per_video = stats['frames_per_video']
    shots_per_video = stats['shots_per_video']
    
    print(f"\n{'='*60}")
    print("Shot Statistics")
    print(f"{'='*60}\n")
    
    # 1. Total number of videos
    print(f"Total number of videos: {total_videos}")
    
    # 2. Average number of frames in videos
    if len(frames_per_video) > 0:
        avg_frames = sum(frames_per_video) / len(frames_per_video)
        print(f"Average number of frames per video: {avg_frames:.2f}")
    
    # 3. Average number of shots in videos
    if len(shots_per_video) > 0:
        avg_shots = sum(shots_per_video) / len(shots_per_video)
        print(f"Average number of shots per video: {avg_shots:.2f}")
    
    print()
    
    # 4. Number of videos with more than X frames
    frame_thresholds = [10, 20, 50, 100, 200]
    print("Number of videos with more than X frames:")
    for threshold in frame_thresholds:
        count = sum(1 for n in frames_per_video if n > threshold)
        percentage = (count / total_videos * 100) if total_videos > 0 else 0
        print(f"  > {threshold:3d} frames: {count:4d} ({percentage:5.1f}%)")
    
    print()
    
    # 5. Number of videos with more than X shots
    shot_thresholds = [2, 5, 10, 20]
    print("Number of videos with more than X shots:")
    for threshold in shot_thresholds:
        count = sum(1 for n in shots_per_video if n > threshold)
        percentage = (count / total_videos * 100) if total_videos > 0 else 0
        print(f"  > {threshold:2d} shots: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\n{'='*60}")


def print_fps_analysis(fps_analysis, target_frames, min_frames):
    """
    Print FPS analysis results.
    
    Args:
        fps_analysis: Dictionary with FPS analysis from analyze_extraction_fps
        target_frames: Target minimum frames
        min_frames: Minimum frames threshold
    """
    fps_counts = fps_analysis['fps_counts']
    video_fps_decisions = fps_analysis['video_fps_decisions']
    
    total_videos = len(video_fps_decisions)
    
    print(f"\n{'='*60}")
    print("Extraction FPS Analysis")
    print(f"Target: > {target_frames} frames per video (using non-single frame shots)")
    print(f"Minimum: {min_frames} frames (videos below this are skipped)")
    print("Duration between keyframes: 2 seconds")
    print(f"{'='*60}\n")
    
    print("Number of videos for each FPS option:")
    for fps in sorted(fps_counts.keys()):
        count = fps_counts[fps]
        percentage = (count / total_videos * 100) if total_videos > 0 else 0
        print(f"  {fps:4.1f} fps: {count:4d} videos ({percentage:5.1f}%)")
        
        # Print up to 3 example video names
        videos_with_fps = [v for v in video_fps_decisions if v['best_fps'] == fps]
        for i, video in enumerate(videos_with_fps[:3]):
            print(f"      - {video['video_name']} ({video['estimated_frames']:.0f} frames)")
    
    print()
    
    # Calculate average estimated frames per FPS option
    print("Average estimated frames per FPS option:")
    for fps in sorted(fps_counts.keys()):
        videos_with_fps = [v for v in video_fps_decisions if v['best_fps'] == fps]
        if len(videos_with_fps) > 0:
            avg_frames = sum(v['estimated_frames'] for v in videos_with_fps) / len(videos_with_fps)
            print(f"  {fps:4.1f} fps: {avg_frames:6.1f} frames (avg)")
    
    print()
    
    # Total frames for all not skipped videos
    total_frames_all_videos = sum(v['estimated_frames'] for v in video_fps_decisions)
    print(f"Total frames of all not skipped videos: {total_frames_all_videos:.0f}")
    
    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Print statistics about shot JSON files')
    parser.add_argument('--shots_dir', type=str, required=True,
                        help='Directory containing shot JSON files')
    parser.add_argument('--target_frames', type=int, default=150,
                        help='Target minimum frames per video (default: 150)')
    parser.add_argument('--min_frames', type=int, default=100,
                        help='Minimum frames threshold to include video (default: 100)')
    
    args = parser.parse_args()
    
    # Setup
    shots_dir = Path(args.shots_dir)
    
    if not shots_dir.exists():
        raise ValueError(f"Shots directory does not exist: {shots_dir}")
    
    # Collect statistics
    print(f"Collecting statistics from: {shots_dir}")
    stats = collect_stats(shots_dir)
    
    # Print statistics
    print_stats(stats)
    
    # Analyze extraction FPS
    fps_analysis = analyze_extraction_fps(shots_dir, args.target_frames, args.min_frames)
    
    # Print FPS analysis
    print_fps_analysis(fps_analysis, args.target_frames, args.min_frames)


if __name__ == '__main__':
    main()
