#!/usr/bin/env python3
"""
infer_ehmx_parallel.py - Run EHM-X pipeline in parallel across multiple GPUs
Processes multiple videos in parallel by distributing them across available GPUs.
Each video runs through: track_base -> flame -> smplx sequentially.
"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from multiprocessing import Process, Queue, Manager
import time
from tqdm import tqdm


def find_index_files(images_dir, index_mode):
    """
    Find all index JSON files based on index_mode.
    
    Args:
        images_dir: Root directory containing video subdirectories
        index_mode: Either 'video_name' or 'video_name_pshuman'
    
    Returns:
        List of (video_name, json_path) tuples
    """
    images_path = Path(images_dir)
    index_files = []
    
    if index_mode == 'video_name':
        pattern = '*/*.json'
        # Find all <video_name>/<video_name>.json files
        for json_file in images_path.glob(pattern):
            video_name = json_file.parent.name
            expected_name = f"{video_name}.json"
            if json_file.name == expected_name:
                index_files.append((video_name, str(json_file)))
    
    elif 'video_name' in index_mode:
        pattern = f"*/*{index_mode.replace('video_name', '')}.json"
        # Find all <video_name>/<video_name>_pshuman.json files
        for json_file in images_path.glob(pattern):
            video_name = json_file.parent.name
            expected_name = f"{index_mode.replace('video_name', video_name)}.json"
            if json_file.name == expected_name:
                index_files.append((video_name, str(json_file)))
    else:
        raise ValueError(f"Unknown index_mode: {index_mode}. Use 'video_name' or 'video_name_pshuman'")
    
    return sorted(index_files)


def run_video_pipeline(video_name, json_path, args, gpu_id):
    """
    Run the complete EHM-X pipeline for a single video on a specific GPU.
    
    Args:
        video_name: Name of the video
        json_path: Path to the index JSON file
        args: Command line arguments
        gpu_id: GPU device ID to use
    """
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"[GPU {gpu_id}] Starting pipeline for: {video_name}")
    
    # Define output directory for this video
    video_out_dir = os.path.join(args.ehmx_dir, video_name)
    
    # Common arguments for all scripts
    common_args = [
        '--images_dir', args.images_dir,
        '--index_json', json_path,
        '--ehmx_dir', args.ehmx_dir,
    ]
    
    if args.mattes_dir:
        common_args.extend(['--mattes_dir', args.mattes_dir])
    
    if args.pshuman_dir:
        common_args.extend(['--pshuman_dir', args.pshuman_dir])
    
    # Config argument for FLAME and SMPL-X
    config_args = ['--config', args.config] if args.config else []
    
    # Determine overwrite flags for each step
    overwrite_base = False
    overwrite_flame = False
    overwrite_smplx = False
    
    if args.overwrite:
        for item in args.overwrite:
            if item in ['base_track', 'track_base', 'all']:
                overwrite_base = True
            if item in ['flame', 'all']:
                overwrite_flame = True
            if item in ['smplx', 'all']:
                overwrite_smplx = True
    
    # Step 1: Track base
    base_tracking_pkl = os.path.join(video_out_dir, 'base_tracking.pkl')
    if os.path.exists(base_tracking_pkl) and not overwrite_base:
        print(f"[GPU {gpu_id}] [{video_name}] Step 1/3: Skipping track_base (already exists)")
    else:
        print(f"[GPU {gpu_id}] [{video_name}] Step 1/3: Running track_base...")
        cmd_track_base = [
            sys.executable, 'infer_ehmx_track_base.py'
        ] + common_args
        
        if overwrite_base:
            cmd_track_base.append('--overwrite')
        
        if args.check_hand_score is not None:
            cmd_track_base.extend(['--check_hand_score', str(args.check_hand_score)])
        
        try:
            result = subprocess.run(cmd_track_base, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[GPU {gpu_id}] [{video_name}] ✗ track_base failed:")
                print(result.stderr)
                return False
            print(f"[GPU {gpu_id}] [{video_name}] ✓ track_base completed")
        except Exception as e:
            print(f"[GPU {gpu_id}] [{video_name}] ✗ track_base error: {e}")
            return False
    
    # Step 2: FLAME refinement
    track_flame_pkl = os.path.join(video_out_dir, 'track_flame.pkl')
    if os.path.exists(track_flame_pkl) and not overwrite_flame:
        print(f"[GPU {gpu_id}] [{video_name}] Step 2/3: Skipping FLAME (already exists)")
    else:
        print(f"[GPU {gpu_id}] [{video_name}] Step 2/3: Running FLAME refinement...")
        cmd_flame = [
            sys.executable, 'infer_ehmx_flame.py'
        ] + common_args + config_args
        
        if overwrite_flame:
            cmd_flame.append('--overwrite')
        
        try:
            result = subprocess.run(cmd_flame, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[GPU {gpu_id}] [{video_name}] ✗ FLAME failed:")
                print(result.stderr)
                return False
            print(f"[GPU {gpu_id}] [{video_name}] ✓ FLAME completed")
        except Exception as e:
            print(f"[GPU {gpu_id}] [{video_name}] ✗ FLAME error: {e}")
            return False
    
    # Step 3: SMPL-X refinement
    track_smplx_pkl = os.path.join(video_out_dir, 'optim_tracking_ehm.pkl')
    if os.path.exists(track_smplx_pkl) and not overwrite_smplx:
        print(f"[GPU {gpu_id}] [{video_name}] Step 3/3: Skipping SMPL-X (already exists)")
    else:
        print(f"[GPU {gpu_id}] [{video_name}] Step 3/3: Running SMPL-X refinement...")
        cmd_smplx = [
            sys.executable, 'infer_ehmx_smplx.py'
        ] + common_args + config_args
        
        if overwrite_smplx:
            cmd_smplx.append('--overwrite')
        
        try:
            result = subprocess.run(cmd_smplx, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[GPU {gpu_id}] [{video_name}] ✗ SMPL-X failed:")
                print(result.stderr)
                return False
            print(f"[GPU {gpu_id}] [{video_name}] ✓ SMPL-X completed")
        except Exception as e:
            print(f"[GPU {gpu_id}] [{video_name}] ✗ SMPL-X error: {e}")
            return False
    
    print(f"[GPU {gpu_id}] [{video_name}] ✓✓✓ Complete pipeline finished!")
    return True


def worker_process(gpu_id, task_queue, progress_queue, args):
    """
    Worker process that pulls tasks from queue and processes them.
    
    Args:
        gpu_id: GPU device ID to use
        task_queue: Queue containing (video_name, json_path) tuples
        progress_queue: Queue for reporting progress updates
        args: Command line arguments
    """
    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:  # Poison pill to stop worker
                break
            
            video_name, json_path = task
            success = run_video_pipeline(video_name, json_path, args, gpu_id)
            
            # Report completion to progress queue
            progress_queue.put({
                'video_name': video_name,
                'gpu_id': gpu_id,
                'success': success
            })
            
            if success:
                print(f"[GPU {gpu_id}] ✓ Successfully processed: {video_name}")
            else:
                print(f"[GPU {gpu_id}] ✗ Failed to process: {video_name}")
        
        except Exception as e:
            print(f"[GPU {gpu_id}] Worker error: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description='Run EHM-X pipeline in parallel across multiple GPUs'
    )
    
    # Required arguments
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Root directory containing video image subdirectories')
    parser.add_argument('--ehmx_dir', type=str, required=True,
                        help='Directory for EHM-X tracking results')
    
    # Optional arguments
    parser.add_argument('--mattes_dir', type=str, default=None,
                        help='Directory containing matte images (optional)')
    parser.add_argument('--pshuman_dir', type=str, default=None,
                        help='Root directory containing PSHuman images (optional)')
    
    # Parallel processing arguments
    parser.add_argument('--distribute', type=str, default='0',
                        help='Comma-separated GPU IDs to use (e.g., "0,1,2,3")')
    parser.add_argument('--index_mode', type=str, default='video_name',
                        help='Index file naming mode: "video_name" for <video>/<video>.json, '
                             '"video_name_pshuman" for <video>/<video>_pshuman.json')
    parser.add_argument('--overwrite', nargs='*', default=None,
                        help='Overwrite existing results. Options: track_base, flame, smplx, all')
    parser.add_argument('--check_hand_score', type=float, default=None,
                        help='Overwrite config.check_hand_score if set')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to optimization config file')
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(gpu_id.strip()) for gpu_id in args.distribute.split(',')]
    print(f"Using GPUs: {gpu_ids}")
    
    # Find all index files
    print(f"\nSearching for index files in: {args.images_dir}")
    print(f"Index mode: {args.index_mode}")
    
    index_files = find_index_files(args.images_dir, args.index_mode)
    
    if not index_files:
        print(f"No index files found in {args.images_dir}")
        return
    
    print(f"Found {len(index_files)} video(s) to process:")
    for video_name, json_path in index_files:
        print(f"  - {video_name}: {json_path}")
    
    # Create task queue and add all tasks
    task_queue = Queue()
    progress_queue = Queue()
    
    for video_name, json_path in index_files:
        task_queue.put((video_name, json_path))
    
    # Add poison pills (one per worker)
    for _ in gpu_ids:
        task_queue.put(None)
    
    # Start worker processes
    print(f"\nStarting {len(gpu_ids)} worker process(es)...")
    workers = []
    for gpu_id in gpu_ids:
        p = Process(target=worker_process, args=(gpu_id, task_queue, progress_queue, args))
        p.start()
        workers.append(p)
    
    # Monitor progress with tqdm
    total_videos = len(index_files)
    completed = 0
    failed = 0
    
    with tqdm(total=total_videos, desc="Overall Progress", unit="video") as pbar:
        while completed + failed < total_videos:
            try:
                result = progress_queue.get(timeout=1)
                if result['success']:
                    completed += 1
                else:
                    failed += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'completed': completed,
                    'failed': failed,
                    'gpu': result['gpu_id']
                })
            except:
                # Check if all workers are done
                if all(not p.is_alive() for p in workers):
                    break
                continue
    
    # Wait for all workers to finish
    for p in workers:
        p.join()
    
    print("\n" + "="*80)
    print(f"All videos processed! Completed: {completed}, Failed: {failed}")
    print("="*80)


if __name__ == '__main__':
    main()
