#!/usr/bin/env python3
"""
draw_view_histogram.py - Visualize distribution of camera angle (yaw) for GT and Pseudo data.
Reads all_optim_track from pkl and plots comparison histogram.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import torch

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_yaw_from_global_orient(global_orient):
    # global_orient: (3,) or (1, 3) axis-angle
    if isinstance(global_orient, torch.Tensor):
        global_orient = global_orient.detach().cpu().numpy()
    
    global_orient = np.array(global_orient).flatten()
    
    # Create rotation object
    r = R.from_rotvec(global_orient)
    
    # Calculate forward vector (local Z-axis [0,0,1] rotated to global)
    # Assumes canonical SMPL faces +Z
    # We use vector transformation to avoid gimbal lock issues with Euler angles
    forward = r.apply([0, 0, 1])
    
    # Calculate yaw from the forward vector projection on XZ plane
    # We want 0 degrees to correspond to facing the camera (which is -Z direction)
    # atan2(x, z) gives angle relative to +Z
    # If facing camera (-Z): x=0, z=-1 -> atan2(0, -1) = 180
    # We subtract 180 to make facing camera 0 degrees
    
    angle = np.degrees(np.arctan2(forward[0], forward[2]))
    yaw = angle - 180.0
    
    # Normalize to [-180, 180]
    if yaw <= -180:
        yaw += 360
    if yaw > 180:
        yaw -= 360
        
    return yaw

def draw_histogram(data_gt, data_pseudo, bins_range, bin_width, title, xlabel, output_path, log_scale=False):
    plt.figure(figsize=(12, 7))
    
    min_val, max_val = bins_range
    # Setup bins
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    
    plot_data = []
    plot_labels = []
    plot_colors = []
    
    if len(data_gt) > 0:
        plot_data.append(data_gt)
        # plot_labels.append(f'GT Data (N={len(data_gt)})')
        plot_labels.append(f'Raw Frames')
        plot_colors.append('dodgerblue')
        
    if len(data_pseudo) > 0:
        plot_data.append(data_pseudo)
        # plot_labels.append(f'Pseudo Data (N={len(data_pseudo)})')
        plot_labels.append(f'Pseudo Frames')
        plot_colors.append('crimson')
        
    if not plot_data:
        print("No data to plot")
        plt.close()
        return

    # Plot side-by-side
    plt.hist(plot_data, bins=bins, label=plot_labels, color=plot_colors, 
             density=False, stacked=False, alpha=0.8, edgecolor='black', linewidth=0.5, log=log_scale)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Count (Log Scale)' if log_scale else 'Count', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([min_val, max_val])
    
    # Adjust ticks based on range
    step = 30 if (max_val - min_val) > 100 else 15
    plt.xticks(np.arange(min_val, max_val + 1, step))
    
    plt.tight_layout()
    
    print(f"Saving histogram to {output_path}...")
    plt.savefig(output_path, dpi=150)
    plt.close()

def draw_polar_histogram(data_gt, data_pseudo, bin_width, title, output_path, log_scale=False):
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    
    def to_radians(angles):
        return np.radians(angles)

    rad_gt = to_radians(data_gt)
    rad_pseudo = to_radians(data_pseudo)
    
    # Bins: full circle [-pi, pi]
    n_bins = int(360 / bin_width)
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]
    
    # Compute counts manually
    counts_gt, _ = np.histogram(rad_gt, bins=bins)
    counts_pseudo, _ = np.histogram(rad_pseudo, bins=bins)
    
    # Handle Log Scale
    if log_scale:
        # Use log10(count + 1)
        # 0 -> 0 (center)
        # 1 -> 0.3
        # 9 -> 1.0
        # 99 -> 2.0
        counts_gt = np.log10(counts_gt + 1)
        counts_pseudo = np.log10(counts_pseudo + 1)
    
    # Configure bar positions (side-by-side)
    if len(data_gt) > 0 and len(data_pseudo) > 0:
        # Split bin width
        bar_width = width / 2
        # GT on left (counter-clockwise?), Pseudo on right?
        # Let's shift centers.
        # Shift GT slightly "left" (negative angle direction?) and Pseudo "right"?
        # Actually in polar, theta increases counter-clockwise.
        # Let's put GT at theta - width/4, Pseudo at theta + width/4
        theta_gt = bin_centers - bar_width/2
        theta_pseudo = bin_centers + bar_width/2
    else:
        bar_width = width
        theta_gt = bin_centers
        theta_pseudo = bin_centers
        
    # Plot bars
    # Bottom should be 0.
    bottom = 0
    
    if len(data_gt) > 0:
        ax.bar(theta_gt, counts_gt, width=bar_width, bottom=bottom,
               color='dodgerblue', alpha=0.8, edgecolor='black', linewidth=0.5, 
               label=f'Raw Frames (N={len(data_gt)})', align='center')
               
    if len(data_pseudo) > 0:
        ax.bar(theta_pseudo, counts_pseudo, width=bar_width, bottom=bottom,
               color='crimson', alpha=0.8, edgecolor='black', linewidth=0.5, 
               label=f'Pseudo Frames (N={len(data_pseudo)})', align='center')
    
    # Configure polar plot
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1) # Clockwise
    
    # Set ticks
    ticks = np.linspace(0, 2*np.pi, 8, endpoint=False)
    tick_labels = ['0° (Front)', '45°', '90° (Right)', '135°', '180° (Back)', '-135°', '-90° (Left)', '-45°']
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    
    if log_scale:
        # Determine max log value to set appropriate ticks
        max_log_val = 0
        if len(counts_gt) > 0:
            max_log_val = max(max_log_val, np.max(counts_gt))
        if len(counts_pseudo) > 0:
            max_log_val = max(max_log_val, np.max(counts_pseudo))
            
        # We want ticks at 10, 100, 1000...
        # Their positions are log10(10+1), log10(100+1), etc.
        # log10(11)~1.04, log10(101)~2.004...
        
        # Determine power range
        # Start at 10^1
        powers = []
        labels = []
        p = 1
        while True:
            val = 10**p
            pos = np.log10(val + 1)
            if pos > max_log_val * 1.1: # Go slightly above max
                break
            powers.append(pos)
            labels.append(f'$10^{p}$')
            p += 1
            
        if not powers: # Handle case with very small counts
            powers = [np.log10(11)]
            labels = ['$10^1$']
            
        ax.set_yticks(powers)
        ax.set_yticklabels(labels)
    
    plt.title(title, fontsize=14, pad=20)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize=12)
    
    plt.tight_layout()
    print(f"Saving polar histogram to {output_path}...")
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize distribution of camera angle (yaw)")
    parser.add_argument('--pkl_path', required=True, help='Path to optim_tracking_ehm.pkl')
    parser.add_argument('--output', default='view_histogram.png', help='Output image path')
    args = parser.parse_args()
    
    if not os.path.exists(args.pkl_path):
        print(f"Error: Pickle file not found at {args.pkl_path}")
        return

    print(f"Loading data from {args.pkl_path}...")
    try:
        data = load_pkl(args.pkl_path)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return
    
    yaws_gt = []
    yaws_pseudo = []
    
    print("Processing videos...")
    for video_name, video_data in tqdm(data.items()):
        # Check dictionary structure
        if not isinstance(video_data, dict):
            print(f"Skipping {video_name}, expected dict but got {type(video_data)}")
            continue

        for frame_key, params in video_data.items():
            # Extract global_orient from the nested structure based on user inspection
            global_orient = None
            
            # Check for smplx_coeffs dict (primary location)
            if 'smplx_coeffs' in params and isinstance(params['smplx_coeffs'], dict):
                if 'global_pose' in params['smplx_coeffs']:
                    global_orient = params['smplx_coeffs']['global_pose']
                elif 'global_orient' in params['smplx_coeffs']:
                    global_orient = params['smplx_coeffs']['global_orient']
            
            # Fallback to direct keys
            if global_orient is None:
                if 'global_orient' in params:
                    global_orient = params['global_orient']
                elif 'root_pose' in params:
                    global_orient = params['root_pose']
            
            if global_orient is None:
                # print(f"Skipping frame {frame_key}: global pose not found")
                continue
                
            yaw = get_yaw_from_global_orient(global_orient)
            # print(f'[{video_name}][{frame_key}] yaw: {yaw:.2f}')
            
            if 'pshuman' in frame_key:
                if yaw > 0:
                    yaw -= 180
                else:
                    yaw += 180
                yaws_pseudo.append(yaw)
                # yaws_pseudo.append(yaw)
                # yaws_pseudo.append(yaw)
            else:
                yaws_gt.append(yaw)
    
    print(f"Num GT frames: {len(yaws_gt)}")
    print(f"Num Pseudo (PSHuman) frames: {len(yaws_pseudo)}")
    
    if len(yaws_gt) == 0 and len(yaws_pseudo) == 0:
        print("No valid data found to plot.")
        return

    # Plotting
    base_name, ext = os.path.splitext(args.output)
    if not ext:
        ext = '.png'
    
    # 1. Yaw Histogram (-180 to 180) - POLAR
    out_yaw = f"{base_name}_yaw{ext}"
    draw_polar_histogram(yaws_gt, yaws_pseudo, 
                  bin_width=10, 
                  title='Distribution of Yaw Angles (View Direction)', 
                  output_path=out_yaw)

    # 2. Absolute Yaw Histogram (0 to 180) - LINEAR
    # Convert to absolute
    abs_yaws_gt = [abs(y) for y in yaws_gt]
    abs_yaws_pseudo = [abs(y) for y in yaws_pseudo]
    
    out_abs = f"{base_name}_abs_yaw{ext}"
    draw_histogram(abs_yaws_gt, abs_yaws_pseudo, 
                  bins_range=(0, 180), 
                  bin_width=10, 
                  title='Distribution of Absolute Yaw Angles (Offset from Front)', 
                  xlabel='Absolute Yaw Angle (degrees)',
                  output_path=out_abs)
    
    # 3. Log Scale - Absolute Yaw Histogram (0 to 180) - LINEAR
    out_abs_log = f"{base_name}_abs_yaw_log{ext}"
    draw_histogram(abs_yaws_gt, abs_yaws_pseudo, 
                  bins_range=(0, 180), 
                  bin_width=10, 
                  title='Distribution of Absolute Yaw Angles (Log Scale)', 
                  xlabel='Absolute Yaw Angle (degrees)',
                  output_path=out_abs_log,
                  log_scale=True)
    
    # 4. Log Scale - Yaw Histogram (-180 to 180) - POLAR
    out_yaw_log = f"{base_name}_yaw_log{ext}"
    draw_polar_histogram(yaws_gt, yaws_pseudo, 
                  bin_width=10, 
                  title='Distribution of Yaw Angles (Log Scale)', 
                  output_path=out_yaw_log,
                  log_scale=True)
    
    print("Done.")

if __name__ == "__main__":
    main()
