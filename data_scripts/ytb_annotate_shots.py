#!/usr/bin/env python3
"""
Interactive annotation tool for shot preview grids.
Allows users to mark frames as correct/wrong with visual feedback.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict


class ShotAnnotator:
    """Interactive annotator for shot preview grids."""
    
    def __init__(self, shots_dir, anno_dir, grid_size=128):
        """
        Initialize the annotator.
        
        Args:
            shots_dir: Directory containing shot preview images and JSONs
            anno_dir: Directory to save annotations
            grid_size: Size of each grid cell (default: 128)
        """
        self.shots_dir = Path(shots_dir)
        self.anno_dir = Path(anno_dir)
        self.grid_size = grid_size
        
        # Create anno_dir if not exists
        self.anno_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all shots
        self.shot_files = sorted(list(self.shots_dir.glob('*.jpg')))
        if len(self.shot_files) == 0:
            raise ValueError(f"No .jpg files found in {shots_dir}")
        
        # Current state
        self.current_idx = 0
        self.current_video_name = None
        self.current_image = None
        self.current_json_data = None
        self.frames_keys = []
        self.grid_positions = []  # List of (row, col, frame_key)
        self.grid_histograms = {}  # {frame_key: histogram}
        
        # Annotation state
        self.user_anno = {}  # {frame_key: True/False} - user clicked annotations
        self.auto_anno = {}  # {frame_key: True/False} - auto-propagated annotations
        
        # UI state
        self.window_name = "Shot Annotator"
        self.display_image = None
        self.click_count = defaultdict(int)  # Track clicks for double-click detection
        self.last_click_time = defaultdict(float)  # Track time for double-click
        
        # Load first shot
        self.load_shot(0)
    
    def load_shot(self, idx):
        """
        Load a shot by index.
        
        Args:
            idx: Index in shot_files list
        """
        if idx < 0 or idx >= len(self.shot_files):
            print(f"Invalid shot index: {idx}")
            return
        
        # Save current annotations before loading new shot
        if self.current_video_name is not None:
            self.save_annotations()
        
        # Load new shot
        self.current_idx = idx
        shot_file = self.shot_files[idx]
        self.current_video_name = shot_file.stem
        
        # Load image
        self.current_image = cv2.imread(str(shot_file))
        if self.current_image is None:
            print(f"Error: Cannot load image {shot_file}")
            return
        
        # Load JSON
        json_file = self.shots_dir / f"{self.current_video_name}.json"
        if not json_file.exists():
            print(f"Error: JSON not found {json_file}")
            return
        
        with open(json_file, 'r', encoding='utf-8') as f:
            self.current_json_data = json.load(f)
        
        # Get frames_keys
        self.frames_keys = self.current_json_data.get('frames_keys', [])
        
        # Calculate grid positions
        self.calculate_grid_positions()
        
        # Calculate histograms for all grids
        self.calculate_grid_histograms()
        
        # Load existing annotations if available
        self.load_annotations()
        
        # Update display
        self.update_display()
        
        print(f"\nLoaded shot {idx + 1}/{len(self.shot_files)}: {self.current_video_name}")
        print(f"Total frames: {len(self.frames_keys)}")
        print(f"Annotated: {len(self.user_anno)} user, {len(self.auto_anno)} auto")
    
    def calculate_grid_positions(self):
        """Calculate grid positions from image dimensions."""
        h, w = self.current_image.shape[:2]
        cols = w // self.grid_size
        rows = h // self.grid_size
        
        self.grid_positions = []
        for i, frame_key in enumerate(self.frames_keys):
            if i >= rows * cols:
                break
            row = i // cols
            col = i % cols
            self.grid_positions.append((row, col, frame_key))
    
    def calculate_grid_histograms(self):
        """Calculate color histograms for central 1/2 area of each grid."""
        self.grid_histograms = {}
        
        for row, col, frame_key in self.grid_positions:
            # Get grid region
            y0 = row * self.grid_size
            x0 = col * self.grid_size
            y1 = y0 + self.grid_size
            x1 = x0 + self.grid_size
            
            grid = self.current_image[y0:y1, x0:x1]
            
            # Get central 1/2 area
            h, w = grid.shape[:2]
            cy0 = h // 4
            cy1 = h * 3 // 4
            cx0 = w // 4
            cx1 = w * 3 // 4
            central = grid[cy0:cy1, cx0:cx1]
            
            # Calculate histogram (HSV space, using H and S channels)
            hsv = cv2.cvtColor(central, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            self.grid_histograms[frame_key] = hist
    
    def calculate_histogram_similarity(self, hist1, hist2):
        """
        Calculate histogram similarity using correlation.
        
        Args:
            hist1: First histogram
            hist2: Second histogram
        
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        correlation = cv2.compareHist(hist1.reshape(-1, 1), hist2.reshape(-1, 1), cv2.HISTCMP_CORREL)
        return (correlation + 1) / 2  # Normalize to 0-1
    
    def propagate_annotation(self, clicked_frame_key, is_correct):
        """
        Propagate annotation to similar frames based on histogram similarity.
        
        Args:
            clicked_frame_key: Frame key that was clicked
            is_correct: True if marked correct, False if marked wrong
        """
        if clicked_frame_key not in self.grid_histograms:
            return
        
        clicked_hist = self.grid_histograms[clicked_frame_key]
        
        # Calculate similarities and propagate
        for frame_key, hist in self.grid_histograms.items():
            # Skip user-annotated frames
            if frame_key in self.user_anno:
                continue
            
            # Calculate similarity
            similarity = self.calculate_histogram_similarity(clicked_hist, hist)
            
            # Propagate if similarity > threshold (0.8)
            if similarity > 0.8:
                self.auto_anno[frame_key] = is_correct
    
    def update_display(self):
        """Update the display image with annotation colors."""
        self.display_image = self.current_image.copy()
        
        # Draw colored lines at bottom of grids
        for row, col, frame_key in self.grid_positions:
            y0 = row * self.grid_size
            x0 = col * self.grid_size
            y1 = y0 + self.grid_size
            x1 = x0 + self.grid_size
            
            # Determine color based on annotations
            color = None
            thickness = 3
            
            if frame_key in self.user_anno:
                # User annotation (solid line)
                color = (0, 255, 0) if self.user_anno[frame_key] else (0, 0, 255)
                thickness = 5
            elif frame_key in self.auto_anno:
                # Auto annotation (thinner line)
                color = (0, 255, 0) if self.auto_anno[frame_key] else (0, 0, 255)
                thickness = 3
            
            if color is not None:
                # Draw line at bottom of grid
                cv2.line(self.display_image, (x0, y1 - thickness), (x1, y1 - thickness), color, thickness)
        
        # Add info text at top
        info_text = f"Shot {self.current_idx + 1}/{len(self.shot_files)}: {self.current_video_name}"
        anno_text = f"User: {len(self.user_anno)} | Auto: {len(self.auto_anno)} | Total: {len(self.frames_keys)}"
        
        cv2.putText(self.display_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.display_image, anno_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.display_image, "Single click: Correct (green) | Double click: Wrong (red)", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(self.display_image, "Keys: Left/Right arrow, N=next non-annotated, S=save, Q=quit", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show the image
        cv2.imshow(self.window_name, self.display_image)
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events.
        
        Args:
            event: Mouse event type
            x: Mouse x coordinate
            y: Mouse y coordinate
            flags: Additional flags
            param: User data
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find which grid was clicked
            col = x // self.grid_size
            row = y // self.grid_size
            
            # Find frame_key
            clicked_frame_key = None
            for r, c, fk in self.grid_positions:
                if r == row and c == col:
                    clicked_frame_key = fk
                    break
            
            if clicked_frame_key is None:
                return
            
            # Check for double click (within 500ms)
            import time
            current_time = time.time()
            last_time = self.last_click_time.get(clicked_frame_key, 0)
            time_diff = current_time - last_time
            
            self.last_click_time[clicked_frame_key] = current_time
            
            if time_diff < 0.5:  # Double click
                # Mark as wrong (red)
                self.user_anno[clicked_frame_key] = False
                # Remove from auto_anno if present
                self.auto_anno.pop(clicked_frame_key, None)
                print(f"Marked {clicked_frame_key} as WRONG")
                
                # Propagate annotation
                self.propagate_annotation(clicked_frame_key, False)
                
                # Reset click count
                self.last_click_time[clicked_frame_key] = 0
            else:
                # Single click - mark as correct (green)
                self.user_anno[clicked_frame_key] = True
                # Remove from auto_anno if present
                self.auto_anno.pop(clicked_frame_key, None)
                print(f"Marked {clicked_frame_key} as CORRECT")
                
                # Propagate annotation
                self.propagate_annotation(clicked_frame_key, True)
            
            # Update display
            self.update_display()
    
    def save_annotations(self):
        """Save annotations to JSON file."""
        if self.current_video_name is None:
            return
        
        # Prepare annotation data
        correct_frames = []
        wrong_frames = []
        
        # Collect from user annotations
        for frame_key, is_correct in self.user_anno.items():
            if is_correct:
                correct_frames.append(frame_key)
            else:
                wrong_frames.append(frame_key)
        
        # Collect from auto annotations
        for frame_key, is_correct in self.auto_anno.items():
            if is_correct:
                correct_frames.append(frame_key)
            else:
                wrong_frames.append(frame_key)
        
        # Create annotation dict
        anno_data = {
            "correct": sorted(correct_frames),
            "wrong": sorted(wrong_frames),
            "user_anno": {k: v for k, v in sorted(self.user_anno.items())}
        }
        
        # Save to file
        anno_file = self.anno_dir / f"{self.current_video_name}.json"
        with open(anno_file, 'w', encoding='utf-8') as f:
            json.dump(anno_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved annotations to {anno_file}")
    
    def load_annotations(self):
        """Load existing annotations from file."""
        self.user_anno = {}
        self.auto_anno = {}
        
        anno_file = self.anno_dir / f"{self.current_video_name}.json"
        if not anno_file.exists():
            return
        
        try:
            with open(anno_file, 'r', encoding='utf-8') as f:
                anno_data = json.load(f)
            
            # Load user annotations
            self.user_anno = anno_data.get('user_anno', {})
            
            # Load auto annotations (correct + wrong - user_anno)
            correct_set = set(anno_data.get('correct', []))
            wrong_set = set(anno_data.get('wrong', []))
            user_set = set(self.user_anno.keys())
            
            for frame_key in correct_set - user_set:
                self.auto_anno[frame_key] = True
            for frame_key in wrong_set - user_set:
                self.auto_anno[frame_key] = False
            
            print(f"Loaded annotations from {anno_file}")
        except Exception as e:
            print(f"Error loading annotations: {e}")
    
    def find_next_non_annotated(self):
        """Find the next shot with non-annotated frames."""
        start_idx = self.current_idx
        
        for offset in range(1, len(self.shot_files) + 1):
            idx = (start_idx + offset) % len(self.shot_files)
            
            # Peek at this shot's annotation
            shot_file = self.shot_files[idx]
            video_name = shot_file.stem
            anno_file = self.anno_dir / f"{video_name}.json"
            
            # Load JSON to get total frames
            json_file = self.shots_dir / f"{video_name}.json"
            if not json_file.exists():
                continue
            
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            total_frames = len(json_data.get('frames_keys', []))
            
            if not anno_file.exists():
                # No annotations at all
                return idx
            
            # Check if fully annotated
            with open(anno_file, 'r', encoding='utf-8') as f:
                anno_data = json.load(f)
            
            annotated_count = len(anno_data.get('correct', [])) + len(anno_data.get('wrong', []))
            
            if annotated_count < total_frames:
                return idx
        
        print("All shots are fully annotated!")
        return self.current_idx
    
    def run(self):
        """Run the interactive annotation tool."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Create trackbar for navigation
        cv2.createTrackbar('Shot', self.window_name, 0, len(self.shot_files) - 1, 
                          lambda x: self.load_shot(x))
        
        print("\n" + "="*60)
        print("Shot Annotator")
        print("="*60)
        print("Controls:")
        print("  Single click: Mark as CORRECT (green)")
        print("  Double click: Mark as WRONG (red)")
        print("  Left/Right arrow: Previous/Next shot")
        print("  N: Jump to next non-annotated shot")
        print("  S: Save annotations")
        print("  Q: Quit")
        print("="*60 + "\n")
        
        while True:
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                # Save and quit
                self.save_annotations()
                break
            elif key == 81 or key == 2:  # Left arrow
                # Previous shot
                new_idx = (self.current_idx - 1) % len(self.shot_files)
                self.load_shot(new_idx)
                cv2.setTrackbarPos('Shot', self.window_name, new_idx)
            elif key == 83 or key == 3:  # Right arrow
                # Next shot
                new_idx = (self.current_idx + 1) % len(self.shot_files)
                self.load_shot(new_idx)
                cv2.setTrackbarPos('Shot', self.window_name, new_idx)
            elif key == ord('n') or key == ord('N'):
                # Next non-annotated
                new_idx = self.find_next_non_annotated()
                if new_idx != self.current_idx:
                    self.load_shot(new_idx)
                    cv2.setTrackbarPos('Shot', self.window_name, new_idx)
            elif key == ord('s') or key == ord('S'):
                # Save
                self.save_annotations()
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Interactive annotation tool for shot preview grids')
    parser.add_argument('--shots_dir', type=str, required=True,
                       help='Directory containing shot preview images and JSONs')
    parser.add_argument('--anno_dir', type=str, required=True,
                       help='Directory to save annotations')
    parser.add_argument('--grid_size', type=int, default=128,
                       help='Size of each grid cell (default: 128)')
    
    args = parser.parse_args()
    
    # Create annotator
    annotator = ShotAnnotator(args.shots_dir, args.anno_dir, args.grid_size)
    
    # Run interactive tool
    annotator.run()


if __name__ == '__main__':
    main()
