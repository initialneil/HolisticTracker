#!/usr/bin/env python3
"""
Gradio-based interactive annotation tool for shot preview grids.
Allows users to mark frames as correct/wrong with visual feedback.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
import gradio as gr
from PIL import Image
import time


class ShotAnnotatorGradio:
    """Gradio-based annotator for shot preview grids."""
    
    def __init__(self, shots_dir, anno_dir, grid_size=138):
        """
        Initialize the annotator.
        
        Args:
            shots_dir: Directory containing shot preview images and JSONs
            anno_dir: Directory to save annotations
            grid_size: Height of each grid cell including 10px indicator (default: 138)
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
        
        # Click tracking for double-click detection
        self.last_click_time = {}
        self.last_click_frame = {}
    
    def load_shot(self, idx):
        """
        Load a shot by index.
        
        Args:
            idx: Index in shot_files list
        
        Returns:
            Tuple of (display_image, info_text, slider_value)
        """
        if idx < 0 or idx >= len(self.shot_files):
            return None, f"Invalid shot index: {idx}", idx
        
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
            return None, f"Error: Cannot load image {shot_file}", idx
        
        # Load JSON
        json_file = self.shots_dir / f"{self.current_video_name}.json"
        if not json_file.exists():
            return None, f"Error: JSON not found {json_file}", idx
        
        with open(json_file, 'r', encoding='utf-8') as f:
            self.current_json_data = json.load(f)
        
        # Get frames_keys from the video_name key
        video_data = self.current_json_data.get(self.current_video_name, {})
        frames_keys_dict = video_data.get('frames_keys', {})
        # Extract frame keys from dictionary
        self.frames_keys = list(frames_keys_dict.keys())
        
        # Calculate grid positions
        self.calculate_grid_positions()
        
        # Calculate histograms for all grids
        self.calculate_grid_histograms()
        
        # Load existing annotations if available
        self.load_annotations()
        
        # Generate display image
        display_img = self.generate_display_image()
        
        # Generate info text
        info_text = self.generate_info_text()
        
        return display_img, info_text, idx
    
    def calculate_grid_positions(self):
        """Calculate grid positions from image dimensions."""
        h, w = self.current_image.shape[:2]
        cols = w // 128  # Width is 128
        rows = h // self.grid_size  # Height is 138
        
        self.grid_positions = []
        for i, frame_key in enumerate(self.frames_keys):
            if i >= rows * cols:
                break
            row = i // cols
            col = i % cols
            self.grid_positions.append((row, col, frame_key))
    
    def calculate_grid_histograms(self):
        """Calculate color histograms for full 128x128 grid."""
        self.grid_histograms = {}
        
        for row, col, frame_key in self.grid_positions:
            # Get grid region (only 128x128 image area, excluding 10px indicator)
            y0 = row * self.grid_size
            x0 = col * 128  # Width is 128
            y1 = y0 + 128  # Image height is 128
            x1 = x0 + 128
            
            grid = self.current_image[y0:y1, x0:x1]
            
            # Calculate histogram (HSV space, using H and S channels) for full 128x128 area
            hsv = cv2.cvtColor(grid, cv2.COLOR_BGR2HSV)
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
        Finds the closest frame to each unannotated frame and propagates that annotation.
        
        Args:
            clicked_frame_key: Frame key that was clicked
            is_correct: True if marked correct, False if marked wrong
        """
        # For each non-user-annotated frame, find its closest annotated frame
        for frame_key, hist in self.grid_histograms.items():
            # Skip user-annotated frames
            if frame_key in self.user_anno:
                continue
            
            # Find closest user-annotated frame
            best_similarity = -1
            best_annotation = None
            
            for anno_frame_key, anno_value in self.user_anno.items():
                if anno_frame_key not in self.grid_histograms:
                    continue
                
                anno_hist = self.grid_histograms[anno_frame_key]
                similarity = self.calculate_histogram_similarity(hist, anno_hist)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_annotation = anno_value
            
            # Propagate the annotation from closest user-annotated frame
            if best_annotation is not None:
                self.auto_anno[frame_key] = best_annotation
    
    def generate_display_image(self):
        """
        Generate display image with annotation colors.
        Uses the last 2 lines of the 10px indicator area at the bottom.
        
        Returns:
            PIL Image
        """
        display_image = self.current_image.copy()
        
        # Draw colored rectangles on last 2 lines of indicator area
        for row, col, frame_key in self.grid_positions:
            y0 = row * self.grid_size
            x0 = col * 128  # Width is 128
            
            # Determine color based on annotations
            color = None
            
            if frame_key in self.user_anno:
                # User annotation
                color = (0, 255, 0) if self.user_anno[frame_key] else (0, 0, 255)
            elif frame_key in self.auto_anno:
                # Auto annotation
                color = (0, 255, 0) if self.auto_anno[frame_key] else (0, 0, 255)
            
            if color is not None:
                # Draw filled rectangle on last 4 lines (lines 135-138 of the 138px grid)
                # This is within the 10px indicator area (128-137)
                rect_y0 = y0 + 134  # Start at line 135 (0-indexed)
                rect_y1 = y0 + 138  # End at line 138
                rect_x0 = x0
                rect_x1 = x0 + 128
                cv2.rectangle(display_image, (rect_x0, rect_y0), (rect_x1, rect_y1), color, -1)
        
        # Convert BGR to RGB for PIL
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(display_image)
    
    def generate_info_text(self):
        """
        Generate info text for display.
        
        Returns:
            Info text string
        """
        user_correct = sum(1 for v in self.user_anno.values() if v)
        user_wrong = sum(1 for v in self.user_anno.values() if not v)
        auto_correct = sum(1 for v in self.auto_anno.values() if v)
        auto_wrong = sum(1 for v in self.auto_anno.values() if not v)
        total = len(self.frames_keys)
        annotated = len(self.user_anno) + len(self.auto_anno)
        
        info = f"**Shot {self.current_idx + 1}/{len(self.shot_files)}**: {self.current_video_name}\n\n"
        info += f"**Total frames**: {total} | **Annotated**: {annotated} ({annotated*100//total if total > 0 else 0}%)\n\n"
        info += f"**User**: ✅ {user_correct} | ❌ {user_wrong}\n\n"
        info += f"**Auto**: ✅ {auto_correct} | ❌ {auto_wrong}\n\n"
        
        return info
    
    def handle_click(self, is_shift_click, evt: gr.SelectData):
        """
        Handle click events on the image.
        Regular click = mark as correct
        Shift+Click = mark as wrong
        
        Args:
            is_shift_click: Boolean indicating if Shift key was pressed
            evt: Gradio SelectData event
        
        Returns:
            Tuple of (display_image, info_text)
        """
        if self.current_image is None:
            return None, "No image loaded"
        
        x, y = evt.index
        
        # Find which grid was clicked
        col = x // 128  # Width is 128
        row = y // self.grid_size  # Height is 138
        
        # Find frame_key
        clicked_frame_key = None
        for r, c, fk in self.grid_positions:
            if r == row and c == col:
                clicked_frame_key = fk
                break
        
        if clicked_frame_key is None:
            return self.generate_display_image(), self.generate_info_text()
        
        # Determine annotation based on Shift modifier
        if is_shift_click:
            # Shift+Click - mark as wrong (red)
            self.user_anno[clicked_frame_key] = False
            is_correct = False
        else:
            # Regular click - mark as correct (green)
            self.user_anno[clicked_frame_key] = True
            is_correct = True
        
        # Remove from auto_anno if present
        self.auto_anno.pop(clicked_frame_key, None)
        
        # Propagate annotation
        self.propagate_annotation(clicked_frame_key, is_correct)
        
        # Update display
        return self.generate_display_image(), self.generate_info_text()
    
    def save_annotations(self):
        """Save annotations to JSON file."""
        if self.current_video_name is None:
            return "No video loaded"
        
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
        
        return f"✅ Saved annotations to {anno_file.name}"
    
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
        except Exception as e:
            print(f"Error loading annotations: {e}")
    
    def find_next_non_annotated(self):
        """
        Find the next shot with non-annotated frames.
        
        Returns:
            Index of next non-annotated shot
        """
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
            
            # Get frames_keys from video_name key
            video_data = json_data.get(video_name, {})
            frames_keys_dict = video_data.get('frames_keys', {})
            total_frames = len(frames_keys_dict)
            
            if not anno_file.exists():
                # No annotations at all
                return idx
            
            # Check if fully annotated
            with open(anno_file, 'r', encoding='utf-8') as f:
                anno_data = json.load(f)
            
            annotated_count = len(anno_data.get('correct', [])) + len(anno_data.get('wrong', []))
            
            if annotated_count < total_frames:
                return idx
        
        return self.current_idx  # All fully annotated
    
    def go_previous(self):
        """Go to previous shot."""
        new_idx = (self.current_idx - 1) % len(self.shot_files)
        return self.load_shot(new_idx)
    
    def go_next(self):
        """Go to next shot."""
        new_idx = (self.current_idx + 1) % len(self.shot_files)
        return self.load_shot(new_idx)
    
    def go_next_non_annotated(self):
        """Go to next non-annotated shot."""
        new_idx = self.find_next_non_annotated()
        return self.load_shot(new_idx)
    
    def go_to_shot(self, slider_value):
        """
        Go to shot by slider value.
        
        Args:
            slider_value: Slider position (0 to len(shot_files)-1)
        
        Returns:
            Tuple of (display_image, info_text, slider_value)
        """
        idx = int(slider_value)
        return self.load_shot(idx)
    
    def mark_all(self, is_wrong):
        """
        Mark all frames as correct or wrong.
        
        Args:
            is_wrong: If True, mark all as wrong; if False, mark all as correct
        
        Returns:
            Tuple of (display_image, info_text)
        """
        if self.current_image is None:
            return None, "No image loaded"
        
        # Clear auto annotations
        self.auto_anno = {}
        
        # Mark all frames in user_anno
        for frame_key in self.frames_keys:
            self.user_anno[frame_key] = not is_wrong
        
        # Update display
        return self.generate_display_image(), self.generate_info_text()
    
    def manual_save(self):
        """Manually save annotations."""
        msg = self.save_annotations()
        return msg


def create_gradio_interface(annotator):
    """
    Create Gradio interface.
    
    Args:
        annotator: ShotAnnotatorGradio instance
    
    Returns:
        Gradio Blocks interface
    """
    # Custom JavaScript for keyboard shortcuts
    js_code = """
    function setupClickHandlers() {
        // Add keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Q for previous
            if (e.key === 'q' || e.key === 'Q') {
                document.getElementById('prev-btn').click();
            }
            // E for next
            else if (e.key === 'e' || e.key === 'E') {
                document.getElementById('next-btn').click();
            }
            // S for save
            else if (e.key === 's' || e.key === 'S') {
                e.preventDefault(); // Prevent browser save dialog
                document.getElementById('save-btn').click();
            }
            // X for toggle wrong marking checkbox
            else if (e.key === 'x' || e.key === 'X') {
                // Find the checkbox by elem_id and then find the actual input element
                const checkboxContainer = document.getElementById('wrong-checkbox');
                if (checkboxContainer) {
                    const checkboxInput = checkboxContainer.querySelector('input[type="checkbox"]');
                    if (checkboxInput) {
                        checkboxInput.click();
                    }
                }
            }
            // A for mark all
            else if (e.key === 'a' || e.key === 'A') {
                const markAllBtn = document.getElementById('mark-all-btn');
                if (markAllBtn) {
                    markAllBtn.click();
                }
            }
        });
        
        return [];
    }
    """
    
    with gr.Blocks(title="Shot Annotation Tool", js=js_code) as demo:
        gr.Markdown("# 🎬 Shot Annotation Tool")
        gr.Markdown("Click on grids to annotate frames. **Click** = ✅ Correct (green), **Checkbox ON + Click** = ❌ Wrong (red) | Press **X** to toggle | **A** to mark all")
        
        with gr.Row():
            with gr.Column(scale=3):
                # Image display
                image_output = gr.Image(
                    label="Shot Preview",
                    type="pil",
                    interactive=False,
                    height=800
                )
            
            with gr.Column(scale=1):
                # Info panel
                info_output = gr.Markdown(label="Information")
                
                # Wrong marking checkbox (toggle with X key)
                shift_checkbox = gr.Checkbox(label="Mark as WRONG (toggle with X key)", value=False, elem_id="wrong-checkbox")
                
                # Navigation controls
                gr.Markdown("### Navigation")
                
                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous (Q)", variant="secondary", elem_id="prev-btn")
                    next_btn = gr.Button("➡️ Next (E)", variant="secondary", elem_id="next-btn")
                
                next_non_anno_btn = gr.Button("⏭️ Next Non-Annotated", variant="primary")
                
                # Mark all button
                mark_all_btn = gr.Button("✓ Mark All (A)", variant="secondary", elem_id="mark-all-btn")
                
                # Slider
                slider = gr.Slider(
                    minimum=0,
                    maximum=len(annotator.shot_files) - 1,
                    step=1,
                    value=0,
                    label="Shot Index",
                    interactive=True
                )
                
                # Save button
                save_btn = gr.Button("💾 Save Annotations (S)", variant="primary", elem_id="save-btn")
                save_status = gr.Textbox(label="Save Status", interactive=False)
        
        # Event handlers
        
        # Click on image (with Shift modifier support)
        image_output.select(
            fn=annotator.handle_click,
            inputs=[shift_checkbox],
            outputs=[image_output, info_output]
        )
        
        # Previous button
        prev_btn.click(
            fn=annotator.go_previous,
            inputs=None,
            outputs=[image_output, info_output, slider]
        )
        
        # Next button
        next_btn.click(
            fn=annotator.go_next,
            inputs=None,
            outputs=[image_output, info_output, slider]
        )
        
        # Next non-annotated button
        next_non_anno_btn.click(
            fn=annotator.go_next_non_annotated,
            inputs=None,
            outputs=[image_output, info_output, slider]
        )
        
        # Mark all button
        mark_all_btn.click(
            fn=annotator.mark_all,
            inputs=[shift_checkbox],
            outputs=[image_output, info_output]
        )
        
        # Slider
        slider.change(
            fn=annotator.go_to_shot,
            inputs=[slider],
            outputs=[image_output, info_output, slider]
        )
        
        # Save button
        save_btn.click(
            fn=annotator.manual_save,
            inputs=None,
            outputs=[save_status]
        )
        
        # Load initial shot
        demo.load(
            fn=lambda: annotator.load_shot(0),
            inputs=None,
            outputs=[image_output, info_output, slider]
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description='Gradio-based annotation tool for shot preview grids')
    parser.add_argument('--shots_dir', type=str, required=True,
                       help='Directory containing shot preview images and JSONs')
    parser.add_argument('--anno_dir', type=str, required=True,
                       help='Directory to save annotations')
    parser.add_argument('--grid_size', type=int, default=138,
                       help='Height of each grid cell including 10px indicator (default: 138, image is 128x128 + 10px indicator)')
    parser.add_argument('--port', type=int, default=9061,
                       help='Port to run Gradio server (default: 7860)')
    parser.add_argument('--share', action='store_true',
                       help='Create a public share link')
    
    args = parser.parse_args()
    
    # Create annotator
    annotator = ShotAnnotatorGradio(args.shots_dir, args.anno_dir, args.grid_size)
    
    # Create Gradio interface
    demo = create_gradio_interface(annotator)
    
    # Launch
    print("\n" + "="*60)
    print("Starting Gradio Shot Annotation Tool")
    print("="*60)
    print(f"Shots directory: {args.shots_dir}")
    print(f"Annotations directory: {args.anno_dir}")
    print(f"Grid size: {args.grid_size}")
    print(f"Total shots: {len(annotator.shot_files)}")
    print("="*60 + "\n")
    
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0"
    )


if __name__ == '__main__':
    main()
