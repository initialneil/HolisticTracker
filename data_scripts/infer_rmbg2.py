#!/usr/bin/env python3
"""
Apply RMBG-2.0 background removal to images.
Processes all .jpg and .png files in input directory and saves RGBA outputs.
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image
from modelscope import AutoModelForImageSegmentation
import torchvision.transforms as transforms


def load_rmbg2_model(model_name):
    """
    Load the RMBG-2.0 model from HuggingFace.
    
    Args:
        model_name: Model name or path
    
    Returns:
        Loaded model
    """
    print(f"Loading RMBG-2.0 model from: {model_name}")
    model = AutoModelForImageSegmentation.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model.eval()
    print(f"RMBG-2.0 model loaded successfully")
    return model


def process_image(img_path, model, image_size=1024):
    """
    Apply RMBG-2.0 background removal to a single image.
    
    Args:
        img_path: Path to input image
        model: RMBG-2.0 model
        image_size: Image size for processing
    
    Returns:
        RGBA image as PIL Image
    """
    # Load image with PIL
    image = Image.open(str(img_path)).convert('RGB')
    if image is None:
        return None
    
    # Get original size
    orig_size = image.size
    orig_w, orig_h = orig_size
    
    # Calculate new size: keep aspect ratio, max dimension <= image_size, divisible by 4
    scale = min(image_size / orig_w, image_size / orig_h, 1.0)  # Don't upscale
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # Make divisible by 32
    new_w = (new_w // 32) * 32
    new_h = (new_h // 32) * 32
    
    # Resize image
    resized_image = image.resize((new_w, new_h), Image.BILINEAR)
    
    # Prepare transform (without resize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Transform image
    input_tensor = transform(resized_image).unsqueeze(0).to(model.device)
    
    # Run inference
    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()
    
    # Get the mask
    pred = preds[0].squeeze()
    
    # Resize mask back to original size
    mask_pil = transforms.ToPILImage()(pred)
    mask_pil = mask_pil.resize(orig_size, Image.BILINEAR)
    
    # Convert to numpy array and scale to 0-255
    alpha = np.array(mask_pil).astype(np.uint8)
    
    # Create RGBA image
    rgba_image = image.convert('RGBA')
    rgba_array = np.array(rgba_image)
    rgba_array[:, :, 3] = alpha
    
    return Image.fromarray(rgba_array, 'RGBA')






def main():
    parser = argparse.ArgumentParser(description='Apply RMBG-2.0 background removal to images')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images (.jpg, .png)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for RGBA images')
    parser.add_argument('--model_name', type=str, default='AI-ModelScope/RMBG-2.0',
                        help='ModelScope model name for RMBG-2.0')
    
    args = parser.parse_args()
    
    # Setup
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load RMBG-2.0 model
    print("Loading RMBG-2.0 model...")
    model = load_rmbg2_model(args.model_name)
    
    # Get list of image files in root directory
    image_files = []
    for ext in ['*.jpg', '*.png', '*.JPG', '*.PNG']:
        image_files.extend(input_dir.glob(ext))
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        print(f"No .jpg or .png files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image(s)")
    print(f"Output directory: {output_dir}")
    
    # Process each image
    processed_count = 0
    for img_file in tqdm(image_files, desc="Processing images"):
        # Output path (always save as .png for RGBA)
        output_path = output_dir / f"{img_file.stem}.png"
        
        # Skip if already processed
        if output_path.exists():
            processed_count += 1
            continue
        
        # Process image
        rgba_image = process_image(img_file, model)
        
        if rgba_image is not None:
            # Save RGBA PNG
            rgba_image.save(str(output_path), 'PNG')
            processed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total images processed: {processed_count}/{len(image_files)}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
