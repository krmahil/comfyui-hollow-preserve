import sys
import os
import cv2
import numpy as np
from PIL import Image
from hollow_preserve_node import RemoveEnclosedMaskedAreas

# Try to import torch, but don't require it
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - tensor handling will be skipped")

def test_with_mask_file(mask_path, break_thickness=3):
    """
    Test the mask breaking node with a mask file.
    
    Args:
        mask_path: Path to the mask image file
        break_thickness: Thickness of the break line (default: 3)
    """
    # Check if file exists
    if not os.path.exists(mask_path):
        print(f"Error: Mask file '{mask_path}' not found.")
        return
    
    # Load the mask image
    try:
        mask_img = Image.open(mask_path)
        # Convert to grayscale if needed
        if mask_img.mode != "L":
            print(f"Converting image from {mask_img.mode} to grayscale.")
            mask_img = mask_img.convert("L")
    except Exception as e:
        print(f"Error loading mask image: {e}")
        return
    
    # Process the mask with our node
    node = RemoveEnclosedMaskedAreas()
    result_mask = node.process_mask(mask_img, break_thickness)[0]
    
    # Save the results
    base_name = os.path.splitext(mask_path)[0]
    result_path = f"{base_name}_broken_t{break_thickness}.png"
    
    # Ensure result is a PIL Image for saving
    if TORCH_AVAILABLE and isinstance(result_mask, torch.Tensor):
        result_img = Image.fromarray((result_mask.cpu().numpy() * 255).astype(np.uint8), mode="L")
    elif isinstance(result_mask, np.ndarray):
        # Convert float to uint8 for PIL if needed
        if result_mask.dtype == np.float32 or result_mask.dtype == np.float64:
            result_mask = (result_mask * 255).astype(np.uint8)
        result_img = Image.fromarray(result_mask, mode="L")
    else:
        # It's already a PIL Image
        result_img = result_mask
    
    result_img.save(result_path)
    
    # Also save a side-by-side comparison
    # Create a side-by-side image
    width, height = mask_img.size
    comparison = Image.new('L', (width * 2, height))
    comparison.paste(mask_img, (0, 0))
    comparison.paste(result_img, (width, 0))
    
    comparison_path = f"{base_name}_comparison_t{break_thickness}.png"
    comparison.save(comparison_path)
    
    print(f"Original mask: {mask_path}")
    print(f"Processed mask (break thickness: {break_thickness}): {result_path}")
    print(f"Comparison image: {comparison_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_mask.py path_to_mask_file.png [break_thickness]")
        print("  break_thickness: Thickness of break line (default: 3)")
    else:
        mask_path = sys.argv[1]
        break_thickness = 3
        
        if len(sys.argv) >= 3:
            try:
                break_thickness = int(sys.argv[2])
            except ValueError:
                print(f"Warning: Invalid break thickness '{sys.argv[2]}', using default (3)")
                
        test_with_mask_file(mask_path, break_thickness) 