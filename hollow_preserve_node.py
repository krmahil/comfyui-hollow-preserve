import cv2
import numpy as np
from PIL import Image

# Try to import torch, but don't require it
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Only print this when not running in ComfyUI
    if __name__ == "__main__":
        print("PyTorch not available - tensor handling will be skipped")

class RemoveEnclosedMaskedAreas:
    """
    A ComfyUI node that processes masks to keep only directly painted regions,
    by breaking closed loops to prevent enclosed areas from being treated as masked.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "break_thickness": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process_mask"
    CATEGORY = "mask"
    
    def process_mask(self, mask, break_thickness=3):
        # Determine the type of mask input and preserve it
        is_tensor = TORCH_AVAILABLE and isinstance(mask, torch.Tensor)
        original_device = None
        
        # Convert to numpy for processing
        if is_tensor:
            original_device = mask.device
            mask_np = mask.cpu().numpy()
        elif isinstance(mask, Image.Image):
            # Ensure mask is in L mode (grayscale)
            if mask.mode != "L":
                mask = mask.convert("L")
            mask_np = np.array(mask)
        else:
            # If already numpy array, make a copy
            mask_np = np.array(mask)
        
        # Store original shape and number of dimensions
        original_shape = mask_np.shape
        
        # Handle multi-dimensional masks (batches, etc.)
        if len(original_shape) > 2:
            # Flatten to process each mask individually
            original_mask_np = mask_np
            flat_results = []
            
            # Process each mask in the batch
            for i in range(original_shape[0]):
                single_mask = original_mask_np[i]
                if len(single_mask.shape) > 2 and single_mask.shape[0] == 1:
                    single_mask = single_mask[0]  # Remove channel dim if present
                    
                processed = self._process_single_mask(single_mask, break_thickness)
                flat_results.append(processed)
                
            # Reconstruct the original shape
            if len(original_shape) == 3:
                result_np = np.stack(flat_results, axis=0)
            else:
                result_np = np.stack(flat_results, axis=0)
                if original_shape[1] == 1:  # Has channel dimension
                    result_np = result_np[:, np.newaxis, :, :]
        else:
            # Process a single mask
            result_np = self._process_single_mask(mask_np, break_thickness)
        
        # Convert back to the original format
        if is_tensor:
            result = torch.from_numpy(result_np).to(original_device)
        elif isinstance(mask, Image.Image):
            if len(result_np.shape) == 2:
                # Convert float to uint8 for PIL
                if result_np.dtype == np.float32 or result_np.dtype == np.float64:
                    result_np = (result_np * 255).astype(np.uint8)
                result = Image.fromarray(result_np, mode="L")
            else:
                # This should not happen with single PIL image input
                result = mask  # Return original on error
        else:
            result = result_np
            
        return (result,)
        
    def _process_single_mask(self, mask_np, break_thickness):
        """Process a single 2D mask array"""
        # Convert float masks to uint8 for OpenCV
        orig_dtype = mask_np.dtype
        
        if mask_np.dtype == np.float32 or mask_np.dtype == np.float64:
            mask_np = (mask_np * 255).astype(np.uint8)
        
        # Ensure binary mask (0 or 255)
        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        
        # Create a copy of the original mask for the result
        result_mask = binary_mask.copy()
        
        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(
            binary_mask, 
            cv2.RETR_TREE,  # Retrieve all contours in a full hierarchy
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Process only if contours are found
        if contours and hierarchy is not None:
            # Create a mask where we'll draw the breaks
            break_mask = np.zeros_like(binary_mask)
            
            # Create a hierarchy map for faster lookup
            hierarchy = hierarchy[0]
            contour_map = {}
            
            # Build a map of parent-child relationships
            for i, h in enumerate(hierarchy):
                parent_idx = h[3]
                if parent_idx != -1:  # Has a parent
                    if parent_idx not in contour_map:
                        contour_map[parent_idx] = []
                    contour_map[parent_idx].append(i)
                    
            # Check each contour
            for i, contour in enumerate(contours):
                # Only process contours with children (potential holes)
                if i in contour_map and len(contour) > 2:
                    # Get the children of this contour
                    children = contour_map[i]
                    
                    # Check if any child is a proper hole (black area inside white)
                    has_hole = False
                    for child_idx in children:
                        child_contour = contours[child_idx]
                        
                        # Create a small mask to check the inside of this child contour
                        mask_to_check = np.zeros_like(binary_mask)
                        cv2.drawContours(mask_to_check, [child_contour], 0, 255, -1)
                        
                        # Check if this is a hole (black inside white) by sampling center
                        M = cv2.moments(child_contour)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            # Check if this point is black in original mask
                            if binary_mask[cy, cx] == 0:
                                has_hole = True
                                break
                    
                    # If it has a proper hole, add a break line
                    if has_hole:
                        # Find the center of the current contour
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx_outer = int(M["m10"] / M["m00"])
                            cy_outer = int(M["m01"] / M["m00"])
                        else:
                            # Fallback to median point if moment calculation fails
                            cx_outer = np.median(contour[:, 0, 0]).astype(int)
                            cy_outer = np.median(contour[:, 0, 1]).astype(int)
                        
                        # Find the rightmost point of the contour
                        rightmost_idx = np.argmax(contour[:, 0, 0])
                        rightmost_point = tuple(contour[rightmost_idx][0])
                        
                        # Draw a line from center to the rightmost edge
                        cv2.line(break_mask, (cx_outer, cy_outer), rightmost_point, 255, break_thickness)
            
            # Apply the breaks to the result mask
            result_mask[break_mask > 0] = 0
        
        # Convert back to the original data type
        if orig_dtype == np.float32 or orig_dtype == np.float64:
            result_mask = result_mask.astype(orig_dtype) / 255.0
            
        return result_mask

# Node registration info
NODE_CLASS_MAPPINGS = {
    "RemoveEnclosedMaskedAreas": RemoveEnclosedMaskedAreas
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveEnclosedMaskedAreas": "Break Closed Mask Loops"
} 