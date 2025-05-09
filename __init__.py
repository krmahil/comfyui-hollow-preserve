"""
ComfyUI Hollow Preserve - A node that prevents inpainting models from modifying enclosed areas in masks.
"""

__version__ = "0.1.0"

from .hollow_preserve_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# For ComfyUI to detect
NODE_CLASS_MAPPINGS = NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 