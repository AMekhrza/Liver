"""
Configuration module for LiverSeg 3D Application.

This module contains all configuration settings, model paths,
and utility functions for the liver segmentation application.
"""

import os
import torch
from typing import Dict, Any, Optional

# Application Info
APP_NAME = "LiverSeg 3D"
APP_VERSION = "1.0.0"
AUTHOR = "LiverSeg Team"

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# Default model paths
DEFAULT_MODEL_PATHS = {
    "YOLOv11": os.path.join(CHECKPOINTS_DIR, "yolov11_liver.pt"),
    "U-Net": os.path.join(CHECKPOINTS_DIR, "unet_liver.pth"),
    "nnU-Net": os.path.join(CHECKPOINTS_DIR, "nnunet_liver.pth"),
}

# YOLO Configuration
YOLO_CONFIG = {
    "window_center": 40,
    "window_width": 400,
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
}

# U-Net Configuration
UNET_CONFIG = {
    "input_channels": 1,
    "output_channels": 2,
    "initial_features": 64,
    "depth": 4,
}

# Preview Configuration
PREVIEW_CONFIG = {
    "cache_size": 5,
    "default_resolution": "original",
    "use_preview_for_display": True,
}

# Processing Configuration
PROCESSING_CONFIG = {
    "batch_size": 1,
    "num_workers": 4,
    "prefetch_factor": 2,
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    "default_window_center": 0,
    "default_window_width": 400,
    "default_opacity": 0.7,
    "liver_color": "#e74c3c",
    "background_color": "#2c3e50",
}


def get_device() -> str:
    """
    Get the best available device for computation.
    
    Returns:
        str: Device name ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model_path(model_name: str) -> Optional[str]:
    """
    Get the path to a model file.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Path to the model file or None if not found
    """
    return DEFAULT_MODEL_PATHS.get(model_name)


def validate_model_paths() -> Dict[str, bool]:
    """
    Validate that all model files exist.
    
    Returns:
        Dictionary mapping model names to existence status
    """
    results = {}
    for name, path in DEFAULT_MODEL_PATHS.items():
        results[name] = os.path.exists(path)
    return results


def get_config() -> Dict[str, Any]:
    """
    Get the complete application configuration.
    
    Returns:
        Dictionary containing all configuration settings
    """
    return {
        "app": {
            "name": APP_NAME,
            "version": APP_VERSION,
            "author": AUTHOR,
        },
        "model_paths": DEFAULT_MODEL_PATHS,
        "yolo": YOLO_CONFIG,
        "unet": UNET_CONFIG,
        "preview": PREVIEW_CONFIG,
        "processing": PROCESSING_CONFIG,
        "visualization": VISUALIZATION_CONFIG,
        "device": get_device(),
    }


if __name__ == "__main__":
    # Test configuration
    print(f"Application: {APP_NAME} v{APP_VERSION}")
    print(f"Device: {get_device()}")
    print("\nModel Paths:")
    for name, exists in validate_model_paths().items():
        status = "✓" if exists else "✗"
        print(f"  [{status}] {name}: {DEFAULT_MODEL_PATHS[name]}")
