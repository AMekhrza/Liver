"""
DICOM Loader Module.

This module provides classes and functions for loading and processing
DICOM medical imaging data, including series loading, preprocessing,
and conversion to numpy arrays.
"""

import os
import numpy as np
import SimpleITK as sitk
from typing import List, Dict, Tuple, Optional, Union
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import configuration
try:
    from config import YOLO_CONFIG
except ImportError:
    YOLO_CONFIG = {"window_center": 40, "window_width": 400}


class DicomLoader:
    """
    Class for loading and processing DICOM files.
    
    Provides methods for working with DICOM series, including loading,
    getting information, and converting to various formats.
    """
    
    def __init__(self):
        """Initialize the DICOM loader."""
        self._reader = sitk.ImageSeriesReader()
        self._series_cache = {}
        
    def get_dicom_series(self, folder_path: str) -> List[str]:
        """
        Get list of DICOM series IDs from a folder.
        
        Args:
            folder_path: Path to the folder containing DICOM files
            
        Returns:
            List of series IDs found in the folder
        """
        if not os.path.exists(folder_path):
            logger.error(f"Folder does not exist: {folder_path}")
            return []
            
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder_path)
            logger.info(f"Found {len(series_ids)} DICOM series in {folder_path}")
            return list(series_ids)
        except Exception as e:
            logger.error(f"Error reading DICOM series: {e}")
            return []
    
    def load_series(self, folder_path: str, series_id: str) -> sitk.Image:
        """
        Load a specific DICOM series as a SimpleITK image.
        
        Args:
            folder_path: Path to the folder containing DICOM files
            series_id: ID of the series to load
            
        Returns:
            SimpleITK Image object
        """
        try:
            file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                folder_path, series_id
            )
            
            if not file_names:
                raise ValueError(f"No files found for series {series_id}")
            
            self._reader.SetFileNames(file_names)
            self._reader.MetaDataDictionaryArrayUpdateOn()
            self._reader.LoadPrivateTagsOn()
            
            image = self._reader.Execute()
            logger.info(f"Loaded series {series_id} with size {image.GetSize()}")
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading DICOM series: {e}")
            raise
    
    def get_series_info(self, folder_path: str, series_id: str) -> Dict[str, Union[str, int, float, Tuple]]:
        """
        Get information about a DICOM series.
        
        Args:
            folder_path: Path to the folder containing DICOM files
            series_id: ID of the series
            
        Returns:
            Dictionary with series information
        """
        try:
            file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                folder_path, series_id
            )
            
            if not file_names:
                return {"series_id": series_id, "num_slices": 0}
            
            # Read first slice for metadata
            first_slice = sitk.ReadImage(file_names[0])
            
            info = {
                "series_id": series_id,
                "num_slices": len(file_names),
                "size": first_slice.GetSize(),
                "spacing": first_slice.GetSpacing(),
                "origin": first_slice.GetOrigin(),
                "direction": first_slice.GetDirection(),
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting series info: {e}")
            return {"series_id": series_id, "num_slices": 0, "error": str(e)}
    
    def convert_to_numpy(self, sitk_image: sitk.Image) -> np.ndarray:
        """
        Convert a SimpleITK image to a numpy array.
        
        Args:
            sitk_image: SimpleITK Image object
            
        Returns:
            Numpy array representation of the image
        """
        array = sitk.GetArrayFromImage(sitk_image)
        logger.info(f"Converted image to numpy array with shape {array.shape}")
        return array
    
    def get_voxel_volume(self, sitk_image: sitk.Image) -> float:
        """
        Calculate the volume of a single voxel in mm³.
        
        Args:
            sitk_image: SimpleITK Image object
            
        Returns:
            Voxel volume in cubic millimeters
        """
        spacing = sitk_image.GetSpacing()
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        return voxel_volume


def preprocess_for_model(image_array: np.ndarray, model_type: str) -> np.ndarray:
    """
    Preprocess image for a specific model type.
    
    Args:
        image_array: Input image as numpy array
        model_type: Type of model ('YOLO', 'U-Net', 'nnU-Net')
        
    Returns:
        Preprocessed image array
    """
    if image_array is None or image_array.size == 0:
        logger.error("Empty image received for preprocessing")
        raise ValueError("Empty image")
    
    # Convert to float32 if needed
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32)
    
    # Get windowing parameters
    window_center = YOLO_CONFIG.get("window_center", 40)
    window_width = YOLO_CONFIG.get("window_width", 400)
    min_hu = window_center - window_width // 2
    max_hu = window_center + window_width // 2
    
    # Target size for U-Net and nnU-Net (trained on 256x256)
    target_size = (256, 256)
    
    model_type_lower = model_type.lower()
    
    if model_type_lower in ["yolo", "yolov11"]:
        # For YOLO, don't apply normalization - model handles it
        return image_array
        
    elif model_type_lower in ["u-net", "unet"]:
        # Apply windowing and normalize to [0, 1]
        image_array = np.clip(image_array, min_hu, max_hu)
        image_array = (image_array - min_hu) / (max_hu - min_hu)
        
        # Resize if needed
        if image_array.ndim == 3:
            from scipy.ndimage import zoom
            zoom_factors = (1, target_size[0] / image_array.shape[1], 
                          target_size[1] / image_array.shape[2])
            image_array = zoom(image_array, zoom_factors, order=1)
            
        return image_array
        
    elif model_type_lower in ["nnu-net", "nnunet"]:
        # nnU-Net uses its own preprocessing
        # Z-score normalization
        mean_val = np.mean(image_array)
        std_val = np.std(image_array)
        if std_val > 0:
            image_array = (image_array - mean_val) / std_val
        return image_array
    
    else:
        logger.warning(f"Unknown model type: {model_type}, returning original")
        return image_array


def postprocess_mask(
    mask_array: np.ndarray,
    target_shape: Optional[tuple] = None,
    prob_threshold: float = 0.5,
    min_object_size: Optional[int] = None,
    model_type: Optional[str] = None
) -> np.ndarray:
    """
    Post-process segmentation mask.
    
    Args:
        mask_array: Input segmentation mask
        target_shape: Target shape for resizing
        prob_threshold: Threshold for binarization
        min_object_size: Minimum object size in voxels
        model_type: Type of model that generated the mask
        
    Returns:
        Post-processed mask
    """
    if mask_array is None or mask_array.size == 0:
        logger.error("Empty mask received for post-processing")
        return np.zeros_like(mask_array) if mask_array is not None else np.zeros((1,))
    
    # Binarize if needed
    if mask_array.max() > 1:
        mask_array = (mask_array > prob_threshold * mask_array.max()).astype(np.uint8)
    else:
        mask_array = (mask_array > prob_threshold).astype(np.uint8)
    
    # Resize to target shape if specified
    if target_shape is not None and mask_array.shape != target_shape:
        from scipy.ndimage import zoom
        zoom_factors = tuple(t / s for t, s in zip(target_shape, mask_array.shape))
        mask_array = zoom(mask_array, zoom_factors, order=0)
        mask_array = (mask_array > 0.5).astype(np.uint8)
    
    # Remove small objects if specified
    if min_object_size is not None:
        try:
            from scipy.ndimage import label, sum as ndsum
            labeled, num_features = label(mask_array)
            component_sizes = ndsum(mask_array, labeled, range(1, num_features + 1))
            
            for i, size in enumerate(component_sizes):
                if size < min_object_size:
                    mask_array[labeled == i + 1] = 0
        except Exception as e:
            logger.warning(f"Error removing small objects: {e}")
    
    return mask_array


def validate_dicom_folder(folder_path: str) -> bool:
    """
    Validate that a folder contains valid DICOM files.
    
    Args:
        folder_path: Path to check
        
    Returns:
        True if folder contains DICOM files, False otherwise
    """
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return False
    
    try:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder_path)
        return len(series_ids) > 0
    except Exception:
        return False


def get_dicom_files_count(folder_path: str) -> Dict[str, int]:
    """
    Count DICOM files per series in a folder.
    
    Args:
        folder_path: Path to the folder
        
    Returns:
        Dictionary mapping series IDs to file counts
    """
    result = {}
    try:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder_path)
        for series_id in series_ids:
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder_path, series_id)
            result[series_id] = len(files)
    except Exception as e:
        logger.error(f"Error counting DICOM files: {e}")
    
    return result
