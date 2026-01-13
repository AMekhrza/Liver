"""
Worker Threads Module.

This module contains QThread worker classes for performing
background operations like DICOM loading, segmentation, and volume calculation.
"""

import os
import numpy as np
import logging
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

# Import local modules (support running as script or package)
try:
    from utils.dicom_loader import DicomLoader, preprocess_for_model, postprocess_mask
    from models.model_manager import ModelManager
    from config import get_model_path, get_device
except ImportError:
    from ..utils.dicom_loader import DicomLoader, preprocess_for_model, postprocess_mask
    from ..models.model_manager import ModelManager
    from ..config import get_model_path, get_device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DicomLoaderWorker(QThread):
    """
    Worker thread for loading DICOM series.
    
    Performs DICOM file loading in a background thread,
    keeping the GUI responsive.
    """
    
    # Signals
    progress_updated = pyqtSignal(int)
    series_loaded = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """Initialize the DICOM loader worker."""
        super().__init__(parent)
        self.folder_path = ""
        self.series_id = ""
        self.dicom_loader = DicomLoader()
        self._is_running = False
    
    def load_series(self, folder_path: str, series_id: str):
        """
        Set parameters for loading a series.
        
        Args:
            folder_path: Path to the DICOM folder
            series_id: ID of the series to load
        """
        self.folder_path = folder_path
        self.series_id = series_id
    
    def run(self):
        """Execute the DICOM loading operation."""
        self._is_running = True
        
        try:
            self.progress_updated.emit(0)
            
            # Validate folder
            if not os.path.exists(self.folder_path):
                raise FileNotFoundError(f"Folder does not exist: {self.folder_path}")
            
            if not os.path.isdir(self.folder_path):
                raise ValueError(f"Path is not a directory: {self.folder_path}")
            
            self.progress_updated.emit(10)
            
            # Get series info
            series_info = self.dicom_loader.get_series_info(self.folder_path, self.series_id)
            self.progress_updated.emit(30)
            
            # Load series
            sitk_image = self.dicom_loader.load_series(self.folder_path, self.series_id)
            self.progress_updated.emit(60)
            
            # Convert to numpy
            image_array = self.dicom_loader.convert_to_numpy(sitk_image)
            self.progress_updated.emit(80)
            
            # Get voxel volume
            voxel_volume = self.dicom_loader.get_voxel_volume(sitk_image)
            
            # Prepare result
            result = {
                "sitk_image": sitk_image,
                "image_array": image_array,
                "series_info": series_info,
                "voxel_volume": voxel_volume,
                "folder_path": self.folder_path,
                "series_id": self.series_id
            }
            
            if self._is_running:
                self.progress_updated.emit(100)
                self.series_loaded.emit(result)
                
        except Exception as e:
            if self._is_running:
                logger.error(f"Error loading DICOM series: {e}")
                self.error_occurred.emit(f"Error loading DICOM series: {str(e)}")
    
    def stop(self):
        """Stop the worker thread."""
        self._is_running = False


class SegmentationWorker(QThread):
    """
    Worker thread for running segmentation.
    
    Performs model inference in a background thread.
    """
    
    # Signals
    progress_updated = pyqtSignal(int)
    segmentation_completed = pyqtSignal(object)
    volume_calculated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """Initialize the segmentation worker."""
        super().__init__(parent)
        self.image_data = None
        self.model_type = ""
        self.model_path = ""
        self.model_manager = ModelManager()
        self._is_running = False
    
    def segment_image(self, image_data: np.ndarray, model_type: str, model_path: str):
        """
        Set parameters for segmentation.
        
        Args:
            image_data: Input image as numpy array
            model_type: Type of model to use
            model_path: Path to the model file
        """
        self.image_data = image_data
        self.model_type = model_type
        self.model_path = model_path
    
    def run(self):
        """Execute the segmentation operation."""
        self._is_running = True
        
        try:
            self.progress_updated.emit(0)
            
            # Validate input
            if self.image_data is None:
                raise ValueError("No image data provided")
            
            self.progress_updated.emit(10)
            
            # Preprocess image
            processed_image = preprocess_for_model(self.image_data, self.model_type)
            self.progress_updated.emit(20)
            
            # Load model if not already loaded
            if not self.model_manager.is_model_loaded(self.model_type):
                success = self.model_manager.load_model(self.model_type, self.model_path)
                if not success:
                    raise ValueError(f"Failed to load model: {self.model_type}")
            
            self.progress_updated.emit(40)
            
            # Run inference
            segmentation_mask = self.model_manager.predict(self.model_type, processed_image)
            
            if segmentation_mask is None:
                raise ValueError(f"Inference failed for model: {self.model_type}")
            
            self.progress_updated.emit(80)
            
            # Post-process mask
            segmentation_mask = postprocess_mask(
                segmentation_mask,
                target_shape=self.image_data.shape,
                model_type=self.model_type
            )
            
            self.progress_updated.emit(100)
            
            # Emit result
            result = {
                "mask": segmentation_mask,
                "model_type": self.model_type
            }
            
            if self._is_running:
                self.segmentation_completed.emit(result)
                
        except Exception as e:
            if self._is_running:
                logger.error(f"Error during segmentation: {e}")
                self.error_occurred.emit(f"Segmentation error: {str(e)}")
    
    def stop(self):
        """Stop the worker thread."""
        self._is_running = False


class VolumeCalculationWorker(QThread):
    """
    Worker thread for calculating liver volume.
    """
    
    # Signals
    volume_calculated = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """Initialize the volume calculation worker."""
        super().__init__(parent)
        self.mask_data = None
        self.voxel_volume = 1.0
        self._is_running = False
    
    def calculate_volume(self, mask_data: np.ndarray, voxel_volume: float):
        """
        Set parameters for volume calculation.
        
        Args:
            mask_data: Segmentation mask
            voxel_volume: Volume of a single voxel in mm³
        """
        self.mask_data = mask_data
        self.voxel_volume = voxel_volume
    
    def run(self):
        """Execute the volume calculation."""
        self._is_running = True
        
        try:
            if self.mask_data is None:
                raise ValueError("No mask data provided")
            
            # Count non-zero voxels
            num_voxels = np.sum(self.mask_data > 0)
            
            # Calculate volume in mm³ and convert to mL
            volume_mm3 = num_voxels * self.voxel_volume
            volume_ml = volume_mm3 / 1000.0
            
            if self._is_running:
                self.volume_calculated.emit(volume_ml)
                
        except Exception as e:
            if self._is_running:
                logger.error(f"Error calculating volume: {e}")
                self.error_occurred.emit(f"Volume calculation error: {str(e)}")
    
    def stop(self):
        """Stop the worker thread."""
        self._is_running = False


class TotalSegmentatorWorker(QThread):
    """
    Worker thread for running TotalSegmentator segmentation.
    
    TotalSegmentator downloads model weights automatically on first run.
    """
    
    # Signals
    progress_updated = pyqtSignal(int)
    segmentation_completed = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """Initialize the TotalSegmentator worker."""
        super().__init__(parent)
        self.sitk_image = None
        self._is_running = False
    
    def set_image(self, sitk_image):
        """
        Set the SimpleITK image to segment.
        
        Args:
            sitk_image: SimpleITK image object
        """
        self.sitk_image = sitk_image
    
    def run(self):
        """Execute the TotalSegmentator segmentation."""
        self._is_running = True
        
        try:
            self.progress_updated.emit(5)
            
            if self.sitk_image is None:
                raise ValueError("No image provided")
            
            # Import model manager
            model_manager = ModelManager()
            
            # Define progress callback
            def progress_cb(val):
                if self._is_running:
                    self.progress_updated.emit(val)
            
            # Run TotalSegmentator
            liver_mask = model_manager.run_totalsegmentator(
                self.sitk_image, 
                progress_callback=progress_cb
            )
            
            if liver_mask is None:
                raise ValueError("TotalSegmentator failed to produce a liver mask")
            
            self.progress_updated.emit(100)
            
            # Emit result
            result = {
                "mask": liver_mask,
                "model_type": "TotalSegmentator"
            }
            
            if self._is_running:
                self.segmentation_completed.emit(result)
                
        except Exception as e:
            if self._is_running:
                logger.error(f"Error during TotalSegmentator: {e}")
                import traceback
                traceback.print_exc()
                self.error_occurred.emit(f"TotalSegmentator error: {str(e)}")
    
    def stop(self):
        """Stop the worker thread."""
        self._is_running = False
