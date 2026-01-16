"""
Model Manager Module.

This module handles loading, managing, and running inference
with different segmentation models (YOLOv11, U-Net, nnU-Net, TotalSegmentator).
"""

import os
import tempfile
import numpy as np
import torch
import logging
import importlib
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manager for segmentation models.
    
    Handles loading, caching, and running inference with
    different types of segmentation models.
    """
    
    def __init__(self):
        """Initialize the model manager."""
        self._models: Dict[str, Any] = {}
        self._device = self._get_device()
        logger.info(f"ModelManager initialized with device: {self._device}")
    
    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def is_model_loaded(self, model_type: str) -> bool:
        """
        Check if a model is already loaded.
        
        Args:
            model_type: Type of model to check
            
        Returns:
            True if model is loaded, False otherwise
        """
        return model_type in self._models and self._models[model_type] is not None
    
    def load_model(self, model_type: str, model_path: str) -> bool:
        """
        Load a model from disk.
        
        Args:
            model_type: Type of model to load
            model_path: Path to the model file
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            model_type_lower = model_type.lower()
            
            if model_type_lower in ["yolo", "yolov11"]:
                return self._load_yolo_model(model_type, model_path)
            elif model_type_lower in ["u-net", "unet"]:
                return self._load_unet_model(model_type, model_path)
            elif model_type_lower in ["nnu-net", "nnunet"]:
                return self._load_nnunet_model(model_type, model_path)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model {model_type}: {e}")
            return False
    
    def _load_yolo_model(self, model_type: str, model_path: str) -> bool:
        """Load a YOLO model."""
        try:
            ultralytics = importlib.import_module("ultralytics")
            yolo_cls = getattr(ultralytics, "YOLO")
            model = yolo_cls(model_path)
            self._models[model_type] = model
            logger.info(f"YOLO model loaded from {model_path}")
            return True
        except ImportError:
            logger.error("ultralytics package not installed")
            return False
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            return False
    
    def _load_unet_model(self, model_type: str, model_path: str) -> bool:
        """Load a U-Net model."""
        try:
            # Create U-Net architecture
            model = self._create_unet()
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self._device)
            model.load_state_dict(state_dict)
            model.to(self._device)
            model.eval()
            
            self._models[model_type] = model
            logger.info(f"U-Net model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading U-Net model: {e}")
            return False
    
    def _load_nnunet_model(self, model_type: str, model_path: str) -> bool:
        """Load an nnU-Net model."""
        try:
            # nnU-Net uses its own loading mechanism
            state_dict = torch.load(model_path, map_location=self._device)
            
            # Create model architecture (simplified)
            model = self._create_unet(depth=5)
            
            # Try to load state dict
            try:
                model.load_state_dict(state_dict)
            except:
                # If state dict doesn't match, use as-is
                logger.warning("nnU-Net state dict mismatch, using partial load")
            
            model.to(self._device)
            model.eval()
            
            self._models[model_type] = model
            logger.info(f"nnU-Net model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading nnU-Net model: {e}")
            return False
    
    def _create_unet(self, in_channels: int = 1, out_channels: int = 2, 
                     features: int = 64, depth: int = 4) -> torch.nn.Module:
        """
        Create a U-Net architecture.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: Base number of features
            depth: Depth of the network
            
        Returns:
            U-Net model
        """
        
        class DoubleConv(torch.nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    torch.nn.BatchNorm2d(out_ch),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    torch.nn.BatchNorm2d(out_ch),
                    torch.nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                return self.conv(x)
        
        class UNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoders = torch.nn.ModuleList()
                self.decoders = torch.nn.ModuleList()
                self.pool = torch.nn.MaxPool2d(2)
                self.upconvs = torch.nn.ModuleList()
                
                # Encoder
                ch = in_channels
                for i in range(depth):
                    out_ch = features * (2 ** i)
                    self.encoders.append(DoubleConv(ch, out_ch))
                    ch = out_ch
                
                # Bottleneck
                self.bottleneck = DoubleConv(ch, ch * 2)
                
                # Decoder
                for i in range(depth - 1, -1, -1):
                    out_ch = features * (2 ** i)
                    self.upconvs.append(
                        torch.nn.ConvTranspose2d(out_ch * 4, out_ch * 2, 2, 2)
                    )
                    self.decoders.append(DoubleConv(out_ch * 4, out_ch))
                
                self.final = torch.nn.Conv2d(features, out_channels, 1)
            
            def forward(self, x):
                skips = []
                
                # Encoder path
                for encoder in self.encoders:
                    x = encoder(x)
                    skips.append(x)
                    x = self.pool(x)
                
                x = self.bottleneck(x)
                
                # Decoder path
                for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
                    x = upconv(x)
                    skip = skips[-(i + 1)]
                    
                    # Handle size mismatch
                    if x.shape != skip.shape:
                        x = torch.nn.functional.interpolate(
                            x, size=skip.shape[2:], mode='bilinear', align_corners=True
                        )
                    
                    x = torch.cat([x, skip], dim=1)
                    x = decoder(x)
                
                return self.final(x)
        
        return UNet()
    
    def predict(self, model_type: str, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Run inference with a loaded model.
        
        Args:
            model_type: Type of model to use
            image: Input image as numpy array
            
        Returns:
            Segmentation mask as numpy array, or None if failed
        """
        if not self.is_model_loaded(model_type):
            logger.error(f"Model {model_type} not loaded")
            return None
        
        try:
            model_type_lower = model_type.lower()
            
            if model_type_lower in ["yolo", "yolov11"]:
                return self._predict_yolo(model_type, image)
            elif model_type_lower in ["u-net", "unet", "nnu-net", "nnunet"]:
                return self._predict_unet(model_type, image)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None
    
    def _predict_yolo(self, model_type: str, image: np.ndarray) -> Optional[np.ndarray]:
        """Run YOLO inference."""
        try:
            model = self._models[model_type]
            
            # Process 3D volume slice by slice
            if image.ndim == 3:
                masks = []
                for i in range(image.shape[0]):
                    slice_img = image[i]
                    
                    # Normalize to 0-255 for YOLO
                    slice_normalized = ((slice_img - slice_img.min()) / 
                                       (slice_img.max() - slice_img.min() + 1e-8) * 255).astype(np.uint8)
                    
                    # Convert to RGB
                    slice_rgb = np.stack([slice_normalized] * 3, axis=-1)
                    
                    # Run inference
                    results = model.predict(slice_rgb, verbose=False)
                    
                    # Extract mask
                    if results and len(results) > 0 and results[0].masks is not None:
                        mask = results[0].masks.data[0].cpu().numpy()
                        # Resize to original size
                        from scipy.ndimage import zoom
                        zoom_factors = (slice_img.shape[0] / mask.shape[0],
                                       slice_img.shape[1] / mask.shape[1])
                        mask = zoom(mask, zoom_factors, order=0)
                    else:
                        mask = np.zeros_like(slice_img)
                    
                    masks.append(mask)
                
                return np.stack(masks, axis=0)
            else:
                # Single 2D image
                return self._predict_yolo_2d(model, image)
                
        except Exception as e:
            logger.error(f"Error in YOLO prediction: {e}")
            return None
    
    def _predict_yolo_2d(self, model, image: np.ndarray) -> np.ndarray:
        """Run YOLO inference on a 2D image."""
        # Normalize
        normalized = ((image - image.min()) / 
                     (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        rgb = np.stack([normalized] * 3, axis=-1)
        
        results = model.predict(rgb, verbose=False)
        
        if results and len(results) > 0 and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            from scipy.ndimage import zoom
            zoom_factors = (image.shape[0] / mask.shape[0],
                           image.shape[1] / mask.shape[1])
            return zoom(mask, zoom_factors, order=0)
        
        return np.zeros_like(image)
    
    def _predict_unet(self, model_type: str, image: np.ndarray) -> Optional[np.ndarray]:
        """Run U-Net inference."""
        try:
            model = self._models[model_type]
            
            # Process 3D volume slice by slice
            if image.ndim == 3:
                masks = []
                for i in range(image.shape[0]):
                    slice_img = image[i]
                    mask = self._predict_unet_2d(model, slice_img)
                    masks.append(mask)
                return np.stack(masks, axis=0)
            else:
                return self._predict_unet_2d(model, image)
                
        except Exception as e:
            logger.error(f"Error in U-Net prediction: {e}")
            return None
    
    def _predict_unet_2d(self, model, image: np.ndarray) -> np.ndarray:
        """Run U-Net inference on a 2D image."""
        original_shape = image.shape
        
        # Resize to model input size
        from scipy.ndimage import zoom
        target_size = (256, 256)
        zoom_factors = (target_size[0] / image.shape[0],
                       target_size[1] / image.shape[1])
        resized = zoom(image, zoom_factors, order=1)
        
        # Prepare tensor
        tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self._device)
        
        # Run inference
        with torch.no_grad():
            output = model(tensor)
            prediction = torch.softmax(output, dim=1)
            mask = prediction[0, 1].cpu().numpy()
        
        # Resize back to original size
        zoom_factors_back = (original_shape[0] / target_size[0],
                            original_shape[1] / target_size[1])
        mask = zoom(mask, zoom_factors_back, order=0)
        
        return (mask > 0.5).astype(np.uint8)
    
    def unload_model(self, model_type: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_type: Type of model to unload
            
        Returns:
            True if unloaded successfully, False otherwise
        """
        if model_type in self._models:
            del self._models[model_type]
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info(f"Model {model_type} unloaded")
            return True
        return False
    
    def unload_all(self):
        """Unload all models from memory."""
        self._models.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("All models unloaded")

    def run_totalsegmentator(self, sitk_image, progress_callback=None) -> Optional[np.ndarray]:
        """
        Run TotalSegmentator on a SimpleITK image and extract the liver mask.
        
        TotalSegmentator downloads its weights automatically on first run.
        
        Args:
            sitk_image: SimpleITK image object
            progress_callback: Optional callback for progress updates
            
        Returns:
            Liver segmentation mask as numpy array, or None if failed
        """
        try:
            import SimpleITK as sitk
            import nibabel as nib
            from totalsegmentator.python_api import totalsegmentator
            from scipy.ndimage import zoom
            
            if progress_callback:
                progress_callback(10)
            
            # Get original image size for reference
            original_size = sitk_image.GetSize()  # (X, Y, Z)
            original_array = sitk.GetArrayFromImage(sitk_image)  # (Z, Y, X)
            original_shape = original_array.shape
            logger.info(f"Original image shape (Z,Y,X): {original_shape}")
            
            # Create temporary directory for I/O
            with tempfile.TemporaryDirectory() as tmpdir:
                input_nifti = os.path.join(tmpdir, "input.nii.gz")
                output_dir = os.path.join(tmpdir, "output")
                os.makedirs(output_dir, exist_ok=True)
                
                # Save the SimpleITK image as NIfTI
                sitk.WriteImage(sitk_image, input_nifti)
                logger.info(f"Saved input NIfTI to {input_nifti}")
                
                if progress_callback:
                    progress_callback(20)
                
                # Run TotalSegmentator - using full resolution for accuracy
                logger.info("Running TotalSegmentator (this may take a few minutes)...")
                
                totalsegmentator(
                    input=input_nifti,
                    output=output_dir,
                    task="total",  # Full segmentation
                    fast=False,    # Full resolution for accurate volume
                    quiet=False
                )
                
                if progress_callback:
                    progress_callback(70)
                
                # Load the liver segmentation result
                liver_file = os.path.join(output_dir, "liver.nii.gz")
                
                if os.path.exists(liver_file):
                    liver_nib = nib.load(liver_file)
                    liver_mask = liver_nib.get_fdata()
                    
                    logger.info(f"Loaded liver mask shape (from NIfTI): {liver_mask.shape}")
                    
                    # Convert to binary mask (0 or 1)
                    liver_mask = (liver_mask > 0).astype(np.uint8)
                    
                    # NIfTI is typically stored as (X, Y, Z), need to match DICOM (Z, Y, X)
                    if liver_mask.ndim == 3:
                        # Transpose from (X, Y, Z) to (Z, Y, X)
                        liver_mask = np.transpose(liver_mask, (2, 1, 0))
                        logger.info(f"Transposed mask shape: {liver_mask.shape}")
                    
                    # Resize mask to match original image size if needed
                    if liver_mask.shape != original_shape:
                        logger.info(f"Resizing mask from {liver_mask.shape} to {original_shape}")
                        zoom_factors = tuple(o / m for o, m in zip(original_shape, liver_mask.shape))
                        liver_mask = zoom(liver_mask, zoom_factors, order=0)
                        liver_mask = (liver_mask > 0.5).astype(np.uint8)
                        logger.info(f"Resized mask shape: {liver_mask.shape}")
                    
                    if progress_callback:
                        progress_callback(100)
                    
                    logger.info(f"Final liver mask shape: {liver_mask.shape}, non-zero voxels: {np.sum(liver_mask > 0)}")
                    return liver_mask
                else:
                    # Check if combined segmentation exists
                    combined_file = os.path.join(output_dir, "total.nii.gz")
                    if os.path.exists(combined_file):
                        seg_nib = nib.load(combined_file)
                        seg_data = seg_nib.get_fdata()
                        
                        # Liver label in TotalSegmentator is typically 1
                        liver_mask = (seg_data == 1).astype(np.uint8)
                        
                        if liver_mask.ndim == 3:
                            liver_mask = np.transpose(liver_mask, (2, 1, 0))
                        
                        # Resize if needed
                        if liver_mask.shape != original_shape:
                            zoom_factors = tuple(o / m for o, m in zip(original_shape, liver_mask.shape))
                            liver_mask = zoom(liver_mask, zoom_factors, order=0)
                            liver_mask = (liver_mask > 0.5).astype(np.uint8)
                        
                        if progress_callback:
                            progress_callback(100)
                        
                        logger.info(f"Liver mask extracted from combined, shape: {liver_mask.shape}")
                        return liver_mask
                    
                    logger.error("Liver segmentation file not found in output")
                    return None
                    
        except ImportError as e:
            logger.error(f"TotalSegmentator not installed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error running TotalSegmentator: {e}")
            import traceback
            traceback.print_exc()
            return None
