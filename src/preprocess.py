import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
import pydicom
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MRIPreprocessor:
    """Preprocessor for MRI data including loading, skull stripping, and normalization."""
    
    def __init__(self):
        self.target_spacing = (1.0, 1.0, 1.0)  # 1mm isotropic
        self.target_shape = (256, 256, 256)   # Standard volume size
        
    def load_dicom_series(self, dicom_dir: Union[str, Path]) -> sitk.Image:
        """Load DICOM series from directory."""
        dicom_dir = Path(dicom_dir)
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        
        if not series_ids:
            raise ValueError(f"No DICOM series found in {dicom_dir}")
            
        # Use the first series
        series_id = series_ids[0]
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)
        reader.SetFileNames(dicom_names)
        
        image = reader.Execute()
        logger.info(f"Loaded DICOM series with shape: {image.GetSize()}")
        return image
    
    def load_nifti(self, nifti_path: Union[str, Path]) -> sitk.Image:
        """Load NIfTI file."""
        image = sitk.ReadImage(str(nifti_path))
        logger.info(f"Loaded NIfTI with shape: {image.GetSize()}")
        return image
    
    def bias_field_correction(self, image: sitk.Image) -> sitk.Image:
        """Apply N4 bias field correction."""
        logger.info("Applying N4 bias field correction...")
        
        # Convert to float32 for processing
        image = sitk.Cast(image, sitk.sitkFloat32)
        
        # Create mask (simple thresholding)
        mask = sitk.OtsuThreshold(image, 0, 1, 200)
        
        # N4 bias correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
        corrector.SetConvergenceThreshold(0.001)
        
        corrected = corrector.Execute(image, mask)
        return corrected
    
    def skull_strip(self, image: sitk.Image) -> Tuple[sitk.Image, sitk.Image]:
        """Basic skull stripping using thresholding and morphological operations."""
        logger.info("Performing skull stripping...")
        
        # Threshold to create initial brain mask
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(0)
        otsu_filter.SetOutsideValue(1)
        mask = otsu_filter.Execute(image)
        
        # Morphological operations to clean up mask
        mask = sitk.BinaryMorphologicalOpening(mask, [2, 2, 2])
        mask = sitk.BinaryMorphologicalClosing(mask, [2, 2, 2])
        
        # Fill holes
        mask = sitk.BinaryFillhole(mask)
        
        # Apply mask to image
        brain_image = sitk.Mask(image, mask)
        
        return brain_image, mask
    
    def resample_to_isotropic(self, image: sitk.Image, 
                            target_spacing: Optional[Tuple[float, float, float]] = None) -> sitk.Image:
        """Resample image to isotropic spacing."""
        if target_spacing is None:
            target_spacing = self.target_spacing
            
        logger.info(f"Resampling to spacing: {target_spacing}")
        
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        # Calculate new size
        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(original_size, original_spacing, target_spacing)
        ]
        
        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        resampler.SetInterpolator(sitk.sitkLinear)
        
        resampled = resampler.Execute(image)
        return resampled
    
    def normalize_intensity(self, image: sitk.Image, 
                          percentile_low: float = 1.0,
                          percentile_high: float = 99.0) -> sitk.Image:
        """Normalize image intensity to [0, 1] range using percentiles."""
        logger.info("Normalizing intensity...")
        
        # Convert to numpy for percentile calculation
        img_array = sitk.GetArrayFromImage(image)
        
        # Calculate percentiles
        p_low = np.percentile(img_array[img_array > 0], percentile_low)
        p_high = np.percentile(img_array[img_array > 0], percentile_high)
        
        # Clip and normalize
        img_array = np.clip(img_array, p_low, p_high)
        img_array = (img_array - p_low) / (p_high - p_low)
        
        # Convert back to SimpleITK
        normalized = sitk.GetImageFromArray(img_array)
        normalized.CopyInformation(image)
        
        return normalized
    
    def preprocess_mri(self, input_path: Union[str, Path], 
                      output_dir: Union[str, Path],
                      save_intermediate: bool = False) -> dict:
        """Full preprocessing pipeline for MRI data."""
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image
        if input_path.is_dir():
            image = self.load_dicom_series(input_path)
        else:
            image = self.load_nifti(input_path)
        
        # Save original if requested
        if save_intermediate:
            sitk.WriteImage(image, str(output_dir / "01_original.nii.gz"))
        
        # Bias field correction
        image = self.bias_field_correction(image)
        if save_intermediate:
            sitk.WriteImage(image, str(output_dir / "02_bias_corrected.nii.gz"))
        
        # Skull stripping
        brain_image, brain_mask = self.skull_strip(image)
        if save_intermediate:
            sitk.WriteImage(brain_image, str(output_dir / "03_skull_stripped.nii.gz"))
            sitk.WriteImage(brain_mask, str(output_dir / "03_brain_mask.nii.gz"))
        
        # Resample to isotropic
        brain_image = self.resample_to_isotropic(brain_image)
        brain_mask = self.resample_to_isotropic(brain_mask)
        if save_intermediate:
            sitk.WriteImage(brain_image, str(output_dir / "04_resampled.nii.gz"))
        
        # Normalize intensity
        brain_image = self.normalize_intensity(brain_image)
        
        # Save final preprocessed image
        final_path = output_dir / "preprocessed_mri.nii.gz"
        sitk.WriteImage(brain_image, str(final_path))
        
        # Save brain mask
        mask_path = output_dir / "brain_mask.nii.gz"
        sitk.WriteImage(brain_mask, str(mask_path))
        
        logger.info(f"Preprocessing complete. Saved to {final_path}")
        
        return {
            'preprocessed_image': final_path,
            'brain_mask': mask_path,
            'spacing': brain_image.GetSpacing(),
            'size': brain_image.GetSize()
        }