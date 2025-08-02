import logging
import yaml
from pathlib import Path
from typing import Union, Dict, Any, Optional
import json
import numpy as np
import nibabel as nib


def setup_logging(log_level: str = "INFO", log_file: Optional[Union[str, Path]] = None):
    """Set up logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: Union[str, Path]):
    """Save configuration to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def validate_nifti_file(file_path: Union[str, Path]) -> bool:
    """Validate if file is a valid NIfTI file."""
    try:
        nib.load(str(file_path))
        return True
    except Exception:
        return False


def get_image_info(image_path: Union[str, Path]) -> Dict[str, Any]:
    """Get basic information about a NIfTI image."""
    nii = nib.load(str(image_path))
    
    info = {
        'shape': nii.shape,
        'voxel_dims': nii.header.get_zooms()[:3],
        'voxel_volume_mm3': np.prod(nii.header.get_zooms()[:3]),
        'data_type': nii.get_data_dtype(),
        'affine': nii.affine.tolist(),
        'min_value': float(nii.get_fdata().min()),
        'max_value': float(nii.get_fdata().max())
    }
    
    return info


def create_color_lookup_table() -> Dict[int, Dict[str, Any]]:
    """Create color lookup table for visualization."""
    colors = {
        # Lesion types (1-7)
        1: {"name": "intraparenchymal_hemorrhage", "rgb": [255, 0, 0]},      # Red
        2: {"name": "subdural_hematoma", "rgb": [255, 128, 0]},              # Orange
        3: {"name": "epidural_hematoma", "rgb": [255, 255, 0]},              # Yellow
        4: {"name": "subarachnoid_hemorrhage", "rgb": [0, 255, 0]},          # Green
        5: {"name": "intraventricular_hemorrhage", "rgb": [0, 255, 255]},    # Cyan
        6: {"name": "diffuse_axonal_injury", "rgb": [0, 0, 255]},            # Blue
        7: {"name": "contusion", "rgb": [255, 0, 255]},                      # Magenta
        
        # Brain zones (11-20)
        11: {"name": "frontal_lobe_left", "rgb": [128, 128, 255]},
        12: {"name": "frontal_lobe_right", "rgb": [128, 192, 255]},
        13: {"name": "parietal_lobe_left", "rgb": [255, 128, 128]},
        14: {"name": "parietal_lobe_right", "rgb": [255, 192, 128]},
        15: {"name": "temporal_lobe_left", "rgb": [128, 255, 128]},
        16: {"name": "temporal_lobe_right", "rgb": [192, 255, 128]},
        17: {"name": "occipital_lobe_left", "rgb": [255, 255, 128]},
        18: {"name": "occipital_lobe_right", "rgb": [255, 255, 192]},
        19: {"name": "cerebellum", "rgb": [192, 128, 255]},
        20: {"name": "brainstem", "rgb": [128, 255, 255]}
    }
    
    return colors


def save_color_lookup_table(output_path: Union[str, Path]):
    """Save color lookup table to JSON file."""
    colors = create_color_lookup_table()
    with open(output_path, 'w') as f:
        json.dump(colors, f, indent=2)


class NiftiConverter:
    """Convert between different medical image formats."""
    
    @staticmethod
    def dicom_to_nifti(dicom_dir: Union[str, Path], 
                      output_path: Union[str, Path]) -> Path:
        """Convert DICOM series to NIfTI."""
        import SimpleITK as sitk
        
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        
        if not series_ids:
            raise ValueError(f"No DICOM series found in {dicom_dir}")
        
        # Use first series
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
        reader.SetFileNames(dicom_names)
        
        image = reader.Execute()
        sitk.WriteImage(image, str(output_path))
        
        return Path(output_path)
    
    @staticmethod
    def nifti_to_dicom(nifti_path: Union[str, Path],
                      output_dir: Union[str, Path],
                      series_description: str = "Converted from NIfTI") -> Path:
        """Convert NIfTI to DICOM series."""
        import SimpleITK as sitk
        from datetime import datetime
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read NIfTI
        image = sitk.ReadImage(str(nifti_path))
        
        # Set DICOM tags
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        
        # Create DICOM series
        for i in range(image.GetDepth()):
            image_slice = image[:, :, i]
            
            # Set metadata
            image_slice.SetMetaData("0008|0008", "DERIVED\\SECONDARY")  # Image Type
            image_slice.SetMetaData("0008|103e", series_description)     # Series Description
            image_slice.SetMetaData("0020|0013", str(i))                # Instance Number
            
            # Write slice
            output_path = output_dir / f"slice_{i:04d}.dcm"
            writer.SetFileName(str(output_path))
            writer.Execute(image_slice)
        
        return output_dir