import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Union, Dict, List, Optional
import logging
from monai.networks.nets import UNet
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, 
    ScaleIntensityRanged, CropForegroundd, Resized
)
from monai.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class TBISegmentationModel(nn.Module):
    """U-Net based model for multi-class TBI lesion segmentation."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 8):  # 7 lesion types + background
        super().__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
            dropout=0.1
        )
        
    def forward(self, x):
        return self.unet(x)


class TBISegmenter:
    """Main class for TBI lesion segmentation."""
    
    # Define lesion types
    LESION_TYPES = {
        0: "background",
        1: "intraparenchymal_hemorrhage",
        2: "subdural_hematoma",
        3: "epidural_hematoma",
        4: "subarachnoid_hemorrhage",
        5: "intraventricular_hemorrhage",
        6: "diffuse_axonal_injury",
        7: "contusion"
    }
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = TBISegmentationModel(in_channels=1, out_channels=8)
        self.model.to(self.device)
        
        # Load pretrained weights if provided
        if model_path:
            self.load_model(model_path)
        else:
            logger.warning("No pretrained model loaded. Using random initialization.")
            
        self.model.eval()
        
    def load_model(self, model_path: Union[str, Path]):
        """Load pretrained model weights."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from {model_path}")
        
    def preprocess_for_inference(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Preprocess image for model inference."""
        # Load NIfTI image
        nii_img = nib.load(str(image_path))
        img_data = nii_img.get_fdata()
        
        # Add channel dimension and normalize
        img_data = img_data[np.newaxis, ...]  # Add channel dimension
        
        # Normalize to [0, 1] if not already
        if img_data.max() > 1.0:
            img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img_data).float().unsqueeze(0)
        
        return img_tensor, nii_img.affine, nii_img.header
    
    def postprocess_prediction(self, prediction: torch.Tensor, 
                             original_affine: np.ndarray,
                             original_header) -> nib.Nifti1Image:
        """Convert model output to NIfTI format."""
        # Get class predictions
        pred_classes = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
        
        # Create NIfTI image
        pred_nii = nib.Nifti1Image(pred_classes.astype(np.uint8), 
                                  affine=original_affine,
                                  header=original_header)
        
        return pred_nii
    
    def segment_lesions(self, image_path: Union[str, Path], 
                       output_dir: Union[str, Path],
                       save_probabilities: bool = False) -> Dict:
        """Perform lesion segmentation on preprocessed MRI."""
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Segmenting lesions in {image_path}")
        
        # Preprocess image
        img_tensor, affine, header = self.preprocess_for_inference(image_path)
        img_tensor = img_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            prediction = self.model(img_tensor)
            
        # Postprocess
        segmentation_nii = self.postprocess_prediction(prediction, affine, header)
        
        # Save segmentation
        seg_path = output_dir / "lesion_segmentation.nii.gz"
        nib.save(segmentation_nii, str(seg_path))
        logger.info(f"Saved segmentation to {seg_path}")
        
        # Save probability maps if requested
        if save_probabilities:
            prob_maps = torch.softmax(prediction, dim=1).squeeze().cpu().numpy()
            for i, lesion_type in self.LESION_TYPES.items():
                if i == 0:  # Skip background
                    continue
                prob_nii = nib.Nifti1Image(prob_maps[i], affine=affine, header=header)
                prob_path = output_dir / f"prob_{lesion_type}.nii.gz"
                nib.save(prob_nii, str(prob_path))
        
        # Calculate lesion statistics
        seg_data = segmentation_nii.get_fdata()
        lesion_stats = self.calculate_lesion_statistics(seg_data, affine)
        
        return {
            'segmentation_path': seg_path,
            'lesion_statistics': lesion_stats
        }
    
    def calculate_lesion_statistics(self, segmentation: np.ndarray, 
                                  affine: np.ndarray) -> Dict:
        """Calculate volume and count statistics for each lesion type."""
        # Calculate voxel volume from affine matrix
        voxel_dims = np.abs(np.diag(affine)[:3])
        voxel_volume_mm3 = np.prod(voxel_dims)
        
        stats = {}
        for lesion_id, lesion_name in self.LESION_TYPES.items():
            if lesion_id == 0:  # Skip background
                continue
                
            mask = segmentation == lesion_id
            voxel_count = np.sum(mask)
            volume_mm3 = voxel_count * voxel_volume_mm3
            
            stats[lesion_name] = {
                'voxel_count': int(voxel_count),
                'volume_mm3': float(volume_mm3),
                'volume_ml': float(volume_mm3 / 1000)
            }
            
        return stats
    
    def create_training_transforms(self):
        """Create MONAI transforms for training."""
        train_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0)),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-100,
                a_max=200,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Resized(keys=["image", "label"], spatial_size=(128, 128, 128))
        ]
        return train_transforms