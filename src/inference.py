#!/usr/bin/env python3
"""
Inference script for trained TBI segmentation model.
"""

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from pathlib import Path
import logging
import argparse
from typing import Dict, Optional, Tuple
import json
from train import UNet3D

logger = logging.getLogger(__name__)


class TBISegmentationInference:
    """Inference class for TBI segmentation model."""
    
    def __init__(self, model_path: Path, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model checkpoint
        checkpoint = torch.load(str(model_path), map_location=self.device)
        self.config = checkpoint['config']
        
        # Initialize model
        self.model = UNet3D(
            in_channels=self.config['in_channels'],
            out_channels=self.config['out_channels'],
            base_filters=self.config['base_filters']
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Model epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            logger.info(f"Model validation loss: {checkpoint['val_loss']:.4f}")
    
    def preprocess_image(self, t1_path: Path, flair_path: Path) -> Tuple[torch.Tensor, nib.Nifti1Image]:
        """Preprocess input images for inference."""
        # Load images
        t1_nii = nib.load(str(t1_path))
        flair_nii = nib.load(str(flair_path))
        
        t1_data = t1_nii.get_fdata()
        flair_data = flair_nii.get_fdata()
        
        # Ensure same shape
        min_shape = np.minimum(t1_data.shape, flair_data.shape)
        t1_data = t1_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        flair_data = flair_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        # Normalize
        t1_data = self._normalize(t1_data)
        flair_data = self._normalize(flair_data)
        
        # Stack channels and add batch dimension
        image = np.stack([t1_data, flair_data], axis=0)
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # (1, 2, H, W, D)
        
        return image_tensor, t1_nii
    
    def _normalize(self, data):
        """Normalize image data to [0, 1]."""
        data = data.astype(np.float32)
        brain_mask = data > 0
        if np.any(brain_mask):
            data_brain = data[brain_mask]
            p1, p99 = np.percentile(data_brain, [1, 99])
            data = np.clip(data, p1, p99)
            data = (data - p1) / (p99 - p1 + 1e-8)
        return data
    
    def predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Run model inference."""
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            predictions = F.softmax(outputs, dim=1)
            segmentation = torch.argmax(predictions, dim=1)
        return segmentation.cpu()
    
    def postprocess_segmentation(self, segmentation: torch.Tensor, reference_nii: nib.Nifti1Image) -> nib.Nifti1Image:
        """Convert segmentation tensor to NIfTI image."""
        seg_array = segmentation.squeeze().numpy().astype(np.uint8)
        seg_nii = nib.Nifti1Image(seg_array, affine=reference_nii.affine, header=reference_nii.header)
        return seg_nii
    
    def segment_brain(self, t1_path: Path, flair_path: Path, output_path: Path) -> Dict:
        """Perform brain segmentation on input images."""
        logger.info(f"Segmenting brain: T1={t1_path}, FLAIR={flair_path}")
        
        # Preprocess
        image_tensor, reference_nii = self.preprocess_image(t1_path, flair_path)
        
        # Predict
        segmentation = self.predict(image_tensor)
        
        # Postprocess
        seg_nii = self.postprocess_segmentation(segmentation, reference_nii)
        
        # Save result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(seg_nii, str(output_path))
        
        # Calculate statistics
        seg_data = seg_nii.get_fdata()
        statistics = self._calculate_statistics(seg_data, reference_nii)
        
        logger.info(f"Segmentation saved to: {output_path}")
        
        return {
            'segmentation_path': str(output_path),
            'statistics': statistics,
            'class_labels': self.config['class_labels']
        }
    
    def _calculate_statistics(self, segmentation: np.ndarray, reference_nii: nib.Nifti1Image) -> Dict:
        """Calculate segmentation statistics."""
        # Get voxel volume
        voxel_dims = reference_nii.header.get_zooms()[:3]
        voxel_volume_mm3 = np.prod(voxel_dims)
        
        statistics = {}
        unique_labels = np.unique(segmentation)
        
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
                
            label_name = self.config['class_labels'].get(str(int(label)), f'Unknown_{int(label)}')
            voxel_count = np.sum(segmentation == label)
            volume_mm3 = voxel_count * voxel_volume_mm3
            
            statistics[label_name] = {
                'label_id': int(label),
                'voxel_count': int(voxel_count),
                'volume_mm3': float(volume_mm3),
                'volume_ml': float(volume_mm3 / 1000)
            }
        
        return statistics


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run brain segmentation inference')
    parser.add_argument('--model-path', type=Path, default='models/brain_segmentation/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--t1-path', type=Path, required=True,
                       help='Path to T1-weighted image')
    parser.add_argument('--flair-path', type=Path, required=True,
                       help='Path to FLAIR image')
    parser.add_argument('--output-path', type=Path, required=True,
                       help='Output path for segmentation')
    parser.add_argument('--save-stats', type=Path,
                       help='Path to save statistics JSON')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check inputs
    if not args.model_path.exists():
        logger.error(f"Model not found: {args.model_path}")
        return 1
    
    if not args.t1_path.exists():
        logger.error(f"T1 image not found: {args.t1_path}")
        return 1
    
    if not args.flair_path.exists():
        logger.error(f"FLAIR image not found: {args.flair_path}")
        return 1
    
    # Initialize inference
    inference = TBISegmentationInference(args.model_path, args.device)
    
    # Run segmentation
    results = inference.segment_brain(args.t1_path, args.flair_path, args.output_path)
    
    # Save statistics if requested
    if args.save_stats:
        args.save_stats.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_stats, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Statistics saved to: {args.save_stats}")
    
    # Print summary
    print("\n" + "="*50)
    print("BRAIN SEGMENTATION RESULTS")
    print("="*50)
    print(f"Input T1: {args.t1_path}")
    print(f"Input FLAIR: {args.flair_path}")
    print(f"Output segmentation: {args.output_path}")
    
    if results['statistics']:
        print("\nSegmented structures:")
        for structure, stats in results['statistics'].items():
            print(f"  - {structure}: {stats['volume_mm3']:.2f} mm³ ({stats['volume_ml']:.2f} ml)")
    
    return 0


if __name__ == "__main__":
    exit(main())