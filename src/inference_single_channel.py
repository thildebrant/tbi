#!/usr/bin/env python3
"""
Inference script for trained brain segmentation model with single-channel input.
Adapts the 2-channel model to work with single MRI volumes.
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


class SingleChannelBrainSegmentation:
    """Inference class for brain segmentation with single-channel input."""
    
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
        logger.info(f"Model expects {self.config['in_channels']} input channels")
    
    def preprocess_single_channel(self, image_path: Path) -> Tuple[torch.Tensor, nib.Nifti1Image, Tuple]:
        """Preprocess single-channel image for inference."""
        # Load image
        nii = nib.load(str(image_path))
        data = nii.get_fdata()
        
        logger.info(f"Input image shape: {data.shape}")
        original_shape = data.shape
        
        # Resize to standard size that works with U-Net (powers of 2)
        target_shape = (128, 128, 64)  # Divisible by 16 for 4 pooling layers
        
        # Resize using scipy if available, otherwise use simple interpolation
        try:
            from scipy.ndimage import zoom
            zoom_factors = [t/o for t, o in zip(target_shape, original_shape)]
            data_resized = zoom(data, zoom_factors, order=1)
        except ImportError:
            # Fallback: simple nearest neighbor resizing using numpy
            data_resized = self._simple_resize(data, target_shape)
        
        logger.info(f"Resized image shape: {data_resized.shape}")
        
        # Normalize
        data_resized = self._normalize(data_resized)
        
        # Create dual-channel input by duplicating the single channel
        if self.config['in_channels'] == 2:
            image = np.stack([data_resized, data_resized], axis=0)
        else:
            image = data_resized[np.newaxis, ...]
        
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # (1, C, H, W, D)
        
        logger.info(f"Preprocessed tensor shape: {image_tensor.shape}")
        return image_tensor, nii, original_shape
    
    def _simple_resize(self, data, target_shape):
        """Simple resize using nearest neighbor interpolation."""
        original_shape = data.shape
        resized = np.zeros(target_shape, dtype=data.dtype)
        
        x_ratio = original_shape[0] / target_shape[0]
        y_ratio = original_shape[1] / target_shape[1]
        z_ratio = original_shape[2] / target_shape[2]
        
        for i in range(target_shape[0]):
            for j in range(target_shape[1]):
                for k in range(target_shape[2]):
                    orig_i = min(int(i * x_ratio), original_shape[0] - 1)
                    orig_j = min(int(j * y_ratio), original_shape[1] - 1)
                    orig_k = min(int(k * z_ratio), original_shape[2] - 1)
                    resized[i, j, k] = data[orig_i, orig_j, orig_k]
        
        return resized
    
    def _normalize(self, data):
        """Normalize image data to [0, 1]."""
        data = data.astype(np.float32)
        
        # Remove background and normalize
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
    
    def postprocess_segmentation(self, segmentation: torch.Tensor, reference_nii: nib.Nifti1Image, original_shape: Tuple) -> nib.Nifti1Image:
        """Convert segmentation tensor to NIfTI image and resize back to original dimensions."""
        seg_array = segmentation.squeeze().numpy().astype(np.uint8)
        
        # Resize back to original dimensions
        if seg_array.shape != original_shape:
            try:
                from scipy.ndimage import zoom
                zoom_factors = [o/s for o, s in zip(original_shape, seg_array.shape)]
                seg_resized = zoom(seg_array, zoom_factors, order=0)  # Nearest neighbor for labels
            except ImportError:
                seg_resized = self._simple_resize_labels(seg_array, original_shape)
        else:
            seg_resized = seg_array
        
        seg_nii = nib.Nifti1Image(seg_resized, affine=reference_nii.affine, header=reference_nii.header)
        return seg_nii
    
    def _simple_resize_labels(self, data, target_shape):
        """Simple resize for label data using nearest neighbor."""
        original_shape = data.shape
        resized = np.zeros(target_shape, dtype=data.dtype)
        
        x_ratio = original_shape[0] / target_shape[0]
        y_ratio = original_shape[1] / target_shape[1]
        z_ratio = original_shape[2] / target_shape[2]
        
        for i in range(target_shape[0]):
            for j in range(target_shape[1]):
                for k in range(target_shape[2]):
                    orig_i = min(int(i * x_ratio), original_shape[0] - 1)
                    orig_j = min(int(j * y_ratio), original_shape[1] - 1)
                    orig_k = min(int(k * z_ratio), original_shape[2] - 1)
                    resized[i, j, k] = data[orig_i, orig_j, orig_k]
        
        return resized
    
    def segment_brain(self, image_path: Path, output_path: Path) -> Dict:
        """Perform brain segmentation on single-channel input."""
        logger.info(f"Segmenting brain from: {image_path}")
        
        # Preprocess
        image_tensor, reference_nii, original_shape = self.preprocess_single_channel(image_path)
        
        # Predict
        segmentation = self.predict(image_tensor)
        
        # Postprocess
        seg_nii = self.postprocess_segmentation(segmentation, reference_nii, original_shape)
        
        # Save result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(seg_nii, str(output_path))
        
        # Calculate statistics
        seg_data = seg_nii.get_fdata()
        statistics = self._calculate_statistics(seg_data, reference_nii)
        
        logger.info(f"Segmentation saved to: {output_path}")
        
        return {
            'input_path': str(image_path),
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
        
        total_volume = 0
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
                
            label_name = self.config['class_labels'].get(str(int(label)), f'Unknown_{int(label)}')
            voxel_count = np.sum(segmentation == label)
            volume_mm3 = voxel_count * voxel_volume_mm3
            total_volume += volume_mm3
            
            statistics[label_name] = {
                'label_id': int(label),
                'voxel_count': int(voxel_count),
                'volume_mm3': float(volume_mm3),
                'volume_ml': float(volume_mm3 / 1000),
                'percentage': 0.0  # Will calculate after total
            }
        
        # Calculate percentages
        for structure in statistics.values():
            if total_volume > 0:
                structure['percentage'] = (structure['volume_mm3'] / total_volume) * 100
        
        return statistics


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run brain segmentation on single-channel MRI')
    parser.add_argument('--model-path', type=Path, default='models/brain_segmentation/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--input-path', type=Path, required=True,
                       help='Path to input MRI image')
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
    
    if not args.input_path.exists():
        logger.error(f"Input image not found: {args.input_path}")
        return 1
    
    # Initialize inference
    inference = SingleChannelBrainSegmentation(args.model_path, args.device)
    
    # Run segmentation
    results = inference.segment_brain(args.input_path, args.output_path)
    
    # Save statistics if requested
    if args.save_stats:
        args.save_stats.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_stats, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Statistics saved to: {args.save_stats}")
    
    # Print summary
    print("\n" + "="*60)
    print("SINGLE-CHANNEL BRAIN SEGMENTATION RESULTS")
    print("="*60)
    print(f"Input: {args.input_path}")
    print(f"Output segmentation: {args.output_path}")
    
    if results['statistics']:
        print(f"\nTotal brain structures: {len(results['statistics'])}")
        print("\nSegmented structures:")
        
        # Sort by volume for better display
        sorted_structures = sorted(
            results['statistics'].items(), 
            key=lambda x: x[1]['volume_mm3'], 
            reverse=True
        )
        
        for structure, stats in sorted_structures:
            print(f"  • {structure:35} {stats['volume_mm3']:8.1f} mm³ "
                  f"({stats['volume_ml']:6.1f} ml) [{stats['percentage']:5.1f}%]")
        
        total_volume = sum(s['volume_mm3'] for s in results['statistics'].values())
        print(f"\nTotal segmented volume: {total_volume:.1f} mm³ ({total_volume/1000:.1f} ml)")
    
    return 0


if __name__ == "__main__":
    exit(main())