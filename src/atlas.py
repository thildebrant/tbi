import numpy as np
import nibabel as nib
import ants
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional
import logging
import SimpleITK as sitk

logger = logging.getLogger(__name__)


class BrainAtlasManager:
    """Manages brain atlas registration and zone mapping."""
    
    # Define 10 major brain zones
    BRAIN_ZONES = {
        1: "frontal_lobe_left",
        2: "frontal_lobe_right",
        3: "parietal_lobe_left",
        4: "parietal_lobe_right",
        5: "temporal_lobe_left",
        6: "temporal_lobe_right",
        7: "occipital_lobe_left",
        8: "occipital_lobe_right",
        9: "cerebellum",
        10: "brainstem"
    }
    
    def __init__(self, atlas_path: Optional[Union[str, Path]] = None):
        self.atlas_path = atlas_path
        self.atlas_image = None
        self.atlas_labels = None
        
        if atlas_path:
            self.load_atlas(atlas_path)
    
    def create_default_atlas(self, reference_image_path: Union[str, Path],
                           output_path: Union[str, Path]) -> nib.Nifti1Image:
        """Create a simple 10-zone atlas based on spatial divisions."""
        logger.info("Creating default 10-zone brain atlas...")
        
        # Load reference image
        ref_nii = nib.load(str(reference_image_path))
        ref_data = ref_nii.get_fdata()
        
        # Create empty atlas
        atlas_data = np.zeros_like(ref_data, dtype=np.uint8)
        
        # Get image dimensions
        x_dim, y_dim, z_dim = ref_data.shape
        
        # Find brain mask (non-zero voxels)
        brain_mask = ref_data > 0.1
        
        # Simple spatial division for demonstration
        # In practice, you would use a proper atlas
        x_mid = x_dim // 2
        y_thirds = [y_dim // 3, 2 * y_dim // 3]
        z_cerebellum = int(z_dim * 0.3)  # Lower 30% for cerebellum/brainstem
        
        # Assign zones based on spatial location
        for x in range(x_dim):
            for y in range(y_dim):
                for z in range(z_dim):
                    if not brain_mask[x, y, z]:
                        continue
                    
                    # Determine hemisphere
                    is_left = x < x_mid
                    
                    # Determine zone
                    if z < z_cerebellum:
                        # Lower brain regions
                        if y < y_thirds[0]:
                            atlas_data[x, y, z] = 10  # Brainstem
                        else:
                            atlas_data[x, y, z] = 9   # Cerebellum
                    else:
                        # Cerebral regions
                        if y < y_thirds[0]:
                            # Frontal
                            atlas_data[x, y, z] = 1 if is_left else 2
                        elif y < y_thirds[1]:
                            # Parietal/Temporal
                            if abs(x - x_mid) < x_dim * 0.2:
                                # Central - Parietal
                                atlas_data[x, y, z] = 3 if is_left else 4
                            else:
                                # Lateral - Temporal
                                atlas_data[x, y, z] = 5 if is_left else 6
                        else:
                            # Occipital
                            atlas_data[x, y, z] = 7 if is_left else 8
        
        # Create NIfTI image
        atlas_nii = nib.Nifti1Image(atlas_data, affine=ref_nii.affine, header=ref_nii.header)
        
        # Save atlas
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        nib.save(atlas_nii, str(output_path))
        logger.info(f"Saved default atlas to {output_path}")
        
        return atlas_nii
    
    def load_atlas(self, atlas_path: Union[str, Path]):
        """Load pre-existing atlas."""
        self.atlas_path = Path(atlas_path)
        self.atlas_image = ants.image_read(str(self.atlas_path))
        logger.info(f"Loaded atlas from {atlas_path}")
    
    def register_atlas_to_subject(self, subject_image_path: Union[str, Path],
                                output_dir: Union[str, Path],
                                registration_type: str = "SyN") -> Dict:
        """Register atlas to subject space using ANTsPy."""
        subject_image_path = Path(subject_image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Registering atlas to subject: {subject_image_path}")
        
        # Load subject image with ANTs
        subject_img = ants.image_read(str(subject_image_path))
        
        # Perform registration
        logger.info(f"Performing {registration_type} registration...")
        registration = ants.registration(
            fixed=subject_img,
            moving=self.atlas_image,
            type_of_transform=registration_type
        )
        
        # Get warped atlas
        warped_atlas = registration['warpedmovout']
        
        # Save warped atlas
        warped_atlas_path = output_dir / "atlas_in_subject_space.nii.gz"
        ants.image_write(warped_atlas, str(warped_atlas_path))
        
        # Save transform files
        forward_transform_path = output_dir / "atlas_to_subject_transform.h5"
        inverse_transform_path = output_dir / "subject_to_atlas_transform.h5"
        
        logger.info(f"Atlas registered to subject space. Saved to {warped_atlas_path}")
        
        return {
            'warped_atlas_path': warped_atlas_path,
            'forward_transforms': registration['fwdtransforms'],
            'inverse_transforms': registration['invtransforms'],
            'registration_params': registration
        }
    
    def apply_transform_to_lesions(self, lesion_seg_path: Union[str, Path],
                                 transforms: List[str],
                                 reference_image_path: Union[str, Path],
                                 output_path: Union[str, Path]) -> nib.Nifti1Image:
        """Apply transformation to lesion segmentation."""
        logger.info("Applying transform to lesion segmentation...")
        
        # Load images with ANTs
        lesion_img = ants.image_read(str(lesion_seg_path))
        reference_img = ants.image_read(str(reference_image_path))
        
        # Apply transform
        transformed_lesions = ants.apply_transforms(
            fixed=reference_img,
            moving=lesion_img,
            transformlist=transforms,
            interpolator='nearestNeighbor'  # For label data
        )
        
        # Save transformed image
        ants.image_write(transformed_lesions, str(output_path))
        
        return transformed_lesions
    
    def create_zone_masks(self, atlas_path: Union[str, Path],
                         output_dir: Union[str, Path]) -> Dict[int, Path]:
        """Create individual binary masks for each brain zone."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load atlas
        atlas_nii = nib.load(str(atlas_path))
        atlas_data = atlas_nii.get_fdata()
        
        zone_masks = {}
        
        for zone_id, zone_name in self.BRAIN_ZONES.items():
            # Create binary mask for this zone
            zone_mask = (atlas_data == zone_id).astype(np.uint8)
            
            # Save mask
            mask_nii = nib.Nifti1Image(zone_mask, affine=atlas_nii.affine, 
                                      header=atlas_nii.header)
            mask_path = output_dir / f"zone_{zone_id:02d}_{zone_name}.nii.gz"
            nib.save(mask_nii, str(mask_path))
            
            zone_masks[zone_id] = mask_path
            
        logger.info(f"Created {len(zone_masks)} zone masks")
        return zone_masks
    
    def get_zone_volumes(self, atlas_path: Union[str, Path]) -> Dict[str, float]:
        """Calculate volume of each brain zone."""
        atlas_nii = nib.load(str(atlas_path))
        atlas_data = atlas_nii.get_fdata()
        
        # Calculate voxel volume
        voxel_dims = atlas_nii.header.get_zooms()[:3]
        voxel_volume_mm3 = np.prod(voxel_dims)
        
        zone_volumes = {}
        
        for zone_id, zone_name in self.BRAIN_ZONES.items():
            voxel_count = np.sum(atlas_data == zone_id)
            volume_mm3 = voxel_count * voxel_volume_mm3
            zone_volumes[zone_name] = {
                'voxel_count': int(voxel_count),
                'volume_mm3': float(volume_mm3),
                'volume_ml': float(volume_mm3 / 1000)
            }
            
        return zone_volumes