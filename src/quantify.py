import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class LesionQuantifier:
    """Quantify lesion overlap with brain atlas zones."""
    
    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(float))
        
    def calculate_overlap(self, lesion_seg_path: Union[str, Path],
                         atlas_path: Union[str, Path],
                         lesion_types: Dict[int, str],
                         zone_names: Dict[int, str]) -> pd.DataFrame:
        """Calculate volume overlap between lesions and brain zones."""
        logger.info("Calculating lesion-zone overlap...")
        
        # Load images
        lesion_nii = nib.load(str(lesion_seg_path))
        atlas_nii = nib.load(str(atlas_path))
        
        lesion_data = lesion_nii.get_fdata()
        atlas_data = atlas_nii.get_fdata()
        
        # Verify same dimensions
        if lesion_data.shape != atlas_data.shape:
            raise ValueError(f"Dimension mismatch: lesion {lesion_data.shape} vs atlas {atlas_data.shape}")
        
        # Calculate voxel volume
        voxel_dims = lesion_nii.header.get_zooms()[:3]
        voxel_volume_mm3 = np.prod(voxel_dims)
        
        # Calculate overlap for each lesion type and zone
        overlap_results = []
        
        for lesion_id, lesion_name in lesion_types.items():
            if lesion_id == 0:  # Skip background
                continue
                
            lesion_mask = lesion_data == lesion_id
            
            for zone_id, zone_name in zone_names.items():
                zone_mask = atlas_data == zone_id
                
                # Calculate intersection
                overlap_mask = lesion_mask & zone_mask
                overlap_voxels = np.sum(overlap_mask)
                overlap_volume_mm3 = overlap_voxels * voxel_volume_mm3
                
                if overlap_volume_mm3 > 0:
                    overlap_results.append({
                        'Zone': zone_name,
                        'Zone_ID': zone_id,
                        'LesionType': lesion_name,
                        'Lesion_ID': lesion_id,
                        'Volume_mm3': overlap_volume_mm3,
                        'Volume_ml': overlap_volume_mm3 / 1000,
                        'Voxel_Count': int(overlap_voxels)
                    })
                    
                    logger.info(f"{lesion_name} in {zone_name}: {overlap_volume_mm3:.2f} mm³")
        
        # Create DataFrame
        df = pd.DataFrame(overlap_results)
        
        # Sort by zone and lesion type
        if not df.empty:
            df = df.sort_values(['Zone_ID', 'Lesion_ID'])
        
        return df
    
    def create_summary_statistics(self, overlap_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics from overlap data."""
        if overlap_df.empty:
            return pd.DataFrame()
            
        # Group by zone
        zone_summary = overlap_df.groupby('Zone').agg({
            'Volume_mm3': ['sum', 'count'],
            'Volume_ml': 'sum'
        }).round(3)
        
        zone_summary.columns = ['Total_Volume_mm3', 'Lesion_Count', 'Total_Volume_ml']
        zone_summary = zone_summary.reset_index()
        
        # Group by lesion type
        lesion_summary = overlap_df.groupby('LesionType').agg({
            'Volume_mm3': ['sum', 'count'],
            'Volume_ml': 'sum'
        }).round(3)
        
        lesion_summary.columns = ['Total_Volume_mm3', 'Occurrence_Count', 'Total_Volume_ml']
        lesion_summary = lesion_summary.reset_index()
        
        return zone_summary, lesion_summary
    
    def create_heatmap_data(self, overlap_df: pd.DataFrame) -> pd.DataFrame:
        """Create a heatmap-ready pivot table of lesion-zone overlap."""
        if overlap_df.empty:
            return pd.DataFrame()
            
        # Pivot table with zones as rows and lesion types as columns
        heatmap_data = overlap_df.pivot_table(
            index='Zone',
            columns='LesionType',
            values='Volume_mm3',
            fill_value=0
        ).round(2)
        
        return heatmap_data
    
    def export_results(self, overlap_df: pd.DataFrame,
                      output_dir: Union[str, Path],
                      subject_id: str = "subject") -> Dict[str, Path]:
        """Export all quantification results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        # Save main overlap results
        overlap_path = output_dir / f"{subject_id}_lesion_zone_overlap.csv"
        overlap_df.to_csv(overlap_path, index=False)
        output_files['overlap'] = overlap_path
        logger.info(f"Saved overlap results to {overlap_path}")
        
        # Save summary statistics
        zone_summary, lesion_summary = self.create_summary_statistics(overlap_df)
        
        if not zone_summary.empty:
            zone_summary_path = output_dir / f"{subject_id}_zone_summary.csv"
            zone_summary.to_csv(zone_summary_path, index=False)
            output_files['zone_summary'] = zone_summary_path
            
        if not lesion_summary.empty:
            lesion_summary_path = output_dir / f"{subject_id}_lesion_summary.csv"
            lesion_summary.to_csv(lesion_summary_path, index=False)
            output_files['lesion_summary'] = lesion_summary_path
        
        # Save heatmap data
        heatmap_data = self.create_heatmap_data(overlap_df)
        if not heatmap_data.empty:
            heatmap_path = output_dir / f"{subject_id}_heatmap_data.csv"
            heatmap_data.to_csv(heatmap_path)
            output_files['heatmap'] = heatmap_path
        
        # Create a detailed report
        report_path = output_dir / f"{subject_id}_report.txt"
        self.generate_text_report(overlap_df, zone_summary, lesion_summary, report_path)
        output_files['report'] = report_path
        
        return output_files
    
    def generate_text_report(self, overlap_df: pd.DataFrame,
                           zone_summary: pd.DataFrame,
                           lesion_summary: pd.DataFrame,
                           output_path: Union[str, Path]):
        """Generate a human-readable text report."""
        with open(output_path, 'w') as f:
            f.write("TBI LESION QUANTIFICATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            if not overlap_df.empty:
                total_volume = overlap_df['Volume_mm3'].sum()
                f.write(f"Total Lesion Volume: {total_volume:.2f} mm³ ({total_volume/1000:.2f} ml)\n")
                f.write(f"Number of Affected Zones: {overlap_df['Zone'].nunique()}\n")
                f.write(f"Number of Lesion Types Found: {overlap_df['LesionType'].nunique()}\n\n")
            else:
                f.write("No lesions detected.\n\n")
                return
            
            # Zone-wise summary
            f.write("AFFECTED BRAIN ZONES\n")
            f.write("-" * 30 + "\n")
            if not zone_summary.empty:
                for _, row in zone_summary.iterrows():
                    f.write(f"\n{row['Zone']}:\n")
                    f.write(f"  Total Volume: {row['Total_Volume_mm3']:.2f} mm³\n")
                    f.write(f"  Number of Lesions: {int(row['Lesion_Count'])}\n")
            
            # Lesion type summary
            f.write("\n\nLESION TYPES DETECTED\n")
            f.write("-" * 30 + "\n")
            if not lesion_summary.empty:
                for _, row in lesion_summary.iterrows():
                    f.write(f"\n{row['LesionType']}:\n")
                    f.write(f"  Total Volume: {row['Total_Volume_mm3']:.2f} mm³\n")
                    f.write(f"  Found in {int(row['Occurrence_Count'])} zone(s)\n")
            
            # Detailed breakdown
            f.write("\n\nDETAILED LESION-ZONE BREAKDOWN\n")
            f.write("-" * 30 + "\n")
            for _, row in overlap_df.iterrows():
                f.write(f"\n{row['LesionType']} in {row['Zone']}:\n")
                f.write(f"  Volume: {row['Volume_mm3']:.2f} mm³\n")
        
        logger.info(f"Generated text report: {output_path}")
    
    def create_visualization_data(self, overlap_df: pd.DataFrame,
                                lesion_seg_path: Union[str, Path],
                                atlas_path: Union[str, Path],
                                output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Create data for visualization (overlay masks, etc.)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load images
        lesion_nii = nib.load(str(lesion_seg_path))
        atlas_nii = nib.load(str(atlas_path))
        
        lesion_data = lesion_nii.get_fdata()
        atlas_data = atlas_nii.get_fdata()
        
        # Create combined overlay
        overlay = np.zeros_like(lesion_data)
        
        # Assign unique values for visualization
        # Lesions: 1-7, Atlas zones: 11-20
        overlay[lesion_data > 0] = lesion_data[lesion_data > 0]
        overlay[atlas_data > 0] = atlas_data[atlas_data > 0] + 10
        
        # Save overlay
        overlay_nii = nib.Nifti1Image(overlay.astype(np.uint8), 
                                     affine=lesion_nii.affine,
                                     header=lesion_nii.header)
        overlay_path = output_dir / "lesion_atlas_overlay.nii.gz"
        nib.save(overlay_nii, str(overlay_path))
        
        return {'overlay': overlay_path}