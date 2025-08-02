import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class TBIVisualizer:
    """Visualization tools for TBI analysis results."""
    
    def __init__(self):
        self.lesion_colors = {
            1: '#FF0000',  # Red - intraparenchymal_hemorrhage
            2: '#FF8000',  # Orange - subdural_hematoma
            3: '#FFFF00',  # Yellow - epidural_hematoma
            4: '#00FF00',  # Green - subarachnoid_hemorrhage
            5: '#00FFFF',  # Cyan - intraventricular_hemorrhage
            6: '#0000FF',  # Blue - diffuse_axonal_injury
            7: '#FF00FF',  # Magenta - contusion
        }
        
        self.lesion_names = {
            1: "IPH",  # Intraparenchymal hemorrhage
            2: "SDH",  # Subdural hematoma
            3: "EDH",  # Epidural hematoma
            4: "SAH",  # Subarachnoid hemorrhage
            5: "IVH",  # Intraventricular hemorrhage
            6: "DAI",  # Diffuse axonal injury
            7: "CTN",  # Contusion
        }
        
        self.zone_names = {
            1: "FL-L",   # Frontal lobe left
            2: "FL-R",   # Frontal lobe right
            3: "PL-L",   # Parietal lobe left
            4: "PL-R",   # Parietal lobe right
            5: "TL-L",   # Temporal lobe left
            6: "TL-R",   # Temporal lobe right
            7: "OL-L",   # Occipital lobe left
            8: "OL-R",   # Occipital lobe right
            9: "CB",     # Cerebellum
            10: "BS",    # Brainstem
        }
    
    def plot_slice_comparison(self, 
                            mri_path: Union[str, Path],
                            segmentation_path: Union[str, Path],
                            atlas_path: Union[str, Path],
                            slice_idx: Optional[int] = None,
                            axis: str = 'axial',
                            output_path: Optional[Union[str, Path]] = None):
        """Create a comparison plot of MRI, segmentation, and atlas."""
        # Load images
        mri_nii = nib.load(str(mri_path))
        seg_nii = nib.load(str(segmentation_path))
        atlas_nii = nib.load(str(atlas_path))
        
        mri_data = mri_nii.get_fdata()
        seg_data = seg_nii.get_fdata()
        atlas_data = atlas_nii.get_fdata()
        
        # Select slice index
        if slice_idx is None:
            if axis == 'axial':
                slice_idx = mri_data.shape[2] // 2
            elif axis == 'sagittal':
                slice_idx = mri_data.shape[0] // 2
            elif axis == 'coronal':
                slice_idx = mri_data.shape[1] // 2
        
        # Get slices
        if axis == 'axial':
            mri_slice = mri_data[:, :, slice_idx]
            seg_slice = seg_data[:, :, slice_idx]
            atlas_slice = atlas_data[:, :, slice_idx]
        elif axis == 'sagittal':
            mri_slice = mri_data[slice_idx, :, :]
            seg_slice = seg_data[slice_idx, :, :]
            atlas_slice = atlas_data[slice_idx, :, :]
        elif axis == 'coronal':
            mri_slice = mri_data[:, slice_idx, :]
            seg_slice = seg_data[:, slice_idx, :]
            atlas_slice = atlas_data[:, slice_idx, :]
        
        # Create figure
        fig = plt.figure(figsize=(15, 5))
        
        # MRI
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(mri_slice.T, cmap='gray', origin='lower')
        ax1.set_title('MRI')
        ax1.axis('off')
        
        # Segmentation overlay
        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(mri_slice.T, cmap='gray', origin='lower', alpha=0.7)
        
        # Create colored overlay for lesions
        overlay = np.zeros((*seg_slice.shape, 3))
        for lesion_id, color in self.lesion_colors.items():
            mask = seg_slice == lesion_id
            if np.any(mask):
                rgb = [int(color[i:i+2], 16)/255 for i in (1, 3, 5)]
                overlay[mask] = rgb
        
        mask = seg_slice == 0
        masked_overlay = np.ma.masked_where(np.stack([mask, mask, mask], axis=-1), overlay)
        ax2.imshow(masked_overlay.transpose(1, 0, 2), origin='lower', alpha=0.6)
        ax2.set_title('Lesion Segmentation')
        ax2.axis('off')
        
        # Atlas
        ax3 = plt.subplot(1, 3, 3)
        ax3.imshow(mri_slice.T, cmap='gray', origin='lower', alpha=0.7)
        atlas_masked = np.ma.masked_where(atlas_slice == 0, atlas_slice)
        ax3.imshow(atlas_masked.T, cmap='tab20', origin='lower', alpha=0.5)
        ax3.set_title('Brain Atlas Zones')
        ax3.axis('off')
        
        # Add legend
        lesion_patches = []
        for lesion_id in np.unique(seg_slice):
            if lesion_id > 0 and lesion_id in self.lesion_names:
                patch = mpatches.Patch(color=self.lesion_colors[int(lesion_id)], 
                                     label=self.lesion_names[int(lesion_id)])
                lesion_patches.append(patch)
        
        if lesion_patches:
            ax2.legend(handles=lesion_patches, loc='upper right', fontsize=8)
        
        plt.suptitle(f'{axis.capitalize()} View - Slice {slice_idx}', fontsize=16)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        return fig
    
    def plot_3d_overview(self,
                        segmentation_path: Union[str, Path],
                        mri_path: Optional[Union[str, Path]] = None,
                        output_path: Optional[Union[str, Path]] = None):
        """Create a 3D overview with multiple slice views."""
        seg_nii = nib.load(str(segmentation_path))
        seg_data = seg_nii.get_fdata()
        
        mri_data = None
        if mri_path:
            mri_nii = nib.load(str(mri_path))
            mri_data = mri_nii.get_fdata()
        
        # Create figure with GridSpec
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Define slice positions
        axial_slices = np.linspace(10, seg_data.shape[2]-10, 4).astype(int)
        sagittal_slices = np.linspace(10, seg_data.shape[0]-10, 4).astype(int)
        coronal_slices = np.linspace(10, seg_data.shape[1]-10, 4).astype(int)
        
        # Plot axial slices
        for i, slice_idx in enumerate(axial_slices):
            ax = fig.add_subplot(gs[0, i])
            if mri_data is not None:
                ax.imshow(mri_data[:, :, slice_idx].T, cmap='gray', origin='lower')
            
            # Overlay segmentation
            seg_slice = seg_data[:, :, slice_idx]
            overlay = np.zeros((*seg_slice.shape, 3))
            for lesion_id, color in self.lesion_colors.items():
                mask = seg_slice == lesion_id
                if np.any(mask):
                    rgb = [int(color[i:i+2], 16)/255 for i in (1, 3, 5)]
                    overlay[mask] = rgb
            
            mask = seg_slice == 0
            masked_overlay = np.ma.masked_where(np.stack([mask, mask, mask], axis=-1), overlay)
            ax.imshow(masked_overlay.transpose(1, 0, 2), origin='lower', alpha=0.6)
            ax.set_title(f'Axial {slice_idx}', fontsize=10)
            ax.axis('off')
        
        # Plot sagittal slices
        for i, slice_idx in enumerate(sagittal_slices):
            ax = fig.add_subplot(gs[1, i])
            if mri_data is not None:
                ax.imshow(mri_data[slice_idx, :, :].T, cmap='gray', origin='lower')
            
            seg_slice = seg_data[slice_idx, :, :]
            overlay = np.zeros((*seg_slice.shape, 3))
            for lesion_id, color in self.lesion_colors.items():
                mask = seg_slice == lesion_id
                if np.any(mask):
                    rgb = [int(color[i:i+2], 16)/255 for i in (1, 3, 5)]
                    overlay[mask] = rgb
            
            mask = seg_slice == 0
            masked_overlay = np.ma.masked_where(np.stack([mask, mask, mask], axis=-1), overlay)
            ax.imshow(masked_overlay.transpose(1, 0, 2), origin='lower', alpha=0.6)
            ax.set_title(f'Sagittal {slice_idx}', fontsize=10)
            ax.axis('off')
        
        # Plot coronal slices
        for i, slice_idx in enumerate(coronal_slices):
            ax = fig.add_subplot(gs[2, i])
            if mri_data is not None:
                ax.imshow(mri_data[:, slice_idx, :].T, cmap='gray', origin='lower')
            
            seg_slice = seg_data[:, slice_idx, :]
            overlay = np.zeros((*seg_slice.shape, 3))
            for lesion_id, color in self.lesion_colors.items():
                mask = seg_slice == lesion_id
                if np.any(mask):
                    rgb = [int(color[i:i+2], 16)/255 for i in (1, 3, 5)]
                    overlay[mask] = rgb
            
            mask = seg_slice == 0
            masked_overlay = np.ma.masked_where(np.stack([mask, mask, mask], axis=-1), overlay)
            ax.imshow(masked_overlay.transpose(1, 0, 2), origin='lower', alpha=0.6)
            ax.set_title(f'Coronal {slice_idx}', fontsize=10)
            ax.axis('off')
        
        # Add legend
        lesion_patches = []
        unique_lesions = np.unique(seg_data)
        for lesion_id in unique_lesions:
            if lesion_id > 0 and lesion_id in self.lesion_names:
                patch = mpatches.Patch(color=self.lesion_colors[int(lesion_id)], 
                                     label=self.lesion_names[int(lesion_id)])
                lesion_patches.append(patch)
        
        if lesion_patches:
            fig.legend(handles=lesion_patches, loc='center right', 
                      bbox_to_anchor=(0.98, 0.5), fontsize=10)
        
        plt.suptitle('TBI Lesion Segmentation - Multi-View', fontsize=16)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        return fig
    
    def plot_quantification_summary(self,
                                  csv_path: Union[str, Path],
                                  output_path: Optional[Union[str, Path]] = None):
        """Create visualization of quantification results."""
        df = pd.read_csv(csv_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Heatmap of lesion-zone overlap
        ax1 = axes[0, 0]
        pivot_data = df.pivot_table(values='Volume_mm3', 
                                   index='Zone', 
                                   columns='LesionType', 
                                   fill_value=0)
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                   ax=ax1, cbar_kws={'label': 'Volume (mm³)'})
        ax1.set_title('Lesion-Zone Volume Heatmap')
        ax1.set_xlabel('Lesion Type')
        ax1.set_ylabel('Brain Zone')
        
        # 2. Bar plot of total lesion volumes
        ax2 = axes[0, 1]
        lesion_volumes = df.groupby('LesionType')['Volume_mm3'].sum().sort_values(ascending=False)
        bars = ax2.bar(range(len(lesion_volumes)), lesion_volumes.values)
        ax2.set_xticks(range(len(lesion_volumes)))
        ax2.set_xticklabels(lesion_volumes.index, rotation=45, ha='right')
        ax2.set_ylabel('Total Volume (mm³)')
        ax2.set_title('Total Volume by Lesion Type')
        
        # Color bars according to lesion type
        for i, lesion_type in enumerate(lesion_volumes.index):
            matching_ids = [k for k, v in self.lesion_names.items() 
                           if lesion_type.lower().startswith(v.lower())]
            if matching_ids and matching_ids[0] in self.lesion_colors:
                bars[i].set_color(self.lesion_colors[matching_ids[0]])
        
        # 3. Pie chart of zone distribution
        ax3 = axes[1, 0]
        zone_volumes = df.groupby('Zone')['Volume_mm3'].sum()
        ax3.pie(zone_volumes.values, labels=zone_volumes.index, autopct='%1.1f%%',
               startangle=90)
        ax3.set_title('Volume Distribution by Brain Zone')
        
        # 4. Stacked bar chart by zone
        ax4 = axes[1, 1]
        pivot_for_stack = df.pivot_table(values='Volume_mm3', 
                                        index='Zone', 
                                        columns='LesionType', 
                                        fill_value=0)
        pivot_for_stack.plot(kind='bar', stacked=True, ax=ax4, legend=True)
        ax4.set_ylabel('Volume (mm³)')
        ax4.set_title('Lesion Composition by Zone')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle('TBI Quantification Summary', fontsize=16)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        return fig
    
    def create_html_report(self,
                         output_dir: Union[str, Path],
                         subject_id: str,
                         include_images: bool = True) -> Path:
        """Generate an HTML report with all results."""
        output_dir = Path(output_dir)
        
        # Load summary JSON
        summary_path = output_dir / f"{subject_id}_pipeline_summary.json"
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Load CSV data
        csv_path = Path(summary['quantification']['overlap_csv'])
        df = pd.read_csv(csv_path) if csv_path.exists() else None
        
        # Create HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TBI Analysis Report - {subject_id}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .summary-box {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            margin: 10px;
        }}
        .lesion-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>TBI Lesion Analysis Report</h1>
        <h2>Subject: {subject_id}</h2>
        <p>Analysis Date: {summary.get('analysis_date', 'N/A')}</p>
        
        <div class="summary-box">
            <h3>Summary Statistics</h3>
            <div class="lesion-stats">
"""
        
        if df is not None and not df.empty:
            total_volume = df['Volume_mm3'].sum()
            num_zones = df['Zone'].nunique()
            num_lesions = df['LesionType'].nunique()
            
            html_content += f"""
                <div class="stat-card">
                    <h4>Total Lesion Volume</h4>
                    <p>{total_volume:.2f} mm³ ({total_volume/1000:.2f} ml)</p>
                </div>
                <div class="stat-card">
                    <h4>Affected Zones</h4>
                    <p>{num_zones} zones</p>
                </div>
                <div class="stat-card">
                    <h4>Lesion Types</h4>
                    <p>{num_lesions} types detected</p>
                </div>
"""
        
        html_content += """
            </div>
        </div>
        
        <h3>Detected Lesions</h3>
"""
        
        # Add lesion statistics table
        if 'lesion_statistics' in summary['segmentation']:
            html_content += """
        <table>
            <tr>
                <th>Lesion Type</th>
                <th>Volume (mm³)</th>
                <th>Volume (ml)</th>
            </tr>
"""
            for lesion_type, stats in summary['segmentation']['lesion_statistics'].items():
                if stats['volume_mm3'] > 0:
                    html_content += f"""
            <tr>
                <td>{lesion_type.replace('_', ' ').title()}</td>
                <td>{stats['volume_mm3']:.2f}</td>
                <td>{stats['volume_ml']:.2f}</td>
            </tr>
"""
            html_content += "</table>"
        
        # Add zone overlap details
        if df is not None and not df.empty:
            html_content += """
        <h3>Lesion-Zone Overlap Details</h3>
        <table>
            <tr>
                <th>Zone</th>
                <th>Lesion Type</th>
                <th>Volume (mm³)</th>
                <th>Volume (ml)</th>
            </tr>
"""
            for _, row in df.iterrows():
                html_content += f"""
            <tr>
                <td>{row['Zone']}</td>
                <td>{row['LesionType'].replace('_', ' ').title()}</td>
                <td>{row['Volume_mm3']:.2f}</td>
                <td>{row['Volume_ml']:.3f}</td>
            </tr>
"""
            html_content += "</table>"
        
        # Add images if requested
        if include_images:
            html_content += """
        <h3>Visualization</h3>
        <div class="image-container">
"""
            # Check for generated visualization images
            viz_dir = output_dir / "visualization"
            if viz_dir.exists():
                for img_file in viz_dir.glob("*.png"):
                    img_name = img_file.stem.replace('_', ' ').title()
                    html_content += f"""
            <div>
                <h4>{img_name}</h4>
                <img src="visualization/{img_file.name}" alt="{img_name}">
            </div>
"""
        
        html_content += """
        </div>
        
        <h3>Output Files</h3>
        <ul>
            <li>Preprocessed MRI: <code>{}</code></li>
            <li>Lesion Segmentation: <code>{}</code></li>
            <li>Quantification CSV: <code>{}</code></li>
            <li>Detailed Report: <code>{}</code></li>
        </ul>
    </div>
</body>
</html>
""".format(
            summary['preprocessing'].get('preprocessed_mri', 'N/A'),
            summary['segmentation'].get('segmentation_path', 'N/A'),
            summary['quantification'].get('overlap_csv', 'N/A'),
            summary['quantification'].get('report', 'N/A')
        )
        
        # Save HTML report
        report_path = output_dir / f"{subject_id}_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {report_path}")
        return report_path