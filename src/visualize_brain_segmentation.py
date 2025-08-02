#!/usr/bin/env python3
"""
Visualization script for brain segmentation results.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
import logging
import argparse
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class BrainSegmentationVisualizer:
    """Visualization tools for brain segmentation results."""
    
    def __init__(self):
        # Brain structure colors (based on common neuroimaging conventions)
        self.structure_colors = {
            1: '#FF6B6B',  # Cortical gray matter - Red
            2: '#4ECDC4',  # Basal ganglia - Teal
            3: '#45B7D1',  # White matter - Blue
            4: '#FFA07A',  # White matter lesions - Light salmon
            5: '#98D8C8',  # CSF extracerebral - Light green
            6: '#F7DC6F',  # Ventricles - Yellow
            7: '#BB8FCE',  # Cerebellum - Purple
            8: '#F8C471',  # Brain stem - Orange
            9: '#EC7063',  # Infarction - Dark red
            10: '#85C1E9', # Other - Light blue
        }
        
        self.structure_names = {
            1: "Cortical GM",
            2: "Basal Ganglia", 
            3: "White Matter",
            4: "WM Lesions",
            5: "CSF",
            6: "Ventricles",
            7: "Cerebellum",
            8: "Brain Stem",
            9: "Infarction",
            10: "Other"
        }
    
    def plot_sample_comparison(self, sample_paths: List[Dict], output_path: Optional[Path] = None):
        """Create comparison plot of all samples."""
        n_samples = len(sample_paths)
        fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
        
        if n_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i, sample_info in enumerate(sample_paths):
            # Load images
            mri_nii = nib.load(sample_info['mri_path'])
            seg_nii = nib.load(sample_info['seg_path'])
            
            mri_data = mri_nii.get_fdata()
            seg_data = seg_nii.get_fdata()
            
            # Get middle slice
            mid_slice = mri_data.shape[2] // 2
            mri_slice = mri_data[:, :, mid_slice]
            seg_slice = seg_data[:, :, mid_slice]
            
            # Plot original MRI
            axes[0, i].imshow(mri_slice.T, cmap='gray', origin='lower')
            axes[0, i].set_title(f'{sample_info["name"]} - Original', fontsize=12)
            axes[0, i].axis('off')
            
            # Plot segmentation overlay
            axes[1, i].imshow(mri_slice.T, cmap='gray', origin='lower', alpha=0.7)
            
            # Create colored overlay
            overlay = np.zeros((*seg_slice.shape, 3))
            for struct_id, color in self.structure_colors.items():
                mask = seg_slice == struct_id
                if np.any(mask):
                    rgb = [int(color[i:i+2], 16)/255 for i in (1, 3, 5)]
                    overlay[mask] = rgb
            
            mask = seg_slice == 0
            masked_overlay = np.ma.masked_where(np.stack([mask, mask, mask], axis=-1), overlay)
            axes[1, i].imshow(masked_overlay.transpose(1, 0, 2), origin='lower', alpha=0.6)
            axes[1, i].set_title(f'{sample_info["name"]} - Segmented', fontsize=12)
            axes[1, i].axis('off')
        
        # Add legend
        legend_patches = []
        for struct_id in np.unique(seg_data):
            if struct_id > 0 and struct_id in self.structure_names:
                patch = mpatches.Patch(
                    color=self.structure_colors.get(int(struct_id), '#888888'),
                    label=self.structure_names.get(int(struct_id), f'Structure {int(struct_id)}')
                )
                legend_patches.append(patch)
        
        if legend_patches:
            fig.legend(handles=legend_patches, loc='center right', 
                      bbox_to_anchor=(0.98, 0.5), fontsize=10)
        
        plt.suptitle('fTRACTS Brain Segmentation Results', fontsize=16, y=0.95)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Sample comparison saved to: {output_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_volume_analysis(self, stats_files: List[Dict], output_path: Optional[Path] = None):
        """Create volume analysis plots."""
        # Load statistics
        all_stats = []
        for sample_info in stats_files:
            with open(sample_info['stats_path'], 'r') as f:
                stats = json.load(f)
            
            for structure, data in stats['statistics'].items():
                all_stats.append({
                    'Sample': sample_info['name'],
                    'Structure': structure,
                    'Volume_ml': data['volume_ml'],
                    'Volume_mm3': data['volume_mm3'],
                    'Percentage': data.get('percentage', 0)
                })
        
        df = pd.DataFrame(all_stats)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Stacked bar chart of volumes by sample
        pivot_data = df.pivot_table(values='Volume_ml', index='Sample', columns='Structure', fill_value=0)
        pivot_data.plot(kind='bar', stacked=True, ax=axes[0, 0], 
                       colormap='tab20', legend=False)
        axes[0, 0].set_title('Brain Structure Volumes by Sample')
        axes[0, 0].set_ylabel('Volume (ml)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. White matter lesions comparison
        wm_lesions = df[df['Structure'] == 'White matter lesions']
        if not wm_lesions.empty:
            bars = axes[0, 1].bar(wm_lesions['Sample'], wm_lesions['Volume_ml'], 
                                 color='#FFA07A', alpha=0.7)
            axes[0, 1].set_title('White Matter Lesions (Pathology Indicator)')
            axes[0, 1].set_ylabel('Volume (ml)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, wm_lesions['Volume_ml']):
                if value > 0:
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                   f'{value:.1f}', ha='center', va='bottom')
        
        # 3. Heatmap of structure percentages
        percentage_pivot = df.pivot_table(values='Percentage', index='Structure', columns='Sample', fill_value=0)
        sns.heatmap(percentage_pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
                   ax=axes[1, 0], cbar_kws={'label': 'Percentage (%)'})
        axes[1, 0].set_title('Structure Distribution by Sample (%)')
        
        # 4. Total brain volume comparison
        total_volumes = df.groupby('Sample')['Volume_ml'].sum().reset_index()
        bars = axes[1, 1].bar(total_volumes['Sample'], total_volumes['Volume_ml'], 
                             color='steelblue', alpha=0.7)
        axes[1, 1].set_title('Total Brain Volume by Sample')
        axes[1, 1].set_ylabel('Total Volume (ml)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, total_volumes['Volume_ml']):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                           f'{value:.0f}', ha='center', va='bottom')
        
        plt.suptitle('fTRACTS Brain Segmentation - Volume Analysis', fontsize=16)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Volume analysis saved to: {output_path}")
        else:
            plt.show()
        
        return fig, df
    
    def plot_3d_overview(self, sample_info: Dict, output_path: Optional[Path] = None):
        """Create 3D overview for a single sample."""
        # Load images
        mri_nii = nib.load(sample_info['mri_path'])
        seg_nii = nib.load(sample_info['seg_path'])
        
        mri_data = mri_nii.get_fdata()
        seg_data = seg_nii.get_fdata()
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Define slice positions for each axis
        axial_slices = np.linspace(5, seg_data.shape[2]-5, 4).astype(int)
        sagittal_slices = np.linspace(10, seg_data.shape[0]-10, 4).astype(int)
        coronal_slices = np.linspace(10, seg_data.shape[1]-10, 4).astype(int)
        
        # Plot axial slices
        for i, slice_idx in enumerate(axial_slices):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(mri_data[:, :, slice_idx].T, cmap='gray', origin='lower')
            
            # Overlay segmentation
            seg_slice = seg_data[:, :, slice_idx]
            overlay = np.zeros((*seg_slice.shape, 3))
            for struct_id, color in self.structure_colors.items():
                mask = seg_slice == struct_id
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
            ax.imshow(mri_data[slice_idx, :, :].T, cmap='gray', origin='lower')
            
            seg_slice = seg_data[slice_idx, :, :]
            overlay = np.zeros((*seg_slice.shape, 3))
            for struct_id, color in self.structure_colors.items():
                mask = seg_slice == struct_id
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
            ax.imshow(mri_data[:, slice_idx, :].T, cmap='gray', origin='lower')
            
            seg_slice = seg_data[:, slice_idx, :]
            overlay = np.zeros((*seg_slice.shape, 3))
            for struct_id, color in self.structure_colors.items():
                mask = seg_slice == struct_id
                if np.any(mask):
                    rgb = [int(color[i:i+2], 16)/255 for i in (1, 3, 5)]
                    overlay[mask] = rgb
            
            mask = seg_slice == 0
            masked_overlay = np.ma.masked_where(np.stack([mask, mask, mask], axis=-1), overlay)
            ax.imshow(masked_overlay.transpose(1, 0, 2), origin='lower', alpha=0.6)
            ax.set_title(f'Coronal {slice_idx}', fontsize=10)
            ax.axis('off')
        
        # Add legend
        legend_patches = []
        unique_labels = np.unique(seg_data)
        for struct_id in unique_labels:
            if struct_id > 0 and struct_id in self.structure_names:
                patch = mpatches.Patch(
                    color=self.structure_colors.get(int(struct_id), '#888888'),
                    label=self.structure_names.get(int(struct_id), f'Structure {int(struct_id)}')
                )
                legend_patches.append(patch)
        
        if legend_patches:
            fig.legend(handles=legend_patches, loc='center right', 
                      bbox_to_anchor=(0.98, 0.5), fontsize=10)
        
        plt.suptitle(f'{sample_info["name"]} - Multi-View Brain Segmentation', fontsize=16)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"3D overview saved to: {output_path}")
        else:
            plt.show()
        
        return fig


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize brain segmentation results')
    parser.add_argument('--output-dir', type=Path, default='models/brain_segmentation/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--samples-dir', type=Path, default='models/brain_segmentation',
                       help='Directory containing segmentation results')
    parser.add_argument('--show-plots', action='store_true',
                       help='Display plots interactively')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define sample information
    samples = [
        {
            'name': 'Sample 1',
            'mri_path': 'data/input/sample1/fTRACTS_07302024-0007-00001-000001-01.nii',
            'seg_path': 'models/brain_segmentation/fTRACTS_sample1_segmentation.nii.gz',
            'stats_path': 'models/brain_segmentation/fTRACTS_sample1_stats.json'
        },
        {
            'name': 'Sample 2', 
            'mri_path': 'data/input/sample2/fTRACTS_07302024-0008-00001-000001-01.nii',
            'seg_path': 'models/brain_segmentation/fTRACTS_sample2_segmentation.nii.gz',
            'stats_path': 'models/brain_segmentation/fTRACTS_sample2_stats.json'
        },
        {
            'name': 'Sample 3',
            'mri_path': 'data/input/sample3/fTRACTS_07302024-0009-00001-000001-01.nii', 
            'seg_path': 'models/brain_segmentation/fTRACTS_sample3_segmentation.nii.gz',
            'stats_path': 'models/brain_segmentation/fTRACTS_sample3_stats.json'
        },
        {
            'name': 'Sample 4',
            'mri_path': 'data/input/sample4/fTRACTS_07302024-0010-00001-000001-01.nii',
            'seg_path': 'models/brain_segmentation/fTRACTS_sample4_segmentation.nii.gz', 
            'stats_path': 'models/brain_segmentation/fTRACTS_sample4_stats.json'
        }
    ]
    
    # Initialize visualizer
    visualizer = BrainSegmentationVisualizer()
    
    # Create visualizations
    logger.info("Creating sample comparison visualization...")
    fig1 = visualizer.plot_sample_comparison(
        samples, 
        args.output_dir / 'fTRACTS_sample_comparison.png'
    )
    
    logger.info("Creating volume analysis visualization...")
    fig2, df = visualizer.plot_volume_analysis(
        samples,
        args.output_dir / 'fTRACTS_volume_analysis.png'
    )
    
    # Create 3D overviews for samples with significant pathology
    for sample in [samples[2], samples[3]]:  # Samples 3 and 4
        logger.info(f"Creating 3D overview for {sample['name']}...")
        fig3 = visualizer.plot_3d_overview(
            sample,
            args.output_dir / f'fTRACTS_{sample["name"].lower().replace(" ", "_")}_3d_overview.png'
        )
    
    # Save summary statistics
    summary_path = args.output_dir / 'fTRACTS_summary_statistics.csv'
    df.to_csv(summary_path, index=False)
    logger.info(f"Summary statistics saved to: {summary_path}")
    
    if args.show_plots:
        plt.show()
    
    print(f"\nVisualization complete! Files saved to: {args.output_dir}")
    print(f"Generated {len(list(args.output_dir.glob('*.png')))} visualization files")


if __name__ == "__main__":
    main()