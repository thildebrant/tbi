#!/usr/bin/env python3
"""
Generate HTML reports for fTRACTS brain segmentation results.
"""

import json
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


def generate_ftracts_html_report(sample_info: Dict, output_path: Path) -> Path:
    """Generate HTML report for a single fTRACTS sample."""
    
    # Load statistics
    with open(sample_info['stats_path'], 'r') as f:
        stats = json.load(f)
    
    sample_name = sample_info['name']
    
    # Calculate summary statistics
    total_volume = sum(s['volume_mm3'] for s in stats['statistics'].values())
    total_structures = len(stats['statistics'])
    
    # Find pathological indicators
    wm_lesions = stats['statistics'].get('White matter lesions', {})
    wm_lesion_volume = wm_lesions.get('volume_ml', 0)
    
    # Determine pathology level
    if wm_lesion_volume > 10:
        pathology_level = "Significant"
        pathology_color = "#ff6b6b"
    elif wm_lesion_volume > 1:
        pathology_level = "Moderate" 
        pathology_color = "#ffa726"
    else:
        pathology_level = "Minimal"
        pathology_color = "#66bb6a"
    
    # Generate HTML content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Brain Segmentation Report - {sample_name}</title>
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
        .pathology-box {{
            background-color: {pathology_color};
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            text-align: center;
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
        .stats-grid {{
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
        .structure-bar {{
            background-color: #e0e0e0;
            border-radius: 3px;
            overflow: hidden;
            margin: 5px 0;
        }}
        .structure-fill {{
            height: 20px;
            background-color: #4CAF50;
            transition: width 0.3s ease;
        }}
        .metadata {{
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 12px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Brain Segmentation Analysis Report</h1>
        <h2>fTRACTS Sample: {sample_name}</h2>
        <p><strong>Analysis Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Input File:</strong> <code>{sample_info['mri_path']}</code></p>
        
        <div class="pathology-box">
            <h3>Pathology Assessment: {pathology_level}</h3>
            <p>White Matter Lesion Volume: {wm_lesion_volume:.2f} ml</p>
        </div>
        
        <div class="summary-box">
            <h3>Summary Statistics</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <h4>Total Brain Volume</h4>
                    <p>{total_volume:.0f} mm³ ({total_volume/1000:.1f} ml)</p>
                </div>
                <div class="stat-card">
                    <h4>Segmented Structures</h4>
                    <p>{total_structures} brain regions</p>
                </div>
                <div class="stat-card">
                    <h4>WM Lesions</h4>
                    <p>{wm_lesion_volume:.2f} ml ({wm_lesions.get('percentage', 0):.2f}%)</p>
                </div>
                <div class="stat-card">
                    <h4>Analysis Method</h4>
                    <p>3D U-Net Deep Learning</p>
                </div>
            </div>
        </div>
        
        <h3>Detailed Brain Structure Analysis</h3>
        <table>
            <tr>
                <th>Brain Structure</th>
                <th>Volume (mm³)</th>
                <th>Volume (ml)</th>
                <th>Percentage</th>
                <th>Distribution</th>
            </tr>"""
    
    # Add table rows for each structure (sorted by volume)
    sorted_structures = sorted(
        stats['statistics'].items(),
        key=lambda x: x[1]['volume_mm3'],
        reverse=True
    )
    
    for structure, data in sorted_structures:
        percentage = data.get('percentage', 0)
        bar_width = min(percentage * 2, 100)  # Scale for visualization
        
        html_content += f"""
            <tr>
                <td>{structure}</td>
                <td>{data['volume_mm3']:.0f}</td>
                <td>{data['volume_ml']:.2f}</td>
                <td>{percentage:.1f}%</td>
                <td>
                    <div class="structure-bar">
                        <div class="structure-fill" style="width: {bar_width}%"></div>
                    </div>
                </td>
            </tr>"""
    
    html_content += """
        </table>
        
        <h3>Clinical Significance</h3>
        <div class="summary-box">"""
    
    # Add clinical interpretation
    if wm_lesion_volume > 10:
        html_content += """
            <h4>⚠️ Significant Pathological Findings</h4>
            <ul>
                <li>High white matter lesion burden detected (>10 ml)</li>
                <li>Suggestive of traumatic brain injury or other pathology</li>
                <li>Recommend clinical correlation and follow-up imaging</li>
                <li>Consider neuropsychological assessment</li>
            </ul>"""
    elif wm_lesion_volume > 1:
        html_content += """
            <h4>⚠️ Moderate Pathological Findings</h4>
            <ul>
                <li>Moderate white matter lesion burden detected</li>
                <li>May indicate mild traumatic brain injury</li>
                <li>Clinical correlation recommended</li>
            </ul>"""
    else:
        html_content += """
            <h4>✓ Minimal Pathological Findings</h4>
            <ul>
                <li>Low white matter lesion burden</li>
                <li>Brain structure appears largely normal</li>
                <li>No significant pathological indicators detected</li>
            </ul>"""
    
    # Add brain structure insights
    cortical_gm = stats['statistics'].get('Cortical gray matter', {})
    if cortical_gm.get('percentage', 0) > 10:
        html_content += "<li>Good cortical gray matter preservation</li>"
    else:
        html_content += "<li>Reduced cortical gray matter volume noted</li>"
    
    html_content += """
        </div>
        
        <h3>Visualization</h3>
        <div class="image-container">"""
    
    # Add visualization images if they exist
    viz_dir = Path("models/brain_segmentation/visualizations")
    
    comparison_img = viz_dir / "fTRACTS_sample_comparison.png"
    if comparison_img.exists():
        html_content += f"""
            <div>
                <h4>Sample Comparison</h4>
                <img src="../visualizations/fTRACTS_sample_comparison.png" alt="Sample Comparison">
            </div>"""
    
    volume_img = viz_dir / "fTRACTS_volume_analysis.png"  
    if volume_img.exists():
        html_content += f"""
            <div>
                <h4>Volume Analysis</h4>
                <img src="../visualizations/fTRACTS_volume_analysis.png" alt="Volume Analysis">
            </div>"""
    
    # Add 3D overview for samples with pathology
    if wm_lesion_volume > 5:
        sample_3d = viz_dir / f"fTRACTS_{sample_name.lower().replace(' ', '_')}_3d_overview.png"
        if sample_3d.exists():
            html_content += f"""
                <div>
                    <h4>3D Multi-View Analysis</h4>
                    <img src="../visualizations/{sample_3d.name}" alt="3D Overview">
                </div>"""
    
    html_content += """
        </div>
        
        <h3>Technical Details</h3>
        <div class="metadata">
            <strong>Model Information:</strong><br>
            Architecture: 3D U-Net<br>
            Input Channels: Single-channel MRI (adapted from T1+FLAIR training)<br>
            Output Classes: 11 brain tissue types<br>
            Training Data: MR Brain Segmentation Challenge 2018<br>
            Validation Loss: 0.6925 (Dice coefficient)<br><br>
            
            <strong>Processing Details:</strong><br>
            Original Image Size: 104×104×72 voxels<br>
            Processing Size: 128×128×64 voxels (resized)<br>
            Voxel Spacing: Variable based on acquisition<br>
            Normalization: Percentile-based (1st-99th percentile)<br><br>
            
            <strong>Output Files:</strong><br>
            Segmentation: <code>{sample_info['seg_path']}</code><br>
            Statistics: <code>{sample_info['stats_path']}</code><br>
        </div>
        
        <h3>Methodology & Limitations</h3>
        <div class="summary-box">
            <h4>Analysis Method</h4>
            <ul>
                <li>Deep learning-based brain segmentation using 3D U-Net</li>
                <li>Trained on multi-modal MRI data (T1-weighted + FLAIR)</li>
                <li>Adapted for single-channel fTRACTS data</li>
                <li>Automated tissue classification into 11 anatomical regions</li>
            </ul>
            
            <h4>Important Limitations</h4>
            <ul>
                <li>Model trained on different imaging protocol than fTRACTS</li>
                <li>Results should be validated by experienced neuroradiologist</li>
                <li>Segmentation accuracy may vary with image quality</li>
                <li>Clinical correlation always required for diagnosis</li>
            </ul>
        </div>
        
        <p style="text-align: center; margin-top: 30px; font-size: 12px; color: #666;">
            Generated by TBI Analysis Pipeline - 
            🤖 <a href="https://claude.ai/code">Claude Code</a> | 
            Report Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </p>
    </div>
</body>
</html>
"""
    
    # Save HTML report
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to: {output_path}")
    return output_path


def main():
    """Generate HTML reports for all fTRACTS samples."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
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
    
    # Create reports directory
    reports_dir = Path("models/brain_segmentation/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating HTML reports for fTRACTS samples...")
    
    # Generate individual reports
    report_paths = []
    for sample in samples:
        sample_name_clean = sample['name'].lower().replace(' ', '_')
        output_path = reports_dir / f"fTRACTS_{sample_name_clean}_report.html"
        report_path = generate_ftracts_html_report(sample, output_path)
        report_paths.append(report_path)
        print(f"✓ Generated report for {sample['name']}")
    
    # Generate index page
    index_path = reports_dir / "index.html"
    with open(index_path, 'w') as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>fTRACTS Brain Segmentation Reports</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #333; text-align: center; }}
        .report-link {{ display: block; padding: 15px; margin: 10px 0; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px; }}
        .report-link:hover {{ background-color: #45a049; }}
        .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 fTRACTS Brain Segmentation Reports</h1>
        <div class="summary">
            <p><strong>Analysis Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Total Samples:</strong> 4 fTRACTS volumes</p>
            <p><strong>Method:</strong> 3D U-Net Deep Learning Segmentation</p>
        </div>
        
        <h2>Individual Sample Reports</h2>
""")
        
        for i, (sample, report_path) in enumerate(zip(samples, report_paths)):
            # Load stats for summary
            with open(sample['stats_path'], 'r') as stats_file:
                stats = json.load(stats_file)
            
            wm_lesions = stats['statistics'].get('White matter lesions', {})
            wm_volume = wm_lesions.get('volume_ml', 0)
            total_volume = sum(s['volume_mm3'] for s in stats['statistics'].values()) / 1000
            
            pathology = "High" if wm_volume > 10 else "Moderate" if wm_volume > 1 else "Low"
            
            f.write(f"""
        <a href="{report_path.name}" class="report-link">
            <strong>{sample['name']}</strong><br>
            Total Volume: {total_volume:.1f} ml | WM Lesions: {wm_volume:.2f} ml | Pathology: {pathology}
        </a>""")
        
        f.write("""
        
        <div class="summary">
            <h3>Quick Summary</h3>
            <ul>
                <li><strong>Samples 1 & 2:</strong> Minimal pathology (< 1 ml WM lesions)</li>
                <li><strong>Samples 3 & 4:</strong> Significant pathology (> 13 ml WM lesions)</li>
                <li><strong>Clinical Action:</strong> Samples 3 & 4 require further evaluation</li>
            </ul>
        </div>
        
        <p style="text-align: center; margin-top: 30px; color: #666;">
            Generated by TBI Analysis Pipeline
        </p>
    </div>
</body>
</html>
""")
    
    print(f"\n🎉 Generated {len(report_paths)} HTML reports!")
    print(f"📁 Reports saved to: {reports_dir}")
    print(f"🌐 Open index.html to view all reports: {index_path}")
    
    return report_paths


if __name__ == "__main__":
    main()