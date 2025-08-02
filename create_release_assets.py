#!/usr/bin/env python3
"""
Script to create release assets for the TBI Lesion Analysis Pipeline.
This script helps organize and prepare model files and sample data for releases.
"""

import os
import shutil
import zipfile
from pathlib import Path
import json

def create_model_release_package():
    """Create a release package for model files."""
    
    print("📦 Creating model release package...")
    
    # Create temporary directory for model files
    model_dir = Path("release_models")
    model_dir.mkdir(exist_ok=True)
    
    # Copy model files (if they exist)
    source_models = Path("models/brain_segmentation")
    if source_models.exists():
        for model_file in source_models.glob("*.pth"):
            if model_file.exists():
                shutil.copy2(model_file, model_dir / model_file.name)
                print(f"  ✅ Added: {model_file.name}")
    
    # Create model info file
    model_info = {
        "version": "1.0.0",
        "description": "TBI Lesion Analysis Pipeline - Pre-trained Models",
        "models": {
            "best_model.pth": {
                "description": "Best performing model from training",
                "architecture": "3D U-Net",
                "input_channels": 1,
                "output_channels": 8,
                "lesion_types": [
                    "background",
                    "intraparenchymal_hemorrhage",
                    "subdural_hematoma", 
                    "epidural_hematoma",
                    "subarachnoid_hemorrhage",
                    "intraventricular_hemorrhage",
                    "diffuse_axonal_injury",
                    "contusion"
                ]
            },
            "final_model.pth": {
                "description": "Final model from training",
                "architecture": "3D U-Net",
                "input_channels": 1,
                "output_channels": 8
            }
        },
        "usage_instructions": {
            "1": "Download the model files",
            "2": "Place them in the models/brain_segmentation/ directory",
            "3": "Use the --model flag with run_pipeline.py to specify the model path",
            "4": "Example: python run_pipeline.py input.nii.gz output/ --model models/brain_segmentation/best_model.pth"
        },
        "requirements": {
            "python": ">=3.8",
            "pytorch": ">=2.0.0",
            "monai": ">=1.3.0"
        }
    }
    
    with open(model_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Create README for models
    model_readme = """# TBI Lesion Analysis Pipeline - Model Files

This package contains pre-trained models for the TBI Lesion Analysis Pipeline.

## Models Included

- **best_model.pth**: Best performing model from training
- **final_model.pth**: Final model from training

## Installation

1. Extract this package to your project directory
2. Place the .pth files in the `models/brain_segmentation/` directory
3. Use the models with the pipeline:

```bash
python run_pipeline.py input.nii.gz output/ --model models/brain_segmentation/best_model.pth
```

## Model Architecture

- **Architecture**: 3D U-Net
- **Input Channels**: 1 (T1-weighted MRI)
- **Output Channels**: 8 (7 lesion types + background)

## Lesion Types Detected

1. Intraparenchymal hemorrhage
2. Subdural hematoma
3. Epidural hematoma
4. Subarachnoid hemorrhage
5. Intraventricular hemorrhage
6. Diffuse axonal injury
7. Contusion

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- MONAI >= 1.3.0

## Medical Disclaimer

These models are trained for research purposes. Clinical use requires additional validation and regulatory approval.
"""
    
    with open(model_dir / "README.md", "w") as f:
        f.write(model_readme)
    
    # Create zip file
    zip_path = Path("tbi-models-v1.0.0.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(model_dir))
    
    print(f"  ✅ Created: {zip_path}")
    
    # Cleanup
    shutil.rmtree(model_dir)
    
    return zip_path

def create_sample_data_package():
    """Create a release package for sample data."""
    
    print("📊 Creating sample data package...")
    
    # Create temporary directory for sample data
    sample_dir = Path("release_sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample data info
    sample_info = {
        "version": "1.0.0",
        "description": "TBI Lesion Analysis Pipeline - Sample Data",
        "data_description": "Sample NIfTI files for testing the pipeline",
        "files": [
            "sample1.nii.gz",
            "sample2.nii.gz", 
            "sample3.nii.gz",
            "sample4.nii.gz"
        ],
        "format": "NIfTI (.nii.gz)",
        "usage": {
            "1": "Extract the sample data",
            "2": "Use with the pipeline: python run_pipeline.py sample1.nii.gz output/",
            "3": "Compare results with expected outputs"
        },
        "data_privacy": "All sample data is anonymized and contains no patient information",
        "disclaimer": "Sample data is for testing purposes only"
    }
    
    with open(sample_dir / "sample_data_info.json", "w") as f:
        json.dump(sample_info, f, indent=2)
    
    # Create README for sample data
    sample_readme = """# TBI Lesion Analysis Pipeline - Sample Data

This package contains sample NIfTI files for testing the TBI Lesion Analysis Pipeline.

## Sample Files

- sample1.nii.gz - Sample MRI scan 1
- sample2.nii.gz - Sample MRI scan 2  
- sample3.nii.gz - Sample MRI scan 3
- sample4.nii.gz - Sample MRI scan 4

## Usage

1. Extract this package
2. Run the pipeline on sample data:

```bash
python run_pipeline.py sample1.nii.gz output/ --subject-id sample1
```

3. Check the output directory for results

## Data Format

- **Format**: NIfTI (.nii.gz)
- **Type**: T1-weighted MRI scans
- **Anonymized**: All patient information has been removed

## Testing

Use these samples to:
- Verify pipeline installation
- Test preprocessing steps
- Validate segmentation results
- Check quantification outputs

## Data Privacy

All sample data is anonymized and contains no patient information or identifiable data.

## Disclaimer

This sample data is for testing and demonstration purposes only.
"""
    
    with open(sample_dir / "README.md", "w") as f:
        f.write(sample_readme)
    
    # Create zip file
    zip_path = Path("tbi-sample-data-v1.0.0.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in sample_dir.rglob("*"):
            if file_path.is_file():
                zipf.write(file_path, file_path.relative_to(sample_dir))
    
    print(f"  ✅ Created: {zip_path}")
    
    # Cleanup
    shutil.rmtree(sample_dir)
    
    return zip_path

def main():
    """Main function to create release assets."""
    
    print("🚀 TBI Lesion Analysis Pipeline - Release Assets")
    print("=" * 50)
    
    # Create model package
    model_zip = create_model_release_package()
    print()
    
    # Create sample data package
    sample_zip = create_sample_data_package()
    print()
    
    print("📋 Release Assets Created:")
    print(f"  📦 Models: {model_zip}")
    print(f"  📊 Sample Data: {sample_zip}")
    print()
    
    print("📤 Upload Instructions:")
    print("1. Go to: https://github.com/thildebrant/tbi/releases")
    print("2. Click on the latest release")
    print("3. Click 'Edit' or create a new release")
    print("4. Drag and drop the zip files to upload")
    print("5. Add descriptions for each asset")
    print("6. Publish the release")
    print()
    
    print("🎉 Release assets ready for upload!")

if __name__ == "__main__":
    main() 