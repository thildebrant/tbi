# TBI Lesion Analysis Pipeline

A deep learning-based pipeline for analyzing traumatic brain injury (TBI) lesions from MRI scans. This tool segments 7 types of TBI lesions and quantifies their overlap with 10 brain anatomical zones. This was a proof of concept using Claude Code, where I found interesting Computed Tomography based Traumatic Brain Injury Quantification python code at https://github.com/nifm-gin/CT-TIQUA.
I then modified the code to use MRI DICOM images, and trained using MR Brain Segmentation Challenge 2018 Data. Finally, I ran inference on some existing MRI images from a TRACTS study.

Next steps would be to make the inference and validation more rigourous and begin an investigation on the lesions identified.
The sample reports are in the output folder.
 

## Features

- **MRI Preprocessing**: Bias field correction, skull stripping, and normalization
- **Multi-class Lesion Segmentation**: Identifies 7 types of TBI lesions
- **Brain Atlas Integration**: Maps lesions to 10 anatomical brain zones
- **Quantitative Analysis**: Calculates lesion volumes per zone with CSV export
- **Visualization Support**: Generates overlay images and color lookup tables

## Lesion Types Detected

1. Intraparenchymal hemorrhage
2. Subdural hematoma
3. Epidural hematoma
4. Subarachnoid hemorrhage
5. Intraventricular hemorrhage
6. Diffuse axonal injury
7. Contusion

## Brain Zones

1. Frontal lobe (left/right)
2. Parietal lobe (left/right)
3. Temporal lobe (left/right)
4. Occipital lobe (left/right)
5. Cerebellum
6. Brainstem

## Installation

```bash
# Clone the repository
cd tbi

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python run_pipeline.py /path/to/mri/data /path/to/output --subject-id patient001
```

### Advanced Options

```bash
python run_pipeline.py /path/to/mri/data /path/to/output \
    --subject-id patient001 \
    --atlas /path/to/custom/atlas.nii.gz \
    --model /path/to/pretrained/model.pth \
    --save-intermediate \
    --save-probabilities \
    --log-level DEBUG
```

### Command Line Arguments

- `input`: Path to input MRI (DICOM directory or NIfTI file)
- `output`: Output directory for results
- `--subject-id`: Subject identifier (default: "subject")
- `--atlas`: Path to custom brain atlas (optional)
- `--model`: Path to pretrained segmentation model (optional)
- `--skip-preprocessing`: Skip preprocessing step
- `--save-intermediate`: Save intermediate processing files
- `--save-probabilities`: Save lesion probability maps
- `--log-level`: Logging level (DEBUG/INFO/WARNING/ERROR)

## Output Structure

```
output_directory/
├── preprocessing/
│   ├── preprocessed_mri.nii.gz
│   └── brain_mask.nii.gz
├── segmentation/
│   ├── lesion_segmentation.nii.gz
│   └── prob_*.nii.gz (if --save-probabilities)
├── registration/
│   └── atlas_in_subject_space.nii.gz
├── atlas/
│   └── zone_masks/
├── quantification/
│   ├── {subject_id}_lesion_zone_overlap.csv
│   ├── {subject_id}_zone_summary.csv
│   ├── {subject_id}_lesion_summary.csv
│   └── {subject_id}_report.txt
├── visualization/
│   ├── lesion_atlas_overlay.nii.gz
│   └── color_lookup_table.json
├── pipeline.log
└── {subject_id}_pipeline_summary.json
```

## Key Output Files

### Lesion-Zone Overlap CSV
Contains detailed volume measurements for each lesion type in each brain zone:
- Zone: Brain region name
- LesionType: Type of lesion
- Volume_mm3: Volume in cubic millimeters
- Volume_ml: Volume in milliliters

### Report
Human-readable summary including:
- Total lesion volume
- Affected brain zones
- Lesion type breakdown
- Detailed lesion-zone combinations

## Pipeline Architecture

1. **Preprocessing** (`src/preprocess.py`)
   - DICOM/NIfTI loading
   - N4 bias field correction
   - Skull stripping
   - Intensity normalization

2. **Segmentation** (`src/segment.py`)
   - 3D U-Net architecture (MONAI)
   - Multi-class prediction (8 classes: 7 lesions + background)
   - Volume calculation

3. **Atlas Registration** (`src/atlas.py`)
   - ANTsPy-based registration
   - 10-zone brain parcellation
   - Zone mask generation

4. **Quantification** (`src/quantify.py`)
   - Lesion-zone overlap calculation
   - Statistical summaries
   - CSV export

## Model Training

The segmentation module includes training transforms for custom model development:

```python
from src.segment import TBISegmenter

segmenter = TBISegmenter()
train_transforms = segmenter.create_training_transforms()
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for full dependencies

## Notes

- Default model uses random initialization (requires training on TBI data)
- Atlas registration quality depends on preprocessing quality
- Processing time varies with image size and GPU availability

## Citation

- This pipeline adapts concepts and code from CT-TIQUA for MRI-based TBI analysis. Source is https://github.com/nifm-gin/CT-TIQUA 
- Labled training data from https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/E0U32Q with full reference:
@data{E0U32Q_2024,
author = {Hugo J. Kuijf and Edwin Bennink and Koen L. Vincken and Nick Weaver and Geert Jan Biessels and Max A. Viergever},
publisher = {DataverseNL},
title = {{MR Brain Segmentation Challenge 2018 Data}},
year = {2024},
version = {V1},
doi = {10.34894/E0U32Q},
url = {https://doi.org/10.34894/E0U32Q}
}

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.
