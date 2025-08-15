# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AutoFlipbook is a Python-based neuroimaging pipeline for generating digital "flipbooks" from longitudinal brain tumor MRI scans. The pipeline uses FSL FLIRT for image registration and creates interactive HTML visualizations showing tumor progression over time.

## Prerequisites and External Dependencies

- **FSL (FMRIB Software Library)** - Required for image registration via FLIRT command
  - Check installation: `flirt -version`
  - Installation guide: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
- **Python 3.8+** - Core runtime requirement

## Development Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Development installation (editable)
pip install -e .

# Install with development dependencies
pip install -e .[dev]
```

### Running the Pipeline
```bash
# Basic command line usage
python cli_script.py --raw_folder data/raw_timepoints --output_folder output/

# With tumor segmentations
python cli_script.py --raw_folder data/raw --segmentation_folder data/segs --output_folder output/

# Custom parameters
python cli_script.py --raw_folder data/raw --registration_dof 12 --tumor_color yellow --show_contour False

# Skip quantitative assessment (faster)
python cli_script.py --raw_folder data/raw --skip_assessment --output_folder output/

# Enhanced visualization features
python cli_script.py --raw_folder data/raw --segmentation_folder data/segs --gif_duration 1500 --summary_slices 3

# Skip specific features
python cli_script.py --raw_folder data/raw --no_summary_slide --no_animated_gif
```

### Running Assessment Only
```bash
# Standalone assessment on existing registered data
python run_assessment.py --registered_folder output/registered --segmentation_folder data/segs --output_folder assessment/

# Generate only difference maps
python run_assessment.py --registered_folder output/registered --only_difference_maps --output_folder assessment/

# Generate only metrics table
python run_assessment.py --registered_folder output/registered --only_metrics_table --output_folder assessment/
```

### Testing
```bash
# Run tests (if available)
pytest tests/

# Note: No test files currently exist in the repository
```

### Code Quality (Development Dependencies)
```bash
# Code formatting
black .

# Linting
flake8 .
```

## Architecture

### Core Components

1. **BrainRegistrationPipeline** (`brain_registration_pipeline.py`):
   - Main pipeline class handling FSL FLIRT registration
   - Generates co-registered images and transformation matrices
   - Creates flipbook visualizations with tumor overlays
   - Integrates quantitative registration assessment

2. **RegistrationAssessment** (`registration_assessment.py`):
   - Quantitative assessment of registration quality
   - Generates metrics tables, difference maps, and contour evolution
   - Supports various similarity metrics and tumor volume analysis

3. **CLI Interface** (`cli_script.py`):
   - Command-line wrapper for the full pipeline
   - Handles argument parsing and validation
   - Entry point: `brain-flipbooks` console script

4. **Standalone Assessment** (`run_assessment.py`):
   - Command-line tool for running assessment on existing registered data
   - Supports selective analysis (metrics only, difference maps only, etc.)

5. **Main Function**:
   - `run_brain_registration_and_flipbook_pipeline()` - High-level API for running complete pipeline

### Pipeline Workflow

1. **Image Registration**: Uses FSL FLIRT to co-register all timepoints to a reference
2. **Segmentation Transformation**: Automatically applies the same transformation matrices to tumor segmentations for perfect alignment with registered images
3. **Flipbook Generation**: Creates HTML visualizations with mosaic layouts using transformed segmentations
4. **Quantitative Assessment** (optional):
   - Registration quality metrics (correlation, MI, SSIM, etc.)
   - Voxelwise difference maps with tumor growth/shrinkage visualization
   - Tumor contour evolution over time
   - Volume analysis and overlap metrics
5. **Output Structure**:
   - `registered/` - Co-registered images
     - `<timepoint>/segmentations/` - Transformed tumor segmentations (aligned with registered images)
   - `transformation_matrices/` - FLIRT .mat files
   - `flipbooks/` - HTML visualizations
   - `logs/` - Registration logs
   - `assessment/` - Quantitative analysis results

### Data Structure Requirements

**Input Structure**:
```
raw_timepoints/
├── 5885/                    # Timepoint folders (numeric IDs - legacy)
│   ├── T1.nii.gz           # Required contrasts
│   ├── T2.nii.gz
│   ├── T1CE.nii.gz
│   └── FLAIR.nii.gz
├── 2001-03-10/             # Date-based folders (YYYY-MM-DD format)
│   ├── T1.nii.gz
│   ├── T2.nii.gz
│   ├── T1CE.nii.gz
│   └── FLAIR.nii.gz
└── 2001-08-22/
    ├── T1.nii.gz
    └── ...
```

**Optional Segmentations**:
```
segmentations/
├── 5885/                    # Must match timepoint folder names
│   └── tumor_seg.nii.gz
├── 2001-03-10/
│   └── tumor_seg.nii.gz
└── 2001-08-22/
    └── tumor_seg.nii.gz
```

## Key Configuration Parameters

### Registration Parameters
- `registration_dof`: 6 (rigid) or 12 (affine) degrees of freedom
- `run_assessment`: Boolean to enable/disable quantitative assessment (default: True)

### Visualization Parameters
- `tumor_color`: 'red', 'yellow', 'cyan', 'green'
- `show_contour`: Boolean for contour vs filled overlays
- `contour_style`: 'solid', 'dashed', 'dotted', 'dashdot'
- `rows`, `cols`: Mosaic layout dimensions (default: 3x5)

### Enhanced Visualization Features
- `create_summary_slide`: Boolean to create tumor-focused summary slide (default: True)
- `create_animated_gif`: Boolean to create animated GIF flipbook (default: True)
- `summary_slices`: Number of central tumor slices in summary slide (default: 4)
- `gif_duration`: Duration per frame in animated GIF in milliseconds (default: 1000)

### Assessment Parameters
- `colormap_style`: 'red_blue', 'viridis', 'plasma', 'coolwarm' for difference maps
- Various metrics: Pearson correlation, normalized mutual information, SSIM, Dice coefficient

## External Tool Integration

The pipeline requires FSL FLIRT to be available in the system PATH. The `setup_fsl_environment()` method in the main class handles FSL environment setup. Registration operations call FLIRT via subprocess and expect specific FSL directory structure.

## File Formats

- **Input**: NIFTI format (.nii or .nii.gz) - neuroimaging standard
- **Output**: HTML flipbooks, PNG slide images, tumor summary slides, animated GIFs, transformation matrices (.mat)
- **Logs**: Text files with registration details

## Dependencies

Core neuroimaging and scientific computing stack:
- `nibabel` - NIFTI file I/O
- `numpy`, `pandas` - Data processing  
- `matplotlib` - Visualization
- `Pillow` - Image processing
- `scipy` - Scientific computing for assessment metrics
- `scikit-learn` - Machine learning metrics (mutual information)

## Performance Considerations

- Processing time: ~15-30 minutes for 5 timepoints, 4 contrasts
- Memory usage: ~2GB per timepoint
- FSL FLIRT is CPU-intensive registration step
- Mosaic generation can be memory-intensive for large images