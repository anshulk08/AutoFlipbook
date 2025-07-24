# Brain Tumor Flipbook Generation Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FSL Required](https://img.shields.io/badge/FSL-Required-orange.svg)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)

An automated pipeline for generating digital "flipbooks" from longitudinal brain tumor MRI scans. This tool implements the methodology described in [Cho et al. (2024)](https://academic.oup.com/neuro-oncology/article/26/10/1823/7684619) for enhanced visual assessment of brain tumor progression over time.

## 🎯 Overview

Digital flipbooks enable clinicians to visualize subtle changes in brain tumors by displaying co-registered, consecutive MRI scans in a slide deck format. When viewed sequentially, changes in tumor size, mass effect, and infiltration appear as perceived motion, enhancing detection of progression or treatment response.

### Key Features

- **Automated FSL FLIRT registration** following clinical best practices
- **Multi-contrast support** (T1, T2, T1CE, FLAIR)
- **Tumor segmentation overlay** with volume quantification
- **Interactive HTML flipbooks** with auto-play functionality
- **Clinical-grade workflow** based on published methodology
- **Comprehensive logging** and error handling

## 📋 Table of Contents

- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Data Structure](#data-structure)
- [Configuration Options](#configuration-options)
- [Clinical Applications](#clinical-applications)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## 🛠 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anshulk08/AutoFlipbook.git
cd AutoFlipbook
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt

# If you encounter NumPy compatibility issues, use:
pip install "numpy<2.0"
```

### 3. Verify FSL Installation

The pipeline requires FSL (FMRIB Software Library) for image registration:

```bash
# Check if FSL is installed
flirt -version

# If not installed, follow FSL installation guide:
# https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
```

## ⚡ Prerequisites

### Software Requirements

- **Python 3.8+**
- **FSL 6.0+** (for FLIRT registration)
- **Git** (for cloning repository)

### Python Packages

All required packages are listed in `requirements.txt`:

```
numpy>=1.20.0
nibabel>=3.2.0
matplotlib>=3.5.0
pandas>=1.3.0
Pillow>=8.0.0
```

### Hardware Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~2GB per timepoint (varies with image resolution)
- **CPU**: Multi-core recommended for faster processing

## 🚀 Quick Start

### Basic Usage

```python
from brain_registration_pipeline import run_brain_registration_and_flipbook_pipeline

# Run the complete pipeline
results = run_brain_registration_and_flipbook_pipeline(
    raw_folder="data/raw_timepoints",
    segmentation_folder="data/segmentations",  # Optional
    output_base_folder="output/flipbooks",
    tumor_color='red',
    show_contour=True
)

print("Flipbooks generated successfully!")
```

### Command Line Usage

```bash
python cli_script.py \
    --raw_folder data/raw_timepoints \
    --segmentation_folder data/segmentations \
    --output_folder output/flipbooks \
    --registration_dof 6 \
    --tumor_color red
```

## 📁 Data Structure

### Required Input Structure

```
raw_timepoints/
├── 5885/                    # First timepoint (baseline)
│   ├── T1.nii.gz           # T1-weighted image
│   ├── T2.nii.gz           # T2-weighted image
│   ├── T1CE.nii.gz         # T1 post-contrast
│   └── FLAIR.nii.gz        # FLAIR image
├── 5972/                    # Second timepoint
│   ├── T1.nii.gz
│   ├── T2.nii.gz
│   ├── T1CE.nii.gz
│   └── FLAIR.nii.gz
└── 6070/                    # Third timepoint
    ├── T1.nii.gz
    ├── T2.nii.gz
    ├── T1CE.nii.gz
    └── FLAIR.nii.gz
```

### Optional Segmentation Structure

```
segmentations/
├── 5885/
│   └── tumor_seg.nii.gz    # Tumor segmentation mask
├── 5972/
│   └── tumor_seg.nii.gz
└── 6070/
    └── tumor_seg.nii.gz
```

### Output Structure

```
output/
├── registered/              # Co-registered images
├── transformation_matrices/ # FLIRT .mat files
├── flipbooks/              # Generated flipbooks
│   ├── T1_flipbook/
│   ├── T2_flipbook/
│   ├── T1CE_flipbook/
│   ├── FLAIR_flipbook/
│   └── index.html          # Master index
└── logs/                   # Registration logs
```

## ⚙️ Configuration Options

### Registration Parameters

```python
# Rigid body registration (preserves tumor size)
registration_dof=6          # 6-DOF rigid transformation

# Affine registration (allows scaling/shearing)
registration_dof=12         # 12-DOF affine transformation

# Optional preprocessing
skull_strip=False           # Keep extra-axial structures
bias_correct=False          # Bias field correction
```

### Flipbook Visualization

```python
# Tumor overlay options
tumor_color='red'           # 'red', 'yellow', 'cyan', 'green'
tumor_alpha=0.5             # Transparency (0-1, default: 0.5)
show_contour=True           # Contour vs filled overlay
contour_width=2             # Line width for contours
contour_style='dotted'      # 'solid', 'dashed', 'dotted', 'dashdot'

# Mosaic layout
rows=3                      # Number of rows
cols=5                      # Number of columns (3x5 = 15 slices)

# Display parameters
colormap='gray'             # 'gray', 'bone', 'hot'
window_level=None           # Auto-windowing if None
window_width=None           # Auto-windowing if None
```

### Reference Timepoint Selection

```python
# Automatic selection (earliest timepoint)
reference_timepoint=None

# Manual selection
reference_timepoint="5885"  # Use specific timepoint as reference
```

### Volume Progression Analysis

When segmentation maps are provided, the pipeline automatically generates:

```python
# Tumor volume tracking
- Volume calculations in mL for each timepoint
- Volume change statistics (absolute and percentage)
- Progression plots showing volume over time
- Clinical summary reports
```

## 🔧 Advanced Usage

### Custom Pipeline Configuration

```python
from brain_registration_pipeline import BrainRegistrationPipeline

# Create custom pipeline
pipeline = BrainRegistrationPipeline(
    raw_folder="data/raw",
    output_base_folder="custom_output",
    reference_timepoint="baseline"
)

# Run registration only
registration_results = pipeline.register_all_timepoints(
    registration_dof=6,
    skull_strip=False
)

# Generate flipbooks separately
flipbook_results = pipeline.generate_flipbooks(
    segmentation_folder="data/segs",
    tumor_color='yellow',
    rows=4,
    cols=4
)
```

### Batch Processing Multiple Patients

```python
import os
from pathlib import Path

patients = ["patient_001", "patient_002", "patient_003"]

for patient in patients:
    print(f"Processing {patient}...")
    
    results = run_brain_registration_and_flipbook_pipeline(
        raw_folder=f"data/{patient}/raw_timepoints",
        segmentation_folder=f"data/{patient}/segmentations",
        output_base_folder=f"output/{patient}",
        tumor_color='red'
    )
    
    print(f"✓ {patient} completed")
```

## 🐛 Troubleshooting

### Common Issues

#### FSL Not Found
```bash
# Error: FSL FLIRT not found
# Solution: Install FSL and add to PATH
export FSLDIR=/usr/local/fsl
export PATH=$FSLDIR/bin:$PATH
```

#### Registration Failures
```python
# Check registration logs
registration_log = "output/logs/registration_5972_T1.log"
with open(registration_log, 'r') as f:
    print(f.read())
```

#### Memory Issues
```python
# Reduce mosaic size for large images
results = run_brain_registration_and_flipbook_pipeline(
    raw_folder="data/raw",
    rows=2,          # Reduce from 3x5 to 2x3
    cols=3
)
```

#### File Format Issues
- Ensure all images are in **NIFTI format** (.nii or .nii.gz)
- Use `dcm2niix` to convert DICOM to NIFTI if needed
- Verify consistent image orientations

### Getting Help

1. **Check logs** in the `logs/` output folder
2. **Verify FSL installation**: `flirt -version`
3. **Test with sample data** (see `examples/` folder)
4. **Open an issue** on GitHub with error logs

## 📊 Performance

### Typical Processing Times
- **Registration**: ~2-5 minutes per timepoint per contrast
- **Flipbook generation**: ~30 seconds per contrast
- **Total pipeline**: ~15-30 minutes for 5 timepoints, 4 contrasts

### Optimization Tips
- Use **SSD storage** for faster I/O
- **Parallel processing** for multiple patients
- **Reduce image resolution** if appropriate for clinical use

## 🔬 Scientific Background

This pipeline implements the methodology described in:

> **Cho, N. S., et al.** (2024). Digital "flipbooks" for enhanced visual assessment of simple and complex brain tumors. *Neuro-Oncology*, 26(10), 1823-1836.

### Key Advantages
- **Motion perception**: Exploits visual cortex motion detection
- **Whole-brain assessment**: Shows complete spatial context
- **Temporal coherence**: Perfect spatial correspondence across time
- **Clinical validation**: Based on published methodology

## 📈 Examples

### Example 1: Low-Grade Glioma Monitoring
```python
# Monitor slow-growing tumor over 2 years
results = run_brain_registration_and_flipbook_pipeline(
    raw_folder="data/low_grade_glioma",
    registration_dof=6,        # Preserve tumor size
    tumor_color='red',
    show_contour=True,         # Show tumor boundaries
    tumor_alpha=0.8
)
```

### Example 2: Treatment Response Assessment
```python
# Assess response to therapy with volume tracking and dotted contours
results = run_brain_registration_and_flipbook_pipeline(
    raw_folder="data/treatment_response", 
    segmentation_folder="data/tumor_segs",
    tumor_color='yellow',
    show_contour=True,         # Show contours
    contour_style='dotted',    # Dotted lines for subtle visualization
    tumor_alpha=0.6
)

# Volume progression plots are automatically generated in output folder
print("Volume progression plots saved as: tumor_volume_progression_*.png")
```

### Development Setup
```bash
git clone https://github.com/anshulk08/AutoFlipbook.git
cd AutoFlipbook
pip install -e .
pip install -e .[dev]
```

### Running Tests
```bash
pytest tests/
```

## 📄 Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{cho2024flipbooks,
  title={Digital "flipbooks" for enhanced visual assessment of simple and complex brain tumors},
  author={Cho, Nicholas S and Le, Vi{\^e}n Lam and Sanvito, Francesco and others},
  journal={Neuro-Oncology},
  volume={26},
  number={10},
  pages={1823--1836},
  year={2024},
  publisher={Oxford University Press}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
