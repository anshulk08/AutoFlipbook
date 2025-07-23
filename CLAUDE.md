# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AutoFlipbook** is a medical imaging application for generating digital "flipbooks" from longitudinal brain tumor MRI scans. The project implements the methodology from Cho et al. (2024) for enhanced visual assessment of brain tumor progression over time.

**⚠️ Important**: The project has the core registration pipeline implemented in `brain_registration_pipeline.py`, but the flipbook generation module is still missing from the imports.

## Technology Stack

- **Language**: Python 3.8+
- **Domain**: Medical/Neuroimaging Research
- **External Dependencies**: FSL (FMRIB Software Library) for image registration
- **Core Libraries**: numpy, nibabel, matplotlib, pandas, Pillow

## Development Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements_txt.txt

# Install FSL (required external dependency)
# Follow: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation

# Verify FSL installation
flirt -version

# Install package in development mode (once properly packaged)
pip install -e .
```

### Running the Pipeline
```bash
# Run the CLI with basic options  
python cli_script.py --raw_folder data/raw_timepoints --output_folder output/

# Run with tumor segmentations
python cli_script.py --raw_folder data/raw --segmentation_folder data/segs --output_folder output/

# Run directly using Python module
python brain_registration_pipeline.py

# Run with custom FSL directory
python cli_script.py --raw_folder data/raw --fsl_dir /usr/local/fsl --output_folder output/
```

### Development Tasks
```bash
# Install with development dependencies (once setup.py is configured)
pip install -e .[dev]

# Test FSL integration
python -c "from brain_registration_pipeline import BrainRegistrationPipeline; print('FSL integration available')"
```

### Testing (Not Yet Implemented)
```bash
# Run tests (once implemented)
pytest tests/

# Run tests with coverage (once implemented)  
pytest --cov=brain_flipbook_pipeline tests/
```

### Code Quality (Development Dependencies)  
```bash
# Code formatting
black brain_registration_pipeline.py cli_script.py

# Linting
flake8 brain_registration_pipeline.py cli_script.py
```

## Project Architecture

### Current State
The project has these components:
- `cli_script.py` - Complete CLI interface with argument parsing
- `brain_registration_pipeline.py` - **Core registration pipeline implementation**
- `setup_py.py` - Package configuration for installation
- `github_readme.md` - Comprehensive documentation
- `requirements_txt.txt` - Python dependencies
- `license_file.md` - MIT license

### Core Implementation Status
✅ **Registration Pipeline** (`brain_registration_pipeline.py`) - **IMPLEMENTED**
- Complete `BrainRegistrationPipeline` class with FSL FLIRT integration
- Multi-timepoint image alignment using rigid/affine registration
- Automated contrast type detection (T1, T2, T1CE, FLAIR)
- Comprehensive error handling and logging
- `run_brain_registration_and_flipbook_pipeline()` function available

❌ **Missing Flipbook Generation Module**
The registration pipeline imports a missing flipbook generator on line 414:
```python
from your_existing_module import BrainFlipbookGenerator  # Needs implementation
```

**Still needs implementation:**
1. `BrainFlipbookGenerator` class for HTML flipbook creation
2. Tumor segmentation overlay functionality
3. Interactive HTML templates with auto-play controls
4. Multi-contrast visualization coordination

### Expected Data Flow
1. **Input**: NIFTI brain MRI scans organized by timepoints
2. **Registration**: FSL FLIRT co-registration to reference timepoint
3. **Visualization**: Generate mosaic views with tumor overlays
4. **Output**: Interactive HTML flipbooks showing tumor progression

### Clinical Requirements
- **6-DOF rigid registration** (preserves tumor size) vs 12-DOF affine
- **Multi-contrast support** for different MRI sequences
- **Tumor segmentation overlay** with configurable colors/transparency
- **Mosaic layout** (default 3x5 slices) for whole-brain visualization

## File Structure Conventions

### Current Structure
```
AutoFlipbook/
├── brain_registration_pipeline.py  # ✅ Main registration pipeline (IMPLEMENTED)  
├── cli_script.py                   # ✅ Command-line interface (IMPLEMENTED)
├── setup_py.py                     # ✅ Package setup (IMPLEMENTED)
├── github_readme.md                # ✅ Documentation (IMPLEMENTED)
├── requirements_txt.txt            # ✅ Dependencies (IMPLEMENTED)
└── license_file.md                 # ✅ MIT license (IMPLEMENTED)
```

### Expected Structure for Flipbook Generator (Missing)
```
brain_flipbook_generator.py         # ❌ HTML flipbook creation (NEEDED)
├── BrainFlipbookGenerator class
├── HTML template generation
├── Tumor overlay visualization
└── Interactive controls
```

## Input Data Requirements

### Raw Timepoints Structure
```
raw_timepoints/
├── 5885/                         # Timepoint folders named by ID
│   ├── T1.nii.gz                # Required MRI sequences
│   ├── T2.nii.gz
│   ├── T1CE.nii.gz
│   └── FLAIR.nii.gz
└── 5972/
    ├── T1.nii.gz
    ├── T2.nii.gz
    ├── T1CE.nii.gz
    └── FLAIR.nii.gz
```

### Optional Segmentations
```
segmentations/
├── 5885/
│   └── tumor_seg.nii.gz         # Binary tumor masks
└── 5972/
    └── tumor_seg.nii.gz
```

## Key Integration Points

### Implemented (`brain_registration_pipeline.py`)
- ✅ **FSL Integration**: Complete FLIRT command execution with error handling  
- ✅ **NIFTI Processing**: Uses nibabel for loading/saving neuroimaging data
- ✅ **Timepoint Detection**: Automatic contrast type detection (T1, T2, T1CE, FLAIR)
- ✅ **Registration Pipeline**: 6-DOF rigid and 12-DOF affine registration options
- ✅ **Error Handling**: Comprehensive logging and timeout handling

### Missing (Flipbook Generator)
- ❌ **HTML Generation**: Interactive flipbooks with JavaScript controls  
- ❌ **Tumor Visualization**: Segmentation overlay with customizable colors
- ❌ **Mosaic Generation**: Multi-slice brain visualization layouts

## Development Notes

### Registration Pipeline Features
- Automatic timepoint folder detection (numeric folder names)
- Flexible contrast type recognition with regex patterns
- FSL FLIRT integration with clinical-grade parameters:
  - 6-DOF rigid registration (preserves tumor size) - **recommended**
  - 12-DOF affine registration (allows scaling/shearing)
  - Normalized mutual information cost function
  - Trilinear interpolation
- Comprehensive error handling with 5-minute timeouts
- Reference timepoint auto-selection (earliest) or manual specification

### Data Requirements
- All MRI data must be in NIFTI format (.nii or .nii.gz)
- Timepoint folders must have numeric names (e.g., "5885", "5972")
- Contrast detection supports multiple naming patterns:
  - T1: `T1.nii.gz`, `*_T1.nii.gz`, `*t1*.nii.gz`
  - T1CE: `T1CE.nii.gz`, `T1C.nii.gz`, `*_t1c*.nii.gz`
  - T2: `T2.nii.gz`, `*_T2.nii.gz`, `*t2*.nii.gz`  
  - FLAIR: `FL.nii.gz`, `FLAIR.nii.gz`, `*flair*.nii.gz`

### Clinical Workflow Integration
- Based on Cho et al. (2024) methodology for brain tumor assessment
- Preserves extra-axial structures (no skull stripping by default)
- Registration parameters optimized for tumor progression analysis
- Generates transformation matrices for reproducible analysis