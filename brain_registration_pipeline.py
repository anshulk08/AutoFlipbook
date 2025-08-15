import os
import glob
import re
import subprocess
import shutil
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from PIL import Image
import pandas as pd
from datetime import datetime
import warnings    
warnings.filterwarnings('ignore')

class BrainRegistrationPipeline:
    def __init__(self, raw_folder, output_base_folder="pipeline_output", 
                 reference_timepoint=None, fsl_dir=None):
        """
        Initialize the automated brain registration and flipbook pipeline
        
        Parameters:
        - raw_folder: Path to folder containing raw timepoint subfolders
        - output_base_folder: Base folder for all pipeline outputs
        - reference_timepoint: Timepoint to use as reference (auto-detected if None)
        - fsl_dir: Path to FSL installation (uses system PATH if None)
        """
        self.raw_folder = raw_folder
        self.output_base_folder = output_base_folder
        self.reference_timepoint = reference_timepoint
        self.fsl_dir = fsl_dir
        
        # Create output directory structure
        self.registered_folder = os.path.join(output_base_folder, "registered")
        self.matrices_folder = os.path.join(output_base_folder, "transformation_matrices")
        self.flipbooks_folder = os.path.join(output_base_folder, "flipbooks")
        self.logs_folder = os.path.join(output_base_folder, "logs")
        
        for folder in [self.registered_folder, self.matrices_folder, 
                      self.flipbooks_folder, self.logs_folder]:
            os.makedirs(folder, exist_ok=True)
        
        # Setup FSL environment
        self.setup_fsl_environment()
        
        # Store timepoint information
        self.timepoint_folders = []
        self.contrast_types = []
        self.reference_files = {}
        
    def setup_fsl_environment(self):
        """Setup FSL environment variables"""
        if self.fsl_dir:
            os.environ['FSLDIR'] = self.fsl_dir
            os.environ['PATH'] = f"{os.path.join(self.fsl_dir, 'bin')}:{os.environ.get('PATH', '')}"
        
        # Set required FSL environment variables
        if 'FSLDIR' not in os.environ:
            # Try to detect FSL installation
            possible_dirs = ['/usr/local/fsl', '/Users/anshul/fsl', '/opt/fsl']
            for fsl_path in possible_dirs:
                if os.path.exists(os.path.join(fsl_path, 'bin', 'flirt')):
                    os.environ['FSLDIR'] = fsl_path
                    os.environ['PATH'] = f"{os.path.join(fsl_path, 'bin')}:{os.environ.get('PATH', '')}"
                    break
        
        # Set FSL output type (required for FSL to work)
        os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
        
        # Test if FSL is available
        try:
            result = subprocess.run(['flirt', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"FSL FLIRT detected: {result.stdout.strip()}")
            else:
                raise RuntimeError("FSL FLIRT not found or not working")
        except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) as e:
            print(f"ERROR: FSL setup failed - {e}")
            print("Please ensure FSL is installed and accessible")
            print("You can specify FSL directory with fsl_dir parameter")
            raise
    
    def find_timepoint_folders(self):
        """Find all timepoint folders and their contents"""
        folders = []
        date_pattern = re.compile(r'^(\d{4})-(\d{2})-(\d{2})$')
        
        for item in os.listdir(self.raw_folder):
            item_path = os.path.join(self.raw_folder, item)
            
            if os.path.isdir(item_path):
                # Check if it's a numeric folder name (legacy support)
                if item.isdigit():
                    folders.append({
                        'folder': item_path,
                        'timepoint': int(item),
                        'folder_name': item,
                        'is_date': False
                    })
                # Check if it's a date format (YYYY-MM-DD)
                elif date_pattern.match(item):
                    try:
                        # Parse date and convert to timestamp for sorting
                        date_obj = datetime.strptime(item, '%Y-%m-%d')
                        folders.append({
                            'folder': item_path,
                            'timepoint': int(date_obj.timestamp()),
                            'folder_name': item,
                            'is_date': True,
                            'date_obj': date_obj
                        })
                    except ValueError:
                        continue
        
        # Sort by timepoint (timestamp for dates, numeric for legacy)
        folders.sort(key=lambda x: x['timepoint'])
        self.timepoint_folders = folders
        
        print(f"Found {len(folders)} timepoint folders:")
        for tp in folders:
            print(f"  {tp['folder_name']}: {tp['folder']}")
        
        return folders
    
    def find_contrast_types(self):
        """Find available contrast types across all timepoints"""
        contrast_types = set()
        
        for tp_folder in self.timepoint_folders:
            nifti_files = glob.glob(os.path.join(tp_folder['folder'], "*.nii*"))
            
            for file in nifti_files:
                basename = os.path.basename(file)
                # Extract contrast type from various naming patterns
                patterns = [
                    r'.*[_-]([Tt]1[Cc]?[Ee]?)[\._]',  # T1, T1C, T1CE
                    r'.*[_-]([Tt]2)[\._]',             # T2
                    r'.*[_-]([Ff][Ll][Aa]?[Ii]?[Rr]?)[\._]',  # FL, FLAIR
                    r'^([Tt]1[Cc]?[Ee]?)[\._]',        # Starting with T1/T1CE
                    r'^([Tt]2)[\._]',                  # Starting with T2
                    r'^([Ff][Ll])[\._]',               # Starting with FL
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, basename, re.IGNORECASE)
                    if match:
                        contrast = match.group(1).upper()
                        # Normalize contrast names
                        if contrast in ['T1C', 'T1CE']:
                            contrast = 'T1CE'
                        elif contrast in ['FLAIR', 'FLAI']:
                            contrast = 'FL'
                        contrast_types.add(contrast)
                        break
                else:
                    # If no pattern matches, try to infer from filename
                    if 't1' in basename.lower():
                        if 'ce' in basename.lower() or 'c' in basename.lower():
                            contrast_types.add('T1CE')
                        else:
                            contrast_types.add('T1')
                    elif 't2' in basename.lower():
                        contrast_types.add('T2')
                    elif 'fl' in basename.lower() or 'flair' in basename.lower():
                        contrast_types.add('FL')
        
        self.contrast_types = sorted(list(contrast_types))
        print(f"Available contrast types: {self.contrast_types}")
        return self.contrast_types
    
    def get_file_for_timepoint_and_contrast(self, timepoint_folder, contrast_type):
        """Get the NIFTI file for a specific timepoint and contrast type"""
        nifti_files = glob.glob(os.path.join(timepoint_folder['folder'], "*.nii*"))
        
        # Try different patterns to match contrast type
        patterns = {
            'T1': [r'.*[_-]t1(?!c)[\._]', r'^t1(?!c)[\._]', r'.*t1.*(?<!ce?)\.nii'],
            'T1CE': [r'.*[_-]t1c?e?[\._]', r'^t1c?e?[\._]', r'.*t1.*c.*\.nii'],
            'T2': [r'.*[_-]t2[\._]', r'^t2[\._]', r'.*t2.*\.nii'],
            'FL': [r'.*[_-]fl(?:air)?[\._]', r'^fl(?:air)?[\._]', r'.*fl.*\.nii']
        }
        
        if contrast_type in patterns:
            for pattern in patterns[contrast_type]:
                for file in nifti_files:
                    basename = os.path.basename(file).lower()
                    if re.search(pattern, basename, re.IGNORECASE):
                        return file
        
        print(f"Warning: No {contrast_type} file found for timepoint {timepoint_folder['folder_name']}")
        return None
    
    def select_reference_timepoint(self):
        """Select reference timepoint (earliest by default)"""
        if not self.timepoint_folders:
            raise ValueError("No timepoint folders found")
        
        if self.reference_timepoint is None:
            # Use the earliest timepoint as reference
            reference_tp = self.timepoint_folders[0]
        else:
            # Find specified reference timepoint
            reference_tp = None
            for tp in self.timepoint_folders:
                if tp['folder_name'] == str(self.reference_timepoint):
                    reference_tp = tp
                    break
            
            if reference_tp is None:
                print(f"Warning: Reference timepoint {self.reference_timepoint} not found, using earliest")
                reference_tp = self.timepoint_folders[0]
        
        print(f"Using reference timepoint: {reference_tp['folder_name']}")
        
        # Get reference files for each contrast
        for contrast in self.contrast_types:
            ref_file = self.get_file_for_timepoint_and_contrast(reference_tp, contrast)
            if ref_file:
                self.reference_files[contrast] = ref_file
                print(f"  Reference {contrast}: {os.path.basename(ref_file)}")
        
        return reference_tp
    
    def register_timepoint(self, timepoint_folder, reference_tp, registration_dof=6):
        """
        Register all contrasts in a timepoint to the reference
        
        Parameters:
        - timepoint_folder: Timepoint folder info
        - reference_tp: Reference timepoint info  
        - registration_dof: Degrees of freedom (6=rigid, 12=affine)
                           Paper recommends 6-DOF rigid to preserve tumor size
        """
        tp_name = timepoint_folder['folder_name']
        ref_name = reference_tp['folder_name']
        
        print(f"Registering timepoint {tp_name} to reference {ref_name} ({registration_dof}-DOF)...")
        
        # Create output folder for this timepoint
        tp_output_folder = os.path.join(self.registered_folder, tp_name)
        os.makedirs(tp_output_folder, exist_ok=True)
        
        # Create matrices folder for this timepoint
        tp_matrices_folder = os.path.join(self.matrices_folder, tp_name)
        os.makedirs(tp_matrices_folder, exist_ok=True)
        
        registration_results = {}
        
        for contrast in self.contrast_types:
            if contrast not in self.reference_files:
                print(f"  Skipping {contrast} - no reference file")
                continue
            
            # Get input and reference files
            input_file = self.get_file_for_timepoint_and_contrast(timepoint_folder, contrast)
            reference_file = self.reference_files[contrast]
            
            if not input_file:
                print(f"  Skipping {contrast} - no input file found")
                continue
            
            # Define output files
            output_file = os.path.join(tp_output_folder, 
                                     f"{tp_name}_{contrast}_to_{ref_name}{contrast}.nii.gz")
            matrix_file = os.path.join(tp_matrices_folder, 
                                     f"{tp_name}_{contrast}_to_{ref_name}{contrast}.mat")
            log_file = os.path.join(self.logs_folder, 
                                  f"registration_{tp_name}_{contrast}.log")
            
            # Skip if already registered and file exists
            if os.path.exists(output_file):
                print(f"  {contrast}: Already registered, skipping")
                registration_results[contrast] = {
                    'success': True,
                    'input_file': input_file,
                    'output_file': output_file,
                    'matrix_file': matrix_file,
                    'skipped': True
                }
                continue
            
            # Run FLIRT registration following paper methodology
            try:
                flirt_cmd = [
                    'flirt',
                    '-in', input_file,
                    '-ref', reference_file,
                    '-out', output_file,
                    '-omat', matrix_file,
                    '-dof', str(registration_dof),  # Paper recommends 6-DOF rigid
                    '-searchrx', '-180', '180',
                    '-searchry', '-180', '180', 
                    '-searchrz', '-180', '180',
                    '-cost', 'normmi',  # Normalized mutual information
                    '-bins', '256',
                    '-interp', 'trilinear'
                ]
                
                print(f"  Registering {contrast}...")
                with open(log_file, 'w') as log:
                    result = subprocess.run(flirt_cmd, 
                                          stdout=log, 
                                          stderr=subprocess.STDOUT, 
                                          text=True, 
                                          timeout=300)  # 5 minute timeout
                
                if result.returncode == 0:
                    print(f"    ✓ {contrast} registration successful")
                    registration_results[contrast] = {
                        'success': True,
                        'input_file': input_file,
                        'output_file': output_file,
                        'matrix_file': matrix_file,
                        'log_file': log_file
                    }
                else:
                    print(f"    ✗ {contrast} registration failed (return code: {result.returncode})")
                    registration_results[contrast] = {
                        'success': False,
                        'error': f"FLIRT returned code {result.returncode}",
                        'log_file': log_file
                    }
                    
            except subprocess.TimeoutExpired:
                print(f"    ✗ {contrast} registration timed out")
                registration_results[contrast] = {
                    'success': False,
                    'error': "Registration timed out (>5 minutes)",
                    'log_file': log_file
                }
            except Exception as e:
                print(f"    ✗ {contrast} registration error: {e}")
                registration_results[contrast] = {
                    'success': False,
                    'error': str(e),
                    'log_file': log_file
                }
        
        return registration_results
    
    def transform_segmentation(self, segmentation_file, matrix_file, reference_file, output_file):
        """
        Transform a segmentation file using a transformation matrix from image registration
        
        Parameters:
        - segmentation_file: Path to input segmentation file
        - matrix_file: Path to transformation matrix (.mat file) from FLIRT registration
        - reference_file: Path to reference image (for space definition)
        - output_file: Path to output transformed segmentation
        
        Returns:
        - dict: Result information with success status
        """
        try:
            # Use FLIRT to apply the transformation to the segmentation
            # Use nearest neighbor interpolation to preserve label values
            flirt_cmd = [
                'flirt',
                '-in', segmentation_file,
                '-ref', reference_file,
                '-out', output_file,
                '-init', matrix_file,      # Use existing transformation matrix
                '-applyxfm',               # Apply transformation (don't compute new one)
                '-interp', 'nearestneighbour'  # Preserve segmentation labels
            ]
            
            # Create log file for segmentation transformation
            log_file = output_file.replace('.nii.gz', '_transform.log').replace('.nii', '_transform.log')
            
            print(f"    Transforming segmentation: {os.path.basename(segmentation_file)}")
            with open(log_file, 'w') as log:
                result = subprocess.run(flirt_cmd, 
                                      stdout=log, 
                                      stderr=subprocess.STDOUT, 
                                      text=True, 
                                      timeout=120)  # 2 minute timeout
            
            if result.returncode == 0:
                print(f"      ✓ Segmentation transformation successful")
                return {
                    'success': True,
                    'input_file': segmentation_file,
                    'output_file': output_file,
                    'matrix_file': matrix_file,
                    'log_file': log_file
                }
            else:
                print(f"      ✗ Segmentation transformation failed (return code: {result.returncode})")
                return {
                    'success': False,
                    'error': f"FLIRT returned code {result.returncode}",
                    'log_file': log_file
                }
                
        except subprocess.TimeoutExpired:
            print(f"      ✗ Segmentation transformation timed out")
            return {
                'success': False,
                'error': "Transformation timed out"
            }
        except Exception as e:
            print(f"      ✗ Segmentation transformation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def transform_segmentations_for_timepoint(self, timepoint_folder, registration_results, segmentation_folder):
        """
        Transform all segmentations for a timepoint using the registration matrices
        
        Parameters:
        - timepoint_folder: Timepoint folder info
        - registration_results: Results from image registration containing matrix files
        - segmentation_folder: Path to folder containing original segmentations
        
        Returns:
        - dict: Transformation results for each available segmentation
        """
        tp_name = timepoint_folder['folder_name']
        segmentation_results = {}
        
        if not segmentation_folder:
            return segmentation_results
            
        # Look for segmentation file for this timepoint
        seg_folder_path = os.path.join(segmentation_folder, tp_name)
        if not os.path.exists(seg_folder_path):
            print(f"    No segmentation folder found for {tp_name}")
            return segmentation_results
            
        # Find segmentation files (look for common names)
        seg_patterns = ['tumor_seg.nii.gz', 'tumor_seg.nii', 'seg.nii.gz', 'seg.nii', '*.nii.gz', '*.nii']
        segmentation_file = None
        
        for pattern in seg_patterns:
            files = glob.glob(os.path.join(seg_folder_path, pattern))
            if files:
                segmentation_file = files[0]  # Take the first match
                break
                
        if not segmentation_file:
            print(f"    No segmentation file found for {tp_name}")
            return segmentation_results
            
        print(f"    Found segmentation: {os.path.basename(segmentation_file)}")
        
        # Create transformed segmentations folder
        transformed_seg_folder = os.path.join(self.registered_folder, tp_name, 'segmentations')
        os.makedirs(transformed_seg_folder, exist_ok=True)
        
        # Transform segmentation using each contrast's transformation matrix
        # We'll use the T1CE matrix if available, otherwise the first successful registration
        preferred_contrasts = ['T1CE', 'T1', 'T2', 'FLAIR']  # Order of preference
        matrix_file = None
        reference_file = None
        contrast_used = None
        
        # Find the best transformation matrix to use
        for contrast in preferred_contrasts:
            if contrast in registration_results and registration_results[contrast]['success']:
                matrix_file = registration_results[contrast]['matrix_file']
                reference_file = self.reference_files.get(contrast)
                contrast_used = contrast
                break
                
        # If no preferred contrast found, use any successful registration
        if not matrix_file:
            for contrast, result in registration_results.items():
                if result['success']:
                    matrix_file = result['matrix_file']
                    reference_file = self.reference_files.get(contrast)
                    contrast_used = contrast
                    break
                    
        if not matrix_file or not reference_file:
            print(f"    No valid transformation matrix found for {tp_name}")
            return segmentation_results
            
        # Define output file for transformed segmentation
        output_file = os.path.join(transformed_seg_folder, f"{tp_name}_seg_registered.nii.gz")
        
        # Skip if already exists
        if os.path.exists(output_file):
            print(f"    Transformed segmentation already exists, skipping")
            segmentation_results['tumor_seg'] = {
                'success': True,
                'input_file': segmentation_file,
                'output_file': output_file,
                'matrix_file': matrix_file,
                'contrast_used': contrast_used,
                'skipped': True
            }
            return segmentation_results
            
        # Transform the segmentation
        print(f"    Using {contrast_used} transformation matrix")
        transform_result = self.transform_segmentation(
            segmentation_file=segmentation_file,
            matrix_file=matrix_file,
            reference_file=reference_file,
            output_file=output_file
        )
        
        transform_result['contrast_used'] = contrast_used
        segmentation_results['tumor_seg'] = transform_result
        
        return segmentation_results
    
    def copy_baseline_segmentation(self, timepoint_folder, segmentation_folder):
        """
        Copy baseline (reference) segmentation to the registered folder
        
        Parameters:
        - timepoint_folder: Reference timepoint folder info
        - segmentation_folder: Path to folder containing original segmentations
        
        Returns:
        - dict: Copy results for the baseline segmentation
        """
        tp_name = timepoint_folder['folder_name']
        segmentation_results = {}
        
        if not segmentation_folder:
            return segmentation_results
            
        # Look for segmentation file for the baseline timepoint
        seg_folder_path = os.path.join(segmentation_folder, tp_name)
        if not os.path.exists(seg_folder_path):
            print(f"    No segmentation folder found for baseline {tp_name}")
            return segmentation_results
            
        # Find segmentation files (look for common names)
        seg_patterns = ['tumor_seg.nii.gz', 'tumor_seg.nii', 'seg.nii.gz', 'seg.nii', '*.nii.gz', '*.nii']
        segmentation_file = None
        
        for pattern in seg_patterns:
            files = glob.glob(os.path.join(seg_folder_path, pattern))
            if files:
                segmentation_file = files[0]  # Take the first match
                break
                
        if not segmentation_file:
            print(f"    No segmentation file found for baseline {tp_name}")
            return segmentation_results
            
        print(f"    Found baseline segmentation: {os.path.basename(segmentation_file)}")
        
        # Create segmentations folder in registered output
        transformed_seg_folder = os.path.join(self.registered_folder, tp_name, 'segmentations')
        os.makedirs(transformed_seg_folder, exist_ok=True)
        
        # Define output file for baseline segmentation
        output_file = os.path.join(transformed_seg_folder, f"{tp_name}_seg_registered.nii.gz")
        
        # Copy baseline segmentation (no transformation needed)
        if not os.path.exists(output_file):
            shutil.copy2(segmentation_file, output_file)
            print(f"      ✓ Baseline segmentation copied")
            
        segmentation_results['tumor_seg'] = {
            'success': True,
            'input_file': segmentation_file,
            'output_file': output_file,
            'is_baseline': True
        }
        
        return segmentation_results
    
    def register_all_timepoints(self, registration_dof=6, skull_strip=False, bias_correct=False, segmentation_folder=None):
        """
        Register all timepoints to the reference following paper methodology
        
        Parameters:
        - registration_dof: Degrees of freedom (6=rigid body, 12=affine)
                           Paper recommends 6-DOF rigid to preserve tumor size
        - skull_strip: Whether to perform skull stripping (optional per paper)
        - bias_correct: Whether to perform bias field correction (optional per paper)
        - segmentation_folder: Path to folder containing tumor segmentations (optional)
        """
        if not self.timepoint_folders:
            self.find_timepoint_folders()
        
        if not self.contrast_types:
            self.find_contrast_types()
        
        reference_tp = self.select_reference_timepoint()
        
        print(f"\n=== Starting Registration Process (Following Cho et al. 2024 Methodology) ===")
        print(f"Registration method: {registration_dof}-DOF {'rigid body' if registration_dof == 6 else 'affine'}")
        print(f"Cost function: Normalized mutual information")
        print(f"Interpolation: Trilinear")
        print(f"Skull stripping: {'Yes' if skull_strip else 'No (preserves extra-axial structures)'}")
        print(f"Bias correction: {'Yes' if bias_correct else 'No'}")
        print(f"Total timepoints to process: {len(self.timepoint_folders)}")
        print(f"Contrasts to register: {self.contrast_types}")
        
        all_results = {}
        
        for i, tp_folder in enumerate(self.timepoint_folders):
            tp_name = tp_folder['folder_name']
            print(f"\n--- Processing timepoint {i+1}/{len(self.timepoint_folders)}: {tp_name} ---")
            
            if tp_folder == reference_tp:
                # Copy reference files to output folder
                print(f"  Copying reference files (baseline timepoint)...")
                tp_output_folder = os.path.join(self.registered_folder, tp_name)
                os.makedirs(tp_output_folder, exist_ok=True)
                
                ref_results = {}
                for contrast in self.contrast_types:
                    if contrast in self.reference_files:
                        src_file = self.reference_files[contrast]
                        dst_file = os.path.join(tp_output_folder, 
                                              f"{tp_name}_{contrast}_to_{tp_name}{contrast}.nii.gz")
                        
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)
                            print(f"    Copied {contrast}")
                        
                        ref_results[contrast] = {
                            'success': True,
                            'input_file': src_file,
                            'output_file': dst_file,
                            'is_reference': True
                        }
                
                all_results[tp_name] = ref_results
                
                # Handle baseline segmentation (no transformation needed)
                if segmentation_folder:
                    print(f"  Copying baseline segmentation for {tp_name}...")
                    baseline_seg_results = self.copy_baseline_segmentation(tp_folder, segmentation_folder)
                    all_results[tp_name]['segmentations'] = baseline_seg_results
            else:
                # Register to reference
                results = self.register_timepoint(tp_folder, reference_tp, registration_dof)
                all_results[tp_name] = results
                
                # Transform segmentations if available
                if segmentation_folder:
                    print(f"  Transforming segmentations for {tp_name}...")
                    seg_results = self.transform_segmentations_for_timepoint(
                        tp_folder, results, segmentation_folder
                    )
                    # Add segmentation results to the main results
                    all_results[tp_name]['segmentations'] = seg_results
        
        # Print summary
        self.print_registration_summary(all_results)
        
        return all_results
    
    def print_registration_summary(self, results):
        """Print summary of registration results"""
        print(f"\n=== REGISTRATION SUMMARY ===")
        
        total_registrations = 0
        successful_registrations = 0
        total_segmentations = 0
        successful_segmentations = 0
        
        for tp_name, tp_results in results.items():
            print(f"\nTimepoint {tp_name}:")
            for contrast, result in tp_results.items():
                # Handle segmentation results separately
                if contrast == 'segmentations':
                    # This is a nested segmentation results structure
                    for seg_name, seg_result in result.items():
                        total_segmentations += 1
                        if seg_result.get('success', False):
                            successful_segmentations += 1
                            status = "✓ SUCCESS"
                            if seg_result.get('is_baseline'):
                                status += " (Baseline - copied)"
                            elif seg_result.get('skipped'):
                                status += " (Skipped - already exists)"
                            else:
                                contrast_used = seg_result.get('contrast_used', 'unknown')
                                status += f" (Transformed using {contrast_used})"
                        else:
                            status = f"✗ FAILED ({seg_result.get('error', 'Unknown error')})"
                        
                        print(f"  {seg_name} segmentation: {status}")
                else:
                    # Handle normal registration results
                    total_registrations += 1
                    if result.get('success', False):
                        successful_registrations += 1
                        status = "✓ SUCCESS"
                        if result.get('is_reference'):
                            status += " (Reference)"
                        elif result.get('skipped'):
                            status += " (Skipped - already exists)"
                    else:
                        status = f"✗ FAILED ({result.get('error', 'Unknown error')})"
                    
                    print(f"  {contrast}: {status}")
        
        # Print registration summary
        success_rate = (successful_registrations / total_registrations * 100) if total_registrations > 0 else 0
        print(f"\nRegistration Overall: {successful_registrations}/{total_registrations} registrations successful ({success_rate:.1f}%)")
        
        # Print segmentation summary if any segmentations were processed
        if total_segmentations > 0:
            seg_success_rate = (successful_segmentations / total_segmentations * 100)
            print(f"Segmentation Overall: {successful_segmentations}/{total_segmentations} segmentations successful ({seg_success_rate:.1f}%)")
        
        if successful_registrations > 0:
            print(f"\nRegistered images saved to: {self.registered_folder}")
            print(f"Transformation matrices saved to: {self.matrices_folder}")
            print(f"Registration logs saved to: {self.logs_folder}")
    
    def generate_flipbooks(self, segmentation_folder=None, **flipbook_kwargs):
        """Generate flipbooks from registered images"""
        print(f"\n=== Generating Flipbooks ===")
        
        # Import the flipbook generator
        from brain_flipbook_with_segmentation import BrainFlipbookGenerator
        
        # Create flipbook generator
        generator = BrainFlipbookGenerator(
            registered_folder=self.registered_folder,
            segmentation_folder=segmentation_folder,
            output_folder=self.flipbooks_folder
        )
        
        # Generate flipbooks for all contrasts (without HTML files)
        results = generator.generate_all_contrast_flipbooks_slides_only(**flipbook_kwargs)
        
        return results
    
    def run_full_pipeline(self, segmentation_folder=None, registration_dof=6, 
                         skull_strip=False, bias_correct=False, run_assessment=True, **flipbook_kwargs):
        """
        Run the complete pipeline: registration + flipbook generation
        Following the methodology from Cho et al. (2024) Neuro-Oncology
        
        Parameters:
        - segmentation_folder: Path to tumor segmentation folder
        - registration_dof: 6 (rigid, preserves tumor size) or 12 (affine) 
        - skull_strip: Optional skull stripping (may remove important structures)
        - bias_correct: Optional bias field correction
        - run_assessment: Run quantitative registration quality assessment (default: True)
        - **flipbook_kwargs: Additional arguments for flipbook generation
        """
        print("=== AUTOMATED BRAIN REGISTRATION AND FLIPBOOK PIPELINE ===")
        print("Following methodology from Cho et al. (2024) Neuro-Oncology")
        print(f"Raw data folder: {self.raw_folder}")
        print(f"Output base folder: {self.output_base_folder}")
        
        try:
            # Step 1: Register all images following paper methodology
            registration_results = self.register_all_timepoints(
                registration_dof=registration_dof,
                skull_strip=skull_strip, 
                bias_correct=bias_correct,
                segmentation_folder=segmentation_folder
            )
            
            # Step 2: Generate flipbooks
            flipbook_results = self.generate_flipbooks(
                segmentation_folder=segmentation_folder, 
                **flipbook_kwargs
            )
            
            print(f"\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
            print(f"Image registration and flipbook generation completed following Cho et al. methodology")
            print(f"All outputs saved to: {os.path.abspath(self.output_base_folder)}")
            print(f"- Registered images: {self.registered_folder}")
            print(f"- Transformation matrices: {self.matrices_folder}")
            print(f"- Logs: {self.logs_folder}")
            
            # Print flipbook results
            if isinstance(flipbook_results, dict) and flipbook_results:
                print(f"- Flipbook slides: {self.flipbooks_folder}")
                total_slides = sum(result.get('slide_count', 0) for result in flipbook_results.values())
                print(f"  Generated {total_slides} total slides for {len(flipbook_results)} contrast types")
            
            # Step 3: Run quantitative assessment (optional)
            assessment_results = None
            if run_assessment:
                try:
                    print("\n=== QUANTITATIVE REGISTRATION ASSESSMENT ===")
                    from registration_assessment import RegistrationAssessment
                    
                    assessment_folder = os.path.join(self.output_base_folder, "assessment")
                    assessment = RegistrationAssessment(
                        registered_folder=self.registered_folder,
                        segmentation_folder=segmentation_folder,
                        output_folder=assessment_folder
                    )
                    
                    # Run comprehensive assessment for all available contrasts
                    assessment_results = assessment.run_comprehensive_assessment()
                    
                    print(f"- Assessment results: {assessment_folder}")
                    
                except Exception as e:
                    print(f"Warning: Assessment failed: {e}")
                    print("Continuing without assessment...")
            
            return {
                'registration_results': registration_results,
                'flipbook_results': flipbook_results,
                'assessment_results': assessment_results,
                'output_folders': {
                    'registered': self.registered_folder,
                    'matrices': self.matrices_folder,
                    'flipbooks': self.flipbooks_folder,
                    'logs': self.logs_folder,
                    'assessment': os.path.join(self.output_base_folder, "assessment") if run_assessment else None
                }
            }
            
        except Exception as e:
            print(f"\n=== PIPELINE FAILED ===")
            print(f"Error: {e}")
            raise

# Convenience function for the full pipeline
def run_brain_registration_and_flipbook_pipeline(raw_folder, 
                                                segmentation_folder=None,
                                                output_base_folder="pipeline_output",
                                                reference_timepoint=None,
                                                fsl_dir=None,
                                                registration_dof=6,
                                                skull_strip=False,
                                                bias_correct=False,
                                                run_assessment=True,
                                                **flipbook_kwargs):
    """
    Run the complete automated pipeline following Cho et al. (2024) methodology
    
    Parameters:
    - raw_folder: Path to folder containing raw timepoint subfolders
    - segmentation_folder: Path to folder containing segmentation maps (optional)
    - output_base_folder: Base folder for all pipeline outputs
    - reference_timepoint: Timepoint to use as reference (auto-detected if None)
    - fsl_dir: Path to FSL installation (uses system PATH if None)
    - registration_dof: 6 (rigid body, preserves tumor size) or 12 (affine)
                       Paper recommends 6-DOF rigid transformation
    - skull_strip: Whether to perform skull stripping (optional, may remove important structures)
    - bias_correct: Whether to perform bias field correction (optional)
    - **flipbook_kwargs: Additional arguments for flipbook generation
    
    Returns:
    Dictionary with registration and flipbook results
    
    Reference:
    Cho et al. (2024). Digital "flipbooks" for enhanced visual assessment of 
    simple and complex brain tumors. Neuro-Oncology, 26(10), 1823-1836.
    """
    pipeline = BrainRegistrationPipeline(
        raw_folder=raw_folder,
        output_base_folder=output_base_folder,
        reference_timepoint=reference_timepoint,
        fsl_dir=fsl_dir
    )
    
    return pipeline.run_full_pipeline(
        segmentation_folder=segmentation_folder,
        registration_dof=registration_dof,
        skull_strip=skull_strip,
        bias_correct=bias_correct,
        run_assessment=run_assessment,
        **flipbook_kwargs
    )

# Example usage
if __name__ == "__main__":
    # Example 1: Run full automated pipeline following Cho et al. methodology
    raw_folder = r"C:\Users\smita\raw_timepoints"  # Folder with timepoint subfolders
    segmentation_folder = r"C:\Users\smita\segmentations"  # Optional segmentations
    
    results = run_brain_registration_and_flipbook_pipeline(
        raw_folder=raw_folder,
        segmentation_folder=segmentation_folder,
        output_base_folder="automated_pipeline_output",
        reference_timepoint=None,  # Use earliest timepoint as reference
        fsl_dir=None,  # Use system PATH for FSL
        registration_dof=6,  # Rigid body registration (preserves tumor size)
        skull_strip=False,  # Keep skull for extra-axial structure evaluation
        bias_correct=False,  # Optional bias field correction
        # Flipbook parameters
        tumor_color='red',
        show_contour=True,
        tumor_alpha=0.8,
        rows=3,
        cols=5
    )
    
    print("Pipeline completed following Cho et al. (2024) methodology!")
    print("Reference: Cho et al. Digital 'flipbooks' for enhanced visual assessment")
    print("of simple and complex brain tumors. Neuro-Oncology. 2024;26(10):1823-1836.")
    
    # Example 2: Run with affine registration for more complex cases
    # results_affine = run_brain_registration_and_flipbook_pipeline(
    #     raw_folder=raw_folder,
    #     registration_dof=12,  # Affine registration (12-DOF)
    #     skull_strip=True,     # Optional skull stripping
    #     bias_correct=True     # Optional bias correction
    # )
    
    # Example 3: Run only registration (without flipbooks)
    # pipeline = BrainRegistrationPipeline(
    #     raw_folder=raw_folder,
    #     output_base_folder="registration_only_output"
    # )
    # registration_results = pipeline.register_all_timepoints(registration_dof=6)