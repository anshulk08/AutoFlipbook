import os
import glob
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import warnings    
warnings.filterwarnings('ignore')

try:
    from pptx import Presentation
    from pptx.util import Inches
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False
    print("Warning: python-pptx not installed. PPTX generation will be disabled.")

class BrainFlipbookGenerator:
    def __init__(self, registered_folder, segmentation_folder=None, output_folder="flipbooks"):
        """
        Initialize the flipbook generator for folder-based timepoint organization
        
        Parameters:
        - registered_folder: Path to folder containing timepoint subfolders (e.g., 5885, 5972, 6070)
        - segmentation_folder: Path to folder containing segmentation maps (same structure as registered_folder)
        - output_folder: Path to output folder for flipbooks
        """
        self.registered_folder = registered_folder
        self.segmentation_folder = segmentation_folder
        self.output_folder = output_folder
        self.timepoint_folders = []
        self.contrast_types = []
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
    def find_timepoint_folders(self):
        """Find all timepoint folders (numeric folder names)"""
        folders = []
        for item in os.listdir(self.registered_folder):
            item_path = os.path.join(self.registered_folder, item)
            if os.path.isdir(item_path) and item.isdigit():
                folders.append({
                    'folder': item_path,
                    'timepoint': int(item),
                    'folder_name': item
                })
        
        # Sort by timepoint number
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
                # Parse contrast type from filename like "6070_T1_to_5885T1CE.nii"
                match = re.search(r'^\d+_([^_]+)_to_', basename)
                if match:
                    contrast_types.add(match.group(1))
        
        self.contrast_types = sorted(list(contrast_types))
        print(f"Available contrast types: {self.contrast_types}")
        return self.contrast_types
    
    def get_file_for_timepoint_and_contrast(self, timepoint_folder, contrast_type):
        """Get the NIFTI file for a specific timepoint and contrast type"""
        folder_name = os.path.basename(timepoint_folder['folder'])
        
        # Look for file with pattern: {timepoint}_{contrast}_to_{reference}T1CE.nii
        pattern = f"{folder_name}_{contrast_type}_to_*.nii*"
        files = glob.glob(os.path.join(timepoint_folder['folder'], pattern))
        
        if files:
            return files[0]  # Return first match
        else:
            print(f"Warning: No {contrast_type} file found for timepoint {folder_name}")
            return None
    
    def get_segmentation_file(self, timepoint_folder):
        """Get the segmentation file for a specific timepoint"""
        if not self.segmentation_folder:
            return None
            
        folder_name = os.path.basename(timepoint_folder['folder'])
        
        # First, try looking for files in a subfolder structure
        seg_folder_path = os.path.join(self.segmentation_folder, folder_name)
        
        if os.path.exists(seg_folder_path):
            # Look in subfolder
            patterns = [
                "*.nii*",  # Any NIFTI file
                "*seg*.nii*",  # Files with 'seg' in name
                "*mask*.nii*",  # Files with 'mask' in name
                "*tumor*.nii*",  # Files with 'tumor' in name
                f"{folder_name}*.nii*"  # Files starting with timepoint number
            ]
            
            for pattern in patterns:
                files = glob.glob(os.path.join(seg_folder_path, pattern))
                if files:
                    print(f"Found segmentation file: {files[0]}")
                    return files[0]
        
        # If no subfolder, look directly in the segmentation folder for files named with timepoint
        direct_patterns = [
            f"{folder_name}_seg.nii*",  # timepoint_seg.nii
            f"{folder_name}_mask.nii*",  # timepoint_mask.nii
            f"{folder_name}_tumor.nii*",  # timepoint_tumor.nii
            f"{folder_name}.nii*",  # timepoint.nii
            f"*{folder_name}*seg*.nii*",  # any file with timepoint and seg
            f"*{folder_name}*.nii*"  # any file with timepoint
        ]
        
        for pattern in direct_patterns:
            files = glob.glob(os.path.join(self.segmentation_folder, pattern))
            if files:
                print(f"Found segmentation file: {files[0]}")
                return files[0]
        
        print(f"Warning: No segmentation file found for timepoint {folder_name}")
        return None
    
    def calculate_tumor_volume(self, segmentation_file):
        """Calculate tumor volume from segmentation file"""
        if not segmentation_file or not os.path.exists(segmentation_file):
            return None
        
        try:
            # Load segmentation
            seg_img = nib.load(segmentation_file)
            seg_data = seg_img.get_fdata()
            
            # Handle 4D data
            if len(seg_data.shape) == 4:
                seg_data = seg_data[:, :, :, 0]
            
            # Count tumor voxels (using higher threshold to avoid noise)
            tumor_threshold = 0.5
            tumor_voxels = np.sum(seg_data > tumor_threshold)
            
            # Get voxel dimensions from header
            header = seg_img.header
            voxel_dims = header.get_zooms()[:3]  # x, y, z dimensions in mm
            
            # Calculate volume in mm³
            voxel_volume_mm3 = np.prod(voxel_dims)
            tumor_volume_mm3 = tumor_voxels * voxel_volume_mm3
            
            # Convert to cm³ (mL)
            tumor_volume_ml = tumor_volume_mm3 / 1000.0
            
            return {
                'volume_mm3': tumor_volume_mm3,
                'volume_ml': tumor_volume_ml,
                'voxel_count': tumor_voxels,
                'voxel_dims': voxel_dims
            }
            
        except Exception as e:
            print(f"Error calculating tumor volume: {e}")
            return None
    
    def create_tumor_overlay_colormap(self, base_colormap='gray', tumor_color='red', tumor_alpha=0.7):
        """Create a colormap that combines base image with tumor overlay"""
        from matplotlib.colors import to_rgba
        
        # Get the tumor color as RGBA
        tumor_rgba = to_rgba(tumor_color, alpha=tumor_alpha)
        
        return {
            'base_cmap': base_colormap,
            'tumor_color': tumor_rgba,
            'tumor_alpha': tumor_alpha
        }
    
    def create_mosaic_view(self, nifti_file, rows=3, cols=5, slice_gap=2, 
                          contrast='T1', window_level=None, window_width=None,
                          colormap='gray', timepoint_info="", segmentation_file=None,
                          tumor_color='red', tumor_alpha=0.7, show_contour=True,
                          contour_width=2, tumor_volume_info=None):
        """
        Create a mosaic view of brain slices with optional tumor segmentation overlay
        
        Parameters:
        - nifti_file: Path to NIFTI file
        - rows, cols: Mosaic grid dimensions
        - slice_gap: Gap between slices
        - contrast: Image contrast type (for labeling)
        - window_level, window_width: Windowing parameters
        - colormap: Matplotlib colormap for brain image
        - timepoint_info: Additional info for title
        - segmentation_file: Path to tumor segmentation file
        - tumor_color: Color for tumor overlay ('red', 'yellow', 'cyan', etc.)
        - tumor_alpha: Transparency of tumor overlay (0-1)
        - show_contour: Whether to show tumor contour instead of filled overlay
        - contour_width: Width of contour lines
        - tumor_volume_info: Dictionary with tumor volume information
        """
        if not nifti_file or not os.path.exists(nifti_file):
            print(f"Error: File not found: {nifti_file}")
            return None
            
        # Load NIFTI file
        img = nib.load(nifti_file)
        data = img.get_fdata()
        
        # Load segmentation if provided
        seg_data = None
        if segmentation_file and os.path.exists(segmentation_file):
            seg_img = nib.load(segmentation_file)
            seg_data = seg_img.get_fdata()
            print(f"Loaded segmentation: {segmentation_file}")
        
        # Handle different orientations - use axial slices (last dimension)
        if len(data.shape) == 4:
            data = data[:, :, :, 0]  # Take first volume if 4D
        
        if seg_data is not None and len(seg_data.shape) == 4:
            seg_data = seg_data[:, :, :, 0]  # Take first volume if 4D
        
        # Get slice indices
        total_slices = data.shape[2]
        # Select slices from central 60% of the volume to avoid edges
        start_idx = int(total_slices * 0.2)
        end_idx = int(total_slices * 0.8)

        # Ensure enough slices
        num_slices = rows * cols
        slice_indices = np.linspace(start_idx, end_idx, num_slices, dtype=int)
        
        # Create figure
        fig = plt.figure(figsize=(cols * 3, rows * 3))
        fig.patch.set_facecolor('black')  # Make figure background black

        gs = GridSpec(rows, cols, figure=fig, hspace=0.1, wspace=0.1)
        
        # Set windowing if provided
        if window_level is not None and window_width is not None:
            vmin = window_level - window_width / 2
            vmax = window_level + window_width / 2
        else:
            # Auto-windowing
            vmin, vmax = np.percentile(data[data > 0], [2, 98])
        
        for i, slice_idx in enumerate(slice_indices):
            row = i // cols
            col = i % cols
            
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor('black')  # Make axes (subplot) background black
            
            # Get slice and flip for proper orientation
            slice_data = data[:, :, slice_idx].T
            slice_data = np.flipud(slice_data)
            
            # Display brain slice
            im = ax.imshow(slice_data, cmap=colormap, vmin=vmin, vmax=vmax, 
                          aspect='equal', interpolation='nearest')
            
            # Add tumor overlay if segmentation is available
            if seg_data is not None:
                seg_slice = seg_data[:, :, slice_idx].T
                seg_slice = np.flipud(seg_slice)
                
                # Much more robust tumor detection
                # First, check the actual range of values in this slice
                slice_max = np.max(seg_slice)
                slice_min = np.min(seg_slice[seg_slice > 0]) if np.any(seg_slice > 0) else 0
                
                # Adaptive threshold based on actual data values
                if slice_max > 0.8:
                    tumor_threshold = 0.5  # Standard threshold for binary masks
                elif slice_max > 0.1:
                    tumor_threshold = slice_max * 0.5  # Use 50% of max value
                else:
                    tumor_threshold = 0.05  # Very low threshold for normalized data
                
                tumor_mask = seg_slice > tumor_threshold
                
                # Require substantial tumor area (at least 25 pixels) and check that
                # the tumor region actually corresponds to visible brain tissue
                if np.sum(tumor_mask) > 25:
                    # Additional check: make sure tumor is in a region with brain tissue
                    # (avoid artifacts in background regions)
                    brain_slice = slice_data
                    brain_threshold = np.percentile(brain_slice[brain_slice > 0], 10) if np.any(brain_slice > 0) else 0
                    
                    # Only show tumor if it overlaps with brain tissue
                    brain_mask = brain_slice > brain_threshold
                    tumor_in_brain = tumor_mask & brain_mask
                    
                    if np.sum(tumor_in_brain) > 15:  # At least 15 pixels of tumor in brain tissue
                        if show_contour:
                            # Show tumor contour with adaptive threshold
                            contours = ax.contour(seg_slice, levels=[tumor_threshold], colors=[tumor_color], 
                                                linewidths=contour_width, alpha=tumor_alpha)
                        else:
                            # Show filled tumor overlay only where there's brain tissue
                            tumor_overlay = np.zeros((*tumor_in_brain.shape, 4))  # RGBA
                            from matplotlib.colors import to_rgba
                            tumor_rgba = to_rgba(tumor_color, alpha=tumor_alpha)
                            tumor_overlay[tumor_in_brain] = tumor_rgba
                            ax.imshow(tumor_overlay, aspect='equal', interpolation='nearest')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Slice {slice_idx}', fontsize=8, color='white')
        
        # Add title with timepoint info, segmentation status, and tumor volume
        seg_status = " (with tumor overlay)" if seg_data is not None else ""
        volume_info = ""
        if tumor_volume_info:
            volume_info = f" | Volume: {tumor_volume_info['volume_ml']:.2f} mL"
        
        title = f'{contrast} - {timepoint_info}{seg_status}{volume_info}'
        fig.suptitle(title, fontsize=16, fontweight='bold', color='white')
        
        plt.tight_layout()
        return fig
    
    def create_flipbook_slides(self, contrast_type, rows=3, cols=5, 
                              window_level=None, window_width=None,
                              colormap='gray', tumor_color='red', 
                              tumor_alpha=0.7, show_contour=True,
                              contour_width=2):
        """Create individual slides for the flipbook with tumor overlay"""
        if not self.timepoint_folders:
            self.find_timepoint_folders()
            
        if not self.contrast_types:
            self.find_contrast_types()
            
        if contrast_type not in self.contrast_types:
            print(f"Error: Contrast type '{contrast_type}' not found.")
            print(f"Available types: {self.contrast_types}")
            return []
        
        slide_files = []
        volume_data = []  # Store volume information for all timepoints
        
        for i, tp_folder in enumerate(self.timepoint_folders):
            timepoint_name = tp_folder['folder_name']
            print(f"Creating slide {i+1}/{len(self.timepoint_folders)}: {timepoint_name}")
            
            # Get the NIFTI file for this timepoint and contrast
            nifti_file = self.get_file_for_timepoint_and_contrast(tp_folder, contrast_type)
            
            if not nifti_file:
                print(f"Skipping timepoint {timepoint_name} - no {contrast_type} file found")
                continue
            
            # Get segmentation file for this timepoint
            seg_file = self.get_segmentation_file(tp_folder)
            
            # Calculate tumor volume
            tumor_volume_info = None
            if seg_file:
                tumor_volume_info = self.calculate_tumor_volume(seg_file)
                if tumor_volume_info:
                    volume_data.append({
                        'timepoint': timepoint_name,
                        'volume_ml': tumor_volume_info['volume_ml'],
                        'volume_mm3': tumor_volume_info['volume_mm3'],
                        'voxel_count': tumor_volume_info['voxel_count']
                    })
                    print(f"  Tumor volume: {tumor_volume_info['volume_ml']:.2f} mL")
                
            # Create mosaic view with tumor overlay and volume info
            fig = self.create_mosaic_view(
                nifti_file, rows=rows, cols=cols, contrast=contrast_type,
                window_level=window_level, window_width=window_width,
                colormap=colormap, timepoint_info=f"Timepoint {timepoint_name}",
                segmentation_file=seg_file, tumor_color=tumor_color,
                tumor_alpha=tumor_alpha, show_contour=show_contour,
                contour_width=contour_width, tumor_volume_info=tumor_volume_info
            )
            
            if fig is None:
                continue
                
            # Save slide
            slide_file = os.path.join(self.output_folder, f'slide_{i+1:03d}_{contrast_type}_{timepoint_name}.png')
            
            # Force overwrite by removing existing file
            if os.path.exists(slide_file):
                os.remove(slide_file)
                print(f"Removed existing file: {slide_file}")
            
            plt.savefig(slide_file, dpi=150, bbox_inches='tight', facecolor='black', edgecolor='none')
            plt.close(fig)
            
            print(f"Saved new slide: {slide_file}")
            slide_files.append(slide_file)
        
        # Print volume progression summary
        if volume_data:
            print(f"\n=== TUMOR VOLUME PROGRESSION ({contrast_type}) ===")
            for vol_info in volume_data:
                print(f"Timepoint {vol_info['timepoint']}: {vol_info['volume_ml']:.2f} mL")
            
            # Calculate volume change
            if len(volume_data) > 1:
                initial_vol = volume_data[0]['volume_ml']
                final_vol = volume_data[-1]['volume_ml']
                vol_change = final_vol - initial_vol
                vol_percent_change = (vol_change / initial_vol) * 100 if initial_vol > 0 else 0
                print(f"Volume change: {vol_change:+.2f} mL ({vol_percent_change:+.1f}%)")
        
        return slide_files
    
    
    def generate_flipbook(self, contrast_type, rows=3, cols=5, 
                         window_level=None, window_width=None,
                         colormap='gray', title=None, tumor_color='red',
                         tumor_alpha=0.7, show_contour=True, contour_width=2):
        """
        Generate complete flipbook for a specific contrast type with tumor overlay
        
        Parameters:
        - contrast_type: Type of contrast (T1, T2, T1CE, FL, etc.)
        - rows, cols: Mosaic grid dimensions
        - window_level, window_width: Windowing parameters
        - colormap: Matplotlib colormap for brain image
        - title: Flipbook title
        - tumor_color: Color for tumor overlay ('red', 'yellow', 'cyan', 'green', 'magenta')
        - tumor_alpha: Transparency of tumor overlay (0-1)
        - show_contour: If True, show tumor contour; if False, show filled overlay
        - contour_width: Width of contour lines
        """
        if title is None:
            seg_suffix = " with Tumor Overlay" if self.segmentation_folder else ""
            title = f"Brain Tumor Flipbook - {contrast_type}{seg_suffix}"
            
        print("=== Enhanced Brain Tumor Flipbook Generator ===")
        print(f"Input folder: {self.registered_folder}")
        print(f"Segmentation folder: {self.segmentation_folder}")
        print(f"Output folder: {self.output_folder}")
        print(f"Contrast type: {contrast_type}")
        print(f"Tumor visualization: {'Contour' if show_contour else 'Filled overlay'} in {tumor_color}")
        
        # Find timepoint folders and contrast types
        self.find_timepoint_folders()
        self.find_contrast_types()
        
        # Create slides with tumor overlay
        slide_files = self.create_flipbook_slides(
            contrast_type=contrast_type, rows=rows, cols=cols,
            window_level=window_level, window_width=window_width,
            colormap=colormap, tumor_color=tumor_color,
            tumor_alpha=tumor_alpha, show_contour=show_contour,
            contour_width=contour_width
        )
        
        if not slide_files:
            print("Error: No slides were created. Check your contrast type and files.")
            return []
        
        print(f"\nFlipbook generated successfully!")
        print(f"- {len(slide_files)} slides created")
        print(f"- Individual slides saved in: {self.output_folder}")
        
        return slide_files
    
    def generate_all_contrast_flipbooks(self, rows=3, cols=5, 
                                       window_level=None, window_width=None,
                                       colormap='gray', tumor_color='red',
                                       tumor_alpha=0.7, show_contour=True, 
                                       contour_width=2):
        """Generate flipbooks for all available contrast types with tumor overlay"""
        self.find_timepoint_folders()
        self.find_contrast_types()
        
        print("=== Generating Flipbooks for All Contrast Types ===")
        print(f"Found contrast types: {self.contrast_types}")
        print(f"Tumor visualization: {'Contour' if show_contour else 'Filled overlay'} in {tumor_color}")
        
        results = {}
        
        for i, contrast in enumerate(self.contrast_types):
            print(f"\n=== Generating flipbook {i+1}/{len(self.contrast_types)} for {contrast} ===")
            
            # Create contrast-specific output folder
            contrast_output_folder = os.path.join(self.output_folder, f"{contrast}_flipbook")
            os.makedirs(contrast_output_folder, exist_ok=True)
            
            # Temporarily change the output folder for this contrast
            original_output = self.output_folder
            self.output_folder = contrast_output_folder
            
            slide_files = self.generate_flipbook(
                contrast_type=contrast, rows=rows, cols=cols,
                window_level=window_level, window_width=window_width,
                colormap=colormap, tumor_color=tumor_color,
                tumor_alpha=tumor_alpha, show_contour=show_contour,
                contour_width=contour_width
            )
            
            # Restore original output folder
            self.output_folder = original_output
            
            results[contrast] = {
                'slide_files': slide_files,
                'output_folder': contrast_output_folder
            }
        
        print(f"\n=== ALL FLIPBOOKS GENERATED SUCCESSFULLY ===")
        print(f"Generated {len(results)} flipbooks for contrast types: {list(results.keys())}")
        print(f"\n*** FILES SAVED TO: {os.path.abspath(self.output_folder)} ***")
        
        # Print absolute paths for verification
        for contrast, result in results.items():
            if result['slide_files']:
                print(f"  {contrast}: {len(result['slide_files'])} slides in {os.path.abspath(result['output_folder'])}")
        
        return results
    
    def generate_all_contrast_flipbooks_slides_only(self, rows=3, cols=5, 
                                                   window_level=None, window_width=None,
                                                   colormap='gray', tumor_color='red',
                                                   tumor_alpha=0.7, show_contour=True, 
                                                   contour_width=2, create_pptx=True):
        """Generate flipbook slides for all available contrast types with optional PPTX output"""
        self.find_timepoint_folders()
        self.find_contrast_types()
        
        print("=== Generating Flipbook Slides for All Contrast Types ===")
        print(f"Found contrast types: {self.contrast_types}")
        print(f"Tumor visualization: {'Contour' if show_contour else 'Filled overlay'} in {tumor_color}")
        print(f"PowerPoint output: {'Enabled' if create_pptx and PPTX_AVAILABLE else 'Disabled'}")
        
        results = {}
        
        for i, contrast in enumerate(self.contrast_types):
            print(f"\n=== Generating slides {i+1}/{len(self.contrast_types)} for {contrast} ===")
            
            # Create contrast-specific output folder
            contrast_output_folder = os.path.join(self.output_folder, f"{contrast}_flipbook")
            os.makedirs(contrast_output_folder, exist_ok=True)
            
            # Temporarily change the output folder for this contrast
            original_output = self.output_folder
            self.output_folder = contrast_output_folder
            
            # Generate slides
            slide_files = self.create_flipbook_slides(
                contrast_type=contrast, rows=rows, cols=cols,
                window_level=window_level, window_width=window_width,
                colormap=colormap, tumor_color=tumor_color,
                tumor_alpha=tumor_alpha, show_contour=show_contour,
                contour_width=contour_width
            )
            
            # Generate PowerPoint if requested and available
            pptx_file = None
            if create_pptx and PPTX_AVAILABLE and slide_files:
                seg_suffix = " with Tumor Overlay" if self.segmentation_folder else ""
                title = f"Brain Tumor Flipbook - {contrast}{seg_suffix}"
                pptx_file = self.create_powerpoint_flipbook(slide_files, contrast, title=title)
            
            # Restore original output folder
            self.output_folder = original_output
            
            results[contrast] = {
                'slide_files': slide_files,
                'output_folder': contrast_output_folder,
                'slide_count': len(slide_files),
                'pptx_file': pptx_file
            }
        
        print(f"\n=== ALL FLIPBOOK SLIDES GENERATED SUCCESSFULLY ===")
        print(f"Generated slides for {len(results)} contrast types: {list(results.keys())}")
        print(f"Slides saved to: {os.path.abspath(self.output_folder)}")
        
        # Print absolute paths for verification
        pptx_count = 0
        for contrast, result in results.items():
            if result['slide_files']:
                print(f"  {contrast}: {len(result['slide_files'])} slides in {os.path.abspath(result['output_folder'])}")
                if result.get('pptx_file'):
                    print(f"    PowerPoint: {os.path.abspath(result['pptx_file'])}")
                    pptx_count += 1
        
        if pptx_count > 0:
            print(f"\n🎉 Generated {pptx_count} PowerPoint flipbooks ready for presentation!")
        
        return results
    
    def create_powerpoint_flipbook(self, slide_files, contrast_type, title="Brain Tumor Flipbook"):
        """Create a PowerPoint presentation from slide files with black background and centered images"""
        if not PPTX_AVAILABLE:
            print("Error: python-pptx not installed. Cannot create PowerPoint file.")
            print("Install with: pip install python-pptx")
            return None
            
        if not slide_files:
            print("No slide files provided for PowerPoint creation")
            return None
        
        from pptx.dml.color import RGBColor
        from PIL import Image as PILImage
        
        # Create presentation
        prs = Presentation()
        
        # Set slide dimensions for widescreen (16:9)
        prs.slide_width = Inches(13.33)
        prs.slide_height = Inches(7.5)
        
        # Add title slide with black background
        title_slide_layout = prs.slide_layouts[0]  # Title slide layout
        slide = prs.slides.add_slide(title_slide_layout)
        
        # Set black background
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor(0, 0, 0)  # Black
        
        title_placeholder = slide.shapes.title
        subtitle_placeholder = slide.placeholders[1]
        
        title_placeholder.text = title
        seg_info = " with Tumor Overlay" if self.segmentation_folder else ""
        subtitle_placeholder.text = f"{contrast_type} Contrast{seg_info}\nLongitudinal Brain Tumor Analysis\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Make title text white
        title_placeholder.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
        subtitle_placeholder.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
        
        # Add slides for each timepoint
        blank_slide_layout = prs.slide_layouts[6]  # Blank slide layout
        
        for i, slide_file in enumerate(slide_files):
            if not os.path.exists(slide_file):
                print(f"Warning: Slide file not found: {slide_file}")
                continue
                
            # Extract timepoint from filename
            timepoint_match = re.search(r'(\d+)\.png$', os.path.basename(slide_file))
            timepoint = timepoint_match.group(1) if timepoint_match else str(i+1)
            
            # Create new slide with black background
            slide = prs.slides.add_slide(blank_slide_layout)
            
            # Set black background
            background = slide.background
            fill = background.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor(0, 0, 0)  # Black
            
            # Add image centered without distortion
            try:
                # Get original image dimensions
                with PILImage.open(slide_file) as img:
                    img_width_px, img_height_px = img.size
                    aspect_ratio = img_width_px / img_height_px
                
                # Calculate slide dimensions
                slide_width = Inches(13.33)
                slide_height = Inches(7.5)
                slide_aspect_ratio = 13.33 / 7.5
                
                # Calculate image size to fit without distortion
                if aspect_ratio > slide_aspect_ratio:
                    # Image is wider than slide aspect ratio - fit to width
                    img_width = slide_width * 0.9  # Leave 10% margin
                    img_height = img_width / aspect_ratio
                else:
                    # Image is taller than slide aspect ratio - fit to height
                    img_height = slide_height * 0.9  # Leave 10% margin
                    img_width = img_height * aspect_ratio
                
                # Center the image
                img_left = (slide_width - img_width) / 2
                img_top = (slide_height - img_height) / 2
                
                slide.shapes.add_picture(slide_file, img_left, img_top, img_width, img_height)
                print(f"Added slide {i+1}: Timepoint {timepoint} (centered, no distortion)")
                
            except Exception as e:
                print(f"Error adding image {slide_file}: {e}")
                continue
        
        # Save PowerPoint file
        pptx_file = os.path.join(self.output_folder, f'flipbook_{contrast_type}.pptx')
        
        try:
            prs.save(pptx_file)
            print(f"PowerPoint flipbook created: {pptx_file}")
            return pptx_file
        except Exception as e:
            print(f"Error saving PowerPoint file: {e}")
            return None
    
    def generate_flipbook_with_pptx(self, contrast_type, rows=3, cols=5, 
                                   window_level=None, window_width=None,
                                   colormap='gray', title=None, tumor_color='red',
                                   tumor_alpha=0.7, show_contour=True, contour_width=2):
        """
        Generate complete flipbook with PowerPoint output
        """
        if title is None:
            seg_suffix = " with Tumor Overlay" if self.segmentation_folder else ""
            title = f"Brain Tumor Flipbook - {contrast_type}{seg_suffix}"
            
        print("=== Brain Tumor Flipbook Generator with PowerPoint Output ===")
        print(f"Input folder: {self.registered_folder}")
        print(f"Segmentation folder: {self.segmentation_folder}")
        print(f"Output folder: {self.output_folder}")
        print(f"Contrast type: {contrast_type}")
        print(f"Tumor visualization: {'Contour' if show_contour else 'Filled overlay'} in {tumor_color}")
        print(f"PowerPoint output: {'Available' if PPTX_AVAILABLE else 'Not available (install python-pptx)'}")
        
        # Find timepoint folders and contrast types
        self.find_timepoint_folders()
        self.find_contrast_types()
        
        # Create slides
        slide_files = self.create_flipbook_slides(
            contrast_type=contrast_type, rows=rows, cols=cols,
            window_level=window_level, window_width=window_width,
            colormap=colormap, tumor_color=tumor_color,
            tumor_alpha=tumor_alpha, show_contour=show_contour,
            contour_width=contour_width
        )
        
        if not slide_files:
            print("Error: No slides were created. Check your contrast type and files.")
            return None, []
        
        # Create PowerPoint flipbook
        pptx_file = None
        if PPTX_AVAILABLE:
            pptx_file = self.create_powerpoint_flipbook(slide_files, contrast_type, title=title)
        
        print(f"\nFlipbook generated successfully!")
        print(f"- {len(slide_files)} slides created")
        if pptx_file:
            print(f"- PowerPoint flipbook: {pptx_file}")
        print(f"- Individual slides saved in: {self.output_folder}")
        
        return pptx_file, slide_files


# Enhanced convenience functions
def create_brain_flipbook_with_segmentation(registered_folder, segmentation_folder, 
                                          contrast_type, output_folder="flipbooks", 
                                          rows=3, cols=5, window_level=None, window_width=None,
                                          colormap='gray', title=None, tumor_color='red',
                                          tumor_alpha=0.7, show_contour=True, contour_width=2):
    """
    Convenience function to create a brain flipbook with tumor segmentation overlay
    
    Parameters:
    - registered_folder: Path to folder containing timepoint subfolders (5885, 5972, 6070, etc.)
    - segmentation_folder: Path to folder containing segmentation maps (same structure)
    - contrast_type: Type of contrast (T1, T2, T1CE, FL, etc.)
    - output_folder: Output folder for flipbook
    - rows, cols: Mosaic grid size (default 3x5 = 15 slices)
    - window_level, window_width: Manual windowing (None for auto)
    - colormap: Matplotlib colormap for brain ('gray', 'bone', etc.)
    - title: Flipbook title
    - tumor_color: Color for tumor ('red', 'yellow', 'cyan', 'green', 'magenta')
    - tumor_alpha: Transparency of tumor overlay (0-1)
    - show_contour: If True, show contour; if False, show filled overlay
    - contour_width: Width of contour lines
    """
    generator = BrainFlipbookGenerator(registered_folder, segmentation_folder, output_folder)
    return generator.generate_flipbook(
        contrast_type=contrast_type, rows=rows, cols=cols,
        window_level=window_level, window_width=window_width,
        colormap=colormap, title=title, tumor_color=tumor_color,
        tumor_alpha=tumor_alpha, show_contour=show_contour, contour_width=contour_width
    )

def create_all_contrast_flipbooks_with_segmentation(registered_folder, segmentation_folder,
                                                  output_folder="all_flipbooks", rows=3, cols=5, 
                                                  window_level=None, window_width=None,
                                                  colormap='gray', tumor_color='red',
                                                  tumor_alpha=0.7, show_contour=True, 
                                                  contour_width=2):
    """
    Create flipbooks for ALL available contrast types with tumor segmentation overlay
    
    Parameters:
    - registered_folder: Path to folder containing timepoint subfolders
    - segmentation_folder: Path to folder containing segmentation maps
    - output_folder: Output folder for all flipbooks (will create subfolders for each contrast)
    - rows, cols: Mosaic grid size
    - window_level, window_width: Manual windowing (None for auto)
    - colormap: Matplotlib colormap for brain
    - tumor_color: Color for tumor overlay
    - tumor_alpha: Transparency of tumor overlay (0-1)
    - show_contour: If True, show contour; if False, show filled overlay
    - contour_width: Width of contour lines
    """
    generator = BrainFlipbookGenerator(registered_folder, segmentation_folder, output_folder)
    return generator.generate_all_contrast_flipbooks(
        rows=rows, cols=cols, window_level=window_level, 
        window_width=window_width, colormap=colormap,
        tumor_color=tumor_color, tumor_alpha=tumor_alpha, 
        show_contour=show_contour, contour_width=contour_width
    )

# Example usage:
if __name__ == "__main__":
    # Example usage with segmentation overlay for ALL contrasts
    registered_folder = r"C:\Users\smita\registered"  # Your registered images folder
    segmentation_folder = r"C:\Users\smita\segmentations"  # Your segmentation folder
    
    # Create flipbooks for ALL available contrasts (T1, T2, T1CE, FL)
    print("=== Creating flipbooks for ALL contrast types ===")
    results = create_all_contrast_flipbooks_with_segmentation(
        registered_folder=registered_folder,
        segmentation_folder=segmentation_folder,
        output_folder="all_contrast_flipbooks",
        tumor_color='red',          # Red tumor overlay
        show_contour=True,          # Show contour instead of filled
        contour_width=2,            # Contour line width
        tumor_alpha=0.8             # Opacity of overlay
    )
    
    print(f"\n*** SUCCESS! Generated flipbooks for: {list(results.keys())}")
    print(f"*** All flipbooks saved in separate folders")
    
    # Alternative: Create individual flipbook for specific contrast
    # slides = create_brain_flipbook_with_segmentation(
    #     registered_folder=registered_folder,
    #     segmentation_folder=segmentation_folder,
    #     contrast_type="T1CE",  # Just T1CE
    #     output_folder="T1CE_only",
    #     tumor_color='yellow',
    #     show_contour=False,     # Filled overlay instead
    #     tumor_alpha=0.5
    # )