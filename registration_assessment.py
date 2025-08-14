"""
Registration Assessment Module for AutoFlipbook

This module provides comprehensive quantitative assessment of registration quality,
including metrics tables, difference maps, and contour evolution analysis.

Author: AutoFlipbook Pipeline
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from scipy import ndimage
from scipy.stats import pearsonr
from sklearn.metrics import normalized_mutual_info_score
from skimage.metrics import structural_similarity as ssim
from skimage.measure import find_contours
import warnings
warnings.filterwarnings('ignore')


class RegistrationAssessment:
    """
    Comprehensive registration quality assessment and longitudinal analysis
    """
    
    def __init__(self, registered_folder, segmentation_folder=None, output_folder="assessment"):
        """
        Initialize registration assessment
        
        Parameters:
        - registered_folder: Path to folder containing registered images
        - segmentation_folder: Path to folder containing tumor segmentations
        - output_folder: Output folder for assessment results
        """
        self.registered_folder = registered_folder
        self.segmentation_folder = segmentation_folder
        self.output_folder = output_folder
        self.timepoint_folders = []
        self.contrast_types = []
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Find timepoints and contrasts
        self.find_timepoint_folders()
        self.find_contrast_types()
        
    def find_timepoint_folders(self):
        """Find all timepoint folders (numeric or date format)"""
        import re
        from datetime import datetime
        
        folders = []
        date_pattern = re.compile(r'^(\d{4})-(\d{2})-(\d{2})$')
        
        for item in os.listdir(self.registered_folder):
            item_path = os.path.join(self.registered_folder, item)
            
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
        
        print(f"Found {len(folders)} timepoint folders for assessment:")
        for tp in folders:
            print(f"  {tp['folder_name']}: {tp['folder']}")
        
        return folders
    
    def find_contrast_types(self):
        """Find available contrast types across all timepoints"""
        import re
        import glob
        
        contrast_types = set()
        
        for tp_folder in self.timepoint_folders:
            nifti_files = glob.glob(os.path.join(tp_folder['folder'], "*.nii*"))
            
            for file in nifti_files:
                basename = os.path.basename(file)
                # Parse contrast type from filename patterns:
                # Legacy: "6070_T1_to_5885T1CE.nii"
                # Date-based: "2000-03-25_FL_to_2000-03-25FL.nii.gz"
                match = re.search(r'^(?:\d+|\d{4}-\d{2}-\d{2})_([^_]+)_to_', basename)
                if match:
                    contrast_types.add(match.group(1))
        
        self.contrast_types = sorted(list(contrast_types))
        print(f"Available contrast types for assessment: {self.contrast_types}")
        return self.contrast_types
    
    def get_file_for_timepoint_and_contrast(self, timepoint_folder, contrast_type):
        """Get the NIFTI file for a specific timepoint and contrast type"""
        import glob
        
        folder_name = os.path.basename(timepoint_folder['folder'])
        
        # Look for file with pattern: {timepoint}_{contrast}_to_{reference}.nii
        pattern = f"{folder_name}_{contrast_type}_to_*.nii*"
        files = glob.glob(os.path.join(timepoint_folder['folder'], pattern))
        
        if files:
            return files[0]  # Return first match
        else:
            print(f"Warning: No {contrast_type} file found for timepoint {folder_name}")
            return None
    
    def calculate_registration_metrics(self, img1_path, img2_path):
        """
        Calculate registration quality metrics between two images
        
        Returns:
        - Dictionary with various similarity metrics
        """
        # Load images
        img1 = nib.load(img1_path)
        img2 = nib.load(img2_path)
        
        data1 = img1.get_fdata()
        data2 = img2.get_fdata()
        
        # Ensure same dimensions
        if data1.shape != data2.shape:
            print(f"Warning: Shape mismatch {data1.shape} vs {data2.shape}")
            return None
        
        # Create brain mask (non-zero voxels)
        mask = (data1 > 0) & (data2 > 0)
        
        if np.sum(mask) == 0:
            print("Warning: No overlapping brain tissue found")
            return None
        
        # Extract brain voxels
        brain1 = data1[mask]
        brain2 = data2[mask]
        
        # Calculate metrics
        metrics = {}
        
        # Pearson Correlation
        try:
            corr, p_val = pearsonr(brain1.flatten(), brain2.flatten())
            metrics['pearson_correlation'] = corr
            metrics['pearson_p_value'] = p_val
        except:
            metrics['pearson_correlation'] = np.nan
            metrics['pearson_p_value'] = np.nan
        
        # Normalized Mutual Information
        try:
            # Discretize for MI calculation
            bins = 64
            hist_range = [(brain1.min(), brain1.max()), (brain2.min(), brain2.max())]
            brain1_disc = np.digitize(brain1, np.linspace(brain1.min(), brain1.max(), bins))
            brain2_disc = np.digitize(brain2, np.linspace(brain2.min(), brain2.max(), bins))
            nmi = normalized_mutual_info_score(brain1_disc, brain2_disc)
            metrics['normalized_mutual_info'] = nmi
        except:
            metrics['normalized_mutual_info'] = np.nan
        
        # Structural Similarity Index (SSIM) - on 2D slices
        try:
            ssim_values = []
            for z in range(data1.shape[2]):
                slice1 = data1[:, :, z]
                slice2 = data2[:, :, z]
                if np.sum(slice1) > 0 and np.sum(slice2) > 0:
                    # Normalize slices
                    slice1 = (slice1 - slice1.min()) / (slice1.max() - slice1.min() + 1e-8)
                    slice2 = (slice2 - slice2.min()) / (slice2.max() - slice2.min() + 1e-8)
                    ssim_val = ssim(slice1, slice2, data_range=1.0)
                    ssim_values.append(ssim_val)
            
            metrics['mean_ssim'] = np.mean(ssim_values) if ssim_values else np.nan
            metrics['std_ssim'] = np.std(ssim_values) if ssim_values else np.nan
        except:
            metrics['mean_ssim'] = np.nan
            metrics['std_ssim'] = np.nan
        
        # Mean Squared Error
        mse = np.mean((brain1 - brain2) ** 2)
        metrics['mean_squared_error'] = mse
        
        # Mean Absolute Error
        mae = np.mean(np.abs(brain1 - brain2))
        metrics['mean_absolute_error'] = mae
        
        return metrics
    
    def generate_registration_metrics_table(self, contrast_type='T1CE', reference_idx=0):
        """
        Generate comprehensive registration quality metrics table
        
        Parameters:
        - contrast_type: Image contrast to analyze
        - reference_idx: Index of reference timepoint (default: first timepoint)
        """
        if not self.timepoint_folders or len(self.timepoint_folders) < 2:
            print("Need at least 2 timepoints for registration assessment")
            return None
        
        if contrast_type not in self.contrast_types:
            print(f"Contrast {contrast_type} not found. Available: {self.contrast_types}")
            return None
        
        reference_tp = self.timepoint_folders[reference_idx]
        reference_file = self.get_file_for_timepoint_and_contrast(reference_tp, contrast_type)
        
        if not reference_file:
            print(f"Reference file not found for {contrast_type}")
            return None
        
        # Calculate metrics for each timepoint vs reference
        results = []
        
        for i, tp_folder in enumerate(self.timepoint_folders):
            timepoint_name = tp_folder['folder_name']
            
            if i == reference_idx:
                # Reference vs itself
                metrics = {
                    'timepoint': timepoint_name,
                    'vs_reference': reference_tp['folder_name'],
                    'pearson_correlation': 1.0,
                    'pearson_p_value': 0.0,
                    'normalized_mutual_info': 1.0,
                    'mean_ssim': 1.0,
                    'std_ssim': 0.0,
                    'mean_squared_error': 0.0,
                    'mean_absolute_error': 0.0,
                    'registration_quality': 'Reference'
                }
            else:
                target_file = self.get_file_for_timepoint_and_contrast(tp_folder, contrast_type)
                
                if target_file:
                    print(f"Calculating metrics: {timepoint_name} vs {reference_tp['folder_name']}")
                    metrics = self.calculate_registration_metrics(reference_file, target_file)
                    
                    if metrics:
                        metrics['timepoint'] = timepoint_name
                        metrics['vs_reference'] = reference_tp['folder_name']
                        
                        # Assign registration quality score
                        corr = metrics.get('pearson_correlation', 0)
                        if corr > 0.9:
                            quality = 'Excellent'
                        elif corr > 0.8:
                            quality = 'Good'
                        elif corr > 0.7:
                            quality = 'Fair'
                        else:
                            quality = 'Poor'
                        metrics['registration_quality'] = quality
                    else:
                        metrics = {
                            'timepoint': timepoint_name,
                            'vs_reference': reference_tp['folder_name'],
                            'registration_quality': 'Failed'
                        }
                else:
                    metrics = {
                        'timepoint': timepoint_name,
                        'vs_reference': reference_tp['folder_name'],
                        'registration_quality': 'File Not Found'
                    }
            
            results.append(metrics)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        csv_file = os.path.join(self.output_folder, f'registration_metrics_{contrast_type}.csv')
        df.to_csv(csv_file, index=False)
        
        # Create formatted table plot
        self.create_metrics_table_plot(df, contrast_type)
        
        print(f"Registration metrics saved to: {csv_file}")
        return df
    
    def create_metrics_table_plot(self, df, contrast_type):
        """Create a formatted table visualization of metrics"""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Select key columns for display
        display_cols = ['timepoint', 'pearson_correlation', 'normalized_mutual_info', 
                       'mean_ssim', 'mean_squared_error', 'registration_quality']
        
        display_df = df[display_cols].copy()
        
        # Round numerical values
        for col in ['pearson_correlation', 'normalized_mutual_info', 'mean_ssim', 'mean_squared_error']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)
        
        # Create table
        table = ax.table(cellText=display_df.values,
                        colLabels=display_df.columns,
                        cellLoc='center',
                        loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Color code quality column
        for i in range(len(display_df)):
            quality = display_df.iloc[i]['registration_quality']
            if quality == 'Excellent':
                color = '#90EE90'  # Light green
            elif quality == 'Good':
                color = '#FFFFE0'  # Light yellow
            elif quality == 'Fair':
                color = '#FFB6C1'  # Light pink
            elif quality == 'Poor':
                color = '#FFB6C1'  # Light pink
            else:
                color = '#D3D3D3'  # Light gray
            
            table[(i+1, display_cols.index('registration_quality'))].set_facecolor(color)
        
        plt.title(f'Registration Quality Assessment - {contrast_type}', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Save plot
        plot_file = os.path.join(self.output_folder, f'registration_metrics_table_{contrast_type}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Metrics table plot saved to: {plot_file}")
        return plot_file
    
    def create_difference_map(self, baseline_path, followup_path, seg_baseline=None, seg_followup=None, 
                             contrast_type='T1CE', colormap_style='red_blue'):
        """
        Create voxelwise difference map with tumor growth/shrinkage overlay
        
        Parameters:
        - baseline_path: Path to baseline image
        - followup_path: Path to follow-up image
        - seg_baseline: Path to baseline segmentation (optional)
        - seg_followup: Path to follow-up segmentation (optional)
        - colormap_style: 'red_blue', 'viridis', 'plasma', 'coolwarm'
        """
        # Load baseline and follow-up images
        baseline_img = nib.load(baseline_path)
        followup_img = nib.load(followup_path)
        
        baseline_data = baseline_img.get_fdata()
        followup_data = followup_img.get_fdata()
        
        if baseline_data.shape != followup_data.shape:
            print(f"Shape mismatch: {baseline_data.shape} vs {followup_data.shape}")
            return None
        
        # Calculate difference map
        diff_map = followup_data - baseline_data
        
        # Load segmentations if provided
        seg_baseline_data = None
        seg_followup_data = None
        
        if seg_baseline and os.path.exists(seg_baseline):
            seg_baseline_data = nib.load(seg_baseline).get_fdata()
        
        if seg_followup and os.path.exists(seg_followup):
            seg_followup_data = nib.load(seg_followup).get_fdata()
        
        return self.create_difference_map_visualization(
            baseline_data, followup_data, diff_map, 
            seg_baseline_data, seg_followup_data,
            contrast_type, colormap_style
        )
    
    def create_difference_map_visualization(self, baseline_data, followup_data, diff_map,
                                          seg_baseline_data, seg_followup_data,
                                          contrast_type, colormap_style):
        """Create visualization of difference maps with tumor overlays"""
        
        # Select representative slices
        z_center = diff_map.shape[2] // 2
        slice_indices = np.linspace(z_center - 20, z_center + 20, 8, dtype=int)
        slice_indices = np.clip(slice_indices, 0, diff_map.shape[2] - 1)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor('black')
        
        gs = GridSpec(3, 8, figure=fig, hspace=0.3, wspace=0.1)
        
        # Define colormaps
        if colormap_style == 'red_blue':
            cmap = plt.cm.RdBu_r
        elif colormap_style == 'viridis':
            cmap = plt.cm.viridis
        elif colormap_style == 'plasma':
            cmap = plt.cm.plasma
        else:  # coolwarm
            cmap = plt.cm.coolwarm
        
        # Calculate difference map range for consistent scaling
        brain_mask = (baseline_data > 0) | (followup_data > 0)
        diff_brain = diff_map[brain_mask]
        diff_range = np.percentile(np.abs(diff_brain), 95)
        
        for i, slice_idx in enumerate(slice_indices):
            # Row 1: Baseline
            ax1 = fig.add_subplot(gs[0, i])
            baseline_slice = baseline_data[:, :, slice_idx].T
            ax1.imshow(baseline_slice, cmap='gray', aspect='equal')
            ax1.set_title(f'Baseline\nSlice {slice_idx}', fontsize=8, color='white')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # Row 2: Follow-up
            ax2 = fig.add_subplot(gs[1, i])
            followup_slice = followup_data[:, :, slice_idx].T
            ax2.imshow(followup_slice, cmap='gray', aspect='equal')
            ax2.set_title(f'Follow-up\nSlice {slice_idx}', fontsize=8, color='white')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            # Row 3: Difference map
            ax3 = fig.add_subplot(gs[2, i])
            diff_slice = diff_map[:, :, slice_idx].T
            
            # Show difference map
            im = ax3.imshow(diff_slice, cmap=cmap, aspect='equal',
                           vmin=-diff_range, vmax=diff_range)
            
            # Add tumor overlays if available
            if seg_baseline_data is not None:
                seg_baseline_slice = seg_baseline_data[:, :, slice_idx].T
                # Show baseline tumor contour in blue
                if np.any(seg_baseline_slice > 0):
                    contours = find_contours(seg_baseline_slice, 0.5)
                    for contour in contours:
                        ax3.plot(contour[:, 1], contour[:, 0], 'blue', linewidth=2, alpha=0.8)
            
            if seg_followup_data is not None:
                seg_followup_slice = seg_followup_data[:, :, slice_idx].T
                # Show follow-up tumor contour in red
                if np.any(seg_followup_slice > 0):
                    contours = find_contours(seg_followup_slice, 0.5)
                    for contour in contours:
                        ax3.plot(contour[:, 1], contour[:, 0], 'red', linewidth=2, alpha=0.8)
            
            ax3.set_title(f'Difference\nSlice {slice_idx}', fontsize=8, color='white')
            ax3.set_xticks([])
            ax3.set_yticks([])
            
            # Add orientation annotations to first column
            if i == 0:
                for ax in [ax1, ax2, ax3]:
                    ax.text(0.02, 0.5, 'L', transform=ax.transAxes, 
                           fontsize=10, color='yellow', ha='left', va='center', fontweight='bold')
                    ax.text(0.98, 0.5, 'R', transform=ax.transAxes, 
                           fontsize=10, color='yellow', ha='right', va='center', fontweight='bold')
                    ax.text(0.5, 0.02, 'P', transform=ax.transAxes, 
                           fontsize=10, color='yellow', ha='center', va='bottom', fontweight='bold')
                    ax.text(0.5, 0.98, 'A', transform=ax.transAxes, 
                           fontsize=10, color='yellow', ha='center', va='top', fontweight='bold')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Intensity Difference', rotation=270, labelpad=20, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Add title and legend
        fig.suptitle(f'Voxelwise Difference Analysis - {contrast_type}', 
                    fontsize=16, fontweight='bold', color='white', y=0.95)
        
        # Add legend for tumor contours
        if seg_baseline_data is not None or seg_followup_data is not None:
            legend_elements = []
            if seg_baseline_data is not None:
                legend_elements.append(plt.Line2D([0], [0], color='blue', lw=2, label='Baseline Tumor'))
            if seg_followup_data is not None:
                legend_elements.append(plt.Line2D([0], [0], color='red', lw=2, label='Follow-up Tumor'))
            
            fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
                      fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.02))
        
        return fig
    
    def generate_all_difference_maps(self, contrast_type='T1CE', reference_idx=0, 
                                   colormap_style='red_blue'):
        """
        Generate difference maps for all timepoints vs baseline
        """
        if not self.timepoint_folders or len(self.timepoint_folders) < 2:
            print("Need at least 2 timepoints for difference analysis")
            return []
        
        baseline_tp = self.timepoint_folders[reference_idx]
        baseline_file = self.get_file_for_timepoint_and_contrast(baseline_tp, contrast_type)
        
        if not baseline_file:
            print(f"Baseline file not found for {contrast_type}")
            return []
        
        # Get baseline segmentation if available
        baseline_seg = None
        if self.segmentation_folder:
            seg_path = os.path.join(self.segmentation_folder, baseline_tp['folder_name'], 'tumor_seg.nii.gz')
            if os.path.exists(seg_path):
                baseline_seg = seg_path
        
        results = []
        
        for i, tp_folder in enumerate(self.timepoint_folders):
            if i == reference_idx:
                continue  # Skip baseline vs baseline
            
            timepoint_name = tp_folder['folder_name']
            followup_file = self.get_file_for_timepoint_and_contrast(tp_folder, contrast_type)
            
            if not followup_file:
                print(f"Follow-up file not found for {timepoint_name}")
                continue
            
            # Get follow-up segmentation if available
            followup_seg = None
            if self.segmentation_folder:
                seg_path = os.path.join(self.segmentation_folder, timepoint_name, 'tumor_seg.nii.gz')
                if os.path.exists(seg_path):
                    followup_seg = seg_path
            
            print(f"Creating difference map: {baseline_tp['folder_name']} vs {timepoint_name}")
            
            fig = self.create_difference_map(
                baseline_file, followup_file, 
                baseline_seg, followup_seg,
                contrast_type, colormap_style
            )
            
            if fig:
                # Save the figure
                output_file = os.path.join(
                    self.output_folder, 
                    f'difference_map_{contrast_type}_{baseline_tp["folder_name"]}_vs_{timepoint_name}.png'
                )
                fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
                plt.close(fig)
                
                results.append({
                    'baseline': baseline_tp['folder_name'],
                    'followup': timepoint_name,
                    'file': output_file
                })
                
                print(f"Difference map saved: {output_file}")
        
        return results
    
    def create_contour_evolution_visualization(self, contrast_type='T1CE', slice_idx=None):
        """
        Create tumor contour evolution visualization over time
        
        Parameters:
        - contrast_type: Image contrast to use as background
        - slice_idx: Specific slice to analyze (if None, uses middle slice)
        """
        if not self.segmentation_folder:
            print("Segmentation folder required for contour evolution")
            return None
        
        if len(self.timepoint_folders) < 2:
            print("Need at least 2 timepoints for contour evolution")
            return None
        
        # Load first timepoint to determine slice if not specified
        first_tp = self.timepoint_folders[0]
        first_img_file = self.get_file_for_timepoint_and_contrast(first_tp, contrast_type)
        
        if not first_img_file:
            print(f"Could not find {contrast_type} image for first timepoint")
            return None
        
        first_img = nib.load(first_img_file)
        first_data = first_img.get_fdata()
        
        if slice_idx is None:
            slice_idx = first_data.shape[2] // 2
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.patch.set_facecolor('black')
        axes = axes.flatten()
        
        # Colors for different timepoints
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        
        # Process each timepoint
        contour_data = []
        
        for i, tp_folder in enumerate(self.timepoint_folders):
            timepoint_name = tp_folder['folder_name']
            
            # Load background image
            img_file = self.get_file_for_timepoint_and_contrast(tp_folder, contrast_type)
            if not img_file:
                print(f"Skipping {timepoint_name} - no {contrast_type} image")
                continue
            
            img_data = nib.load(img_file).get_fdata()
            
            # Load segmentation
            seg_path = os.path.join(self.segmentation_folder, timepoint_name, 'tumor_seg.nii.gz')
            if not os.path.exists(seg_path):
                print(f"Skipping {timepoint_name} - no segmentation")
                continue
            
            seg_data = nib.load(seg_path).get_fdata()
            
            # Get slice data
            img_slice = img_data[:, :, slice_idx].T
            seg_slice = seg_data[:, :, slice_idx].T
            
            # Calculate tumor volume for this slice
            slice_volume = np.sum(seg_slice > 0)
            
            contour_data.append({
                'timepoint': timepoint_name,
                'img_slice': img_slice,
                'seg_slice': seg_slice,
                'volume': slice_volume,
                'color': colors[i % len(colors)]
            })
        
        if not contour_data:
            print("No valid timepoints found for contour evolution")
            return None
        
        # Create individual timepoint plots (top 2 rows)
        for i, data in enumerate(contour_data[:6]):  # Max 6 timepoints
            ax = axes[i]
            
            # Show background image
            ax.imshow(data['img_slice'], cmap='gray', aspect='equal')
            
            # Add tumor contour
            if np.any(data['seg_slice'] > 0):
                contours = find_contours(data['seg_slice'], 0.5)
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], 
                           color=data['color'], linewidth=2, alpha=0.8)
            
            ax.set_title(f"{data['timepoint']}\nVol: {data['volume']} voxels", 
                        fontsize=10, color='white')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add orientation to first subplot
            if i == 0:
                ax.text(0.02, 0.5, 'L', transform=ax.transAxes, 
                       fontsize=10, color='yellow', ha='left', va='center', fontweight='bold')
                ax.text(0.98, 0.5, 'R', transform=ax.transAxes, 
                       fontsize=10, color='yellow', ha='right', va='center', fontweight='bold')
                ax.text(0.5, 0.02, 'P', transform=ax.transAxes, 
                       fontsize=10, color='yellow', ha='center', va='bottom', fontweight='bold')
                ax.text(0.5, 0.98, 'A', transform=ax.transAxes, 
                       fontsize=10, color='yellow', ha='center', va='top', fontweight='bold')
        
        # Hide unused subplots
        for i in range(len(contour_data), len(axes)):
            axes[i].set_visible(False)
        
        # Create overlay plot showing all contours together
        if len(contour_data) > 1:
            # Use last available subplot for overlay
            overlay_ax = axes[-1]
            overlay_ax.set_visible(True)
            
            # Use first timepoint as background
            overlay_ax.imshow(contour_data[0]['img_slice'], cmap='gray', aspect='equal')
            
            # Add all contours
            for data in contour_data:
                if np.any(data['seg_slice'] > 0):
                    contours = find_contours(data['seg_slice'], 0.5)
                    for contour in contours:
                        overlay_ax.plot(contour[:, 1], contour[:, 0], 
                                       color=data['color'], linewidth=2, alpha=0.7,
                                       label=data['timepoint'])
            
            overlay_ax.set_title('All Timepoints Overlay', fontsize=10, color='white')
            overlay_ax.set_xticks([])
            overlay_ax.set_yticks([])
            
            # Add legend
            overlay_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                            fontsize=8, frameon=False)
        
        plt.suptitle(f'Tumor Contour Evolution - {contrast_type} (Slice {slice_idx})', 
                    fontsize=16, fontweight='bold', color='white', y=0.95)
        
        plt.tight_layout()
        return fig
    
    def generate_volume_analysis(self, contrast_type='T1CE'):
        """
        Generate comprehensive tumor volume analysis over time
        """
        if not self.segmentation_folder:
            print("Segmentation folder required for volume analysis")
            return None
        
        volume_data = []
        
        for tp_folder in self.timepoint_folders:
            timepoint_name = tp_folder['folder_name']
            
            # Load segmentation
            seg_path = os.path.join(self.segmentation_folder, timepoint_name, 'tumor_seg.nii.gz')
            if not os.path.exists(seg_path):
                print(f"Skipping {timepoint_name} - no segmentation")
                continue
            
            seg_data = nib.load(seg_path).get_fdata()
            
            # Calculate volume (assuming 1mm³ voxels, adjust if needed)
            volume_voxels = np.sum(seg_data > 0)
            volume_ml = volume_voxels * 0.001  # Convert to mL (assuming 1mm³ voxels)
            
            volume_data.append({
                'timepoint': timepoint_name,
                'volume_voxels': volume_voxels,
                'volume_ml': volume_ml
            })
        
        if len(volume_data) < 2:
            print("Need at least 2 timepoints with segmentations for volume analysis")
            return None
        
        # Create volume evolution plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('white')
        
        timepoints = [d['timepoint'] for d in volume_data]
        volumes_ml = [d['volume_ml'] for d in volume_data]
        
        # Plot 1: Absolute volumes
        ax1.plot(timepoints, volumes_ml, 'o-', linewidth=2, markersize=8, color='red')
        ax1.set_title('Tumor Volume Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Timepoint', fontsize=12)
        ax1.set_ylabel('Volume (mL)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Relative change from baseline
        baseline_volume = volumes_ml[0]
        relative_changes = [(v - baseline_volume) / baseline_volume * 100 for v in volumes_ml]
        
        colors = ['green' if x >= 0 else 'red' for x in relative_changes]
        ax2.bar(timepoints, relative_changes, color=colors, alpha=0.7)
        ax2.set_title('Relative Volume Change from Baseline', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Timepoint', fontsize=12)
        ax2.set_ylabel('Volume Change (%)', fontsize=12)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save volume data to CSV
        df = pd.DataFrame(volume_data)
        csv_file = os.path.join(self.output_folder, f'tumor_volumes_{contrast_type}.csv')
        df.to_csv(csv_file, index=False)
        
        print(f"Volume analysis saved to: {csv_file}")
        return fig, df
    
    def run_comprehensive_assessment(self, contrast_types=None, reference_idx=0, 
                                   colormap_style='red_blue'):
        """
        Run complete longitudinal assessment including all analyses
        
        Parameters:
        - contrast_types: List of contrasts to analyze (if None, analyzes all available)
        - reference_idx: Index of reference timepoint for comparisons
        - colormap_style: Colormap style for difference maps
        """
        if contrast_types is None:
            contrast_types = self.contrast_types
        
        print("=== COMPREHENSIVE LONGITUDINAL ASSESSMENT ===")
        print(f"Analyzing contrasts: {contrast_types}")
        print(f"Reference timepoint: {self.timepoint_folders[reference_idx]['folder_name']}")
        
        results = {}
        
        for contrast in contrast_types:
            print(f"\n--- Analyzing {contrast} ---")
            
            # 1. Registration quality metrics
            print("Generating registration metrics table...")
            metrics_df = self.generate_registration_metrics_table(contrast, reference_idx)
            
            # 2. Difference maps
            print("Creating voxelwise difference maps...")
            diff_maps = self.generate_all_difference_maps(contrast, reference_idx, colormap_style)
            
            # 3. Contour evolution
            if self.segmentation_folder:
                print("Creating contour evolution visualization...")
                contour_fig = self.create_contour_evolution_visualization(contrast)
                if contour_fig:
                    contour_file = os.path.join(self.output_folder, f'contour_evolution_{contrast}.png')
                    contour_fig.savefig(contour_file, dpi=300, bbox_inches='tight', facecolor='black')
                    plt.close(contour_fig)
                    print(f"Contour evolution saved: {contour_file}")
                
                # 4. Volume analysis
                print("Performing tumor volume analysis...")
                volume_result = self.generate_volume_analysis(contrast)
                if volume_result:
                    volume_fig, volume_df = volume_result
                    volume_file = os.path.join(self.output_folder, f'volume_analysis_{contrast}.png')
                    volume_fig.savefig(volume_file, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(volume_fig)
                    print(f"Volume analysis saved: {volume_file}")
            
            results[contrast] = {
                'metrics_table': metrics_df,
                'difference_maps': diff_maps
            }
        
        print(f"\n=== ASSESSMENT COMPLETE ===")
        print(f"All results saved to: {self.output_folder}")
        
        return results