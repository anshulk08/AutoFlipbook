#!/usr/bin/env python3
"""
Command Line Interface for Brain Tumor Flipbook Pipeline

This script provides a command-line interface for running the automated
brain tumor flipbook generation pipeline.

Usage:
    python run_pipeline.py --raw_folder data/raw --output_folder output/

Author: Your Name
Date: 2025
"""

import argparse
import sys
import os
from pathlib import Path

# Add the package to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain_registration_pipeline import run_brain_registration_and_flipbook_pipeline


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Automated Brain Tumor Flipbook Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python cli_script.py --raw_folder data/raw_timepoints --output_folder output/

    # With tumor segmentations
    python cli_script.py --raw_folder data/raw --segmentation_folder data/segs --output_folder output/

    # Custom registration and visualization
    python cli_script.py --raw_folder data/raw --registration_dof 12 --tumor_color yellow --show_contour False

    # Dotted tumor contours with custom transparency
    python cli_script.py --raw_folder data/raw --show_contour True --contour_style dotted --tumor_alpha 0.8

    # Specify reference timepoint
    python cli_script.py --raw_folder data/raw --reference_timepoint 5885 --output_folder output/

    # Skip quantitative assessment for faster execution
    python cli_script.py --raw_folder data/raw --skip_assessment --output_folder output/

For more information, see: https://github.com/anshulk08/AutoFlipbook
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--raw_folder",
        type=str,
        required=True,
        help="Path to folder containing raw timepoint subfolders (e.g., 5885/, 5972/, 6070/)"
    )
    
    parser.add_argument(
        "--output_folder",
        type=str,
        default="pipeline_output",
        help="Output folder for all pipeline results (default: pipeline_output)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--segmentation_folder",
        type=str,
        default=None,
        help="Path to folder containing tumor segmentation masks (optional)"
    )
    
    parser.add_argument(
        "--reference_timepoint",
        type=str,
        default=None,
        help="Timepoint to use as reference (default: earliest timepoint)"
    )
    
    parser.add_argument(
        "--fsl_dir",
        type=str,
        default=None,
        help="Path to FSL installation directory (default: use system PATH)"
    )
    
    # Registration parameters
    parser.add_argument(
        "--registration_dof",
        type=int,
        choices=[6, 12],
        default=6,
        help="Degrees of freedom for registration: 6 (rigid, preserves tumor size) or 12 (affine) (default: 6)"
    )
    
    parser.add_argument(
        "--skull_strip",
        action="store_true",
        help="Perform skull stripping (may remove important extra-axial structures)"
    )
    
    parser.add_argument(
        "--bias_correct",
        action="store_true",
        help="Perform bias field correction"
    )
    
    # Flipbook visualization parameters
    parser.add_argument(
        "--tumor_color",
        type=str,
        choices=["red", "yellow", "cyan", "green", "magenta", "blue"],
        default="red",
        help="Color for tumor overlay (default: red)"
    )
    
    parser.add_argument(
        "--tumor_alpha",
        type=float,
        default=0.5,
        help="Transparency of tumor overlay (0-1, default: 0.5)"
    )
    
    parser.add_argument(
        "--show_contour",
        type=str,
        choices=["True", "False", "true", "false"],
        default="True",
        help="Show tumor contour (True) or filled overlay (False) (default: True)"
    )
    
    parser.add_argument(
        "--contour_width",
        type=int,
        default=2,
        help="Width of tumor contour lines (default: 2)"
    )
    
    parser.add_argument(
        "--contour_style",
        type=str,
        choices=["solid", "dashed", "dotted", "dashdot"],
        default="solid",
        help="Style of tumor contour lines (default: solid)"
    )
    
    # Mosaic layout parameters
    parser.add_argument(
        "--rows",
        type=int,
        default=3,
        help="Number of rows in mosaic view (default: 3)"
    )
    
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Number of columns in mosaic view (default: 5)"
    )
    
    parser.add_argument(
        "--colormap",
        type=str,
        default="gray",
        help="Colormap for brain images (default: gray)"
    )
    
    # Utility arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--skip_assessment",
        action="store_true",
        help="Skip quantitative registration assessment (faster execution)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without actually running the pipeline"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Brain Tumor Flipbook Pipeline v1.0.0"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments"""
    errors = []
    
    # Check if raw folder exists
    if not os.path.exists(args.raw_folder):
        errors.append(f"Raw folder does not exist: {args.raw_folder}")
    
    # Check if segmentation folder exists (if provided)
    if args.segmentation_folder and not os.path.exists(args.segmentation_folder):
        errors.append(f"Segmentation folder does not exist: {args.segmentation_folder}")
    
    # Check FSL directory (if provided)
    if args.fsl_dir and not os.path.exists(args.fsl_dir):
        errors.append(f"FSL directory does not exist: {args.fsl_dir}")
    
    # Validate tumor_alpha range
    if not 0 <= args.tumor_alpha <= 1:
        errors.append(f"tumor_alpha must be between 0 and 1, got: {args.tumor_alpha}")
    
    # Validate mosaic dimensions
    if args.rows < 1 or args.cols < 1:
        errors.append(f"rows and cols must be positive integers, got: {args.rows}x{args.cols}")
    
    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


def print_configuration(args):
    """Print pipeline configuration"""
    print("🧠 Brain Tumor Flipbook Pipeline")
    print("=" * 50)
    print(f"📁 Raw folder: {args.raw_folder}")
    print(f"📁 Output folder: {args.output_folder}")
    if args.segmentation_folder:
        print(f"📁 Segmentation folder: {args.segmentation_folder}")
    print(f"🔧 Registration: {args.registration_dof}-DOF {'rigid' if args.registration_dof == 6 else 'affine'}")
    print(f"🎨 Tumor visualization: {args.tumor_color} ({'contour' if args.show_contour.lower() == 'true' else 'filled'})")
    print(f"📊 Mosaic layout: {args.rows}x{args.cols}")
    if args.reference_timepoint:
        print(f"📅 Reference timepoint: {args.reference_timepoint}")
    print()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Validate arguments
    validate_arguments(args)
    
    # Print configuration
    if args.verbose or args.dry_run:
        print_configuration(args)
    
    # Dry run mode
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be created")
        print("✅ Configuration is valid. Run without --dry_run to execute.")
        return
    
    try:
        # Convert string boolean arguments
        show_contour = args.show_contour.lower() == 'true'
        
        # Prepare flipbook kwargs
        flipbook_kwargs = {
            'tumor_color': args.tumor_color,
            'tumor_alpha': args.tumor_alpha,
            'show_contour': show_contour,
            'contour_width': args.contour_width,
            'contour_style': args.contour_style,
            'rows': args.rows,
            'cols': args.cols,
            'colormap': args.colormap,
        }
        
        print("🚀 Starting Brain Tumor Flipbook Pipeline...")
        print()
        
        # Run the pipeline
        results = run_brain_registration_and_flipbook_pipeline(
            raw_folder=args.raw_folder,
            segmentation_folder=args.segmentation_folder,
            output_base_folder=args.output_folder,
            reference_timepoint=args.reference_timepoint,
            fsl_dir=args.fsl_dir,
            registration_dof=args.registration_dof,
            skull_strip=args.skull_strip,
            bias_correct=args.bias_correct,
            run_assessment=not args.skip_assessment,  # Run assessment unless explicitly skipped
            **flipbook_kwargs
        )
        
        print()
        print("🎉 Pipeline completed successfully!")
        print(f"📁 Results saved to: {os.path.abspath(args.output_folder)}")
        
        # Print flipbook locations
        if 'flipbook_results' in results and isinstance(results['flipbook_results'], dict) and results['flipbook_results']:
            print("\n📚 Generated flipbook outputs:")
            pptx_files = []
            for contrast, result in results['flipbook_results'].items():
                if result and 'slide_files' in result and result['slide_files']:
                    slide_count = len(result['slide_files'])
                    folder_path = os.path.abspath(result['output_folder'])
                    print(f"  {contrast}: {slide_count} slides in {folder_path}")
                    
                    # Check for PowerPoint file
                    if result.get('pptx_file'):
                        pptx_path = os.path.abspath(result['pptx_file'])
                        print(f"    📊 PowerPoint: {pptx_path}")
                        pptx_files.append(pptx_path)
            
            total_slides = sum(result.get('slide_count', 0) for result in results['flipbook_results'].values())
            print(f"\n🎯 Total: {total_slides} flipbook slides generated")
            if pptx_files:
                print(f"📊 PowerPoint presentations: {len(pptx_files)} files ready for clinical review")
        
        print()
        if 'flipbook_results' in results and any(result.get('pptx_file') for result in results['flipbook_results'].values()):
            print("💡 Open the PowerPoint (.pptx) files for clinical presentation and review!")
        else:
            print("💡 View the generated PNG slide images to analyze brain tumor progression!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()