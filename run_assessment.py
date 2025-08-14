#!/usr/bin/env python3
"""
Standalone Registration Assessment Script

Run comprehensive quantitative assessment on existing registered data.
Generates metrics tables, difference maps, and contour evolution analysis.

Usage:
    python run_assessment.py --registered_folder output/registered --segmentation_folder data/segs --output_folder assessment/
    python run_assessment.py --registered_folder output/registered --only_difference_maps --output_folder assessment/
    python run_assessment.py --registered_folder output/registered --only_metrics_table --output_folder assessment/

Author: AutoFlipbook Pipeline  
Date: 2025
"""

import argparse
import os
import sys
from registration_assessment import RegistrationAssessment


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive registration quality assessment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full assessment with all analyses
  python run_assessment.py --registered_folder results/registered --segmentation_folder data/segs --output_folder assessment/

  # Only generate difference maps
  python run_assessment.py --registered_folder results/registered --only_difference_maps --output_folder assessment/

  # Only generate metrics table
  python run_assessment.py --registered_folder results/registered --only_metrics_table --output_folder assessment/

  # Custom contrast and colormap
  python run_assessment.py --registered_folder results/registered --contrast T2 --colormap_style viridis --output_folder assessment/
        """
    )
    
    # Required arguments
    parser.add_argument('--registered_folder', required=True,
                        help='Path to folder containing registered images')
    
    # Optional arguments
    parser.add_argument('--segmentation_folder', 
                        help='Path to folder containing tumor segmentations (optional)')
    
    parser.add_argument('--output_folder', default='assessment',
                        help='Output folder for assessment results (default: assessment)')
    
    parser.add_argument('--contrast', default='T1CE',
                        help='Image contrast to analyze (default: T1CE)')
    
    parser.add_argument('--reference_idx', type=int, default=0,
                        help='Index of reference timepoint (default: 0 = first timepoint)')
    
    parser.add_argument('--colormap_style', choices=['red_blue', 'viridis', 'plasma', 'coolwarm'],
                        default='red_blue',
                        help='Colormap style for difference maps (default: red_blue)')
    
    # Analysis type flags
    parser.add_argument('--only_difference_maps', action='store_true',
                        help='Generate only difference maps')
    
    parser.add_argument('--only_metrics_table', action='store_true',
                        help='Generate only registration metrics table') 
    
    parser.add_argument('--only_contour_evolution', action='store_true',
                        help='Generate only contour evolution visualization')
    
    parser.add_argument('--only_volume_analysis', action='store_true',
                        help='Generate only tumor volume analysis')
    
    # Visualization options
    parser.add_argument('--slice_idx', type=int,
                        help='Specific slice for contour evolution (default: middle slice)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.registered_folder):
        print(f"Error: Registered folder does not exist: {args.registered_folder}")
        sys.exit(1)
    
    if args.segmentation_folder and not os.path.exists(args.segmentation_folder):
        print(f"Error: Segmentation folder does not exist: {args.segmentation_folder}")
        sys.exit(1)
    
    # Initialize assessment
    print("=== REGISTRATION QUALITY ASSESSMENT ===")
    print(f"Registered data: {args.registered_folder}")
    print(f"Segmentation data: {args.segmentation_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Target contrast: {args.contrast}")
    
    assessment = RegistrationAssessment(
        registered_folder=args.registered_folder,
        segmentation_folder=args.segmentation_folder,
        output_folder=args.output_folder
    )
    
    # Check if target contrast is available
    if args.contrast not in assessment.contrast_types:
        print(f"Error: Contrast '{args.contrast}' not found.")
        print(f"Available contrasts: {assessment.contrast_types}")
        sys.exit(1)
    
    # Run specific analyses based on flags
    if args.only_metrics_table:
        print("\n--- Generating Registration Metrics Table ---")
        metrics_df = assessment.generate_registration_metrics_table(
            args.contrast, args.reference_idx)
        if metrics_df is not None:
            print("✅ Registration metrics table completed successfully")
        else:
            print("❌ Failed to generate registration metrics table")
    
    elif args.only_difference_maps:
        print("\n--- Generating Difference Maps ---")
        diff_maps = assessment.generate_all_difference_maps(
            args.contrast, args.reference_idx, args.colormap_style)
        if diff_maps:
            print(f"✅ Generated {len(diff_maps)} difference maps successfully")
            for dm in diff_maps:
                print(f"   📁 {dm['file']}")
        else:
            print("❌ Failed to generate difference maps")
    
    elif args.only_contour_evolution:
        if not args.segmentation_folder:
            print("❌ Error: Segmentation folder required for contour evolution")
            sys.exit(1)
        
        print("\n--- Generating Contour Evolution ---")
        contour_fig = assessment.create_contour_evolution_visualization(
            args.contrast, args.slice_idx)
        
        if contour_fig:
            contour_file = os.path.join(args.output_folder, f'contour_evolution_{args.contrast}.png')
            contour_fig.savefig(contour_file, dpi=300, bbox_inches='tight', facecolor='black')
            print(f"✅ Contour evolution saved: {contour_file}")
        else:
            print("❌ Failed to generate contour evolution")
    
    elif args.only_volume_analysis:
        if not args.segmentation_folder:
            print("❌ Error: Segmentation folder required for volume analysis")
            sys.exit(1)
        
        print("\n--- Generating Volume Analysis ---")
        volume_result = assessment.generate_volume_analysis(args.contrast)
        
        if volume_result:
            volume_fig, volume_df = volume_result
            volume_file = os.path.join(args.output_folder, f'volume_analysis_{args.contrast}.png')
            volume_fig.savefig(volume_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Volume analysis saved: {volume_file}")
        else:
            print("❌ Failed to generate volume analysis")
    
    else:
        # Run comprehensive assessment (default)
        print("\n--- Running Comprehensive Assessment ---")
        results = assessment.run_comprehensive_assessment(
            contrast_types=[args.contrast],
            reference_idx=args.reference_idx,
            colormap_style=args.colormap_style
        )
        
        if results:
            print("✅ Comprehensive assessment completed successfully")
        else:
            print("❌ Assessment failed")
    
    print(f"\n🎉 Assessment complete! Results saved to: {args.output_folder}")


if __name__ == "__main__":
    main()