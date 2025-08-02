#!/usr/bin/env python3
"""
Main pipeline script for TBI lesion analysis from MRI images.
"""

import argparse
import logging
from pathlib import Path
import sys
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from preprocess import MRIPreprocessor
from segment import TBISegmenter
from atlas import BrainAtlasManager
from quantify import LesionQuantifier
from utils import setup_logging, get_image_info, save_color_lookup_table


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TBI Lesion Analysis Pipeline - Analyze MRI for traumatic brain injuries"
    )
    
    # Required arguments
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input MRI (DICOM directory or NIfTI file)"
    )
    
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory for results"
    )
    
    # Optional arguments
    parser.add_argument(
        "--subject-id",
        type=str,
        default="subject",
        help="Subject identifier for output files"
    )
    
    parser.add_argument(
        "--atlas",
        type=Path,
        help="Path to brain atlas file (if not provided, creates default)"
    )
    
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to pretrained segmentation model"
    )
    
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing (input must be preprocessed)"
    )
    
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate processing files"
    )
    
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Save lesion probability maps"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


def main():
    """Main pipeline execution."""
    args = parse_arguments()
    
    # Setup logging
    log_file = args.output / "pipeline.log"
    args.output.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level, log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting TBI Lesion Analysis Pipeline")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    
    try:
        # Step 1: Preprocessing
        if not args.skip_preprocessing:
            logger.info("=" * 50)
            logger.info("STEP 1: MRI Preprocessing")
            logger.info("=" * 50)
            
            preprocessor = MRIPreprocessor()
            preprocess_output = preprocessor.preprocess_mri(
                args.input,
                args.output / "preprocessing",
                save_intermediate=args.save_intermediate
            )
            
            mri_path = preprocess_output['preprocessed_image']
            brain_mask_path = preprocess_output['brain_mask']
        else:
            logger.info("Skipping preprocessing - using input as preprocessed MRI")
            mri_path = args.input
            brain_mask_path = None
        
        # Step 2: Lesion Segmentation
        logger.info("=" * 50)
        logger.info("STEP 2: Lesion Segmentation")
        logger.info("=" * 50)
        
        segmenter = TBISegmenter(model_path=args.model)
        seg_output = segmenter.segment_lesions(
            mri_path,
            args.output / "segmentation",
            save_probabilities=args.save_probabilities
        )
        
        lesion_seg_path = seg_output['segmentation_path']
        lesion_stats = seg_output['lesion_statistics']
        
        # Log lesion statistics
        logger.info("Lesion statistics:")
        for lesion_type, stats in lesion_stats.items():
            if stats['volume_mm3'] > 0:
                logger.info(f"  {lesion_type}: {stats['volume_mm3']:.2f} mm³")
        
        # Step 3: Atlas Registration
        logger.info("=" * 50)
        logger.info("STEP 3: Brain Atlas Registration")
        logger.info("=" * 50)
        
        atlas_manager = BrainAtlasManager(atlas_path=args.atlas)
        
        # Create default atlas if not provided
        if not args.atlas:
            logger.info("Creating default 10-zone brain atlas")
            atlas_path = args.output / "atlas" / "default_atlas.nii.gz"
            atlas_manager.create_default_atlas(mri_path, atlas_path)
            atlas_manager.load_atlas(atlas_path)
        
        # Register atlas to subject space
        reg_output = atlas_manager.register_atlas_to_subject(
            mri_path,
            args.output / "registration"
        )
        
        warped_atlas_path = reg_output['warped_atlas_path']
        
        # Create zone masks
        zone_masks = atlas_manager.create_zone_masks(
            warped_atlas_path,
            args.output / "atlas" / "zone_masks"
        )
        
        # Step 4: Quantification
        logger.info("=" * 50)
        logger.info("STEP 4: Lesion-Zone Quantification")
        logger.info("=" * 50)
        
        quantifier = LesionQuantifier()
        
        # Calculate overlap
        overlap_df = quantifier.calculate_overlap(
            lesion_seg_path,
            warped_atlas_path,
            TBISegmenter.LESION_TYPES,
            BrainAtlasManager.BRAIN_ZONES
        )
        
        # Export results
        output_files = quantifier.export_results(
            overlap_df,
            args.output / "quantification",
            subject_id=args.subject_id
        )
        
        # Create visualization data
        viz_output = quantifier.create_visualization_data(
            overlap_df,
            lesion_seg_path,
            warped_atlas_path,
            args.output / "visualization"
        )
        
        # Save color lookup table
        color_lut_path = args.output / "visualization" / "color_lookup_table.json"
        save_color_lookup_table(color_lut_path)
        
        # Create final summary
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 50)
        
        summary = {
            "subject_id": args.subject_id,
            "input_path": str(args.input),
            "output_path": str(args.output),
            "preprocessing": {
                "completed": not args.skip_preprocessing,
                "preprocessed_mri": str(mri_path) if mri_path else None,
                "brain_mask": str(brain_mask_path) if brain_mask_path else None
            },
            "segmentation": {
                "model_used": str(args.model) if args.model else "random_initialization",
                "segmentation_path": str(lesion_seg_path),
                "lesion_statistics": lesion_stats
            },
            "atlas": {
                "atlas_path": str(warped_atlas_path),
                "zones": list(BrainAtlasManager.BRAIN_ZONES.values())
            },
            "quantification": {
                "overlap_csv": str(output_files['overlap']),
                "report": str(output_files['report']),
                "total_lesions_found": len(overlap_df) if not overlap_df.empty else 0
            },
            "visualization": {
                "overlay": str(viz_output['overlay']),
                "color_lookup": str(color_lut_path)
            }
        }
        
        # Save summary
        summary_path = args.output / f"{args.subject_id}_pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Pipeline summary saved to: {summary_path}")
        logger.info(f"All results saved to: {args.output}")
        
        # Print key results
        print("\n" + "=" * 50)
        print("TBI LESION ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"Subject: {args.subject_id}")
        print(f"Output directory: {args.output}")
        print(f"\nKey files:")
        print(f"  - Lesion segmentation: {lesion_seg_path}")
        print(f"  - Quantification CSV: {output_files['overlap']}")
        print(f"  - Report: {output_files['report']}")
        print(f"  - Summary: {summary_path}")
        
        if not overlap_df.empty:
            print(f"\nLesions detected: {overlap_df['LesionType'].nunique()} types")
            print(f"Affected zones: {overlap_df['Zone'].nunique()} zones")
            print(f"Total lesion volume: {overlap_df['Volume_mm3'].sum():.2f} mm³")
        else:
            print("\nNo lesions detected.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"\nERROR: Pipeline failed - {str(e)}")
        print(f"Check log file for details: {log_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()