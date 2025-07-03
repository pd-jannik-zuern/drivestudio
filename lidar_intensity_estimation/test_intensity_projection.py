#!/usr/bin/env python3
"""
Test script to verify lidar intensity projection functionality.
"""

import os
import sys
import warnings

# Suppress TensorFlow warnings about missing TensorRT libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', message='.*libnvinfer_plugin.*')
warnings.filterwarnings('ignore', message='.*Could not load dynamic library.*')

import torch
import numpy as np
from omegaconf import OmegaConf
import logging

# Add the parent directory to the path to import from the main repository
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidar_intensity_estimation.lidar_intensity_dataloader import create_lidar_intensity_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_intensity_projection():
    """
    Test the lidar intensity projection functionality.
    """
    # Example configuration - you'll need to adjust this for your dataset
    config = OmegaConf.create({
        "data": {
            "dataset": {
                "type": "waymo",  # or "argoverse", "nuplan", etc.
                "data_path": "/path/to/your/dataset",  # Update this path
                "start_timestep": 0,
                "end_timestep": 10,
            },
            "pixel_source": {
                "type": "waymo",  # or corresponding dataset type
                "load_images": True,
                "test_image_stride": 0,
            },
            "lidar_source": {
                "type": "waymo",  # or corresponding dataset type
                "load_lidar": True,
                "only_use_top_lidar": True,  # for Waymo
            }
        }
    })
    
    try:
        # Create dataloader with intensity projection enabled
        logger.info("Creating dataloader with intensity projection...")
        dataloader = create_lidar_intensity_dataloader(
            data_cfg=config,
            split="train",
            batch_size=2,
            num_workers=0,  # Use 0 for debugging
            shuffle=False,
            target_size=(224, 224),
            device=torch.device("cpu"),
            project_intensity=True
        )
        
        logger.info(f"Dataloader created successfully with {len(dataloader)} batches")
        
        # Test loading a few samples
        for batch_idx, batch in enumerate(dataloader):
            logger.info(f"Batch {batch_idx}:")
            logger.info(f"  RGB shape: {batch['rgb'].shape}")
            logger.info(f"  Intensity shape: {batch['intensity'].shape}")
            logger.info(f"  Depth shape: {batch['depth'].shape}")
            logger.info(f"  Valid mask shape: {batch['valid_mask'].shape}")
            
            # Check intensity values
            intensity = batch['intensity']
            valid_mask = batch['valid_mask']
            
            logger.info(f"  Intensity range: [{intensity.min():.4f}, {intensity.max():.4f}]")
            logger.info(f"  Valid pixels: {valid_mask.sum().item()}")
            logger.info(f"  Total pixels: {valid_mask.numel()}")
            logger.info(f"  Coverage: {valid_mask.sum().item() / valid_mask.numel():.2%}")
            
            if batch_idx >= 2:  # Only test first 3 batches
                break
                
        logger.info("Intensity projection test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during intensity projection test: {e}")
        import traceback
        traceback.print_exc()

def test_without_intensity_projection():
    """
    Test the dataloader without intensity projection (should use fallback).
    """
    # Example configuration
    config = OmegaConf.create({
        "data": {
            "dataset": {
                "type": "waymo",
                "data_path": "/path/to/your/dataset",
                "start_timestep": 0,
                "end_timestep": 10,
            },
            "pixel_source": {
                "type": "waymo",
                "load_images": True,
                "test_image_stride": 0,
            },
            "lidar_source": {
                "type": "waymo",
                "load_lidar": True,
                "only_use_top_lidar": True,
            }
        }
    })
    
    try:
        # Create dataloader without intensity projection
        logger.info("Creating dataloader without intensity projection...")
        dataloader = create_lidar_intensity_dataloader(
            data_cfg=config,
            split="train",
            batch_size=2,
            num_workers=0,
            shuffle=False,
            target_size=(224, 224),
            device=torch.device("cpu"),
            project_intensity=False
        )
        
        logger.info(f"Dataloader created successfully with {len(dataloader)} batches")
        
        # Test loading a sample
        for batch_idx, batch in enumerate(dataloader):
            logger.info(f"Batch {batch_idx} (no projection):")
            logger.info(f"  RGB shape: {batch['rgb'].shape}")
            logger.info(f"  Intensity shape: {batch['intensity'].shape}")
            logger.info(f"  Depth shape: {batch['depth'].shape}")
            logger.info(f"  Valid mask shape: {batch['valid_mask'].shape}")
            
            # Check that intensity is all zeros (fallback behavior)
            intensity = batch['intensity']
            logger.info(f"  Intensity range: [{intensity.min():.4f}, {intensity.max():.4f}]")
            logger.info(f"  All zeros: {torch.all(intensity == 0).item()}")
            
            break
                
        logger.info("No projection test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during no projection test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("Starting lidar intensity projection tests...")
    
    # Test with intensity projection
    test_intensity_projection()
    
    # Test without intensity projection
    test_without_intensity_projection()
    
    logger.info("All tests completed!") 