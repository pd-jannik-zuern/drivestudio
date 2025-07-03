#!/usr/bin/env python3
"""
Test script for the LiDAR intensity dataloader.
"""

import os
import warnings

# Suppress TensorFlow warnings about missing TensorRT libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', message='.*libnvinfer_plugin.*')
warnings.filterwarnings('ignore', message='.*Could not load dynamic library.*')

import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import argparse

from lidar_intensity_dataloader import create_lidar_intensity_dataloader

def visualize_batch(batch, save_dir: str, batch_idx: int):
    """
    Visualize a batch of data.
    
    Args:
        batch: Batch of data from dataloader
        save_dir: Directory to save visualizations
        batch_idx: Batch index for naming
    """
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = batch["rgb"].shape[0]
    
    for i in range(min(batch_size, 3)):  # Visualize up to 3 samples per batch
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # RGB image
        rgb = batch["rgb"][i].permute(1, 2, 0).cpu().numpy()
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title(f"RGB Image (Sample {i})")
        axes[0, 0].axis('off')
        
        # Intensity map
        intensity = batch["intensity"][i].squeeze().cpu().numpy()
        im1 = axes[0, 1].imshow(intensity, cmap='viridis')
        axes[0, 1].set_title(f"LiDAR Intensity (Sample {i})")
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Depth map
        depth = batch["depth"][i].squeeze().cpu().numpy()
        im2 = axes[1, 0].imshow(depth, cmap='plasma')
        axes[1, 0].set_title(f"LiDAR Depth (Sample {i})")
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Valid mask
        valid_mask = batch["valid_mask"][i].squeeze().cpu().numpy()
        axes[1, 1].imshow(valid_mask, cmap='gray')
        axes[1, 1].set_title(f"Valid Mask (Sample {i}) - {valid_mask.sum()} valid pixels")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"batch_{batch_idx}_sample_{i}.png"), dpi=150, bbox_inches='tight')
        plt.close()

def test_dataloader(config_path: str, num_samples: int = 5, save_dir: str = "test_visualizations"):
    """
    Test the LiDAR intensity dataloader.
    
    Args:
        config_path: Path to dataset configuration
        num_samples: Number of samples to test
        save_dir: Directory to save visualizations
    """
    print(f"Testing dataloader with config: {config_path}")
    
    # Load configuration
    data_cfg = OmegaConf.load(config_path)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Create dataloader
        print("Creating dataloader...")
        dataloader = create_lidar_intensity_dataloader(
            data_cfg=data_cfg,
            split="train",
            batch_size=2,
            num_workers=0,  # Use 0 for debugging
            shuffle=True,
            device=device
        )
        
        print(f"Dataloader created successfully with {len(dataloader)} batches")
        
        # Test a few batches
        batch_count = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_count >= num_samples:
                break
                
            print(f"\nBatch {batch_idx}:")
            print(f"  RGB shape: {batch['rgb'].shape}")
            print(f"  Intensity shape: {batch['intensity'].shape}")
            print(f"  Depth shape: {batch['depth'].shape}")
            print(f"  Valid mask shape: {batch['valid_mask'].shape}")
            print(f"  Image indices: {batch['image_idx']}")
            
            # Check data ranges
            print(f"  RGB range: [{batch['rgb'].min():.3f}, {batch['rgb'].max():.3f}]")
            print(f"  Intensity range: [{batch['intensity'].min():.3f}, {batch['intensity'].max():.3f}]")
            print(f"  Depth range: [{batch['depth'].min():.3f}, {batch['depth'].max():.3f}]")
            print(f"  Valid mask range: [{batch['valid_mask'].min():.3f}, {batch['valid_mask'].max():.3f}]")
            
            # Count valid pixels
            valid_pixels = batch['valid_mask'].sum().item()
            total_pixels = batch['valid_mask'].numel()
            print(f"  Valid pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
            
            # Visualize batch
            visualize_batch(batch, save_dir, batch_idx)
            
            batch_count += 1
        
        print(f"\n✓ Successfully tested {batch_count} batches")
        print(f"Visualizations saved to: {save_dir}")
        
    except Exception as e:
        print(f"✗ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Test LiDAR intensity dataloader")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset config")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--save_dir", type=str, default="test_visualizations", help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Testing LiDAR Intensity Dataloader")
    print("=" * 50)
    
    test_dataloader(args.config, args.num_samples, args.save_dir)

if __name__ == "__main__":
    main() 