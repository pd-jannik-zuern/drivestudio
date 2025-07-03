#!/usr/bin/env python3
"""
Test script to verify that the configuration and dataloader work correctly.
"""

import sys
import os
import torch
from omegaconf import OmegaConf

# Add the parent directory to the path so we can import from the main repository
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidar_intensity_estimation.lidar_intensity_dataloader import create_lidar_intensity_dataloader
from lidar_intensity_estimation.segformer_intensity import create_segformer_intensity_model

def test_config_loading():
    """Test that we can load the example configuration."""
    print("Testing configuration loading...")
    
    try:
        # Load the example configuration
        config_path = os.path.join(os.path.dirname(__file__), "example_config.yaml")
        config = OmegaConf.load(config_path)
        
        print("✓ Configuration loaded successfully")
        print(f"  Dataset: {config.data.dataset}")
        print(f"  Scene index: {config.data.scene_idx}")
        print(f"  Data root: {config.data.data_root}")
        
        return config
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return None

def test_dataloader_creation(config):
    """Test that we can create a dataloader with the configuration."""
    print("\nTesting dataloader creation...")
    
    try:
        # Create a simple dataloader
        dataloader = create_lidar_intensity_dataloader(
            data_cfg=config,
            split="train",
            batch_size=2,
            num_workers=0,  # Use 0 workers for testing
            shuffle=False,
            target_size=(224, 224),
            device=torch.device("cpu")
        )
        
        print("✓ Dataloader created successfully")
        print(f"  Number of batches: {len(dataloader)}")
        
        return dataloader
        
    except Exception as e:
        print(f"✗ Dataloader creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_creation():
    """Test that we can create a model."""
    print("\nTesting model creation...")
    
    try:
        model = create_segformer_intensity_model(
            model_name="nvidia/segformer-b0-finetuned-ade-512-512",
            use_depth=True,
            freeze_backbone=False,
            freeze_decoder=False
        )
        
        print("✓ Model created successfully")
        print(f"  Model type: {type(model)}")
        
        return model
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_single_batch(dataloader, model):
    """Test that we can process a single batch."""
    print("\nTesting single batch processing...")
    
    try:
        # Get a single batch
        batch = next(iter(dataloader))
        
        print("✓ Batch loaded successfully")
        print(f"  RGB shape: {batch['rgb'].shape}")
        print(f"  Intensity shape: {batch['intensity'].shape}")
        print(f"  Depth shape: {batch['depth'].shape}")
        print(f"  Valid mask shape: {batch['valid_mask'].shape}")
        
        # Test model forward pass
        model.eval()
        with torch.no_grad():
            rgb = batch["rgb"]
            depth = batch["depth"]
            output = model(rgb)
            
        print("✓ Model forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing LiDAR Intensity Estimation Setup")
    print("=" * 60)
    
    # Test configuration loading
    config = test_config_loading()
    if config is None:
        print("Configuration loading failed, stopping tests.")
        return
    
    # Test dataloader creation
    dataloader = test_dataloader_creation(config)
    if dataloader is None:
        print("Dataloader creation failed, stopping tests.")
        return
    
    # Test model creation
    model = test_model_creation()
    if model is None:
        print("Model creation failed, stopping tests.")
        return
    
    # Test single batch processing
    batch_success = test_single_batch(dataloader, model)
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Configuration: {'✓' if config is not None else '✗'}")
    print(f"  Dataloader: {'✓' if dataloader is not None else '✗'}")
    print(f"  Model: {'✓' if model is not None else '✗'}")
    print(f"  Batch Processing: {'✓' if batch_success else '✗'}")
    print("=" * 60)
    
    if all([config is not None, dataloader is not None, model is not None, batch_success]):
        print("🎉 All tests passed! The setup is working correctly.")
        print("\nYou can now run training with:")
        print("python train_segformer_lidar_intensity.py --config example_config.yaml")
    else:
        print("❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 