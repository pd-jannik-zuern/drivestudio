#!/usr/bin/env python3
"""
Test script for the Segformer intensity estimation architecture.
"""

import torch
import numpy as np
from segformer_intensity import create_segformer_intensity_model, SegformerIntensityLoss


def test_model_creation():
    """Test that the model can be created successfully."""
    print("Testing model creation...")
    
    try:
        model = create_segformer_intensity_model(
            model_name="nvidia/segformer-b0-finetuned-ade-512-512",
            use_depth=True,
            # freeze_backbone=False,
            # freeze_decoder=False
        )
        print("✓ Model created successfully")
        return model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return None

def test_forward_pass(model, device):
    """Test the forward pass with dummy data."""
    print("Testing forward pass...")
    
    try:
        # Create dummy data
        batch_size = 2
        height, width = 224, 224
        
        # RGB images
        rgb_images = torch.randn(batch_size, 3, height, width)
        rgb_images = torch.clamp(rgb_images, 0, 1).to(device)
        
        # Depth maps
        # depth_maps = torch.randn(batch_size, 1, height, width).to(device)
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            intensity_pred = model(rgb_images)
            print(intensity_pred.shape)
        
        print(f"✓ Forward pass successful")
        print(f"  Input RGB shape: {rgb_images.shape}")
        print(f"  Output intensity shape: {intensity_pred.shape}")
        print(f"  Intensity range: [{intensity_pred.min():.3f}, {intensity_pred.max():.3f}]")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def test_loss_function():
    """Test the loss function."""
    print("Testing loss function...")
    
    try:
        # Create loss function
        loss_fn = SegformerIntensityLoss(loss_type="mse")
        
        # Create dummy data
        batch_size = 2
        height, width = 224, 224
        
        # Predictions and targets
        predictions = torch.randn(batch_size, 1, height, width)
        targets = torch.randn(batch_size, 1, height, width)

        
        # Valid mask (randomly mask some pixels)
        valid_mask = torch.randint(0, 2, (batch_size, 1, height, width), dtype=torch.bool)
        
        # Compute loss
        loss = loss_fn(predictions, targets, valid_mask)
        
        print(f"✓ Loss computation successful")
        print(f"  Loss value: {loss.item():.6f}")
        print(f"  Number of valid pixels: {valid_mask.sum().item()}")
        
        return True
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        return False

def test_model_parameters(model, device):
    """Test that model parameters can be accessed and counted."""
    print("Testing model parameters...")
    
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Parameter counting successful")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test parameter gradients
        model.train()
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_input = torch.clamp(dummy_input, 0, 1)

        output = model(dummy_input)
        loss = output.mean()
        loss.backward()
        
        # Check if gradients are computed
        has_gradients = any(p.grad is not None for p in model.parameters())
        print(f"  Gradients computed: {has_gradients}")
        
        return True
    except Exception as e:
        print(f"✗ Parameter test failed: {e}")
        return False

def test_different_segformer_models(device):
    """Test different Segformer model variants."""
    print("Testing different Segformer models...")
    
    model_variants = [
        "nvidia/segformer-b0-finetuned-ade-512-512",
        "nvidia/segformer-b1-finetuned-ade-512-512",
        "nvidia/segformer-b2-finetuned-ade-512-512",
    ]
    
    success_count = 0
    for model_name in model_variants:
        try:
            print(f"  Testing {model_name}...")
            model = create_segformer_intensity_model(
                model_name=model_name,
                use_depth=True,
                # freeze_backbone=False,
                # freeze_decoder=False
            ).to(device)
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dummy_input = torch.clamp(dummy_input, 0, 1)
            # dummy_depth = torch.randn(1, 1, 224, 224).to(device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"    ✓ {model_name} works")
            success_count += 1
            
        except Exception as e:
            print(f"    ✗ {model_name} failed: {e}")
    
    print(f"  {success_count}/{len(model_variants)} models work")
    return success_count == len(model_variants)

def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Segformer Intensity Estimation Architecture")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()
    
    # Test model creation
    model = test_model_creation()


    if model is None:
        print("Model creation failed, stopping tests.")
        return
    
    # Test forward pass
    forward_success = test_forward_pass(model, device)
    
    print()
    
    # Test loss function
    loss_success = test_loss_function()
    
    print()
    
    # Test model parameters
    param_success = test_model_parameters(model, device)
    
    print()
    
    # Test different model variants
    variants_success = test_different_segformer_models(device)
    
    print()
    print("=" * 50)
    print("Test Summary:")
    print(f"  Model Creation: {'✓' if model is not None else '✗'}")
    print(f"  Forward Pass: {'✓' if forward_success else '✗'}")
    print(f"  Loss Function: {'✓' if loss_success else '✗'}")
    print(f"  Parameters: {'✓' if param_success else '✗'}")
    print(f"  Model Variants: {'✓' if variants_success else '✗'}")
    print("=" * 50)
    
    if all([model is not None, forward_success, loss_success, param_success, variants_success]):
        print("🎉 All tests passed! The architecture is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 