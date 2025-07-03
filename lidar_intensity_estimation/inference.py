import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from omegaconf import OmegaConf
import torchvision.transforms as transforms

from segformer_intensity import create_segformer_intensity_model

def load_model(checkpoint_path: str, device: torch.device):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Create model
    model = create_segformer_intensity_model(
        model_name="nvidia/segformer-b0-finetuned-ade-512-512",
        use_depth=True,
        freeze_backbone=False,
        freeze_decoder=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['metrics']['val_loss']:.4f}")
    
    return model

def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> torch.Tensor:
    """
    Preprocess an image for inference.
    
    Args:
        image_path: Path to the input image
        target_size: Target size for the image (H, W)
        
    Returns:
        Preprocessed image tensor (1, 3, H, W)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def predict_intensity_with_depth(model, image_tensor: torch.Tensor, depth_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Predict LiDAR intensity for an input image using RGB + depth.
    
    Args:
        model: Trained model
        image_tensor: Input image tensor (1, 3, H, W)
        depth_tensor: Input depth tensor (1, 1, H, W)
        device: Device to run inference on
        
    Returns:
        Predicted intensity tensor (1, 1, H, W)
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        depth_tensor = depth_tensor.to(device)
        intensity_pred = model(image_tensor, depth_tensor)
    
    return intensity_pred

def visualize_prediction(image_tensor: torch.Tensor, intensity_pred: torch.Tensor, save_path: str = None):
    """
    Visualize the prediction.
    
    Args:
        image_tensor: Input image tensor (1, 3, H, W)
        intensity_pred: Predicted intensity tensor (1, 1, H, W)
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Input image
    image_np = image_tensor[0].permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(image_np)
    axes[0].set_title("Input RGB Image")
    axes[0].axis('off')
    
    # Predicted intensity
    intensity_np = intensity_pred[0].squeeze().cpu().numpy()
    im = axes[1].imshow(intensity_np, cmap='viridis')
    axes[1].set_title("Predicted LiDAR Intensity")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def batch_inference(model, image_dir: str, output_dir: str, device: torch.device):
    """
    Run inference on a batch of images.
    
    Args:
        model: Trained model
        image_dir: Directory containing input images
        output_dir: Directory to save predictions
        device: Device to run inference on
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    for i, image_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {image_file}")
        
        # Load and preprocess image
        image_path = os.path.join(image_dir, image_file)
        image_tensor = preprocess_image(image_path)
        
        # Create dummy depth (you can replace this with actual depth data)
        depth_tensor = torch.randn(1, 1, 224, 224)  # Dummy depth
        
        # Predict intensity
        intensity_pred = predict_intensity_with_depth(model, image_tensor, depth_tensor, device)
        
        # Save prediction
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_intensity.png")
        visualize_prediction(image_tensor, intensity_pred, output_path)
        
        # Also save raw prediction as numpy array
        intensity_np = intensity_pred[0].squeeze().cpu().numpy()
        np.save(os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_intensity.npy"), intensity_np)

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained Segformer LiDAR intensity model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, help="Path to single input image")
    parser.add_argument("--image_dir", type=str, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="inference_outputs", help="Output directory")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224], help="Input image size (H W)")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.image:
        # Single image inference
        print(f"Processing single image: {args.image}")
        
        # Load and preprocess image
        image_tensor = preprocess_image(args.image, tuple(args.image_size))
        
        # Create dummy depth (you can replace this with actual depth data)
        depth_tensor = torch.randn(1, 1, args.image_size[0], args.image_size[1])  # Dummy depth
        
        # Predict intensity
        intensity_pred = predict_intensity_with_depth(model, image_tensor, depth_tensor, device)
        
        # Visualize and save
        output_path = os.path.join(args.output_dir, "prediction.png")
        visualize_prediction(image_tensor, intensity_pred, output_path)
        
        # Save raw prediction
        intensity_np = intensity_pred[0].squeeze().cpu().numpy()
        np.save(os.path.join(args.output_dir, "intensity_prediction.npy"), intensity_np)
        
    elif args.image_dir:
        # Batch inference
        print(f"Processing images in directory: {args.image_dir}")
        batch_inference(model, args.image_dir, args.output_dir, device)
        
    else:
        print("Please provide either --image or --image_dir")

if __name__ == "__main__":
    main() 