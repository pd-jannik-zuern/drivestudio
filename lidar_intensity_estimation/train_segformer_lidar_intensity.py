import os
import warnings

# Suppress TensorFlow warnings about missing TensorRT libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', message='.*libnvinfer_plugin.*')
warnings.filterwarnings('ignore', message='.*Could not load dynamic library.*')

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import argparse
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import wandb
from typing import Dict, Any

from lidar_intensity_dataloader import create_lidar_intensity_dataloader
from segformer_intensity import create_segformer_intensity_model, SegformerIntensityLoss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool = False
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        use_wandb: Whether to log to wandb
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        rgb = batch["rgb"].to(device)
        intensity = batch["intensity"].to(device)
        # depth = batch["depth"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        intensity_pred = model(rgb)
        
        # Compute loss
        intensity_pred = torch.nn.functional.interpolate(
            intensity_pred,
            size=intensity.shape[-2:],
            mode='nearest',
        )

        loss = criterion(intensity_pred, intensity, valid_mask)

        # cv2 imshow the RGB, intensity, intensity_pred, valid_mask
        rgb_np = rgb[0].permute(1, 2, 0).detach().cpu().numpy()
        intensity_np = intensity[0].squeeze().detach().cpu().numpy()
        intensity_pred_np = intensity_pred[0].squeeze().detach().cpu().numpy()
        valid_mask_np = valid_mask[0].squeeze().detach().cpu().numpy()

        # normalize intensity_pred and intensity to [0, 1]
        i_min, i_max = intensity_np.min(), intensity_np.max()
        i_min = -12
        i_max = 0
        intensity_np = (intensity_np - i_min) / (i_max - i_min)
        intensity_pred_np = (intensity_pred_np - i_min) / (i_max - i_min)

        # color-map intensity_np and intensity_pred_np
        intensity_np = cv2.applyColorMap((intensity_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
        intensity_pred_np = cv2.applyColorMap((intensity_pred_np * 255).astype(np.uint8), cv2.COLORMAP_JET)

        cv2.imshow("RGB", rgb_np)
        cv2.imshow("Intensity", intensity_np)
        cv2.imshow("Intensity Pred", intensity_pred_np)
        cv2.imshow("Valid Mask", valid_mask_np)
        cv2.waitKey(1)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
        
        # Log to wandb
        if use_wandb and batch_idx % 10 == 0:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/epoch': epoch,
                'train/batch': batch_idx
            })
    
    avg_loss = total_loss / num_batches
    
    metrics = {
        'train_loss': avg_loss
    }
    
    if use_wandb:
        wandb.log({
            'train/epoch_loss': avg_loss,
            'train/epoch': epoch
        })
    
    return metrics

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_wandb: bool = False
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        use_wandb: Whether to log to wandb
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            rgb = batch["rgb"].to(device)
            intensity = batch["intensity"].to(device)
            depth = batch["depth"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            
            # Forward pass
            intensity_pred = model(rgb)

            # resize intensity_pred to the same size as intensity
            intensity_pred = torch.nn.functional.interpolate(
                intensity_pred,
                size=intensity.shape[-2:],
                mode='nearest',
            )
            
            # Compute loss
            loss = criterion(intensity_pred, intensity, valid_mask)
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'val_loss': f'{loss.item():.4f}',
                'avg_val_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
    
    avg_loss = total_loss / num_batches
    
    metrics = {
        'val_loss': avg_loss
    }
    
    if use_wandb:
        wandb.log({
            'val/epoch_loss': avg_loss,
            'val/epoch': epoch
        })
    
    return metrics

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")

def visualize_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str,
    num_samples: int = 5
):
    """Visualize model predictions."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            # Move data to device
            rgb = batch["rgb"].to(device)
            intensity = batch["intensity"].to(device)
            depth = batch["depth"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            
            # Get predictions
            intensity_pred = model(rgb)
            
            # Visualize
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # RGB image
            rgb_np = rgb[0].permute(1, 2, 0).cpu().numpy()
            axes[0, 0].imshow(rgb_np)
            axes[0, 0].set_title("RGB Image")
            axes[0, 0].axis('off')
            
            # Ground truth intensity
            gt_intensity = intensity[0].squeeze().cpu().numpy()
            im1 = axes[0, 1].imshow(gt_intensity, cmap='viridis')
            axes[0, 1].set_title("Ground Truth Intensity")
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
            
            # Predicted intensity
            pred_intensity = intensity_pred[0].squeeze().cpu().numpy()
            im2 = axes[0, 2].imshow(pred_intensity, cmap='viridis')
            axes[0, 2].set_title("Predicted Intensity")
            axes[0, 2].axis('off')
            plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
            
            # Input depth
            input_depth = depth[0].squeeze().cpu().numpy()
            im3 = axes[1, 0].imshow(input_depth, cmap='plasma')
            axes[1, 0].set_title("Input Depth")
            axes[1, 0].axis('off')
            plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            # Valid mask
            mask = valid_mask[0].squeeze().cpu().numpy()
            axes[1, 1].imshow(mask, cmap='gray')
            axes[1, 1].set_title(f"Valid Mask ({mask.sum()} valid pixels)")
            axes[1, 1].axis('off')
            
            # Intensity error
            intensity_error = np.abs(gt_intensity - pred_intensity)
            im4 = axes[1, 2].imshow(intensity_error, cmap='hot')
            axes[1, 2].set_title("Intensity Error")
            axes[1, 2].axis('off')
            plt.colorbar(im4, ax=axes[1, 2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"prediction_{i}.png"), dpi=150, bbox_inches='tight')
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train Segformer for LiDAR intensity estimation")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset config")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--wandb_project", type=str, default="lidar-intensity-segformer", help="Wandb project name")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--val_interval", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--model_name", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512", help="Segformer model name")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone")
    parser.add_argument("--freeze_decoder", action="store_true", help="Freeze decoder")
    parser.add_argument("--use_depth", action="store_true", default=True, help="Use depth as input")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load configuration
    data_cfg = OmegaConf.load(args.config)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            entity="paralleldomain",
            project="lidar-intensity-estimation",
            config={
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "dataset_config": args.config,
                "model_name": args.model_name,
                "freeze_backbone": args.freeze_backbone,
                "freeze_decoder": args.freeze_decoder,
                "use_depth": args.use_depth
            }
        )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader = create_lidar_intensity_dataloader(
        data_cfg=data_cfg,
        split="train",
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        device=device,
        target_size=(640, 960)
    )
    
    # val_dataloader = create_lidar_intensity_dataloader(
    #     data_cfg=data_cfg,
    #     split="test",
    #     batch_size=args.batch_size,
    #     num_workers=4,
    #     shuffle=False,
    #     device=device,
        # target_size=(640, 960)
    # )


    val_dataloader = train_dataloader # use train dataloader for validation
    
    logger.info(f"Train dataloader: {len(train_dataloader)} batches")
    logger.info(f"Val dataloader: {len(val_dataloader)} batches")
    
    # Create model
    logger.info("Creating model...")
    model = create_segformer_intensity_model(
        model_name=args.model_name,
        use_depth=args.use_depth,
        # freeze_backbone=args.freeze_backbone,
        # freeze_decoder=args.freeze_decoder
    ).to(device)
    
    # Create loss function and optimizer
    criterion = SegformerIntensityLoss(loss_type="mse")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_dataloader, criterion, optimizer, device, epoch, args.use_wandb
        )
        
        # Update learning rate
        scheduler.step()
        
        # Validate
        if epoch % args.val_interval == 0:
            val_metrics = validate(
                model, val_dataloader, criterion, device, epoch, args.use_wandb
            )
            
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}, Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                save_checkpoint(
                    model, optimizer, epoch, val_metrics,
                    os.path.join(args.output_dir, "best_model.pth")
                )
                logger.info(f"New best model saved with val loss: {best_val_loss:.4f}")
        
        # Save checkpoint periodically
        if epoch % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, train_metrics,
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            )
    
    # Final validation and visualization
    logger.info("Training completed. Generating final visualizations...")
    visualize_predictions(
        model, val_dataloader, device,
        os.path.join(args.output_dir, "final_predictions")
    )
    
    if args.use_wandb:
        wandb.finish()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 