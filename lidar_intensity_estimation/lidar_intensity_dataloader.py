import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from typing import Dict, Tuple, Optional, List, Any
import logging
from tqdm import tqdm
from PIL import Image

# Import from the existing repository
from datasets.driving_dataset import DrivingDataset

logger = logging.getLogger(__name__)

class LiDARIntensityDataset(Dataset):
    """
    Dataset for LiDAR intensity estimation from RGB images and LiDAR depth maps.
    """
    
    def __init__(
        self,
        data_cfg: OmegaConf,
        split: str = "train",
        transform: Optional[nn.Module] = None,
        target_size: Tuple[int, int] = (224, 224),
        device: torch.device = torch.device("cpu"),
        project_intensity: bool = True
    ):
        """
        Initialize the LiDAR intensity dataset.
        
        Args:
            data_cfg: Dataset configuration
            split: Dataset split ("train", "val", "test")
            transform: Optional transforms to apply
            target_size: Target size for images (H, W)
            device: Device to load data on
            project_intensity: Whether to project lidar intensity onto images
        """
        self.data_cfg = data_cfg
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.device = device
        self.project_intensity = project_intensity
        
        # Clear CUDA cache to prevent initialization errors
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Create the base driving dataset
        # The data_cfg should have a 'data' section with the actual configuration
        if hasattr(data_cfg, 'data'):
            # If the config has a 'data' section, use that
            actual_data_cfg = data_cfg.data
        else:
            # Otherwise, assume the config is already in the correct format
            actual_data_cfg = data_cfg
            
        self.base_dataset = DrivingDataset(actual_data_cfg)
                
        # Get the appropriate image set based on split
        if split == "train":
            self.image_set = self.base_dataset.train_image_set
        elif split == "test":
            self.image_set = self.base_dataset.test_image_set
        elif split == "full":
            self.image_set = self.base_dataset.full_image_set
        else:
            raise ValueError(f"Unknown split: {split}")
        
        logger.info(f"Created LiDAR intensity dataset with {len(self.image_set)} samples for {split} split")
        
    def __len__(self) -> int:
        return len(self.image_set)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - rgb: RGB image (3, H, W) in [0, 1] range
                - intensity: LiDAR intensity map (1, H, W) in [0, 1] range
                - depth: LiDAR depth map (1, H, W) in [0, 1] range
                - valid_mask: Valid mask (1, H, W) indicating where intensity is available
                - image_idx: Image index
        """
        # Get image from the image set
        img_info, _ = self.image_set.get_image(idx, camera_downscale=1.0)
        
        # Extract RGB image
        rgb = img_info["pixels"]  # Should be (H, W, 3) in [0, 1] range
        rgb = rgb.cpu().numpy()
        
        # Get LiDAR data if available
        intensity = None
        depth = None
        
        # Try to get LiDAR depth and intensity from the camera data
        cam_id, frame_idx = self.base_dataset.pixel_source.parse_img_idx(idx)
        camera = self.base_dataset.pixel_source.camera_data[cam_id]
        
        # Get depth data
        if hasattr(camera, 'lidar_depth_maps') and camera.lidar_depth_maps is not None:
            depth = camera.lidar_depth_maps[frame_idx].cpu().numpy()
        else:
            depth = np.zeros((camera.HEIGHT, camera.WIDTH), dtype=np.float32)
        
        # Get intensity data
        if hasattr(camera, 'lidar_intensity_maps') and camera.lidar_intensity_maps is not None:
            intensity = camera.lidar_intensity_maps[frame_idx].cpu().numpy()
        else:
            intensity = np.zeros((camera.HEIGHT, camera.WIDTH), dtype=np.float32)
        
        # Ensure proper shape and range
        if intensity.ndim == 2:
            intensity = intensity[..., None]  # Add channel dimension
        if depth.ndim == 2:
            depth = depth[..., None]  # Add channel dimension

        # normalize nonzero intensity to log scale
        intensity[intensity > 0] = np.log(intensity[intensity > 0] + 1e-6)
        
        # Normalize depth to [0, 1] range if not already
        if depth.max() > 1.0:
            depth = depth / depth.max()
        
        # Resize to target size
        print(f"Resizing from {rgb.shape} to {self.target_size}")
        rgb = self._resize_image(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        intensity = self._resize_image(intensity, self.target_size, interpolation=cv2.INTER_NEAREST)
        depth = self._resize_image(depth, self.target_size, interpolation=cv2.INTER_NEAREST)

        valid_mask = (intensity != 0).astype(np.float32)
        
        # Convert to tensors and move to device
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()  # (3, H, W)
        intensity_tensor = torch.from_numpy(intensity)[None, ...].squeeze(-1).float()  # (1, H, W)
        depth_tensor = torch.from_numpy(depth)[None, ...].float().squeeze(-1)  # (1, H, W)
        valid_mask_tensor = torch.from_numpy(valid_mask)[None, ...].float().squeeze(-1)  # (1, H, W)
        

        # Apply transforms if provided
        if self.transform is not None:
            rgb_tensor = self.transform(rgb_tensor)
            intensity_tensor = self.transform(intensity_tensor)
            depth_tensor = self.transform(depth_tensor)
            valid_mask_tensor = self.transform(valid_mask_tensor)
        
        return {
            "rgb": rgb_tensor,
            "intensity": intensity_tensor,
            "depth": depth_tensor,
            "valid_mask": valid_mask_tensor,
            "image_idx": torch.tensor(idx, dtype=torch.long)
        }
    
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int], interpolation = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            target_size: Target size (H, W)
            
        Returns:
            Resized image
        """
        if image.shape[:2] != target_size:
            if image.ndim == 3:
                # Multi-channel image
                resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
            else:
                # Single-channel image
                resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)
                resized = resized[..., None]  # Add channel dimension
            return resized
        return image


def create_lidar_intensity_dataloader(
    data_cfg: OmegaConf,
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    transform: Optional[nn.Module] = None,
    target_size: Tuple[int, int] = (224, 224),
    device: torch.device = torch.device("cpu"),
    project_intensity: bool = True
) -> DataLoader:
    """
    Create a DataLoader for LiDAR intensity estimation.
    
    Args:
        data_cfg: Dataset configuration
        split: Dataset split ("train", "val", "test")
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the dataset
        transform: Optional transforms to apply
        target_size: Target size for images (H, W)
        device: Device to load data on
        project_intensity: Whether to project lidar intensity onto images
        
    Returns:
        DataLoader for LiDAR intensity estimation
    """
    # Create dataset
    dataset = LiDARIntensityDataset(
        data_cfg=data_cfg,
        split=split,
        transform=transform,
        target_size=target_size,
        device=device,
        project_intensity=project_intensity
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Force single process to avoid CUDA issues
        pin_memory=False,  # Disable pin_memory to avoid CUDA issues
        drop_last=True if split == "train" else False
    )
    
    return dataloader 