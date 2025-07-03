# LiDAR Intensity Estimation with Segformer

This project implements LiDAR intensity estimation from RGB images and LiDAR depth maps using the Segformer model from HuggingFace. The implementation reuses the existing dataloaders from the DriveStudio repository to load paired RGB images, LiDAR depth maps, and **projected LiDAR intensity maps**.

## New Feature: LiDAR Intensity Projection

The dataloader now supports **automatic projection of LiDAR intensity values onto camera images**. This feature:

- **Projects LiDAR intensity data** onto the image plane using the same geometric transformations as depth projection
- **Works with all supported datasets** (Waymo, KITTI, NuScenes, ArgoVerse, PandaSet, NuPlan)
- **Provides fallback mechanisms** when intensity data is not available
- **Can be enabled/disabled** via the `project_intensity` parameter
- **Stores pre-computed intensity maps** for efficient loading during training

## Overview

The project consists of four main components:

1. **LiDAR Intensity Dataloader** (`lidar_intensity_dataloader.py`): A custom dataloader that loads RGB images, LiDAR depth maps, and **automatically projects LiDAR intensity values** onto the image plane
2. **Segformer Architecture** (`segformer_intensity.py`): A modified Segformer model that replaces the semantic segmentation head with an intensity regression head
3. **Test Scripts** (`test_dataloader.py`, `test_segformer_architecture.py`): Scripts to test and visualize the dataloader output and model architecture
4. **Training Script** (`train_segformer_lidar_intensity.py`): A complete training pipeline using Segformer for intensity estimation

## Installation

1. Install the base requirements from the DriveStudio repository:
```bash
pip install -r requirements.txt
```

2. Install additional requirements for LiDAR intensity estimation:
```bash
pip install -r requirements_lidar_intensity.txt
```

## Configuration

The training script expects a dataset configuration file. You can use any of the existing configuration files in the `configs/datasets/` directory, or create your own.

### Configuration Structure

The configuration should have the following structure:

```yaml
data:
  data_root: data/waymo/processed/training  # Path to your dataset
  dataset: waymo  # Dataset type: waymo, nuscenes, kitti, argoverse, pandaset, nuplan
  scene_idx: 0  # Scene index to use
  start_timestep: 0  # Start timestep
  end_timestep: -1  # End timestep (-1 means use all)
  preload_device: cuda  # Device to preload data on
  
  pixel_source:
    type: datasets.waymo.waymo_sourceloader.WaymoPixelSource
    cameras: [0]  # Which cameras to use
    downscale_when_loading: [1]  # Image loading scale
    downscale: 1  # Downscale factor
    undistort: True  # Whether to undistort images
    test_image_stride: 0  # Test image stride
    load_sky_mask: True  # Load sky masks
    load_dynamic_mask: True  # Load dynamic masks
    load_objects: True  # Load object annotations
    load_smpl: True  # Load SMPL templates
    
  lidar_source:
    type: datasets.waymo.waymo_sourceloader.WaymoLiDARSource
    load_lidar: True  # Whether to load LiDAR
    only_use_top_lidar: False  # Use only top LiDAR
    truncated_max_range: 80  # Max range
    truncated_min_range: -2  # Min range
    lidar_downsample_factor: 4  # Downsample factor for AABB computation
    lidar_percentile: 0.02  # Percentile for AABB computation
```

An example configuration file `example_config.yaml` is provided in this directory.

## Usage

### 0. Test Intensity Projection

Test the new intensity projection functionality:

```bash
python test_intensity_projection.py
```

This will test both with and without intensity projection enabled.

### 1. Test the Dataloader

First, test the dataloader to ensure it works correctly with your dataset:

```bash
python test_dataloader.py --config configs/datasets/waymo/1cams.yaml --num_samples 5
```

This will:
- Load the dataset using the specified configuration
- Create visualizations showing RGB images, LiDAR intensity maps, depth maps, and valid masks
- Save the visualizations to `test_visualizations/`

### 2. Test the Architecture

Test the Segformer architecture to ensure it works correctly:

```bash
python test_segformer_architecture.py
```

This will verify that:
- The model can be created successfully
- Forward pass works with dummy data
- Loss function computes correctly
- Parameters can be accessed and gradients computed
- Different Segformer variants work

### 3. Train the Model

To train the Segformer model for LiDAR intensity estimation:

```bash
python train_segformer_lidar_intensity.py \
    --config configs/datasets/waymo/1cams.yaml \
    --output_dir outputs \
    --batch_size 4 \
    --num_epochs 50 \
    --lr 1e-4 \
    --use_wandb \
    --model_name nvidia/segformer-b0-finetuned-ade-512-512 \```

#### Training Parameters

- `--config`: Path to the dataset configuration file
- `--output_dir`: Directory to save model checkpoints and outputs
- `--batch_size`: Training batch size (default: 4)
- `--num_epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay for optimizer (default: 1e-4)
- `--use_wandb`: Enable wandb logging
- `--wandb_project`: Wandb project name (default: "lidar-intensity-segformer")
- `--save_interval`: Save checkpoint every N epochs (default: 5)
- `--val_interval`: Validate every N epochs (default: 1)
- `--model_name`: Segformer model name (default: "nvidia/segformer-b0-finetuned-ade-512-512")
- `--freeze_backbone`: Freeze the backbone for transfer learning
- `--freeze_decoder`: Freeze the decoder for transfer learning
- `--use_depth`: Use depth as additional input (default: True)
- `--project_intensity`: Enable LiDAR intensity projection (default: True)

## Model Architecture

The `SegformerForIntensityEstimation` model:

1. **Uses Segformer as backbone**: Leverages the pre-trained Segformer encoder-decoder for RGB feature extraction
2. **Depth encoder** (optional): A separate CNN encoder to process LiDAR depth maps
3. **Feature fusion**: Combines RGB and depth features when depth is used
4. **Intensity regression head**: Replaces the semantic segmentation head with a regression head for per-pixel intensity values
5. **Flexible training**: Option to freeze backbone/decoder for transfer learning
6. **Masked loss**: Only computes loss on pixels where LiDAR intensity is available

### Key Features

- **Direct Architecture Integration**: Modifies Segformer's final layers instead of adding external heads
- **Flexible Depth Usage**: Can use RGB-only or RGB+depth inputs
- **Transfer Learning Support**: Option to freeze backbone/decoder for efficient training
- **Multiple Loss Options**: MSE, L1, or SmoothL1 loss functions
- **Multiple Model Variants**: Support for different Segformer sizes (B0, B1, B2, etc.)

## LiDAR Intensity Projection

### How It Works

The intensity projection feature works in two modes:

1. **Pre-computed Mode** (default): Projects all intensity data upfront and stores it in memory
2. **On-the-fly Mode** (fallback): Projects intensity data during data loading if pre-computed maps are not available

### Projection Process

1. **Load LiDAR Data**: Extract intensity values from the lidar source
2. **Geometric Transformation**: Apply the same camera projection matrix used for depth projection
3. **Image Plane Mapping**: Map 3D lidar points to 2D image coordinates
4. **Intensity Assignment**: Assign intensity values to corresponding image pixels
5. **Validation**: Only use points that are visible in the camera view

### Usage

```python
# Enable intensity projection (default)
dataloader = create_lidar_intensity_dataloader(
    data_cfg=config,
    project_intensity=True  # This is the default
)

# Disable intensity projection (uses zeros as fallback)
dataloader = create_lidar_intensity_dataloader(
    data_cfg=config,
    project_intensity=False
)
```

### Supported Datasets

The intensity projection works with all datasets that provide intensity data:

- **Waymo**: ✅ Full support (intensity in column 11)
- **PandaSet**: ✅ Full support (intensity in column 3)  
- **ArgoVerse**: ✅ Full support (intensity in column 3)
- **NuPlan**: ✅ Full support (intensity in column 3)
- **KITTI**: ⚠️ Limited support (may need custom preprocessing)
- **NuScenes**: ⚠️ Limited support (may need custom preprocessing)

### Performance Considerations

- **Memory Usage**: Pre-computed mode uses more memory but faster loading
- **Computation Time**: Initial projection takes time but subsequent loads are fast
- **Coverage**: Intensity maps typically have sparse coverage (5-20% of pixels)

## Data Format

The dataloader outputs batches with the following structure:

```python
{
    "rgb": torch.Tensor,        # RGB images (B, 3, H, W) in [0, 1] range
    "intensity": torch.Tensor,  # LiDAR intensity maps (B, 1, H, W) in [0, 1] range
    "depth": torch.Tensor,      # LiDAR depth maps (B, 1, H, W) in [0, 1] range
    "valid_mask": torch.Tensor, # Valid mask (B, 1, H, W) indicating where intensity is available
    "image_idx": torch.Tensor   # Image indices (B,)
}
```

## Loss Function

The training uses a `SegformerIntensityLoss` that:
- Computes MSE/L1/SmoothL1 loss between predicted and ground truth intensity
- Only considers pixels where LiDAR intensity is available (valid_mask > 0)
- Normalizes the loss by the number of valid pixels

## Outputs

The training script generates:

1. **Model checkpoints**: Saved periodically and for the best validation loss
2. **Training logs**: Console output and optional wandb logging
3. **Final predictions**: Visualizations of model predictions vs ground truth
4. **Best model**: The model with the lowest validation loss

## Supported Datasets

The implementation works with all datasets supported by DriveStudio:
- Waymo
- KITTI
- NuScenes
- ArgoVerse
- PandaSet
- NuPlan

## Customization

### Using Different Segformer Models

You can change the Segformer model by modifying the `model_name` parameter:

```python
model = create_segformer_intensity_model(
    model_name="nvidia/segformer-b1-finetuned-ade-512-512",
    use_depth=True,
    freeze_backbone=False,
    freeze_decoder=False
)
```

Available models:
- `nvidia/segformer-b0-finetuned-ade-512-512` (smallest, fastest)
- `nvidia/segformer-b1-finetuned-ade-512-512`
- `nvidia/segformer-b2-finetuned-ade-512-512`
- `nvidia/segformer-b3-finetuned-ade-512-512`
- `nvidia/segformer-b4-finetuned-ade-512-512` (largest, most accurate)

### Modifying the Regression Head

You can customize the intensity regression head by modifying the `intensity_head` in the `SegformerForIntensityEstimation` class:

```python
self.intensity_head = nn.Sequential(
    nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(hidden_dim // 4, 1, kernel_size=1),  # Single channel for intensity
    nn.Sigmoid()  # Output in [0, 1] range
)
```

### Testing the Architecture

You can test the new architecture using the provided test script:

```bash
python test_segformer_architecture.py
```

This will verify that:
- The model can be created successfully
- Forward pass works with dummy data
- Loss function computes correctly
- Parameters can be accessed and gradients computed
- Different Segformer variants work

### Adding Data Augmentation

You can add data augmentation by implementing transforms and passing them to the dataloader:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

dataloader = create_lidar_intensity_dataloader(
    data_cfg=data_cfg,
    transform=transform,
    # ... other parameters
)
```

## Inference

To run inference with a trained model:

```bash
python inference.py \
    --checkpoint outputs/best_model.pth \
    --image path/to/image.jpg \
    --output_dir inference_outputs
```

For batch inference:

```bash
python inference.py \
    --checkpoint outputs/best_model.pth \
    --image_dir path/to/images/ \
    --output_dir inference_outputs
```

## Troubleshooting

### Memory Issues

If you encounter memory issues:
- Reduce the batch size
- Use a smaller Segformer model (B0 instead of B4)
- Use gradient accumulation
- Reduce the image resolution in the dataset config

### No Valid Pixels

If you see "0 valid pixels" in the output:
- Check that LiDAR data is properly loaded
- Verify the camera calibration and projection matrices
- Ensure the LiDAR points are within the camera field of view

### Slow Training

To speed up training:
- Use multiple GPUs with DataParallel or DistributedDataParallel
- Increase the number of workers in the dataloader
- Use mixed precision training (FP16)
- Use a smaller Segformer model

### Model Loading Issues

If you encounter issues loading the Segformer model:
- Ensure you have the correct transformers version
- Check that the model name is valid
- Try downloading the model manually first

## License

This implementation follows the same license as the DriveStudio repository. 