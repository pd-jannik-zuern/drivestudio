import torch
import torch.nn as nn
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from typing import Optional, Tuple, Union

class SegformerForIntensityEstimation(nn.Module):
    """
    Segformer model modified for LiDAR intensity estimation.
    Replaces the semantic segmentation head with an intensity regression head.
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        use_depth: bool = False,
    ):
        super().__init__()
        
        # Load the base Segformer model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)

        self.model.decode_head.classifier = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        self.model.decode_head.activation = nn.Sigmoid() 



    def forward(
        self,
        pixel_values: torch.Tensor,
        # depth_values: Optional[torch.Tensor] = None,
        # output_hidden_states: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Forward pass for intensity prediction.
        
        Args:
            pixel_values: RGB images (B, 3, H, W)
            depth_values: Depth maps (B, 1, H, W), optional
            output_hidden_states: Whether to output hidden states
            output_attentions: Whether to output attentions
            return_dict: Whether to return as dict
            
        Returns:
            intensity_pred: Predicted intensity map (B, 1, H, W)
        """
        # Get the base Segformer features


        
        with torch.no_grad() if not self.training else torch.enable_grad():
            inputs = self.processor(pixel_values, return_tensors="pt")
            output = self.model(pixel_values)

            intensity_pred = output.logits

            return intensity_pred


    def get_processor(self):
        """Get the processor for preprocessing inputs."""
        return self.processor
    
    def preprocess_inputs(self, images, depths=None):
        """
        Preprocess inputs using the Segformer processor.
        
        Args:
            images: List of PIL images or numpy arrays
            depths: List of depth maps (optional)
            
        Returns:
            processed_inputs: Processed inputs ready for the model
        """
        # Process images
        processed_inputs = self.processor(
            images=images,
            return_tensors="pt"
        )
        
        # Add depth if provided
        if depths is not None:
            # Ensure depths are in the right format
            if isinstance(depths, list):
                depths = torch.stack([torch.tensor(d) for d in depths])
            processed_inputs["depth_values"] = depths
            
        return processed_inputs


class SegformerIntensityLoss(nn.Module):
    """
    Loss function for intensity estimation with masking support.
    """
    
    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked loss.
        
        Args:
            predictions: Predicted intensity (B, 1, H, W)
            targets: Target intensity (B, 1, H, W)
            valid_mask: Valid mask (B, 1, H, W)
            
        Returns:
            loss: Masked loss value
        """
        # Compute loss for all pixels
        loss = self.loss_fn(predictions, targets)
        
        # Apply mask
        masked_loss = loss * valid_mask.float()
        
        # Compute mean only over valid pixels
        num_valid = valid_mask.sum()
        if num_valid > 0:
            total_loss = masked_loss.sum() / num_valid
        else:
            total_loss = masked_loss.sum() * 0.0
        
        return total_loss


def create_segformer_intensity_model(
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    use_depth: bool = True,
) -> SegformerForIntensityEstimation:
    """
    Factory function to create a Segformer model for intensity estimation.
    
    Args:
        model_name: Name of the pre-trained Segformer model
        use_depth: Whether to use depth as additional input
        freeze_backbone: Whether to freeze the backbone
        freeze_decoder: Whether to freeze the decoder
        
    Returns:
        Configured Segformer model for intensity estimation
    """
    return SegformerForIntensityEstimation(
        model_name=model_name,
        use_depth=use_depth,
    ) 