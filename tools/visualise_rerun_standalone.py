#!/usr/bin/env python3
"""
Standalone script for visualizing driving dataset in Rerun.
This script can be run independently to visualize lidar point clouds and camera poses over time.
"""

import argparse
import os
import sys
import time
import logging
from typing import Dict, List

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import rerun as rr
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from datasets.driving_dataset import DrivingDataset
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Please install required packages:")
    print("pip install rerun-sdk omegaconf torch numpy")
    sys.exit(1)

logger = logging.getLogger()

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def visualise_in_rerun(cfg, dataset):
    """
    Visualize the driving dataset in Rerun.
    
    Args:
        cfg: Configuration object
        dataset: DrivingDataset instance
    """
    try:
        # Initialize Rerun
        rr.init("drivestudio_visualization", spawn=True)
        logger.info("Rerun visualization started")
        
        # Set up the coordinate system
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)
        
        # Get dataset information
        num_frames = dataset.num_img_timesteps
        num_cams = dataset.num_cams
        camera_list = dataset.pixel_source.camera_list
        
        logger.info(f"Visualizing {num_frames} frames with {num_cams} cameras")
        
        # Visualize camera poses over time
        visualize_camera_poses(dataset, num_frames, camera_list)
        
        # Visualize lidar point clouds over time
        visualize_lidar_pointclouds(dataset, num_frames)
        
        # # Visualize instance bounding boxes if available
        if hasattr(dataset.pixel_source, 'instances_pose') and dataset.pixel_source.instances_pose is not None:
            visualize_instance_boxes(dataset, num_frames)
        
        logger.info("Rerun visualization setup complete. Check the Rerun viewer.")
        
    except Exception as e:
        logger.error(f"Failed to initialize Rerun visualization: {e}")
        raise

def visualize_camera_poses(dataset, num_frames: int, camera_list: List[int]):
    """
    Visualize camera poses over time.
    
    Args:
        dataset: DrivingDataset instance
        num_frames: Number of frames
        camera_list: List of camera IDs
    """
    logger.info("Setting up camera pose visualization...")
    
    for cam_id in camera_list:
        cam_data = dataset.pixel_source.camera_data[cam_id]
        cam_name = cam_data.cam_name
        
        # Log camera trajectory
        for frame_idx in range(min(num_frames, len(cam_data))):
            # Get camera pose (4x4 transformation matrix)
            cam_to_world = cam_data.cam_to_worlds[frame_idx].cpu().numpy()
            intrinsics = cam_data.intrinsics[frame_idx].cpu().numpy()
            
            # Extract position and orientation
            position = cam_to_world[:3, 3]
            rotation_matrix = cam_to_world[:3, :3]


            rr.log(
                f"world/cameras/{cam_name}",
                rr.Transform3D(translation=position, mat3x3=rotation_matrix, axis_length=2),
                rr.ViewCoordinates.RDF,
                rr.Pinhole(
                    resolution=[1920, 1080],
                    focal_length=[intrinsics[0, 0], intrinsics[1, 1]],
                    principal_point=[intrinsics[0, 2], intrinsics[1, 2]],
                ),
            )


            
            # # Log camera frustum (optional)
            # if frame_idx % 10 == 0:  # Log every 10th frame to avoid clutter
            #     visualize_camera_frustum(cam_data, frame_idx, cam_name)

def visualize_camera_frustum(cam_data, frame_idx: int, cam_name: str):
    """
    Visualize camera frustum for debugging.
    
    Args:
        cam_data: Camera data
        frame_idx: Frame index
        cam_name: Camera name
    """
    try:
        rr.set_time("frame_nr", sequence=frame_idx)


        # Get camera intrinsics
        intrinsic = cam_data.intrinsics[frame_idx].cpu().numpy()
        cam_to_world = cam_data.cam_to_worlds[frame_idx].cpu().numpy()
        
        # Create frustum points (simplified)
        width, height = cam_data.WIDTH, cam_data.HEIGHT
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        # Create frustum corners at different depths
        depths = [1.0, 10.0, 50.0]  # Near, middle, far planes
        
        for depth in depths:
            # Frustum corners in camera space
            corners_cam = np.array([
                [-(width/2 - cx) * depth / fx, -(height/2 - cy) * depth / fy, depth],
                [ (width/2 - cx) * depth / fx, -(height/2 - cy) * depth / fy, depth],
                [ (width/2 - cx) * depth / fx,  (height/2 - cy) * depth / fy, depth],
                [-(width/2 - cx) * depth / fx,  (height/2 - cy) * depth / fy, depth],
            ])
            
            # Transform to world space
            corners_world = transform_points(corners_cam, cam_to_world)
            
            # Log frustum as points
            rr.log(
                f"world/cameras/{cam_name}/frustum_{depth}m",
                rr.Points3D(
                    positions=corners_world,
                    colors=[255, 0, 0] if depth == 1.0 else [0, 255, 0] if depth == 10.0 else [0, 0, 255],
                    radii=0.1
                ),
                # rr.Time.from_time(frame_idx)
            )
    except Exception as e:
        logger.warning(f"Failed to visualize camera frustum: {e}")

def visualize_lidar_pointclouds(dataset, num_frames: int):
    """
    Visualize lidar point clouds over time.
    
    Args:
        dataset: DrivingDataset instance
        num_frames: Number of frames
    """
    logger.info("Setting up lidar point cloud visualization...")
    
    if dataset.lidar_source is None:
        logger.warning("No lidar source available")
        return
    
    # Get unique timestamps
    unique_timestamps = dataset.lidar_source.unique_normalized_timestamps
    
    for frame_idx in range(min(num_frames, len(unique_timestamps))):
        rr.set_time("frame_nr", sequence=frame_idx)


        print(frame_idx)
        try:
            # Get lidar data for this frame
            lidar_dict = dataset.lidar_source.get_lidar_rays(frame_idx)
            
            # Compute 3D points
            lidar_pts = lidar_dict["lidar_origins"] + lidar_dict["lidar_viewdirs"] * lidar_dict["lidar_ranges"]
            
            # Get colors if available
            if dataset.lidar_source.colors is not None:
                colors = dataset.lidar_source.colors[lidar_dict["lidar_mask"]].cpu().numpy()
            else:
                # Use intensity-based colors
                ranges = lidar_dict["lidar_ranges"].cpu().numpy()
                colors = np.zeros((len(ranges), 3))
                colors[:, 0] = np.clip(ranges / 100.0, 0, 1) * 255  # Red channel based on range
                colors[:, 1] = 255 - colors[:, 0]  # Green channel inverse to red
                colors[:, 2] = 128  # Blue channel constant
            
            # Convert to numpy
            lidar_pts = lidar_pts.cpu().numpy()
            
            # Downsample for performance (optional)
            num_max = 1000000
            if len(lidar_pts) > num_max:
                indices = np.random.choice(len(lidar_pts), num_max, replace=False)
                lidar_pts = lidar_pts[indices]
                colors = colors[indices]
            
            # Log point cloud
            rr.log(
                "world/lidar",
                rr.Points3D(
                    positions=lidar_pts,
                    colors=colors.astype(np.uint8),
                    radii=0.05
                ),
            )
            
        except Exception as e:
            logger.warning(f"Failed to visualize lidar frame {frame_idx}: {e}")

def visualize_instance_boxes(dataset, num_frames: int):
    """
    Visualize instance bounding boxes over time.
    
    Args:
        dataset: DrivingDataset instance
        num_frames: Number of frames
    """
    logger.info("Setting up instance bounding box visualization...")
    
    instances_pose = dataset.pixel_source.instances_pose
    instances_size = dataset.pixel_source.instances_size
    per_frame_instance_mask = dataset.pixel_source.per_frame_instance_mask
    
    if instances_pose is None:
        return

    
    num_instances = instances_pose.shape[1]
    
    for frame_idx in range(min(num_frames, instances_pose.shape[0])):
        rr.set_time("frame_nr", sequence=frame_idx)


        for instance_id in range(num_instances):
            # Check if instance is active in this frame
            if not per_frame_instance_mask[frame_idx, instance_id]:
                continue
            
            try:
                # Get instance pose and size
                o2w = instances_pose[frame_idx, instance_id].cpu().numpy()
                size = instances_size[instance_id].cpu().numpy()
                
                # Create bounding box corners
                half_size = size / 2.0

                rr.log(
                    f"world/instances/{instance_id}/bbox",
                    rr.Boxes3D(
                        centers=o2w[:3, 3],
                        half_sizes=half_size.reshape(1, 3),
                        # quaternions=rotation_matrix_to_quaternion(o2w[:3, :3]),
                        quaternions=o2w[:3, :3],
                        radii=0.025,
                        colors=[(0, 0, 255)],
                        labels=[f"instance_{instance_id}"],
                    ),
                )



                
                
            except Exception as e:
                logger.warning(f"Failed to visualize instance {instance_id} at frame {frame_idx}: {e}")

def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion.
    
    Args:
        rotation_matrix: 3x3 rotation matrix
        
    Returns:
        Quaternion as [w, x, y, z]
    """
    # Simple conversion - for more robust conversion, consider using scipy
    trace = np.trace(rotation_matrix)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        s = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
        w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        x = 0.25 * s
        y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        s = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
        w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
        y = 0.25 * s
        z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
        w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])

def transform_points(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Transform points using a 4x4 transformation matrix.
    
    Args:
        points: Points to transform, shape (N, 3)
        transform_matrix: 4x4 transformation matrix
        
    Returns:
        Transformed points, shape (N, 3)
    """
    # Add homogeneous coordinate
    points_homogeneous = np.column_stack([points, np.ones(len(points))])
    
    # Apply transformation
    transformed_points_homogeneous = (transform_matrix @ points_homogeneous.T).T
    
    # Remove homogeneous coordinate
    return transformed_points_homogeneous[:, :3]

def main():
    """Main function to run the visualization."""
    parser = argparse.ArgumentParser(description="Visualize driving dataset in Rerun")
    parser.add_argument("--config_file", required=True, help="Path to config file")
    parser.add_argument("--dataset", help="Dataset type (e.g., waymo, nuscenes)")
    parser.add_argument("--scene_idx", type=int, default=0, help="Scene index to visualize")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to visualize")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        # Load config
        cfg = OmegaConf.load(args.config_file)
        
        # Parse dataset config if specified
        if args.dataset:
            dataset_cfg = OmegaConf.load(os.path.join("configs", "datasets", f"{args.dataset}.yaml"))
            dataset_cfg.data.start_timestep = 0
            # dataset_cfg.end_timestep = args.max_frames
            dataset_cfg.data.end_timestep = 20

            cfg = OmegaConf.merge(cfg, dataset_cfg)

        
        # Set scene index
        if "data" in cfg:
            cfg.data.scene_idx = args.scene_idx
        else:
            cfg.scene_idx = args.scene_idx
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Build dataset
        logger.info("Building dataset...")
        dataset = DrivingDataset(data_cfg=cfg.data)
        # dataset.to(device)
        
        # Visualize in Rerun
        visualise_in_rerun(cfg, dataset)
        
        logger.info("Visualization complete! The Rerun viewer should be open.")
        logger.info("Press Ctrl+C to exit.")
        
        # Keep the script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Exiting...")
            
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        raise

if __name__ == "__main__":
    main() 