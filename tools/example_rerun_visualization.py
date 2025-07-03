#!/usr/bin/env python3
"""
Example script showing how to use the Rerun visualization for driving datasets.

This script demonstrates how to visualize:
1. Camera poses over time
2. LiDAR point clouds over time
3. Instance bounding boxes (if available)

Usage:
    python tools/example_rerun_visualization.py --config_file configs/omnire.yaml --dataset waymo --scene_idx 0
"""

import os
import sys
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.visualise_rerun_standalone import main as run_visualization

def main():
    """Example usage of the rerun visualization."""
    parser = argparse.ArgumentParser(description="Example Rerun visualization for driving datasets")
    parser.add_argument("--config_file", required=True, help="Path to config file")
    parser.add_argument("--dataset", required=True, help="Dataset type (e.g., waymo, nuscenes, kitti)")
    parser.add_argument("--scene_idx", type=int, default=0, help="Scene index to visualize")
    parser.add_argument("--max_frames", type=int, default=10, help="Maximum number of frames to visualize")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DriveStudio Rerun Visualization Example")
    print("=" * 60)
    print(f"Config file: {args.config_file}")
    print(f"Dataset: {args.dataset}")
    print(f"Scene index: {args.scene_idx}")
    print(f"Max frames: {args.max_frames}")
    print("=" * 60)
    
    # Check if config file exists
    if not os.path.exists(args.config_file):
        print(f"Error: Config file {args.config_file} not found!")
        return
    
    # Check if dataset config exists
    dataset_config_path = os.path.join("configs", "datasets", f"{args.dataset}.yaml")
    if not os.path.exists(dataset_config_path):
        print(f"Error: Dataset config {dataset_config_path} not found!")
        print("Available datasets:")
        datasets_dir = os.path.join("configs", "datasets")
        if os.path.exists(datasets_dir):
            for file in os.listdir(datasets_dir):
                if file.endswith(".yaml"):
                    print(f"  - {file[:-5]}")
        return
    
    print("Starting visualization...")
    print("The Rerun viewer will open in your browser.")
    print("You can:")
    print("  - Navigate the 3D scene with mouse")
    print("  - Use the timeline to scrub through frames")
    print("  - Toggle different visualization layers")
    print("  - Press Ctrl+C to exit")
    print("=" * 60)
    
    # Run the visualization
    try:
        run_visualization()
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    main() 