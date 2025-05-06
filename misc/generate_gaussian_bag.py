"""
Script Name: generate_gaussian_bag.py

Description:
    This script loads Gaussian data from a specified PLY file, initializes
    a ROS2 node using rclpy, and continuously publishes a GaussianArray message 
    that contains 1000 SingleGaussian messages every 0.1 seconds on the /gaussian_test topic.
    After reaching the end of the Gaussian data, it cycles back to the beginning.

Requirements:
    - ROS2 (rclpy)
    - numpy
    - plyfile
    - Custom message types:
        gaussian_interface/SingleGaussian, with:
            geometry_msgs/msg/Point xyz
            geometry_msgs/msg/Quaternion rotation
            float32 opacity
            geometry_msgs/msg/Vector3 scale
            float32[] spherical_harmonics
        gaussian_interface/GaussianArray, with:
            gaussian_interface/SingleGaussian[] gaussians

Usage:
    1. Run it (optionally passing in a path to the PLY file):
         python generate_gaussian_bag.py --ply_path /path/to/your_file.ply
    2. In a separate terminal, record the published data or visualize it using ROSplat:
         ros2 bag record /gaussian_test
"""

import sys
import argparse
import numpy as np
from dataclasses import dataclass
from plyfile import PlyData

import rclpy
from rclpy.node import Node

# ROS2 message imports
from geometry_msgs.msg import Point, Quaternion, Vector3
from gaussian_interface.msg import SingleGaussian, GaussianArray

@dataclass
class GaussianData:
    """
    Holds arrays for Gaussian data: positions, rotations, scale, opacity, and SH coefficients.
    """
    xyz: np.ndarray      # shape: (N, 3)
    rot: np.ndarray      # shape: (N, 4)
    scale: np.ndarray    # shape: (N, 3)
    opacity: np.ndarray  # shape: (N, 1)
    sh: np.ndarray       # shape: (N, sh_dim)

    def flat(self) -> np.ndarray:
        """
        Flatten all parameters into a single contiguous array of shape (N, <all_dims>).
        """
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)
    
    def __len__(self) -> int:
        """
        Return the number of Gaussian entries (N).
        """
        return len(self.xyz)
    
    @property
    def sh_dim(self) -> int:
        """
        Return the dimensionality of the spherical harmonics (sh).
        """
        return self.sh.shape[-1]

def from_ply(path: str) -> GaussianData:
    """
    Loads Gaussian data from a PLY file and returns a GaussianData object.
    This function assumes the PLY file contains the properties:
      - x, y, z for positions
      - opacity
      - f_dc_0, f_dc_1, f_dc_2 for DC components
      - f_rest_X for other SH components
      - scale_0, scale_1, scale_2 for scale
      - rot_0, rot_1, rot_2, rot_3 for quaternion rotations
    """
    max_sh_degree = 3
    plydata = PlyData.read(path)
    
    # Load positions.
    xyz = np.stack((
        np.asarray(plydata.elements[0]["x"]),
        np.asarray(plydata.elements[0]["y"]),
        np.asarray(plydata.elements[0]["z"])
    ), axis=1).astype(np.float32)
    
    # Load opacities and apply a sigmoid.
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)
    opacities = 1 / (1 + np.exp(-opacities))
    
    # Load direct current (DC) features for spherical harmonics.
    features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    
    # Load extra SH features.
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3, "Unexpected number of extra features"
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])
    
    # Load scales.
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    scales = np.exp(scales).astype(np.float32)
    
    # Load rotations.
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Normalize quaternion
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    
    # Concatenate DC and extra features to form SH coefficients.
    sh = np.concatenate([
        features_dc.reshape(-1, 3),
        features_extra.reshape(xyz.shape[0], -1)
    ], axis=-1).astype(np.float32)
    
    return GaussianData(xyz, rots, scales, opacities, sh)

class GaussianPublisher(Node):
    """
    ROS2 Node that publishes a GaussianArray message containing 1000 SingleGaussian messages
    every 0.1 seconds.
    """
    def __init__(self, gaussian_data: GaussianData):
        super().__init__('gaussian_publisher')
        self.gaussian_data = gaussian_data
        self.publisher_ = self.create_publisher(GaussianArray, '/gaussian_test', 10)
        self.idx = 0
        self.total = len(self.gaussian_data)
        self.batch_size = 1000
        self.get_logger().info(
            f"Loaded {self.total} gaussians. Publishing GaussianArray with {self.batch_size} gaussians every 0.1 seconds."
        )
        
        # Create a timer that calls publish_array every 0.1 seconds.
        self.timer = self.create_timer(0.1, self.publish_array)

    def publish_array(self):
        """Publish a GaussianArray containing a batch of 1000 SingleGaussian messages."""
        array_msg = GaussianArray()
        
        # Create 1000 SingleGaussian messages.
        for i in range(self.batch_size):
            curr_idx = (self.idx + i) % self.total
            single_msg = SingleGaussian()
            # Set position.
            single_msg.xyz = Point(
                x=float(self.gaussian_data.xyz[curr_idx, 0]),
                y=float(self.gaussian_data.xyz[curr_idx, 1]),
                z=float(self.gaussian_data.xyz[curr_idx, 2])
            )
            # Set rotation (quaternion).
            single_msg.rotation = Quaternion(
                x=float(self.gaussian_data.rot[curr_idx, 0]),
                y=float(self.gaussian_data.rot[curr_idx, 1]),
                z=float(self.gaussian_data.rot[curr_idx, 2]),
                w=float(self.gaussian_data.rot[curr_idx, 3])
            )
            # Set opacity.
            single_msg.opacity = float(self.gaussian_data.opacity[curr_idx, 0])
            # Set scale.
            single_msg.scale = Vector3(
                x=float(self.gaussian_data.scale[curr_idx, 0]),
                y=float(self.gaussian_data.scale[curr_idx, 1]),
                z=float(self.gaussian_data.scale[curr_idx, 2])
            )
            # Set spherical harmonics.
            sh_coeffs = self.gaussian_data.sh[curr_idx]
            single_msg.spherical_harmonics = [float(val) for val in sh_coeffs]
            
            array_msg.gaussians.append(single_msg)
        
        # Advance the index by batch_size.
        self.idx = (self.idx + self.batch_size) % self.total
        
        self.publisher_.publish(array_msg)
        self.get_logger().info(f"Published GaussianArray with {len(array_msg.gaussians)} gaussians.")

def main(args=None):
    parser = argparse.ArgumentParser(
        description='ROS2 Node to publish a GaussianArray message containing 1000 gaussians every 0.1 seconds.'
    )
    parser.add_argument('--ply_path', default='gaussians.ply',
                        help='Path to the PLY file containing the Gaussian data.')
    cli_args = parser.parse_args()

    rclpy.init(args=args)

    try:
        g_data = from_ply(cli_args.ply_path)
    except Exception as e:
        print(f"Failed to load PLY file: {e}")
        sys.exit(1)

    node = GaussianPublisher(g_data)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
