from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import numpy as np
from plyfile import PlyData
from dataclasses import dataclass
from . import util


@dataclass
class GaussianData:
    xyz: np.ndarray      # shape: (N, 3)
    rot: np.ndarray      # shape: (N, 4)
    scale: np.ndarray    # shape: (N, 3)
    opacity: np.ndarray  # shape: (N, 1)
    sh: np.ndarray       # shape: (N, sh_dim)

    def flat(self) -> np.ndarray:
        """
        Returns a contiguous 2D array (N x total_dims) where each row is the concatenation of:
          [xyz, rot, scale, opacity, sh]
        """
        combined = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(combined)
    
    def __len__(self) -> int:
        return len(self.xyz)
    
    @property 
    def sh_dim(self) -> int:
        return self.sh.shape[-1]


def combine_gaussians(gaussians: List[Optional[GaussianData]]) -> Optional[GaussianData]:
    """
    Combines multiple GaussianData instances into one.
    Filters out any None entries.
    """
    valid_gaussians = [g for g in gaussians if g is not None]
    if not valid_gaussians:
        return None
    if len(valid_gaussians) == 1:
        return valid_gaussians[0]
    
    combined = GaussianData(
        xyz=np.concatenate([g.xyz for g in valid_gaussians], axis=0),
        rot=np.concatenate([g.rot for g in valid_gaussians], axis=0),
        scale=np.concatenate([g.scale for g in valid_gaussians], axis=0),
        opacity=np.concatenate([g.opacity for g in valid_gaussians], axis=0),
        sh=np.concatenate([g.sh for g in valid_gaussians], axis=0)
    )
    return combined


def naive_gaussian() -> GaussianData:
    """
    Creates a set of 4 naive Gaussians with hard-coded values.
    """
    xyz = np.array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ], dtype=np.float32).reshape(-1, 3)

    rot = np.array([
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
    ], dtype=np.float32).reshape(-1, 4)

    scale = np.array([
        0.03, 0.03, 0.03,
        0.2,  0.03, 0.03,
        0.03, 0.2,  0.03,
        0.03, 0.03, 0.2,
    ], dtype=np.float32).reshape(-1, 3)

    opacity = np.array([1, 1, 1, 1], dtype=np.float32).reshape(-1, 1)

    sh = np.array([
        1, 0, 1,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ], dtype=np.float32).reshape(-1, 3)
    # Normalize the spherical-harmonics coefficients as in the reference.
    sh = (sh - 0.5) / 0.28209

    return GaussianData(xyz, rot, scale, opacity, sh)


def from_ply(ply_path: Path, max_sh_degree: int = 3) -> GaussianData:
    """
    Loads Gaussians from a PLY file and returns a GaussianData instance.
    Expects the PLY file to have properties corresponding to positions, opacity,
    spherical harmonics (DC and extra), scales, and rotations.
    """
    plydata = PlyData.read(ply_path)
    element = plydata.elements[0]

    # Load positions.
    xyz = np.stack([
        np.asarray(element["x"]),
        np.asarray(element["y"]),
        np.asarray(element["z"])
    ], axis=1).astype(np.float32)

    # Load and transform opacities.
    opacities = np.asarray(element["opacity"], dtype=np.float32)[..., np.newaxis]
    opacities = 1 / (1 + np.exp(-opacities))

    # Load DC features for spherical harmonics.
    features_dc = np.empty((xyz.shape[0], 3, 1), dtype=np.float32)
    features_dc[:, 0, 0] = np.asarray(element["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(element["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(element["f_dc_2"])

    # Load extra SH features.
    extra_f_names = sorted(
        [prop.name for prop in element.properties if prop.name.startswith("f_rest_")],
        key=lambda x: int(x.split('_')[-1])
    )
    expected_num = 3 * (max_sh_degree + 1) ** 2 - 3
    if len(extra_f_names) != expected_num:
        raise ValueError(
            f"Unexpected number of extra features: found {len(extra_f_names)}, expected {expected_num}"
        )

    features_extra = np.empty((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(element[attr_name])
    features_extra = features_extra.reshape((xyz.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, (0, 2, 1))

    # Load scales.
    scale_names = sorted(
        [prop.name for prop in element.properties if prop.name.startswith("scale_")],
        key=lambda x: int(x.split('_')[-1])
    )
    scales = np.empty((xyz.shape[0], len(scale_names)), dtype=np.float32)
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(element[attr_name])
    scales = np.exp(scales).astype(np.float32)

    # Load rotations.
    rot_names = sorted(
        [prop.name for prop in element.properties if prop.name.startswith("rot")],
        key=lambda x: int(x.split('_')[-1])
    )
    rots = np.empty((xyz.shape[0], len(rot_names)), dtype=np.float32)
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(element[attr_name])
    # Normalize quaternions.
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)

    # Concatenate DC and extra features to form SH coefficients.
    sh = np.concatenate([
        features_dc.reshape(-1, 3),
        features_extra.reshape(xyz.shape[0], -1)
    ], axis=-1).astype(np.float32)

    return GaussianData(xyz, rots, scales, opacities, sh)


def main():
    # Test using naive gaussians.
    naive = naive_gaussian()
    print("Naive gaussians flat shape:", naive.flat().shape)

    # Example: Loading from a PLY file.
    ply_file = Path("../../dataset/ply/bonsai/point_cloud/iteration_7000/point_cloud.ply")
    try:
        gaussians = from_ply(ply_file)
        print("Loaded Gaussians count:", len(gaussians))
        print("Flat shape:", gaussians.flat().shape)
        # Print properties for Gaussian at index 99 if it exists.
        if len(gaussians) > 99:
            print("Gaussian index 99:")
            print(" Position:", gaussians.xyz[99])
            print(" Opacity:", gaussians.opacity[99])
            print(" Scale:", gaussians.scale[99])
            print(" Rotation:", gaussians.rot[99])
        else:
            print("Less than 100 gaussians loaded.")
    except Exception as e:
        print(f"Error loading PLY data: {e}")


if __name__ == "__main__":
    main()
