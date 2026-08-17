#!/usr/bin/env python3
"""Exercise ROS imports and one real gsplat CUDA rasterization."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from gaussian_interface.msg import GaussianArray, SingleGaussian
from gsplat import rasterization
from rosplat.core.gaussian_representation import naive_gaussian
from rosplat.render.camera import Camera


def main() -> None:
    if os.environ.get("ROS_DISTRO") != "lyrical":
        raise RuntimeError("ROS 2 Lyrical is not sourced")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access an NVIDIA GPU")

    device = torch.device("cuda")
    gaussian_set = naive_gaussian()
    camera = Camera(64, 64)
    means = torch.from_numpy(gaussian_set.xyz).to(device)
    quats = torch.from_numpy(gaussian_set.rot).to(device)
    scales = torch.from_numpy(gaussian_set.scale).to(device)
    opacities = torch.from_numpy(gaussian_set.opacity.reshape(-1)).to(device)
    colors = torch.from_numpy(gaussian_set.sh.reshape(-1, 1, 3)).to(device)
    viewmats = torch.from_numpy(camera.get_view_matrix_opencv()).to(device).unsqueeze(0)
    intrinsics = torch.from_numpy(camera.get_intrinsics_matrix()).to(device).unsqueeze(0)

    rendered, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=intrinsics,
        width=64,
        height=64,
        packed=True,
        sh_degree=0,
    )

    if rendered.shape != (1, 64, 64, 3):
        raise RuntimeError(f"Unexpected render shape: {rendered.shape}")
    if alpha.shape != (1, 64, 64, 1):
        raise RuntimeError(f"Unexpected alpha shape: {alpha.shape}")
    if not torch.isfinite(rendered).all() or not torch.isfinite(alpha).all():
        raise RuntimeError("gsplat returned non-finite values")
    if alpha.max().item() <= 0:
        raise RuntimeError("Default camera culled every Gaussian")

    message = GaussianArray()
    message.gaussians.append(SingleGaussian())
    capability = ".".join(str(part) for part in torch.cuda.get_device_capability())
    print(
        f"ROS_DISTRO=lyrical torch={torch.__version__} "
        f"cuda={torch.version.cuda} gpu={torch.cuda.get_device_name()} "
        f"compute_capability={capability} render_shape={tuple(rendered.shape)}"
    )


if __name__ == "__main__":
    main()
