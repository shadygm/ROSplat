#!/usr/bin/env python3
"""Exercise ROS imports and one real Spirula Vulkan rasterization."""

from types import SimpleNamespace

import numpy as np

from gaussian_interface.msg import GaussianArray, SingleGaussian  # noqa: F401
from rosplat.core.gaussian_representation import naive_gaussian
from rosplat.render.camera import Camera
from rosplat.render.renderer import RenderOutputMode, SpirulaRenderer


def main() -> None:
    camera = Camera(64, 64)
    world = SimpleNamespace(world_camera=camera)
    renderer = SpirulaRenderer(64, 64, world)
    renderer.update_gaussian_data(naive_gaussian(), full_update=True)
    for mode in RenderOutputMode:
        renderer.set_render_output(mode)
        rgba = renderer.render_rgba8()
        if not np.any(rgba[..., :3]):
            raise RuntimeError(f"Spirula returned an all-black {mode.name} image")
    print(
        f"ROS_DISTRO=lyrical backend=spirula-vulkan "
        f"splats={renderer.splat_count} render_shape={renderer.latest_rgba.shape} "
        "outputs=color,depth,opacity"
    )
    renderer.shutdown()


if __name__ == "__main__":
    main()
