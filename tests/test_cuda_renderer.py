import threading
import unittest
import importlib
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rosplat.core.gaussian_representation import naive_gaussian
from rosplat.render.camera import Camera
from rosplat.render.renderer.CUDARenderer import CUDARenderer


cuda_renderer_module = importlib.import_module("rosplat.render.renderer.CUDARenderer")


def cpu_renderer() -> CUDARenderer:
    renderer = object.__new__(CUDARenderer)
    renderer.device = torch.device("cpu")
    renderer._data_lock = threading.Lock()
    renderer._cache_means = []
    renderer._cache_quats = []
    renderer._cache_scales = []
    renderer._cache_opacs = []
    renderer._cache_colors = []
    renderer._cache_sh_deg = None
    return renderer


class CUDARendererTest(unittest.TestCase):
    def test_camera_places_visible_points_at_positive_z(self):
        camera = Camera(64, 64)
        renderer = cpu_renderer()
        renderer.world_settings = SimpleNamespace(world_camera=camera)

        renderer.update_camera_pose()

        origin = torch.tensor([0.0, 0.0, 0.0, 1.0])
        camera_origin = renderer.viewmats[0] @ origin
        self.assertGreater(camera_origin[2].item(), 0.0)
        self.assertEqual(camera.right.tolist(), [1.0, 0.0, 0.0])

    def test_single_gaussian_opacity_remains_a_vector(self):
        renderer = cpu_renderer()
        gaussian = naive_gaussian()
        gaussian.xyz = gaussian.xyz[:1]
        gaussian.rot = gaussian.rot[:1]
        gaussian.scale = gaussian.scale[:1]
        gaussian.opacity = gaussian.opacity[:1]
        gaussian.sh = gaussian.sh[:1]

        renderer.update_gaussian_data(gaussian)

        self.assertEqual(tuple(renderer.opacities.shape), (1,))

    def test_unchanged_or_zero_resolution_does_not_reallocate_texture(self):
        renderer = object.__new__(CUDARenderer)
        renderer.width = 640
        renderer.height = 480
        renderer.texture_id = 1
        renderer._texture_needs_clear = False

        with patch.object(cuda_renderer_module.gl, "glTexImage2D") as allocate:
            renderer.set_render_resolution(640, 480)
            renderer.set_render_resolution(0, 480)

        allocate.assert_not_called()

    def test_changed_resolution_uses_the_byte_texture_format(self):
        renderer = object.__new__(CUDARenderer)
        renderer.width = 640
        renderer.height = 480
        renderer.texture_id = 1
        renderer._texture_needs_clear = False

        with (
            patch.object(cuda_renderer_module.gl, "glBindTexture"),
            patch.object(cuda_renderer_module.gl, "glTexImage2D") as allocate,
        ):
            renderer.set_render_resolution(800, 600)

        self.assertEqual((renderer.width, renderer.height), (800, 600))
        self.assertEqual(allocate.call_args.args[2], cuda_renderer_module.gl.GL_RGBA8)
        self.assertTrue(renderer._texture_needs_clear)


if __name__ == "__main__":
    unittest.main()
