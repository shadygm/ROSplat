import unittest
from unittest.mock import Mock

from gaussian_interface.msg import GaussianArray, SingleGaussian

from rosplat.config.world_settings import WorldSettings
from rosplat.render.renderer.CUDARenderer import CUDARenderer


def gaussian_message() -> SingleGaussian:
    message = SingleGaussian()
    message.xyz = [0.0, 0.0, 0.0]
    message.rotation = [1.0, 0.0, 0.0, 0.0]
    message.scale = [0.1, 0.1, 0.1]
    message.opacity = 255
    message.spherical_harmonics = [1.0, 0.0, 0.0]
    return message


def fake_cuda_renderer() -> CUDARenderer:
    renderer = object.__new__(CUDARenderer)
    renderer.reset_gaussians = Mock()
    renderer.add_gaussians_from_ros = Mock()
    return renderer


class WorldSettingsTest(unittest.TestCase):
    def test_render_panel_resize_updates_resolution_and_intrinsics(self):
        settings = WorldSettings()
        settings.gauss_renderer = Mock()

        settings.update_window_size(800, 600)

        self.assertEqual((settings.world_camera.w, settings.world_camera.h), (800, 600))
        settings.gauss_renderer.set_render_resolution.assert_called_once_with(800, 600)
        settings.gauss_renderer.update_camera_intrin.assert_called_once_with()
        self.assertFalse(settings.world_camera.dirty_intrinsic)

    def test_empty_refresh_clears_without_enqueuing_none(self):
        settings = WorldSettings()
        settings.gauss_renderer = fake_cuda_renderer()
        message = GaussianArray()
        message.refresh = True

        settings.append_gaussians(message)

        self.assertIsNone(settings.gaussian_set)
        settings.gauss_renderer.reset_gaussians.assert_called_once_with()
        settings.gauss_renderer.add_gaussians_from_ros.assert_not_called()
        self.assertFalse(settings.have_new_gaussians)

    def test_cuda_overwrite_replaces_the_cpu_and_gpu_scene(self):
        settings = WorldSettings()
        settings.gauss_renderer = fake_cuda_renderer()
        settings.is_original = False
        settings.overwrite_gaussians = True
        message = GaussianArray()
        message.gaussians = [gaussian_message()]

        settings.append_gaussians(message)

        self.assertEqual(settings.get_num_gaussians(), 1)
        settings.gauss_renderer.reset_gaussians.assert_called_once_with()
        queued = settings.gauss_renderer.add_gaussians_from_ros.call_args.args[0]
        self.assertEqual(len(queued), 1)


if __name__ == "__main__":
    unittest.main()
