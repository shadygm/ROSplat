import unittest
from unittest.mock import Mock

from gaussian_interface.msg import GaussianArray, SingleGaussian

from rosplat.config.world_settings import WorldSettings
from rosplat.render.renderer import RenderOutputMode


def gaussian_message(x=0.0) -> SingleGaussian:
    message = SingleGaussian()
    message.xyz = [x, 0.0, 0.0]
    message.rotation = [1.0, 0.0, 0.0, 0.0]
    message.scale = [0.1, 0.1, 0.1]
    message.opacity = 255
    message.spherical_harmonics = [1.0, 0.0, 0.0]
    return message


def fake_renderer():
    renderer = Mock()
    renderer.append_gaussians = Mock()
    renderer.update_gaussian_data = Mock()
    renderer.reset_gaussians = Mock()
    return renderer


class WorldSettingsTest(unittest.TestCase):
    def test_render_panel_resize_updates_resolution_and_intrinsics(self):
        settings = WorldSettings()
        settings.gauss_renderer = fake_renderer()

        settings.update_window_size(800, 600)

        self.assertEqual((settings.world_camera.w, settings.world_camera.h), (800, 600))
        settings.gauss_renderer.set_render_resolution.assert_called_once_with(800, 600)
        settings.gauss_renderer.update_camera_intrin.assert_called_once_with()
        self.assertFalse(settings.world_camera.dirty_intrinsic)

    def test_empty_refresh_is_applied_on_the_render_thread(self):
        settings = WorldSettings()
        settings.gauss_renderer = fake_renderer()
        message = GaussianArray()
        message.refresh = True

        settings.append_gaussians(message)

        self.assertEqual(settings.get_num_gaussians(), 0)
        settings.gauss_renderer.reset_gaussians.assert_not_called()
        settings.update_activated_render_state()
        settings.gauss_renderer.reset_gaussians.assert_called_once_with()
        settings.gauss_renderer.append_gaussians.assert_not_called()
        self.assertFalse(settings.have_new_gaussians)

    def test_stream_batches_append_without_rebuilding_the_host_scene(self):
        settings = WorldSettings()
        settings.gauss_renderer = fake_renderer()
        first = GaussianArray()
        first.refresh = True
        first.gaussians = [gaussian_message(1.0), gaussian_message(2.0)]
        second = GaussianArray()
        second.gaussians = [gaussian_message(3.0)]

        settings.append_gaussians(first)
        settings.append_gaussians(second)
        settings.update_activated_render_state()

        self.assertEqual(settings.get_num_gaussians(), 3)
        self.assertEqual(settings.gauss_renderer.append_gaussians.call_count, 2)
        first_call, second_call = settings.gauss_renderer.append_gaussians.call_args_list
        self.assertTrue(first_call.kwargs["refresh"])
        self.assertFalse(second_call.kwargs["refresh"])
        self.assertEqual(len(first_call.args[0]), 2)
        self.assertEqual(len(second_call.args[0]), 1)
        # The retained CPU snapshot is only the refresh batch, not a repeatedly
        # concatenated copy of all streamed data.
        self.assertEqual(len(settings.gaussian_set), 2)

    def test_batch_conversion_rejects_mixed_sh_layouts(self):
        settings = WorldSettings()
        message = GaussianArray()
        first = gaussian_message()
        second = gaussian_message()
        second.spherical_harmonics = [1.0] * 12
        message.gaussians = [first, second]

        with self.assertRaisesRegex(ValueError, "mix SH layouts"):
            settings.append_gaussians(message)

    def test_renderer_options_are_forwarded_without_reloading_the_scene(self):
        settings = WorldSettings()
        settings.gauss_renderer = fake_renderer()

        settings.update_render_output(RenderOutputMode.OPACITY)
        settings.update_sh_degree(0)

        self.assertEqual(settings.render_output, RenderOutputMode.OPACITY)
        self.assertEqual(settings.active_sh_degree, 0)
        settings.gauss_renderer.set_render_output.assert_called_once_with(
            RenderOutputMode.OPACITY
        )
        settings.gauss_renderer.set_sh_degree.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
