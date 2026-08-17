import ctypes
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from rosplat.core.gaussian_representation import GaussianData
from rosplat.render.camera import Camera
from rosplat.render.renderer.SpirulaRenderer import SpirulaRenderer, _DeviceInfo


class FakeBridge:
    def __init__(self):
        self.appended = None
        self.camera = None
        self.lib = SimpleNamespace(
            rosplat_spirula_device_count=Mock(return_value=1),
            rosplat_spirula_device_info=Mock(side_effect=self._device_info),
            rosplat_spirula_select_device=Mock(return_value=1),
            rosplat_spirula_reset_scene=Mock(return_value=1),
            rosplat_spirula_append=Mock(side_effect=self._append),
            rosplat_spirula_set_camera=Mock(side_effect=self._set_camera),
            rosplat_spirula_render_rgba8=Mock(side_effect=self._render),
            rosplat_spirula_splat_count=Mock(return_value=2),
            rosplat_spirula_capacity=Mock(return_value=4096),
            rosplat_spirula_shutdown=Mock(),
        )

    @staticmethod
    def _device_info(index, output):
        assert index == 0
        output.name = b"Fake Vulkan GPU"
        output.vram_bytes = 8 * 1024**3
        output.usable = 1
        return 1

    def _append(self, count, means, quats, scales, opacities, dc, higher):
        def read(pointer, elements):
            if not pointer:
                return None
            array_type = ctypes.c_float * elements
            return np.ctypeslib.as_array(array_type.from_address(pointer.value)).copy()

        self.appended = {
            "count": count,
            "means": read(means, count * 3).reshape(count, 3),
            "quats": read(quats, count * 4).reshape(count, 4),
            "scales": read(scales, count * 3).reshape(count, 3),
            "opacities": read(opacities, count),
            "dc": read(dc, count * 3).reshape(count, 3),
            "higher": read(higher, count * 9).reshape(count, 3, 3),
        }
        return 1

    def _set_camera(self, width, height, view, intrinsics):
        float16 = ctypes.c_float * 16
        float4 = ctypes.c_float * 4
        self.camera = {
            "size": (width, height),
            "view": np.ctypeslib.as_array(float16.from_address(view.value)).copy(),
            "intrinsics": np.ctypeslib.as_array(
                float4.from_address(intrinsics.value)
            ).copy(),
        }
        return 1

    @staticmethod
    def _render(output, output_bytes):
        ctypes.memset(output.value, 127, output_bytes)
        return 1

    @staticmethod
    def require(result):
        assert result

    @staticmethod
    def error():
        return RuntimeError("fake bridge error")


def gaussian_batch():
    return GaussianData(
        xyz=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        rot=np.array([[1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32),
        scale=np.array([[0.5, 1, 2], [0.25, 0.5, 1]], dtype=np.float32),
        opacity=np.array([[0.25], [0.75]], dtype=np.float32),
        sh=np.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            ],
            dtype=np.float32,
        ),
    )


def make_renderer(width=320, height=180):
    camera = Camera(height, width)
    settings = SimpleNamespace(world_camera=camera)
    bridge = FakeBridge()
    return SpirulaRenderer(width, height, settings, bridge=bridge), bridge


def test_append_converts_activated_rosplat_values_to_spirula_layout():
    renderer, bridge = make_renderer()
    batch = gaussian_batch()

    renderer.append_gaussians(batch, refresh=True)

    bridge.lib.rosplat_spirula_reset_scene.assert_called_once_with(1, 2)
    np.testing.assert_array_equal(bridge.appended["means"], batch.xyz)
    np.testing.assert_array_equal(bridge.appended["quats"], batch.rot)
    np.testing.assert_allclose(bridge.appended["scales"], np.log(batch.scale))
    np.testing.assert_allclose(
        bridge.appended["opacities"],
        np.log(batch.opacity[:, 0]) - np.log1p(-batch.opacity[:, 0]),
    )
    np.testing.assert_array_equal(bridge.appended["dc"], batch.sh[:, :3])
    np.testing.assert_array_equal(
        bridge.appended["higher"], batch.sh[:, 3:].reshape(2, 3, 3)
    )


def test_camera_uses_vertical_fov_and_model_transform():
    renderer, bridge = make_renderer(320, 180)
    model = np.eye(4, dtype=np.float32)
    model[0, 3] = 2.0
    renderer.set_model_matrix(model)

    renderer._sync_camera()

    assert bridge.camera["size"] == (320, 180)
    np.testing.assert_allclose(bridge.camera["intrinsics"], [90, 90, 160, 90])
    expected = (
        np.diag([1.0, -1.0, -1.0, 1.0])
        @ renderer.world_settings.world_camera.get_view_matrix()
        @ model
    )
    np.testing.assert_allclose(bridge.camera["view"].reshape(4, 4), expected)


def test_draw_uploads_a_vulkan_texture_only_when_render_is_dirty():
    renderer, bridge = make_renderer(8, 4)
    texture = SimpleNamespace(texture_id=Mock(return_value=42))

    with patch(
        "rosplat.render.renderer.SpirulaRenderer.hello_imgui.create_texture_gpu_from_rgba_data",
        return_value=texture,
    ) as create_texture:
        assert renderer.draw() == 42
        assert renderer.draw() == 42

    bridge.lib.rosplat_spirula_render_rgba8.assert_called_once()
    create_texture.assert_called_once()
    assert np.all(renderer._rgba == 127)
