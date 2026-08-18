from unittest.mock import Mock

import numpy as np

from rosplat.gui.vulkan_texture import VulkanTextureRegistry


class FakeTexture:
    def __init__(self) -> None:
        self.view = object()
        self.destroyed = False

    def create_view(self):
        return self.view

    def destroy(self) -> None:
        self.destroyed = True


def test_upload_reuses_texture_until_dimensions_change():
    device = Mock()
    device.queue = Mock()
    first_texture = FakeTexture()
    second_texture = FakeTexture()
    device.create_texture.side_effect = [first_texture, second_texture]

    backend = Mock()
    first_ref = object()
    second_ref = object()
    backend.register_texture.side_effect = [first_ref, second_ref]
    registry = VulkanTextureRegistry(device, backend)

    owner = object()
    pixels = np.zeros((4, 8, 4), dtype=np.uint8)
    assert registry.upload(owner, pixels) is first_ref
    assert registry.upload(owner, pixels) is first_ref
    assert device.create_texture.call_count == 1
    assert device.queue.write_texture.call_count == 2

    resized = np.zeros((2, 3, 4), dtype=np.uint8)
    assert registry.upload(owner, resized) is second_ref
    backend.unregister_texture.assert_called_once_with(first_ref)
    assert first_texture.destroyed


def test_upload_rejects_non_rgba_images():
    registry = VulkanTextureRegistry(Mock(), Mock())

    try:
        registry.upload("bad", np.zeros((3, 4, 3), dtype=np.uint8))
    except ValueError as error:
        assert "RGBA" in str(error)
    else:
        raise AssertionError("expected a non-RGBA texture to be rejected")
