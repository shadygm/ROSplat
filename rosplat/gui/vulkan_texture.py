from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np
import wgpu
from imgui_bundle import imgui


@dataclass
class _TextureRecord:
    width: int
    height: int
    texture: object
    texture_ref: imgui.ImTextureRef


class VulkanTextureRegistry:
    """Upload reusable RGBA textures for wgpu-py's Dear ImGui renderer."""

    def __init__(self, device, imgui_backend) -> None:
        self._device = device
        self._imgui_backend = imgui_backend
        self._textures: dict[Hashable, _TextureRecord] = {}

    def upload(self, owner: Hashable, rgba: np.ndarray) -> imgui.ImTextureRef:
        pixels = np.ascontiguousarray(rgba, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError("Vulkan textures require a height x width x 4 RGBA array")

        height, width = pixels.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Vulkan textures cannot be empty")

        record = self._textures.get(owner)
        if record is None or (record.width, record.height) != (width, height):
            self.release(owner)
            texture = self._device.create_texture(
                label="ROSplat RGBA texture",
                size=(width, height, 1),
                format=wgpu.TextureFormat.rgba8unorm,
                usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
            )
            texture_ref = self._imgui_backend.register_texture(texture.create_view())
            record = _TextureRecord(width, height, texture, texture_ref)
            self._textures[owner] = record

        self._device.queue.write_texture(
            {
                "texture": record.texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            pixels,
            {"offset": 0, "bytes_per_row": width * 4},
            (width, height, 1),
        )
        return record.texture_ref

    def release(self, owner: Hashable) -> None:
        record = self._textures.pop(owner, None)
        if record is None:
            return
        self._imgui_backend.unregister_texture(record.texture_ref)
        record.texture.destroy()

    def shutdown(self) -> None:
        for owner in list(self._textures):
            self.release(owner)
