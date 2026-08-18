from __future__ import annotations

import ctypes
from enum import IntEnum
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

from rosplat.core import util
from rosplat.core.gaussian_representation import GaussianData
from rosplat.render.renderer.base_gaussian_renderer import GaussianRenderBase


class SpirulaError(RuntimeError):
    """Raised when the native Spirula Vulkan bridge reports an error."""


class RenderOutputMode(IntEnum):
    """Visual buffers exposed by the native Spirula renderer."""

    COLOR = 0
    DEPTH = 1
    OPACITY = 2


class _DeviceInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("vram_bytes", ctypes.c_uint64),
        ("usable", ctypes.c_int),
    ]


def _library_candidates() -> list[Path]:
    project_root = Path(__file__).resolve().parents[3]
    candidates = []
    configured = os.environ.get("ROSPLAT_SPIRULA_LIBRARY")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            project_root / "build-native" / "librosplat_spirula.so",
            project_root / "build" / "native" / "librosplat_spirula.so",
            Path("/opt/rosplat-native/lib/rosplat/librosplat_spirula.so"),
            Path("/usr/local/lib/rosplat/librosplat_spirula.so"),
        ]
    )
    return candidates


def _load_library() -> ctypes.CDLL:
    for candidate in _library_candidates():
        if candidate.is_file():
            return ctypes.CDLL(str(candidate))
    searched = "\n  ".join(str(path) for path in _library_candidates())
    raise SpirulaError(
        "The ROSplat Spirula Vulkan bridge is not built. Searched:\n"
        f"  {searched}\n"
        "Build it with: cmake -S . -B build-native -G Ninja "
        "-DCMAKE_BUILD_TYPE=Release && cmake --build build-native"
    )


def _as_pointer(array: Optional[np.ndarray]) -> ctypes.c_void_p:
    if array is None or array.size == 0:
        return ctypes.c_void_p()
    return ctypes.c_void_p(array.ctypes.data)


class _Bridge:
    def __init__(self, library: Optional[ctypes.CDLL] = None) -> None:
        self.lib = library if library is not None else _load_library()
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        lib = self.lib
        lib.rosplat_spirula_last_error.restype = ctypes.c_char_p
        lib.rosplat_spirula_device_count.restype = ctypes.c_int
        lib.rosplat_spirula_device_info.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(_DeviceInfo),
        ]
        lib.rosplat_spirula_device_info.restype = ctypes.c_int
        lib.rosplat_spirula_select_device.argtypes = [ctypes.c_int]
        lib.rosplat_spirula_select_device.restype = ctypes.c_int
        lib.rosplat_spirula_reset_scene.argtypes = [ctypes.c_int, ctypes.c_int64]
        lib.rosplat_spirula_reset_scene.restype = ctypes.c_int
        lib.rosplat_spirula_append.argtypes = [
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.rosplat_spirula_append.restype = ctypes.c_int
        lib.rosplat_spirula_set_camera.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.rosplat_spirula_set_camera.restype = ctypes.c_int
        lib.rosplat_spirula_set_render_options.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.rosplat_spirula_set_render_options.restype = ctypes.c_int
        lib.rosplat_spirula_set_scale_modifier.argtypes = [ctypes.c_float]
        lib.rosplat_spirula_set_scale_modifier.restype = ctypes.c_int
        lib.rosplat_spirula_render_rgba8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.rosplat_spirula_render_rgba8.restype = ctypes.c_int
        lib.rosplat_spirula_splat_count.restype = ctypes.c_int64
        lib.rosplat_spirula_capacity.restype = ctypes.c_int64
        lib.rosplat_spirula_shutdown.restype = None

    def error(self) -> SpirulaError:
        raw = self.lib.rosplat_spirula_last_error()
        message = raw.decode("utf-8", errors="replace") if raw else "unknown error"
        return SpirulaError(message)

    def require(self, result: int) -> None:
        if not result:
            raise self.error()


class SpirulaRenderer(GaussianRenderBase):
    """Spirula's Vulkan 3DGS renderer with an ImGui-compatible texture."""

    def __init__(
        self,
        width: int,
        height: int,
        world_settings,
        bridge=None,
        texture_registry=None,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.world_settings = world_settings
        self._bridge = bridge if bridge is not None else _Bridge()
        self._texture_registry = texture_registry
        self._texture = None
        self._rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self._model_matrix = np.eye(4, dtype=np.float32)
        self._scale_modifier = 1.0
        self._sh_degree = 0
        self._active_sh_degree: Optional[int] = None
        self._output_mode = RenderOutputMode.COLOR
        self._scene_ready = False
        self._camera_dirty = True
        self._render_dirty = True
        self._texture_dirty = True

        count = self._bridge.lib.rosplat_spirula_device_count()
        if count < 0:
            raise self._bridge.error()
        if count == 0:
            raise SpirulaError("No usable Vulkan 1.2 device was found")
        info = _DeviceInfo()
        self._bridge.require(self._bridge.lib.rosplat_spirula_device_info(0, info))
        if not info.usable:
            raise SpirulaError(f"Vulkan device is not usable: {info.name.decode()}")
        self._bridge.require(self._bridge.lib.rosplat_spirula_select_device(0))
        self._device_name = info.name.decode("utf-8", errors="replace")
        self._device_vram_bytes = int(info.vram_bytes)
        util.logger.info(
            "Spirula Vulkan renderer initialized on {} ({:.1f} GiB)",
            self._device_name,
            self._device_vram_bytes / (1024**3),
        )

    @staticmethod
    def _sh_layout(gaussians: GaussianData) -> tuple[int, int]:
        if gaussians.sh_dim % 3 != 0:
            raise ValueError("SH data must contain RGB coefficient triplets")
        coefficients = gaussians.sh_dim // 3
        root = math.isqrt(coefficients)
        if root * root != coefficients:
            raise ValueError(f"Invalid SH coefficient count: {coefficients}")
        degree = root - 1
        if not 0 <= degree <= 4:
            raise ValueError(f"Spirula supports SH degrees 0 through 4, got {degree}")
        return degree, coefficients

    @staticmethod
    def _native_arrays(gaussians: GaussianData, scale_modifier: float):
        count = len(gaussians)
        _, coefficients = SpirulaRenderer._sh_layout(gaussians)
        means = np.ascontiguousarray(gaussians.xyz, dtype=np.float32)
        quats = np.ascontiguousarray(gaussians.rot, dtype=np.float32)
        activated_scales = np.asarray(gaussians.scale, dtype=np.float32) * scale_modifier
        log_scales = np.ascontiguousarray(
            np.log(np.clip(activated_scales, 1e-12, None)), dtype=np.float32
        )
        activated_opacity = np.asarray(gaussians.opacity, dtype=np.float32).reshape(-1)
        activated_opacity = np.clip(activated_opacity, 1e-6, 1.0 - 1e-6)
        opacity_logits = np.ascontiguousarray(
            np.log(activated_opacity) - np.log1p(-activated_opacity),
            dtype=np.float32,
        )
        sh = np.ascontiguousarray(
            np.asarray(gaussians.sh, dtype=np.float32).reshape(count, coefficients, 3)
        )
        dc = np.ascontiguousarray(sh[:, 0, :])
        higher = np.ascontiguousarray(sh[:, 1:, :]) if coefficients > 1 else None
        return means, quats, log_scales, opacity_logits, dc, higher

    def _reset_for(self, gaussians: GaussianData, capacity: Optional[int] = None) -> None:
        degree, _ = self._sh_layout(gaussians)
        self._sh_degree = degree
        reserve = len(gaussians) if capacity is None else max(len(gaussians), capacity)
        self._bridge.require(
            self._bridge.lib.rosplat_spirula_reset_scene(degree, reserve)
        )
        self._scene_ready = True

    def append_gaussians(self, gaussian_set: GaussianData, refresh: bool = False) -> None:
        if gaussian_set is None or len(gaussian_set) == 0:
            return
        degree, _ = self._sh_layout(gaussian_set)
        if refresh or not self._scene_ready:
            self._reset_for(gaussian_set)
        elif degree != self._sh_degree:
            raise SpirulaError(
                f"A streamed scene cannot change SH degree ({self._sh_degree} to {degree})"
            )

        arrays = self._native_arrays(gaussian_set, self._scale_modifier)
        self._bridge.require(
            self._bridge.lib.rosplat_spirula_append(
                len(gaussian_set), *(_as_pointer(array) for array in arrays)
            )
        )
        self.gaussians = gaussian_set
        self._render_dirty = True

    def update_gaussian_data(
        self, gaussian_set: GaussianData, full_update: bool = False
    ) -> None:
        self.append_gaussians(gaussian_set, refresh=True)

    def reset_gaussians(self) -> None:
        self._bridge.require(self._bridge.lib.rosplat_spirula_reset_scene(0, 0))
        self.gaussians = None
        self._scene_ready = False
        self._sh_degree = 0
        self._render_dirty = True

    def set_render_resolution(self, width: int, height: int) -> None:
        width, height = int(width), int(height)
        if width <= 0 or height <= 0:
            return
        if (width, height) != (self.width, self.height):
            self.width, self.height = width, height
            self._rgba = np.zeros((height, width, 4), dtype=np.uint8)
            self._texture = None
            self._camera_dirty = True
            self._render_dirty = True
            self._texture_dirty = True

    def set_model_matrix(self, model_matrix) -> None:
        matrix = np.ascontiguousarray(model_matrix, dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError("model matrix must be 4x4")
        if not np.array_equal(matrix, self._model_matrix):
            self._model_matrix = matrix
            self._camera_dirty = True
            self._render_dirty = True

    def set_scale_modifier(self, modifier: float) -> None:
        modifier = float(modifier)
        if modifier <= 0:
            raise ValueError("scale modifier must be positive")
        if modifier == self._scale_modifier:
            return
        self._bridge.require(
            self._bridge.lib.rosplat_spirula_set_scale_modifier(modifier)
        )
        self._scale_modifier = modifier
        self._render_dirty = True

    def _apply_render_options(self) -> None:
        degree = -1 if self._active_sh_degree is None else self._active_sh_degree
        self._bridge.require(
            self._bridge.lib.rosplat_spirula_set_render_options(
                int(self._output_mode), degree
            )
        )
        self._render_dirty = True

    def set_render_output(self, mode: RenderOutputMode) -> None:
        mode = RenderOutputMode(mode)
        if mode == self._output_mode:
            return
        self._output_mode = mode
        self._apply_render_options()

    def set_sh_degree(self, degree: Optional[int]) -> None:
        if degree is not None:
            degree = int(degree)
            if not 0 <= degree <= 4:
                raise ValueError("active SH degree must be between 0 and 4")
        if degree == self._active_sh_degree:
            return
        self._active_sh_degree = degree
        self._apply_render_options()

    def sort_and_update(self) -> None:
        # Spirula performs tile-key generation and radix sorting every render.
        return

    def update_vsync(self) -> None:
        # rendercanvas/wgpu owns the Vulkan swapchain and present mode.
        return

    def update_camera_pose(self) -> None:
        self._camera_dirty = True
        self._render_dirty = True

    def update_camera_intrin(self) -> None:
        self._camera_dirty = True
        self._render_dirty = True

    def _sync_camera(self) -> None:
        if not self._camera_dirty:
            return
        camera = self.world_settings.world_camera
        view_gl = np.asarray(camera.get_view_matrix(), dtype=np.float32)
        gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        view_cv = np.ascontiguousarray(gl_to_cv @ view_gl @ self._model_matrix)

        focal = camera.h / (2.0 * math.tan(camera.fovy / 2.0))
        intrinsics = np.ascontiguousarray(
            [focal, focal, camera.w / 2.0, camera.h / 2.0], dtype=np.float32
        )
        self._bridge.require(
            self._bridge.lib.rosplat_spirula_set_camera(
                self.width,
                self.height,
                _as_pointer(view_cv),
                _as_pointer(intrinsics),
            )
        )
        self._camera_dirty = False

    def render_rgba8(self) -> np.ndarray:
        """Render into the reusable host RGBA8 buffer without requiring a UI."""
        self._sync_camera()
        if self._render_dirty:
            self._bridge.require(
                self._bridge.lib.rosplat_spirula_render_rgba8(
                    _as_pointer(self._rgba), self._rgba.nbytes
                )
            )
            self._render_dirty = False
            self._texture_dirty = True
        return self._rgba

    def draw(self):
        self.render_rgba8()
        if self._texture_dirty:
            if self._texture_registry is None:
                raise SpirulaError("The Vulkan ImGui texture registry is not initialized")
            self._texture = self._texture_registry.upload(self, self._rgba)
            self._texture_dirty = False
        return self._texture

    @property
    def splat_count(self) -> int:
        return int(self._bridge.lib.rosplat_spirula_splat_count())

    @property
    def capacity(self) -> int:
        return int(self._bridge.lib.rosplat_spirula_capacity())

    @property
    def scene_sh_degree(self) -> int:
        return self._sh_degree

    @property
    def active_sh_degree(self) -> int:
        if self._active_sh_degree is None:
            return self._sh_degree
        return min(self._active_sh_degree, self._sh_degree)

    @property
    def output_mode(self) -> RenderOutputMode:
        return self._output_mode

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def device_vram_bytes(self) -> int:
        return self._device_vram_bytes

    @property
    def latest_rgba(self) -> np.ndarray:
        return self._rgba

    def shutdown(self) -> None:
        if self._texture_registry is not None:
            self._texture_registry.release(self)
        self._texture = None
        self._bridge.lib.rosplat_spirula_shutdown()
