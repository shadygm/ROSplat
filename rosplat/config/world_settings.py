from __future__ import annotations

from collections import deque
from enum import Enum
import threading
from typing import Optional

import numpy as np

from gaussian_interface.msg import GaussianArray, SingleGaussian
from rosplat.core import gaussian_representation, util
from rosplat.core.gaussian_representation import GaussianData
from rosplat.render.camera import Camera
from rosplat.render.renderer import RenderOutputMode, SpirulaRenderer


MAX_GAUSSIANS = 50_000_000


class RendererType(Enum):
    SPIRULA_VULKAN = "Spirula Vulkan"
    UNKNOWN = "Unknown"

    def __str__(self) -> str:
        return self.value


class WorldSettings:
    """Camera, streamed scene state, and the single Spirula renderer."""

    def __init__(self) -> None:
        self.world_camera = Camera(720, 1280)
        self.input_handler = None
        self.texture_registry = None
        self.gauss_renderer: Optional[SpirulaRenderer] = None
        self.gauss_renderer_type = RendererType.UNKNOWN

        self.gaussian_set: Optional[GaussianData] = gaussian_representation.naive_gaussian()
        self._gaussian_count = len(self.gaussian_set)
        self._full_scene_dirty = True
        self._pending_batches: deque[tuple[Optional[GaussianData], bool]] = deque()
        self._pending_lock = threading.Lock()
        self.have_new_gaussians = True

        self.time_scale = 5.0
        self.model_transform_speed = 100.0
        self.scale_modifier = 1.0
        self.render_output = RenderOutputMode.COLOR
        self.active_sh_degree: Optional[int] = None
        self.inverse_movements = False
        self.overwrite_gaussians = False
        self.model_transform = np.eye(4, dtype=np.float32)

    def process_model_translation(self, dx: float, dy: float) -> None:
        translation = np.eye(4, dtype=np.float32)
        translation[0, 3] = dx * self.model_transform_speed
        translation[1, 3] = dy * self.model_transform_speed
        self.model_transform = translation @ self.model_transform
        if self.gauss_renderer:
            self.gauss_renderer.set_model_matrix(self.model_transform)

    def update_camera_pose(self) -> None:
        if self.gauss_renderer:
            self.gauss_renderer.update_camera_pose()

    def update_camera_intrin(self) -> None:
        if self.gauss_renderer:
            self.gauss_renderer.update_camera_intrin()

    def update_window_size(self, width: int, height: int) -> None:
        width, height = int(width), int(height)
        if width <= 0 or height <= 0:
            return
        resized = (width, height) != (self.world_camera.w, self.world_camera.h)
        if resized:
            self.world_camera.w = width
            self.world_camera.h = height
            self.world_camera.dirty_intrinsic = True
        if self.gauss_renderer:
            self.gauss_renderer.set_render_resolution(width, height)
            if resized:
                self.gauss_renderer.update_camera_intrin()
                self.world_camera.dirty_intrinsic = False

    def check_inputs(self) -> None:
        if self.input_handler:
            self.input_handler.check_inputs()

    def update_render_output(self, mode: RenderOutputMode) -> None:
        self.render_output = RenderOutputMode(mode)
        if self.gauss_renderer:
            self.gauss_renderer.set_render_output(self.render_output)

    def update_sh_degree(self, degree: Optional[int]) -> None:
        self.active_sh_degree = degree
        if self.gauss_renderer:
            self.gauss_renderer.set_sh_degree(degree)

    def update_scale_modifier(self, modifier: float) -> None:
        self.scale_modifier = float(modifier)
        if self.gauss_renderer:
            self.gauss_renderer.set_scale_modifier(self.scale_modifier)

    def get_camera_pose(self):
        return self.world_camera.get_pose()

    def create_gaussian_renderer(self, texture_registry=None) -> None:
        if self.gauss_renderer:
            self.gauss_renderer.shutdown()
        self.texture_registry = texture_registry
        self.gauss_renderer = SpirulaRenderer(
            self.world_camera.w,
            self.world_camera.h,
            self,
            texture_registry=texture_registry,
        )
        self.gauss_renderer_type = RendererType.SPIRULA_VULKAN
        self._full_scene_dirty = True
        self.update_activated_render_state(full_update=True)

    def get_renderer_type(self) -> RendererType:
        return self.gauss_renderer_type

    def process_translation(self, dx: float, dy: float, dz: float) -> None:
        self.world_camera.process_translation(
            dx * self.time_scale,
            dy * self.time_scale,
            dz * self.time_scale,
        )

    def get_num_gaussians(self) -> int:
        with self._pending_lock:
            return self._gaussian_count

    @staticmethod
    def convert_gaussian(gaussian: SingleGaussian) -> GaussianData:
        return GaussianData(
            xyz=np.asarray(gaussian.xyz, dtype=np.float32).reshape(1, 3),
            rot=np.asarray(gaussian.rotation, dtype=np.float32).reshape(1, 4),
            scale=np.asarray(gaussian.scale, dtype=np.float32).reshape(1, 3),
            opacity=np.asarray([[gaussian.opacity / 255.0]], dtype=np.float32),
            sh=np.asarray(gaussian.spherical_harmonics, dtype=np.float32).reshape(1, -1),
        )

    @staticmethod
    def convert_gaussian_array(gaussians: GaussianArray) -> Optional[GaussianData]:
        messages = gaussians.gaussians
        count = len(messages)
        if count == 0:
            return None
        sh_dim = len(messages[0].spherical_harmonics)
        if sh_dim == 0:
            raise ValueError("Gaussian messages must contain spherical harmonics")

        xyz = np.empty((count, 3), dtype=np.float32)
        rot = np.empty((count, 4), dtype=np.float32)
        scale = np.empty((count, 3), dtype=np.float32)
        opacity = np.empty((count, 1), dtype=np.float32)
        sh = np.empty((count, sh_dim), dtype=np.float32)
        for index, gaussian in enumerate(messages):
            if len(gaussian.spherical_harmonics) != sh_dim:
                raise ValueError("A streamed batch cannot mix SH layouts")
            xyz[index] = gaussian.xyz
            rot[index] = gaussian.rotation
            scale[index] = gaussian.scale
            opacity[index, 0] = gaussian.opacity / 255.0
            sh[index] = gaussian.spherical_harmonics
        return GaussianData(xyz, rot, scale, opacity, sh)

    def _queue_batch(self, batch: Optional[GaussianData], refresh: bool) -> None:
        batch_count = len(batch) if batch is not None else 0
        with self._pending_lock:
            if refresh:
                self._pending_batches.clear()
                self._gaussian_count = batch_count
                self.gaussian_set = batch
            else:
                available = MAX_GAUSSIANS - self._gaussian_count
                if available <= 0:
                    return
                if batch is not None and batch_count > available:
                    batch = GaussianData(
                        batch.xyz[:available],
                        batch.rot[:available],
                        batch.scale[:available],
                        batch.opacity[:available],
                        batch.sh[:available],
                    )
                    batch_count = available
                self._gaussian_count += batch_count
            self._pending_batches.append((batch, refresh))
            self._full_scene_dirty = False
            self.have_new_gaussians = True

    def append_gaussian(self, gaussian: SingleGaussian) -> None:
        self._queue_batch(
            self.convert_gaussian(gaussian),
            refresh=self.overwrite_gaussians,
        )

    def append_gaussians(self, gaussians: GaussianArray) -> None:
        batch = self.convert_gaussian_array(gaussians)
        refresh = bool(gaussians.refresh or self.overwrite_gaussians)
        if refresh:
            util.logger.info("Received refresh signal, replacing the Vulkan scene")
        self._queue_batch(batch, refresh=refresh)

    def _drain_pending(self) -> list[tuple[Optional[GaussianData], bool]]:
        with self._pending_lock:
            pending = list(self._pending_batches)
            self._pending_batches.clear()
            return pending

    def reset_gaussians(self) -> None:
        with self._pending_lock:
            self._pending_batches.clear()
            self.gaussian_set = gaussian_representation.naive_gaussian()
            self._gaussian_count = len(self.gaussian_set)
            self._full_scene_dirty = True
            self.have_new_gaussians = True

    def load_ply(self, file_path: str) -> None:
        loaded = gaussian_representation.from_ply(file_path)
        with self._pending_lock:
            self._pending_batches.clear()
            self.gaussian_set = loaded
            self._gaussian_count = len(loaded)
            self._full_scene_dirty = True
            self.have_new_gaussians = True
        self.update_activated_render_state(full_update=True)

    def update_activated_render_state(self, full_update: bool = False) -> None:
        """Apply queued ROS batches on the GUI/render thread."""
        if not self.gauss_renderer:
            return

        pending = self._drain_pending()
        if pending:
            for batch, refresh in pending:
                if batch is None:
                    if refresh:
                        self.gauss_renderer.reset_gaussians()
                    continue
                self.gauss_renderer.append_gaussians(batch, refresh=refresh)
        elif full_update or self._full_scene_dirty:
            if self.gaussian_set is None or len(self.gaussian_set) == 0:
                self.gauss_renderer.reset_gaussians()
            else:
                self.gauss_renderer.update_gaussian_data(
                    self.gaussian_set,
                    full_update=True,
                )

        self.gauss_renderer.set_scale_modifier(self.scale_modifier)
        self.gauss_renderer.set_render_output(self.render_output)
        self.gauss_renderer.set_sh_degree(self.active_sh_degree)
        self.gauss_renderer.set_model_matrix(self.model_transform)
        self.gauss_renderer.update_camera_pose()
        self.gauss_renderer.update_camera_intrin()
        self.gauss_renderer.set_render_resolution(
            self.world_camera.w,
            self.world_camera.h,
        )
        with self._pending_lock:
            self._full_scene_dirty = False
            self.have_new_gaussians = bool(self._pending_batches)

    def shutdown(self) -> None:
        if self.gauss_renderer:
            self.gauss_renderer.shutdown()
            self.gauss_renderer = None
            self.gauss_renderer_type = RendererType.UNKNOWN
