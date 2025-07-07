from typing import Optional, List
import math
import numpy as np
import torch
import threading
from gsplat import rasterization
from OpenGL import GL as gl
import rosplat.core.util as util
from rosplat.render.renderer.base_gaussian_renderer import GaussianRenderBase
from rosplat.core.gaussian_representation import GaussianData
import cupy as cp

class CUDARenderer(GaussianRenderBase):
    def __init__(self, w: int, h: int, world_settings):
        super().__init__()
        self.width = w
        self.height = h
        self.world_settings = world_settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        self.scale_modifier = 1.0
        self.render_mode = 'RGB'
        self.sh_degree: Optional[int] = None
        self.model_matrix: np.ndarray = np.eye(4, dtype=np.float32)
        self._data_lock = threading.Lock()
        self._cache_means: List[torch.Tensor] = []
        self._cache_quats: List[torch.Tensor] = []
        self._cache_scales: List[torch.Tensor] = []
        self._cache_opacs: List[torch.Tensor] = []
        self._cache_colors: List[torch.Tensor] = []
        self._cache_sh_deg: Optional[int] = None


        # Create a single OpenGL texture
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8,
            self.width, self.height, 0,
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
            None
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def update_vsync(self) -> None:
        # Not used for CUDA renderer
        pass

    def reset_gaussians(self) -> None:
        with self._data_lock:
            # Clear main data
            self.means = None
            self.quats = None
            self.scales = None
            self.opacities = None
            self.colors = None
            self.sh_degree = None
            # Clear cache
            self._cache_means.clear()
            self._cache_quats.clear()
            self._cache_scales.clear()
            self._cache_opacs.clear()
            self._cache_colors.clear()
            self._cache_sh_deg = None
        # Force memory cleanup
        torch.cuda.empty_cache()

    def update_gaussian_data(self, gaussian_set: GaussianData, full_update: bool = False) -> None:
        if gaussian_set is None or len(gaussian_set) == 0:
            util.logger.error("Gaussian data not loaded")
            return

        # Build new tensors
        means = torch.from_numpy(gaussian_set.xyz.astype(np.float32)).to(self.device)
        quats = torch.from_numpy(gaussian_set.rot.astype(np.float32)).to(self.device)
        scales = torch.from_numpy(gaussian_set.scale.astype(np.float32)).to(self.device)
        opacs = torch.from_numpy(gaussian_set.opacity.astype(np.float32).squeeze()).to(self.device)

        total_sh = gaussian_set.sh_dim
        K = total_sh // 3
        sh_deg = int(math.sqrt(K) - 1)
        colors_np = gaussian_set.sh.astype(np.float32).reshape(-1, K, 3)
        colors = torch.from_numpy(colors_np).to(self.device)

        # Atomically replace main data and clear cache
        with self._data_lock:
            self.means = means
            self.quats = quats
            self.scales = scales
            self.opacities = opacs
            self.colors = colors
            self.sh_degree = sh_deg
            self._cache_means.clear()
            self._cache_quats.clear()
            self._cache_scales.clear()
            self._cache_opacs.clear()
            self._cache_colors.clear()
            self._cache_sh_deg = None

    def add_gaussians_from_ros(self, gaussian_set: GaussianData) -> None:
        """
        Enqueue new Gaussians into a cache; they will be
        merged into the main data only when write_through()
        is called (before each draw).
        """
        new_means  = torch.from_numpy(gaussian_set.xyz.astype(np.float32)).to(self.device)
        new_quats  = torch.from_numpy(gaussian_set.rot.astype(np.float32)).to(self.device)
        new_scales = torch.from_numpy(gaussian_set.scale.astype(np.float32)).to(self.device)
        new_opacs  = torch.from_numpy(gaussian_set.opacity.astype(np.float32).squeeze()).to(self.device)

        total_sh = gaussian_set.sh_dim
        k = total_sh // 3
        sh_deg = int(math.sqrt(k) - 1)
        colors_np = gaussian_set.sh.astype(np.float32).reshape(-1, k, 3)
        new_colors = torch.from_numpy(colors_np).to(self.device)

        with self._data_lock:
            self._cache_means.append(new_means)
            self._cache_quats.append(new_quats)
            self._cache_scales.append(new_scales)
            self._cache_opacs.append(new_opacs)
            self._cache_colors.append(new_colors)
            self._cache_sh_deg = sh_deg

    def write_through(self) -> None:
        """
        Atomically merge cached Gaussian updates into the
        main tensors, then clear the cache.
        """
        with self._data_lock:
            if not self._cache_means:
                return

            if self.means is None:
                # First initialization
                self.means     = torch.cat(self._cache_means,  dim=0)
                self.quats     = torch.cat(self._cache_quats,  dim=0)
                self.scales    = torch.cat(self._cache_scales, dim=0)
                self.opacities = torch.cat(self._cache_opacs,  dim=0)
                self.colors    = torch.cat(self._cache_colors, dim=0)
            else:
                # Append to existing
                self.means     = torch.cat([self.means,     *self._cache_means],  dim=0)
                self.quats     = torch.cat([self.quats,     *self._cache_quats],  dim=0)
                self.scales    = torch.cat([self.scales,    *self._cache_scales], dim=0)
                self.opacities = torch.cat([self.opacities, *self._cache_opacs],  dim=0)
                self.colors    = torch.cat([self.colors,    *self._cache_colors], dim=0)

            # Update final SH degree
            if self._cache_sh_deg is not None:
                self.sh_degree = self._cache_sh_deg

            # Clear cache
            self._cache_means.clear()
            self._cache_quats.clear()
            self._cache_scales.clear()
            self._cache_opacs.clear()
            self._cache_colors.clear()
            self._cache_sh_deg = None

    def sort_and_update(self) -> None:
        # GSplat handles sorting internally
        pass

    def set_scale_modifier(self, modifier: float) -> None:
        self.scale_modifier = modifier
        # TODO: apply to scales

    def set_model_matrix(self, model_mat: np.ndarray) -> None:
        self.model_matrix = model_mat.astype(np.float32)
        # TODO: upload into shader or transform pipeline

    def set_render_mode(self, mod: int) -> None:
        modes = {0: 'RGB', 1: 'D', 2: 'ED', 3: 'RGB+D', 4: 'RGB+ED'}
        self.render_mode = modes.get(mod, 'RGB')
        # TODO: adjust rasterization parameters

    def update_camera_pose(self) -> None:
        cam = self.world_settings.world_camera
        view_mat = cam.get_view_matrix().astype(np.float32)
        flip = np.eye(4, dtype=np.float32); flip[0,0] = -1.0
        view_mat = flip @ view_mat
        self.viewmats = torch.from_numpy(view_mat).to(self.device).unsqueeze(0)
    def update_camera_intrin(self) -> None:
        cam = self.world_settings.world_camera
        K = cam.get_intrinsics_matrix().astype(np.float32)
        self.Ks = torch.from_numpy(K).to(self.device).unsqueeze(0)

    def set_render_resolution(self, w: int, h: int) -> None:
        self.width, self.height = w, h
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F,
                        w, h, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glViewport(0, 0, w, h)

    def draw(self) -> int:
        # 1) Ensure data and camera
        if getattr(self, 'viewmats', None) is None:
            self.update_camera_pose()
        if getattr(self, 'Ks', None) is None:
            self.update_camera_intrin()
        self.write_through()

        # 2) Lock data
        with self._data_lock:
            if any(x is None for x in (
                self.means, self.quats, self.scales,
                self.opacities, self.colors,
                self.viewmats, self.Ks
            )):
                util.logger.error("CUDARenderer: missing data")
                return self.texture_id
            means, quats, scales = self.means, self.quats, self.scales
            opacs, colors        = self.opacities, self.colors
            viewmats, Ks         = self.viewmats, self.Ks
            sh_deg, w, h         = self.sh_degree, self.width, self.height

        # 3) Rasterize via CUDA
        cols, alps, _ = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacs, colors=colors,
            viewmats=viewmats, Ks=Ks,
            width=w, height=h,
            sh_degree=sh_deg,
            packed=True, tile_size=32,
            radius_clip=0.5, sparse_grad=True,
        )
        img = torch.cat([cols, alps], dim=-1)[0].cpu().numpy()  # (H, W, 4), float32 [0,1]
        max_val = img.max()
        min_val = img.min()
        mean_val = img.mean()

        if max_val == 0.0 and mean_val == 0.0:
            # Render solid black with full alpha
            img8 = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            img8[..., 3] = 255
        else:
            img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # 4) Upload image to GPU
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D, 0,
            0, 0, w, h,
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
            img8
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return self.texture_id