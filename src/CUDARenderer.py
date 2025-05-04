from gsplat import rasterization
import torch
import numpy as np
from OpenGL import GL as gl
from base_gaussian_renderer import GaussianRenderBase
import util

class CUDARenderer(GaussianRenderBase):
    def __init__(self, w: int, h: int, world_settings):
        super().__init__()
        self.w, self.h = w, h
        self.world_settings = world_settings

        # OpenGL texture setup
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S,   gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T,   gl.GL_CLAMP_TO_EDGE)

        # Shader for fullscreen quad
        self.program = util.load_shaders(
            './ui/shaders/fullscreen_quad_vert.glsl',
            './ui/shaders/fullscreen_quad_frag.glsl'
        )

        # Fullscreen quad geometry
        self.quad_v = np.array([
            [-1,  1,  0, 1],
            [ 1,  1,  1, 1],
            [ 1, -1,  1, 0],
            [-1, -1,  0, 0],
        ], dtype=np.float32)
        self.quad_f = np.array([0,2,1,  0,3,2], dtype=np.uint32).reshape(2,3)
        self.vao, _ = util.set_attributes(
            self.program,
            ["aPos","aUV"],
            [self.quad_v[:, :2], self.quad_v[:, 2:]]
        )
        util.set_faces_to_vao(self.vao, self.quad_f)

        # GL state
        gl.glViewport(0, 0, self.w, self.h)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDisable(gl.GL_DEPTH_TEST)

        util.logger.info("CUDA renderer with gsplat rasterization initialized.")

    def update_vsync(self) -> None:
        util.logger.info("VSync is not supported in CUDA renderer")

    def update_gaussian_data(self, gaussian_set, full_update: bool = False) -> None:
        if gaussian_set is None or len(gaussian_set) == 0:
            util.logger.error("Gaussian data not loaded")
            return
        self.gaussians = gaussian_set

    def sort_and_update(self) -> None:
        # Nothing to do if no data
        if not getattr(self, "gaussians", None) or len(self.gaussians) == 0:
            util.logger.warning("No Gaussians to render.")
            return

        g = self.gaussians

        # --- 1) Compute SH degree and K ---
        sh_len = g.sh.shape[-1]                            # total SH channels = 3*(deg+1)^2
        deg    = int(np.sqrt(sh_len // 3) - 1)              # solves 3*(deg+1)^2 = sh_len
        K      = (deg + 1) ** 2

        # --- 2) Camera matrices (batched) ---
        cam = self.world_settings.world_camera
        viewmats = torch.tensor(
            cam.get_view_matrix(), dtype=torch.float32, device='cuda'
        )[None, :, :]                                      # [1,4,4]
        Ks = torch.tensor(
            cam.get_intrinsics_matrix(), dtype=torch.float32, device='cuda'
        )[None, :, :]                                      # [1,3,3]

        # --- 3) Background color (batched) ---
        background = torch.tensor([[0.0, 0.0, 0.0]],
                                  dtype=torch.float32, device='cuda')  # [1,3]

        # --- 4) Convert GaussianData → torch with correct shapes ---
        means     = torch.from_numpy(g.xyz.astype(np.float32)).to('cuda')        # [N,3]
        quats     = torch.from_numpy(g.rot.astype(np.float32)).to('cuda')        # [N,4]
        scales    = torch.from_numpy(g.scale.astype(np.float32)).to('cuda')      # [N,3]
        opacities = (
            torch.from_numpy(g.opacity.astype(np.float32))
            .squeeze(-1)                          # from [N,1] → [N]
            .to('cuda')
        )
        colors = (
            torch.from_numpy(g.sh.astype(np.float32))
            .view(-1, K, 3)                       # [N, K, 3]
            .to('cuda')
        )

        # Optional: debug shapes
        # print("means:", means.shape, "quats:", quats.shape,
        #       "scales:", scales.shape, "opacities:", opacities.shape,
        #       "colors:", colors.shape,
        #       "viewmats:", viewmats.shape, "Ks:", Ks.shape)

        # --- 5) Call gsplat rasterization ---
        color_img, alpha_img, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=self.w,
            height=self.h,
            backgrounds=background,
            sh_degree=deg,
            packed=True,
            render_mode='RGB',
            rasterize_mode='classic',
        )

        # Store for drawing; permute to [C,H,W]
        self.raster_out = (
            color_img[0].permute(2, 0, 1),   # [3,H,W]
            alpha_img[0].permute(2, 0, 1)    # [1,H,W]
        )
        util.logger.debug("gsplat rasterization complete")

    def draw(self) -> None:
        if not hasattr(self, "raster_out") or self.raster_out is None:
            return

        # Convert to H×W×RGBA uint8
        color_tensor = self.raster_out[0]
        img = color_tensor.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        util.logger.info(f"Rendered image stats: min={img.min()}, max={img.max()}, mean={img.mean():.2f}")

        # Add alpha channel if missing
        if img.shape[2] == 3:
            h, w, _ = img.shape
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            img = np.concatenate([img, alpha], axis=2)

        # Upload to OpenGL
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, img.shape[1], img.shape[0],
            0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img
        )

        # Draw fullscreen quad
        gl.glUseProgram(self.program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        loc = gl.glGetUniformLocation(self.program, "screen_tex")
        gl.glUniform1i(loc, 0)
        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)

    def set_scale_modifier(self, modifier: float) -> None:
        self.world_settings.scale_modifier = modifier

    def set_render_mode(self, mod: int) -> None:
        pass

    def set_render_resolution(self, w: int, h: int) -> None:
        self.w, self.h = w, h

    def update_camera_pose(self) -> None:
        pass

    def update_camera_intrin(self) -> None:
        pass

    def set_model_matrix(self, model_mat) -> None:
        pass
