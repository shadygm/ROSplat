import numpy as np
import torch
from OpenGL import GL as gl
from base_gaussian_renderer import GaussianRenderBase
from imgui_manager import frame_queue

import util
from diff_gaussian_rasterization import rasterize_gaussians, GaussianRasterizationSettings, GaussianRasterizer
def compute_cov3D_from_scale_rotation(scale: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """
    Computes 3D covariance matrices (as 6D vectors) from per-Gaussian scale and rotation.
    Output shape: (N, 6)
    """
    # Convert quaternion to rotation matrix
    # Input shape: (N, 4)
    rot = torch.nn.functional.normalize(rot, dim=-1)
    x, y, z, w = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]

    R = torch.stack([
        1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
        2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
        2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y
    ], dim=-1).reshape(-1, 3, 3)

    S = torch.diag_embed(scale)  # (N, 3, 3)
    Sigma = R @ S @ S @ R.transpose(1, 2)  # (N, 3, 3)

    # Extract symmetric matrix upper triangle: (xx, yy, zz, xy, xz, yz)
    cov3D = torch.stack([
        Sigma[:, 0, 0],
        Sigma[:, 1, 1],
        Sigma[:, 2, 2],
        Sigma[:, 0, 1],
        Sigma[:, 0, 2],
        Sigma[:, 1, 2]
    ], dim=-1)

    return cov3D


class CUDARenderer(GaussianRenderBase):
    def __init__(self, w: int, h: int, world_settings):
        super().__init__()
        self.w, self.h = w, h
        self.world_settings = world_settings

        # Generate the texture that we'll upload into each frame
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        # Ensure the texture is complete without mipmaps
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S,     gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T,     gl.GL_CLAMP_TO_EDGE)
        # (If you do want mipmaps instead, call glGenerateMipmap(GL_TEXTURE_2D) after each glTexImage2D)

        # Load the custom shader for fullscreen textured quad
        self.program = util.load_shaders(
            './ui/shaders/fullscreen_quad_vert.glsl',
            './ui/shaders/fullscreen_quad_frag.glsl'
        )

        # Define fullscreen quad: [x, y, u, v]
        self.quad_v = np.array([
            [-1,  1, 0, 1],  # top-left
            [ 1,  1, 1, 1],  # top-right
            [ 1, -1, 1, 0],  # bottom-right
            [-1, -1, 0, 0],  # bottom-left
        ], dtype=np.float32)

        self.quad_f = np.array([
            0, 2, 1,
            0, 3, 2
        ], dtype=np.uint32).reshape(2, 3)

        # Upload attributes to GPU: position (aPos), texcoord (aUV)
        self.vao, _ = util.set_attributes(
            self.program,
            ["aPos", "aUV"],
            [self.quad_v[:, :2], self.quad_v[:, 2:]]
        )
        util.set_faces_to_vao(self.vao, self.quad_f)

        # OpenGL state
        gl.glViewport(0, 0, self.w, self.h)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        # Disable depth‐test so your full‐screen quad never gets discarded
        gl.glDisable(gl.GL_DEPTH_TEST)

        util.logger.info("CUDA renderer with fullscreen textured quad initialized.")


    def update_vsync(self) -> None:
        util.logger.info("VSync is not supported in CUDA renderer")

    def update_gaussian_data(self, gaussian_set, full_update: bool = False) -> None:
        if gaussian_set is None or len(gaussian_set) == 0:
            util.logger.error("Gaussian data not loaded")
            return
        self.gaussians = gaussian_set

    def sort_and_update(self) -> None:
        if self.gaussians is None or len(self.gaussians) == 0:
            util.logger.warning("No Gaussians to render.")
            return

        g = self.gaussians
        # 1) Check counts
        N = len(g.xyz)
        print(f"[DEBUG] N={N}, shapes:",
            f"xyz{g.xyz.shape},",
            f"sh{g.sh.shape},",
            f"opacity{g.opacity.shape},",
            f"scale{g.scale.shape},",
            f"rot{g.rot.shape}")

        # 2) Check SH length validity
        sh_len = g.sh.shape[-1]
        if sh_len % 3 != 0:
            raise RuntimeError(f"SH vector length {sh_len} not a multiple of 3")
        deg = int(np.sqrt(sh_len//3) - 1)
        if deg < 0 or (deg+1)**2*3 != sh_len:
            raise RuntimeError(f"Computed SH degree {deg} is invalid for length {sh_len}")

        # 3) Check NaNs/Infs
        for name, arr in [("xyz", g.xyz), ("sh", g.sh),
                        ("opa", g.opacity), ("sca", g.scale), ("rot", g.rot)]:
            if np.isnan(arr).any() or np.isinf(arr).any():
                raise RuntimeError(f"Found NaN/Inf in {name}")

        cam = self.world_settings.world_camera

        # Camera & rasterization settings
        view_mat = torch.tensor(cam.get_view_matrix(), dtype=torch.float32, device='cuda')
        proj_mat = torch.tensor(cam.get_project_matrix(), dtype=torch.float32, device='cuda')
        campos = torch.tensor(cam.position, dtype=torch.float32, device='cuda')
        tan_fovx, tan_fovy, _ = cam.get_htanfovxy_focal()
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        raster_settings = GaussianRasterizationSettings(
            image_height=self.h,
            image_width=self.w,
            tanfovx=tan_fovx,
            tanfovy=tan_fovy,
            bg=background,
            scale_modifier=self.world_settings.scale_modifier,
            viewmatrix=view_mat,
            projmatrix=proj_mat,
            sh_degree=int(np.sqrt(g.sh.shape[-1] // 3) - 1),
            campos=campos,
            prefiltered=False,
            debug=True
        )

        rasterizer = GaussianRasterizer(raster_settings)

        # Data tensors
        N = g.xyz.shape[0]
        means3D   = torch.tensor(g.xyz, dtype=torch.float32, device='cuda')
        means2D   = torch.zeros((N, 2), dtype=torch.float32, device='cuda')           
        shs       = torch.tensor(g.sh,  dtype=torch.float32, device='cuda')
        opacities = torch.tensor(g.opacity, dtype=torch.float32, device='cuda').squeeze(-1) 
        scales    = torch.tensor(g.scale, dtype=torch.float32, device='cuda')
        rotations = torch.tensor(g.rot,   dtype=torch.float32, device='cuda')


        # Rasterize using SH and computed covariance
        self.raster_out = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )

        util.logger.debug("CUDA Gaussian forward pass rendering complete")



    def set_scale_modifier(self, modifier: float) -> None:
        self.world_settings.scale_modifier = modifier

    def set_render_mode(self, mod: int) -> None:
        pass # Not needed in CUDA renderer

    def set_render_resolution(self, w: int, h: int) -> None:
        self.w, self.h = w, h

    def update_camera_pose(self) -> None:
        pass  # Already handled in sort_and_update

    def update_camera_intrin(self) -> None:
        pass  # Already handled in sort_and_update

    def set_model_matrix(self, model_mat) -> None:
        pass  # CUDA rasterizer assumes baked transform

    def draw(self) -> None:
        if self.raster_out is None:
            return

        color_tensor = self.raster_out[0]  # [3, H, W]
        image = color_tensor.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()

        # Log whether image has any non-zero pixel
        has_nonzero = np.any(image != 0)
        util.logger.info(f"Min: {image.min()}, Max: {image.max()}, Mean: {image.mean():.2f}")

        if image.shape[2] == 3:
            h, w, _ = image.shape
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            image = np.concatenate([image, alpha], axis=2)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, image.shape[1], image.shape[0],
                        0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, image)

        gl.glUseProgram(self.program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)

        loc = gl.glGetUniformLocation(self.program, "screen_tex")
        gl.glUniform1i(loc, 0)

        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)



