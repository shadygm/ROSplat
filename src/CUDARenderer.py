import queue
import numpy as np
import torch
from gsplat import rasterization
from OpenGL import GL as gl
from base_gaussian_renderer import GaussianRenderBase
import util

# this is the same queue your UI is reading from
from imgui_manager import frame_queue  

class CUDARenderer(GaussianRenderBase):
    def __init__(self, w: int, h: int, world_settings):
        super().__init__()
        self.w, self.h = w, h
        self.world_settings = world_settings

        # 1) Create & allocate the texture once
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S,   gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T,   gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGB,
            self.w, self.h, 0,
            gl.GL_RGB, gl.GL_UNSIGNED_BYTE,
            None
        )

        # 2) Fullscreen‐quad shader & VAO
        self.program = util.load_shaders(
            './ui/shaders/fullscreen_quad_vert.glsl',
            './ui/shaders/fullscreen_quad_frag.glsl'
        )
        quad_v = np.array([
            [-1,  1,  0, 1],
            [ 1,  1,  1, 1],
            [ 1, -1,  1, 0],
            [-1, -1,  0, 0],
        ], dtype=np.float32)
        quad_f = np.array([0,2,1,  0,3,2], dtype=np.uint32).reshape(2,3)
        self.vao, _ = util.set_attributes(
            self.program, ["aPos","aUV"],
            [quad_v[:, :2], quad_v[:, 2:]]
        )
        util.set_faces_to_vao(self.vao, quad_f)

        # 3) GL state
        gl.glViewport(0, 0, self.w, self.h)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDisable(gl.GL_DEPTH_TEST)

        util.logger.info("CUDA renderer initialized.")

    def update_vsync(self) -> None:
        util.logger.info("VSync not supported in CUDA path")

    def update_gaussian_data(self, gaussian_set, full_update: bool = False) -> None:
        if gaussian_set is None or len(gaussian_set) == 0:
            util.logger.error("Gaussian data not loaded")
            return
        self.gaussians = gaussian_set

    def sort_and_update(self) -> None:
        if not getattr(self, "gaussians", None):
            util.logger.warning("No Gaussians to render.")
            return

        g = self.gaussians

        sh_len = g.sh.shape[-1]
        # Total SH coeffs = 3 * (bands)^2, so
        bands = int(np.sqrt(sh_len / 3))
        assert bands*bands*3 == sh_len, (
            f"Expected 3*(deg+1)^2 == {sh_len}, got bands={bands}"
        )
        deg = bands - 1
        K   = bands * bands


        # build proper 4×4 view matrix
        vm = self.world_settings.world_camera.get_view_matrix()
        vm = np.asarray(vm, np.float32)
        if vm.shape == (3,3):
            mm = np.eye(4, np.float32); mm[:3,:3]=vm; vm=mm
        elif vm.shape == (4,4):
            vm[3,:]=[0,0,0,1]
        viewmats = torch.tensor(vm, device='cuda')[None]

        Ks = torch.tensor(
            self.world_settings.world_camera.get_intrinsics_matrix(),
            device='cuda', dtype=torch.float32
        )[None]

        bg = torch.zeros((1,3), device='cuda', dtype=torch.float32)

        # to‐torch
        means     = torch.from_numpy(g.xyz.astype(np.float32)).cuda()
        quats     = torch.from_numpy(g.rot.astype(np.float32)).cuda()
        scales    = torch.from_numpy(g.scale.astype(np.float32)).cuda()
        opacities = torch.from_numpy(g.opacity.astype(np.float32)).squeeze(-1).cuda()
        colors    = torch.from_numpy(g.sh.astype(np.float32)).view(-1, K, 3).cuda()

        # rasterize
        color_img, alpha_img, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=self.w,
            height=self.h,
            backgrounds=bg,
            sh_degree=deg,
            packed=True,
            render_mode='RGB',
            rasterize_mode='classic',
        )

        # store for both GL‐texture upload and frame‐queue
        self.raster_out = (
            color_img[0].permute(2,0,1),
            alpha_img[0].permute(2,0,1)
        )

    def draw(self) -> None:
        if not hasattr(self, "raster_out"):
            return

        # A) Convert to H×W×3 RGB uint8
        color = self.raster_out[0]
        img = (color.clamp(0,1).mul(255).byte()
               .permute(1,2,0).cpu().numpy())
        # B) Upload into the pre-allocated texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D, 0, 0, 0,
            self.w, self.h,
            gl.GL_RGB, gl.GL_UNSIGNED_BYTE,
            # add opaque alpha channel:
            np.dstack([img, np.full((*img.shape[:2],1),255,dtype=np.uint8)])
        )

        # C) Draw full‐screen quad
        gl.glUseProgram(self.program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        loc = gl.glGetUniformLocation(self.program, "screen_tex")
        gl.glUniform1i(loc, 0)
        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)

        # D) Push into the frame‐queue for UI
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        frame_queue.put(img)

    # --- implement the remaining abstract methods ---

    def set_scale_modifier(self, modifier: float) -> None:
        # remember it (if your shader uses it, you can upload here)
        self.world_settings.scale_modifier = modifier

    def set_render_mode(self, mode: int) -> None:
        self.world_settings.render_mode = mode

    def set_render_resolution(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        gl.glViewport(0, 0, w, h)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGB, w, h, 0,
            gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None
        )

    def update_camera_pose(self) -> None:
        # no‐op here; we rebuild it fresh each frame in sort_and_update()
        pass

    def update_camera_intrin(self) -> None:
        pass

    def set_model_matrix(self, model_mat) -> None:
        pass
