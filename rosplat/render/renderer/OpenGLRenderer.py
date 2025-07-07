from pathlib import Path
from OpenGL import GL as gl
import numpy as np
from rosplat.core import util
import sys
from rosplat.render.renderer.base_gaussian_renderer import GaussianRenderBase
wglSwapIntervalEXT = None

# ---------------- Sorting Strategies ----------------

class GaussianSorterBase:
    def sort(self, gaussian_set, view_mat: np.ndarray, force_update: bool = False) -> np.ndarray:
        raise NotImplementedError("sort must be implemented by subclasses")


class CPUSorter(GaussianSorterBase):
    def sort(self, gaussian_set, view_mat: np.ndarray, force_update: bool = False) -> np.ndarray:
        if gaussian_set is None or len(gaussian_set) == 0:
            util.logger.error("Gaussian data not loaded")
            return np.empty((0, 1), dtype=np.int32)
        xyz = np.asarray(gaussian_set.xyz)
        view_mat = np.asarray(view_mat)
        xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
        depth = xyz_view[:, 2, 0]
        index = np.argsort(depth).astype(np.int32).reshape(-1, 1)
        return index

class CupySorter(GaussianSorterBase):
    def __init__(self):
        self._buffer_xyz = None
        self._buffer_gausid = None

    def sort(self, gaussian_set, view_mat: np.ndarray, force_update: bool = False) -> np.ndarray:
        import cupy as cp
        if gaussian_set is None or len(gaussian_set) == 0:
            util.logger.error("Gaussian data not loaded")
            return np.empty((0, 1), dtype=np.int32)
        if (
            self._buffer_gausid != id(gaussian_set)
            or self._buffer_xyz is None
            or self._buffer_xyz.shape[0] != len(gaussian_set)
            or force_update
        ):
            self._buffer_xyz = cp.asarray(gaussian_set.xyz)
            self._buffer_gausid = id(gaussian_set)
        view_mat_cp = cp.asarray(view_mat)
        xyz_view = view_mat_cp[None, :3, :3] @ self._buffer_xyz[..., None] + view_mat_cp[None, :3, 3, None]
        depth = xyz_view[:, 2, 0]
        index = cp.argsort(depth).astype(cp.int32).reshape(-1, 1)
        return cp.asnumpy(index)
    
class TorchSorter(GaussianSorterBase):
    def __init__(self):
        self._buffer_xyz = None
        self._buffer_gausid = None

    def sort(self, gaussian_set, view_mat: np.ndarray, force_update: bool = False) -> np.ndarray:
        import torch
        if gaussian_set is None or len(gaussian_set) == 0:
            util.logger.error("Gaussian data not loaded")
            return np.empty((0, 1), dtype=np.int32)
        if (
            self._buffer_gausid != id(gaussian_set)
            or self._buffer_xyz is None
            or self._buffer_xyz.shape[0] != len(gaussian_set)
            or force_update
        ):
            self._buffer_xyz = torch.tensor(gaussian_set.xyz, device='cuda')
            self._buffer_gausid = id(gaussian_set)
        view_mat_torch = torch.tensor(view_mat, device='cuda')
        xyz_view = view_mat_torch[None, :3, :3] @ self._buffer_xyz[..., None] + view_mat_torch[None, :3, 3, None]
        depth = xyz_view[:, 2, 0]
        index = torch.argsort(depth).type(torch.int32).reshape(-1, 1).cpu().numpy()
        return index


def get_sorter() -> GaussianSorterBase:
    try:
        import torch
        if torch.cuda.is_available():
            util.logger.info("Detected torch cuda installed, will use torch as sorting backend")
            return TorchSorter()
        raise ImportError
    except ImportError:
        try:
            import cupy  # noqa: F401
            util.logger.info("Detected cupy installed, will use cupy as sorting backend")
            return CupySorter()
        except ImportError:
            util.logger.info("Using CPU sorting backend")
            return CPUSorter()


_sort_gaussian = get_sorter()
    

class OpenGLRenderer(GaussianRenderBase):
    def __init__(self, w: int, h: int, world_settings):
        super().__init__()
        gl.glViewport(0, 0, w, h)
        shader_dir = Path(__file__).parent / 'shaders'
        vert_shader_path = shader_dir / 'gau_vert.glsl'
        frag_shader_path = shader_dir / 'gau_frag.glsl'
        self.program = util.load_shaders(str(vert_shader_path), str(frag_shader_path))

        self._prev_gaussian_count = 0
        self.gau_bufferid = None
        self.index_bufferid = None

        self.quad_v = np.array([
            -1,  1,  # Top-left
             1,  1,  # Top-right
             1, -1,  # Bottom-right
            -1, -1   # Bottom-left
        ], dtype=np.float32).reshape(4, 2)

        self.quad_f = np.array([
            0, 2, 1,  # First triangle
            0, 3, 2   # Second triangle
        ], dtype=np.uint32).reshape(2, 3)

        vao, buffer_id = util.set_attributes(self.program, ["position"], [self.quad_v])
        util.set_faces_to_vao(vao, self.quad_f)
        self.vao = vao

        gl.glEnable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.world_settings = world_settings

        # --- Create texture for rendering ---
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # --- Create and bind the framebuffer ---
        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)

        # --- Attach the color texture to the FBO ---
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
            self.texture_id,
            0
        )

        # --- Create and attach the depth buffer while FBO is bound ---
        self.depth_buffer = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.depth_buffer)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, w, h)
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER,
            gl.GL_DEPTH_ATTACHMENT,
            gl.GL_RENDERBUFFER,
            self.depth_buffer
        )

        # --- Check FBO completeness ---
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        assert status == gl.GL_FRAMEBUFFER_COMPLETE, f"Framebuffer incomplete: {status}"

        # --- Unbind the framebuffer ---
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def update_vsync(self) -> None:
        if wglSwapIntervalEXT is not None:
            wglSwapIntervalEXT(1 if self.reduce_updates else 0)
        else:
            print("VSync is not supported")

    def update_gaussian_data(self, gaussian_set, full_update: bool = False) -> None:
        if gaussian_set is None or len(gaussian_set) == 0:
            util.logger.error("Gaussian data not loaded")
            return

        self.gaussians = gaussian_set
        gaussian_data = gaussian_set.flat()

        if self.world_settings.brand_new:
            self._prev_gaussian_count = 0

        self.gau_bufferid, self._prev_gaussian_count = util.set_storage_buffer_data(
            program=self.program,
            key="gaussian_data",
            value=gaussian_data,
            bind_idx=0,
            buffer_id=self.gau_bufferid,
            prev_count=self._prev_gaussian_count,
            full_update=full_update
        )
        util.set_uniform_1int(self.program, gaussian_set.sh_dim, "sh_dim")

    def sort_and_update(self) -> None:
        if self.gaussians is None or len(self.gaussians) == 0:
            return

        camera = self.world_settings.world_camera
        time_start = util.get_time()
        index = _sort_gaussian.sort(self.gaussians, camera.get_view_matrix_glm(), force_update=True)
        time_end = util.get_time()
        util.logger.debug(f"Sorting time: {time_end - time_start:.3f} s")

        self.index_bufferid = util.upload_index_array(
            program=self.program,
            key="gaussian_order",
            index_array=index,
            bind_idx=1,
            buffer_id=self.index_bufferid
        )

    def set_scale_modifier(self, modifier: float) -> None:
        util.set_uniform_1f(self.program, modifier, "scale_modifier")

    def set_render_mode(self, mod: int) -> None:
        util.set_uniform_1int(self.program, mod, "render_mod")

    def set_render_resolution(self, w: int, h: int) -> None:
        gl.glViewport(0, 0, w, h)

    def update_camera_pose(self) -> None:
        camera = self.world_settings.world_camera
        view_mat = camera.get_view_matrix_glm()
        util.set_uniform_mat4(self.program, view_mat, "view_matrix")
        util.set_uniform_v3(self.program, camera.position, "cam_pos")

    def update_camera_intrin(self) -> None:
        camera = self.world_settings.world_camera
        proj_mat = camera.get_project_matrix()
        util.set_uniform_mat4(self.program, proj_mat, "projection_matrix")
        util.set_uniform_v3(self.program, camera.get_htanfovxy_focal(), "hfovxy_focal")

    def set_model_matrix(self, model_mat) -> None:
        util.set_uniform_mat4(self.program, model_mat, "model_matrix")

    def draw(self) -> int:
        if self.gaussians is None or len(self.gaussians) == 0:
            return self.texture_id  # Return blank texture

        # --- Bind the framebuffer to render into texture ---
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, self.world_settings.world_camera.w, self.world_settings.world_camera.h)

        # --- Clear the buffer ---
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # --- Render ---
        gl.glUseProgram(self.program)
        gl.glBindVertexArray(self.vao)

        num_gaussians = len(self.gaussians)
        gl.glDrawElementsInstanced(
            gl.GL_TRIANGLES,
            self.quad_f.size,
            gl.GL_UNSIGNED_INT,
            None,
            num_gaussians
        )

        # --- Unbind framebuffer ---
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # --- Return the texture that now contains the rendered image ---
        return self.texture_id

