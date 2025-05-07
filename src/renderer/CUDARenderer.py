from typing import Optional
import math
import numpy as np
import torch
from gsplat import rasterization
from OpenGL import GL as gl
import util
from renderer.base_gaussian_renderer import GaussianRenderBase
from gaussian_representation import GaussianData

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
        self.original_means: Optional[torch.Tensor] = None
        self.model_matrix: Optional[np.ndarray] = np.eye(4, dtype=np.float32)

        # OpenGL texture setup
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, w, h, 0, gl.GL_RGBA, gl.GL_FLOAT, None)

        # Fullscreen quad shaders
        fullscreen_vert = """
#version 330 core
layout(location = 0) in vec2 position;
out vec2 texCoord;
void main() {
    texCoord = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""
        fullscreen_frag = """
#version 330 core
uniform sampler2D uTexture;
in vec2 texCoord;
out vec4 FragColor;
void main() {
    FragColor = texture(uTexture, texCoord);
}
"""
        # Compile shaders
        vert = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vert, fullscreen_vert)
        gl.glCompileShader(vert)
        frag = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(frag, fullscreen_frag)
        gl.glCompileShader(frag)
        self.program = gl.glCreateProgram()
        gl.glAttachShader(self.program, vert)
        gl.glAttachShader(self.program, frag)
        gl.glLinkProgram(self.program)
        gl.glDeleteShader(vert)
        gl.glDeleteShader(frag)

        # Quad VAO/VBO/EBO setup
        quad_v = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]], dtype=np.float32)
        quad_f = np.array([0, 2, 1, 0, 3, 2], dtype=np.uint32)
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.ebo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad_v.nbytes, quad_v, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 0, None)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, quad_f.nbytes, quad_f, gl.GL_STATIC_DRAW)
        gl.glBindVertexArray(0)

    def update_vsync(self) -> None:
        # VSync not applicable for CUDA renderer
        pass

    def update_gaussian_data(self, gaussian_set: GaussianData, full_update: bool = False) -> None:
        if gaussian_set is None or len(gaussian_set) == 0:
            util.logger.error("Gaussian data not loaded")
            return
        # Store original means
        self.means = torch.from_numpy(gaussian_set.xyz.astype(np.float32)).to(self.device)
        # Quaternions
        self.quats = torch.from_numpy(gaussian_set.rot.astype(np.float32)).to(self.device)
        # Scales
        self.scales = torch.from_numpy(gaussian_set.scale.astype(np.float32)).to(self.device)
        # Opacities
        self.opacities = torch.from_numpy(gaussian_set.opacity.astype(np.float32).squeeze()).to(self.device)
        # SH coefficients
        total_sh = gaussian_set.sh_dim
        K = total_sh // 3
        self.sh_degree = int(math.sqrt(K) - 1)
        colors_np = gaussian_set.sh.astype(np.float32).reshape(-1, K, 3)
        self.colors = torch.from_numpy(colors_np).to(self.device)
    
    def sort_and_update(self) -> None:
        # GSplat handles sorting internally
        pass

    def set_scale_modifier(self, modifier: float) -> None:
        self.scale_modifier = modifier
        # TODO: Implement scale modifier transformation

    def set_model_matrix(self, model_mat: np.ndarray) -> None:
        self.model_matrix = model_mat.astype(np.float32)
        # TODO: Implement model matrix transformation
    
    def set_render_mode(self, mod: int) -> None:
        modes = {0: 'RGB', 1: 'D', 2: 'ED', 3: 'RGB+D', 4: 'RGB+ED'}
        self.render_mode = modes.get(mod, 'RGB')
        # TODO: Actually change the rendering mode in the rasterizer

    def update_camera_pose(self) -> None:
        """
        Update the camera pose in the renderer, correcting for coordinate-flip.
        """
        cam = self.world_settings.world_camera
        # Obtain world-to-camera matrix
        view_mat = cam.get_view_matrix().astype(np.float32)
        # Flip the X axis to correct left/right inversion
        flip = np.eye(4, dtype=np.float32)
        flip[0, 0] = -1.0
        view_mat = flip @ view_mat
        # Upload to GPU with batch dim
        self.viewmats = torch.from_numpy(view_mat).to(self.device).unsqueeze(0)

    def update_camera_intrin(self) -> None:
        """
        Update camera intrinsics in the renderer using the Camera's intrinsic matrix.
        """
        cam = self.world_settings.world_camera
        # Use Camera.get_intrinsics_matrix() to fetch K
        K = cam.get_intrinsics_matrix()
        K = K.astype(np.float32)
        # batch dimension
        self.Ks = torch.from_numpy(K).to(self.device).unsqueeze(0)

    def set_render_resolution(self, w: int, h: int) -> None:
        self.width = w
        self.height = h

        # 1) Resize the texture that the CUDA output will be uploaded into
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F,
            w, h, 0, gl.GL_RGBA, gl.GL_FLOAT, None
        )

        # 2) Tell OpenGL to use the new dimensions for its viewport
        gl.glViewport(0, 0, w, h)
    

    def draw(self) -> None:
        # Rasterize with GSplat
        colors, alphas, _ = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            viewmats=self.viewmats,
            Ks=self.Ks,
            width=self.width,
            height=self.height,
            sh_degree=self.sh_degree,
            packed=True,
            tile_size=16,
            radius_clip=0.5,
            sparse_grad=True,
            rasterize_mode="antialiased",
        )
        
        # Compose RGBA image
        img = torch.cat([colors, alphas], dim=-1)[0]
        data = img.cpu().numpy().astype(np.float32)
        
        # Upload to texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_FLOAT, data)
        
        # Draw textured quad
        gl.glUseProgram(self.program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glUniform1i(gl.glGetUniformLocation(self.program, "uTexture"), 0)
        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
