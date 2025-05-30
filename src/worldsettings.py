import numpy as np
from typing import Optional

from camera import Camera
from gaussian_interface.msg import SingleGaussian, GaussianArray
import gaussian_representation
from gaussian_representation import GaussianData
from renderer.OpenGLRenderer import OpenGLRenderer
from renderer.CUDARenderer import CUDARenderer
import util

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Maximum number of Gaussians allowed in the scene.
MAX_GAUSSIANS = 50_000_000


class WorldSettings:
    """
    Manages world state including camera, Gaussian representation, and renderer.
    """

    def __init__(self) -> None:
        self.world_camera = Camera(720, 1280)
        self.window = None
        self.input_handler = None
        self.imgui_manager = None
        self.gauss_renderer: Optional[OpenGLRenderer] = None

        # Gaussian data and state flags
        self.gaussian_set = gaussian_representation.naive_gaussian()
        self.have_new_gaussians = False
        self.is_original = True
        self.brand_new = False

        # Render parameters
        self.time_scale = 5.0
        self.model_transform_speed = 100.0
        self.scale_modifier = 1.0
        self.render_mode = 7
        self.auto_sort = False
        self.inverse_movements = False
        self.overwrite_gaussians = False
        self.model_transform = np.eye(4, dtype=np.float32)

    def process_model_translation(self, dx: float, dy: float) -> None:
        """
        Translate the model matrix along X and Y axes.
        """
        translation = np.eye(4, dtype=np.float32)
        translation[0, 3] = dx * self.model_transform_speed
        translation[1, 3] = dy * self.model_transform_speed
        self.model_transform = translation @ self.model_transform

        if self.gauss_renderer:
            self.gauss_renderer.set_model_matrix(self.model_transform)

    def update_camera_pose(self) -> None:
        """
        Update the camera pose in the active renderer.
        """
        if self.gauss_renderer:
            self.gauss_renderer.update_camera_pose()

    def update_window_size(self, width: int, height: int) -> None:
        """
        Update renderer's resolution to match window size.
        """
        if self.gauss_renderer:
            self.gauss_renderer.set_render_resolution(width, height)

    def update_render_mode(self, mode: int) -> None:
        """
        Set the rendering mode.
        """
        self.render_mode = mode
        self.update_activated_render_state()

    def update_camera_intrin(self) -> None:
        """
        Update camera intrinsics in the renderer.
        """
        if self.gauss_renderer:
            self.gauss_renderer.update_camera_intrin()

    def get_camera_pose(self):
        """
        Return the current camera pose from the world camera.
        """
        return self.world_camera.get_pose()

    def create_gaussian_renderer(self) -> None:
        """
        Initialize the appropriate Gaussian renderer (CUDA or OpenGL).
        """
        try:
            if HAS_TORCH and torch.cuda.is_available():
                self.gauss_renderer = CUDARenderer(self.world_camera.w, self.world_camera.h, self)
                util.logger.info("CUDA renderer initialized.")
            else:
                raise RuntimeError("Torch not installed or CUDA not available.")
        except Exception as e:
            util.logger.info(f"{e} Falling back to OpenGL renderer.")
            self.gauss_renderer = OpenGLRenderer(self.world_camera.w, self.world_camera.h, self)

        self.update_activated_render_state()

    def process_translation(self, dx: float, dy: float, dz: float) -> None:
        """
        Translate the camera in 3D space.
        """
        if isinstance(self.gauss_renderer, CUDARenderer):
            dx *= -1
            dy *= -1

        self.world_camera.process_translation(
            dx * self.time_scale,
            dy * self.time_scale,
            dz * self.time_scale
        )

    def get_num_gaussians(self) -> int:
        """
        Return the number of Gaussians currently stored.
        """
        return len(self.gaussian_set) if self.gaussian_set is not None else 0

    def convert_gaussian(self, gaussian: SingleGaussian) -> GaussianData:
        """
        Convert a SingleGaussian message (with array-based fields) to a GaussianData instance.
        """
        xyz = np.array([gaussian.xyz], dtype=np.float32)
        rot = np.array([gaussian.rotation], dtype=np.float32)
        scale = np.array([gaussian.scale], dtype=np.float32)
        opacity = np.array([[gaussian.opacity / 255.0]], dtype=np.float32)
        sh = np.array([gaussian.spherical_harmonics], dtype=np.float32)
        return GaussianData(xyz, rot, scale, opacity, sh)


    def switch_renderer(self, type: str) -> None:
        """
        Switch between CUDA and OpenGL renderers.
        """
        type = type.lower()
        self.gauss_renderer = None

        if type == "cuda":
            if not HAS_TORCH or not torch.cuda.is_available():
                util.logger.error("CUDA renderer not available.")
                return
            self.gauss_renderer = CUDARenderer(self.world_camera.w, self.world_camera.h, self)
        elif type == "opengl":
            self.gauss_renderer = OpenGLRenderer(self.world_camera.w, self.world_camera.h, self)
        else:
            util.logger.error(f"Unknown renderer type: {type}")
            return

        self.update_activated_render_state()

    def append_gaussian(self, gaussian: SingleGaussian) -> None:
        """
        Append a single Gaussian to the current set.
        """
        if len(self.gaussian_set) >= MAX_GAUSSIANS:
            return

        new_gaussian = self.convert_gaussian(gaussian)

        if self.overwrite_gaussians:
            self.gaussian_set = new_gaussian
        elif self.is_original:
            self.gaussian_set = new_gaussian
            self.is_original = False
        else:
            self.gaussian_set = gaussian_representation.combine_gaussians(
                [self.gaussian_set, new_gaussian]
            )
            self.brand_new = False

        self.brand_new = True
        self.have_new_gaussians = True

    def append_gaussians(self, gaussians: GaussianArray) -> None:
        """
        Append multiple Gaussians to the current set, or replace them entirely if a refresh is requested.
        """
        new_gaussians = [self.convert_gaussian(g) for g in gaussians.gaussians]

        if gaussians.refresh:
            util.logger.info("Received refresh signal, clearing existing Gaussians.")
            self.gaussian_set = gaussian_representation.combine_gaussians(new_gaussians)
            self.is_original = False
            self.brand_new = True
            self.have_new_gaussians = True

            if self.gauss_renderer:
                self.gauss_renderer.reset_gaussians()
                self.gauss_renderer.add_gaussians_from_ros(self.gaussian_set)
            return

        # Normal append path
        if isinstance(self.gauss_renderer, CUDARenderer):
            if self.overwrite_gaussians or self.is_original:
                self.gauss_renderer.reset_gaussians()

            if self.is_original:
                self.gaussian_set = gaussian_representation.combine_gaussians(new_gaussians)
                self.gauss_renderer.reset_gaussians()
                self.is_original = False
            else:
                self.gaussian_set = gaussian_representation.combine_gaussians(
                    [self.gaussian_set] + new_gaussians
                )

            temp_gaussian_set = gaussian_representation.combine_gaussians(new_gaussians)
            self.gauss_renderer.add_gaussians_from_ros(temp_gaussian_set)
            return

        if len(self.gaussian_set) >= MAX_GAUSSIANS:
            return

        if self.overwrite_gaussians or self.is_original:
            self.gaussian_set = gaussian_representation.combine_gaussians(new_gaussians)
            self.is_original = False
        else:
            self.gaussian_set = gaussian_representation.combine_gaussians(
                [self.gaussian_set] + new_gaussians
            )
            self.brand_new = False

        self.brand_new = True
        self.have_new_gaussians = True



    def reset_gaussians(self) -> None:
        """
        Reset to an empty or naive Gaussian state.
        """
        self.gaussian_set = gaussian_representation.naive_gaussian()
        self.is_original = True
        self.have_new_gaussians = True
        self.brand_new = False

    def load_ply(self, file_path: str) -> None:
        """
        Load Gaussians from a PLY file.
        """
        self.gaussian_set = gaussian_representation.from_ply(file_path)
        self.update_activated_render_state(full_update=True)

    def update_activated_render_state(self, full_update: bool = False) -> None:
        """
        Push all current state to the active renderer.
        """
        if not self.gauss_renderer:
            return

        self.gauss_renderer.update_gaussian_data(self.gaussian_set, full_update=full_update)

        if isinstance(self.gauss_renderer, OpenGLRenderer) and self.auto_sort:
            self.gauss_renderer.sort_and_update()

        self.gauss_renderer.set_scale_modifier(self.scale_modifier)
        self.gauss_renderer.set_render_mode(self.render_mode - 4)
        self.gauss_renderer.set_model_matrix(self.model_transform)
        self.gauss_renderer.update_camera_pose()
        self.gauss_renderer.update_camera_intrin()
        self.gauss_renderer.set_render_resolution(self.world_camera.w, self.world_camera.h)

        self.have_new_gaussians = False
