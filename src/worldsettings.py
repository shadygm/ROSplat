import numpy as np
from typing import Optional

from camera import Camera
from gaussian_interface.msg import SingleGaussian, GaussianArray
import gaussian_representation
from gaussian_representation import GaussianData
from gaussian_renderer import OpenGLRenderer
import util


# Maximum number of gaussians allowed.
MAX_GAUSSIANS = 4000000


class WorldSettings:
    """
    Manages world state including camera, gaussian representation, and rendering.
    """
    def __init__(self) -> None:
        # Initialize camera.
        self.world_camera = Camera(720, 1280)
        self.window = None
        self.input_handler = None
        self.imgui_manager = None
        self.gauss_renderer: Optional[OpenGLRenderer] = None

        # Gaussian data.
        self.gaussian_set = gaussian_representation.naive_gaussian()
        self.have_new_gaussians: bool = False
        self.is_original: bool = True
        self.brand_new: bool = False

        # Parameters.
        self.time_scale: float = 0.1
        self.model_transform_speed: float = 100.0
        self.scale_modifier: float = 1.0
        self.render_mode: int = 7
        self.auto_sort: bool = False

        # Transformation matrix (4x4 identity).
        self.model_transform = np.eye(4, dtype=np.float32)

    def process_model_translation(self, dx: float, dy: float) -> None:
        """
        Update the model transformation matrix based on translation deltas.
        """
        translation = np.eye(4, dtype=np.float32)
        translation[0, 3] = dx * self.model_transform_speed
        translation[1, 3] = dy * self.model_transform_speed

        self.model_transform = translation @ self.model_transform
        if self.gauss_renderer:
            self.gauss_renderer.set_model_matrix(self.model_transform)

    def update_camera_pose(self) -> None:
        """
        Update the camera pose in the renderer.
        """
        if self.gauss_renderer:
            self.gauss_renderer.update_camera_pose()

    def update_render_mode(self, mode: int) -> None:
        """
        Update rendering mode and then apply the changes.
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
        Return the current camera pose.
        """
        return self.world_camera.get_pose()

    def create_gaussian_renderer(self) -> None:
        """
        Create an OpenGL renderer for gaussian visualization.
        """
        self.gauss_renderer = OpenGLRenderer(self.world_camera.w, self.world_camera.h, self)
        self.update_activated_render_state()

    def process_translation(self, dx: float, dy: float, dz: float) -> None:
        """
        Process a translation of the camera.
        """
        self.world_camera.process_translation(dx * self.time_scale,
                                                dy * self.time_scale,
                                                dz * self.time_scale)

    def get_num_gaussians(self) -> int:
        """
        Return the number of gaussians in the current set.
        """
        return len(self.gaussian_set) if self.gaussian_set is not None else 0

    def convert_gaussian(self, gaussian: SingleGaussian) -> GaussianData:
        """
        Convert a SingleGaussian message to a GaussianData instance.
        """
        xyz = np.array([[gaussian.xyz.x, gaussian.xyz.y, gaussian.xyz.z]], dtype=np.float32)
        rot = np.array([[gaussian.rotation.x, gaussian.rotation.y,
                         gaussian.rotation.z, gaussian.rotation.w]], dtype=np.float32)
        scale = np.array([[gaussian.scale.x, gaussian.scale.y, gaussian.scale.z]],
                         dtype=np.float32)
        opacity = np.array([[gaussian.opacity]], dtype=np.float32)
        sh = np.array([gaussian.spherical_harmonics], dtype=np.float32)
        return GaussianData(xyz, rot, scale, opacity, sh)

    def append_gaussian(self, gaussian: SingleGaussian) -> None:
        """
        Convert and add a SingleGaussian to the current gaussian set.
        """
        if len(self.gaussian_set) >= MAX_GAUSSIANS:
            return

        new_gaussian = self.convert_gaussian(gaussian)
        if self.is_original:
            self.is_original = False
            self.brand_new = True
            self.gaussian_set = new_gaussian
        else:
            self.brand_new = False
            self.gaussian_set = gaussian_representation.combine_gaussians(
                [self.gaussian_set, new_gaussian]
            )
        self.have_new_gaussians = True

    def append_gaussians(self, gaussians: GaussianArray) -> None:
        """
        Convert and add multiple gaussians to the current gaussian set.
        """
        if len(self.gaussian_set) >= MAX_GAUSSIANS:
            return

        new_gaussians = [self.convert_gaussian(g) for g in gaussians.gaussians]
        if self.is_original:
            self.is_original = False
            self.brand_new = True
            self.gaussian_set = gaussian_representation.combine_gaussians(new_gaussians)
        else:
            self.brand_new = False
            self.gaussian_set = gaussian_representation.combine_gaussians(
                [self.gaussian_set] + new_gaussians
            )
        self.have_new_gaussians = True

    def reset_gaussians(self) -> None:
        """
        Reset the gaussian set to its initial state.
        """
        self.gaussian_set = gaussian_representation.naive_gaussian()
        self.is_original = True
        self.have_new_gaussians = True
        self.brand_new = False

    def load_ply(self, file_path: str) -> None:
        """
        Load gaussian data from a PLY file.
        """
        self.gaussian_set = gaussian_representation.from_ply(file_path)
        self.update_activated_render_state(full_update=True)

    def update_activated_render_state(self, full_update: bool = False) -> None:
        """
        Update the rendering state of the renderer with the current gaussian set.
        """
        if not self.gauss_renderer:
            return

        self.gauss_renderer.update_gaussian_data(self.gaussian_set, full_update=full_update)
        self.gauss_renderer.sort_and_update()
        self.gauss_renderer.set_scale_modifier(self.scale_modifier)
        self.gauss_renderer.set_render_mode(self.render_mode - 4)
        self.gauss_renderer.set_model_matrix(self.model_transform)
        self.gauss_renderer.update_camera_pose()
        self.gauss_renderer.update_camera_intrin()
        self.gauss_renderer.set_render_resolution(self.world_camera.w, self.world_camera.h)
        self.have_new_gaussians = False