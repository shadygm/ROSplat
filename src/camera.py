import numpy as np
import glm
import util


class Camera:
    """
    A camera class to manage viewing transformations for a 3D scene.
    """

    def __init__(self, height: int, width: int) -> None:
        """
        Initialize the Camera with screen dimensions.
        """
        self.h: int = height
        self.w: int = width
        self.znear: float = 0.01
        self.zfar: float = 100.0
        self.fovy: float = np.pi / 2
        self.position: np.ndarray = np.array([0.0, 0.0, 3.0], dtype=np.float32)
        self.target: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.up: np.ndarray = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        self.yaw: float = -np.pi / 2
        self.pitch: float = 0.0

        self.dirty_pose: bool = True
        self.dirty_intrinsic: bool = True

        self.last_x: float = 640
        self.last_y: float = 360

        self.first_mouse: bool = True

        self.is_leftmouse_pressed: bool = False
        self.is_rightmouse_pressed: bool = False

        self.log = util.logger

        # Sensitivity factors
        self.rot_sensitivity: float = 0.02
        self.trans_sensitivity: float = 0.01
        self.zoom_sensitivity: float = 0.08
        self.roll_sensitivity: float = 0.03  # Currently unused
        self.target_dist: float = 3.0

    def process_translation(self, dx: float, dy: float, dz: float) -> None:
        """
        Process translation in the scene based on provided offsets.
        """
        front: np.ndarray = self.target - self.position
        norm_front = np.linalg.norm(front)
        if norm_front <= 0:
            front = np.array([0.0, 0.0, 1.0])
        else:
            front = front / norm_front

        right: np.ndarray = np.cross(front, self.up)
        up: np.ndarray = np.cross(right, front)

        move_direction: np.ndarray = front * dy + right * dx + self.up * dz
        norm_move = np.linalg.norm(move_direction)
        if norm_move > 0:
            move_direction = move_direction / norm_move

        self.position += move_direction * self.trans_sensitivity
        self.target += move_direction * self.trans_sensitivity
        self.dirty_pose = True

    def process_scroll(self, xoffset: float, yoffset: float) -> None:
        """
        Process zooming based on scroll input.
        """
        front: np.ndarray = self.target - self.position
        front = front / np.linalg.norm(front)
        delta = front * yoffset * self.zoom_sensitivity
        self.position += delta
        self.target += delta
        self.dirty_pose = True

    def process_mouse(self, xpos: float, ypos: float) -> None:
        """
        Process mouse movement for camera rotation or panning.
        """
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False

        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos

        if self.is_leftmouse_pressed:
            self._rotate_camera(xoffset, yoffset)
        if self.is_rightmouse_pressed:
            self._pan_camera(xoffset, yoffset)

    def _rotate_camera(self, xoffset: float, yoffset: float) -> None:
        """
        Rotate the camera based on mouse input.
        """
        self.yaw += xoffset * self.rot_sensitivity
        self.pitch += yoffset * self.rot_sensitivity
        self.pitch = np.clip(self.pitch, -np.pi / 2, np.pi / 2)

        front = np.array([
            np.cos(self.yaw) * np.cos(self.pitch),
            np.sin(self.pitch),
            np.sin(self.yaw) * np.cos(self.pitch)
        ])
        front = self._global_rot_mat() @ front.reshape(3, 1)
        front = front.flatten()
        distance = np.linalg.norm(self.position - self.target)
        self.position = self.target - front * distance
        self.dirty_pose = True

    def _pan_camera(self, xoffset: float, yoffset: float) -> None:
        """
        Pan the camera based on right mouse dragging.
        """
        front = self.target - self.position
        front = front / np.linalg.norm(front)
        right = np.cross(self.up, front)
        pan_horizontal = right * xoffset * self.trans_sensitivity
        self.position += pan_horizontal
        self.target += pan_horizontal

        cam_up = np.cross(right, front)
        pan_vertical = cam_up * yoffset * self.trans_sensitivity
        self.position += pan_vertical
        self.target += pan_vertical

        self.dirty_pose = True

    def get_pose(self) -> np.ndarray:
        """
        Get the current camera position.
        """
        return self.position

    def get_view_matrix(self) -> np.ndarray:
        """
        Compute and return the view matrix using glm.
        """
        pos = glm.vec3(*self.position)
        tgt = glm.vec3(*self.target)
        up_vec = glm.vec3(*self.up)
        return np.array(glm.lookAt(pos, tgt, up_vec))

    def get_project_matrix(self) -> np.ndarray:
        """
        Compute and return the projection matrix.
        """
        proj = glm.perspective(
            self.fovy,
            self.w / self.h,
            self.znear,
            self.zfar
        )
        return np.array(proj, dtype=np.float32)

    def get_htanfovxy_focal(self) -> list:
        """
        Calculate tan(fovx/2), tan(fovy/2) and the focal length.
        """
        htany = np.tan(self.fovy / 2)
        htanx = htany * (self.w / self.h)
        focal = self.w / (2 * htanx) if htanx != 0 else 0
        return [htanx, htany, focal]

    def _global_rot_mat(self) -> np.ndarray:
        """
        Compute a global rotation matrix based on the camera's up vector.
        """
        x_axis = np.array([1, 0, 0], dtype=np.float32)
        z_axis = np.cross(x_axis, self.up)
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(self.up, z_axis)
        return np.stack([x_axis, self.up, z_axis], axis=-1)