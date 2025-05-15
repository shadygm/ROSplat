# camera.py
import math
import numpy as np
import glm
import util
from scipy.spatial.transform import Rotation as R

class Camera:
    """
    A camera class to manage viewing transformations for a 3D scene.
    Internally uses quaternion orientation, externally exposes Euler angles.
    """

    def __init__(self, height: int, width: int) -> None:
        self.h = height
        self.w = width
        self.znear = 0.01
        self.zfar = 100.0
        self.fovy = np.pi / 2

        # Position
        self.position = np.array([0.0, 0.0, 3.0], dtype=np.float32)

        # Orientation as a unit quaternion [x, y, z, w]
        self.rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # World-up is now +Y
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Mouse state
        self.first_mouse = True
        self.last_x = width  / 2
        self.last_y = height / 2
        self.is_leftmouse_pressed  = False
        self.is_rightmouse_pressed = False

        # Dirty flags
        self.dirty_pose      = True
        self.dirty_intrinsic = True

        # Sensitivities
        self.rot_sensitivity   = 0.005
        self.trans_sensitivity = 0.01
        self.zoom_sensitivity  = 0.08
        self.roll_speed  = 45.0

        self.log = util.logger

    def process_translation(self, dx: float, dy: float, dz: float) -> None:
        """
        Move the camera along its local axes:
          dx → right, dy → forward, dz → up (world-up)
        """
        front = self._get_front_vector()
        front /= np.linalg.norm(front)
        right = np.cross(front, self.up)
        right /= np.linalg.norm(right)
        # no normalization of the sum: each axis moves at its own speed
        self.position += (
            right * dx +
            front * dy +
            self.up * dz
        ) * self.trans_sensitivity
        self.dirty_pose = True

    def process_scroll(self, xoffset: float, yoffset: float) -> None:
        """
        Zoom in/out by moving along the view direction.
        """
        front = self._get_front_vector()
        front /= np.linalg.norm(front)
        self.position += front * yoffset * self.zoom_sensitivity
        self.dirty_pose = True

    def get_pose(self) -> tuple:
        """
        Returns the current camera position and rotation in Euler angles (yaw, pitch, roll).
        """
        rot = R.from_quat(self.rotation)
        yaw, pitch, roll = rot.as_euler('yxz', degrees=False)
        return self.position, (yaw, pitch, roll)

    def process_mouse(self, xpos: float, ypos: float) -> None:
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False

        dx = xpos - self.last_x
        dy = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos

        if self.is_leftmouse_pressed:
            self._rotate_camera(dx, dy)
        if self.is_rightmouse_pressed:
            self._pan_camera(dx, dy)

    def _rotate_camera(self, dx: float, dy: float) -> None:
        """
        Yaw around world-up; pitch around camera's right axis.
        Drag right → look right; drag up → look up.
        """
        yaw = dx * self.rot_sensitivity
        pitch = dy * self.rot_sensitivity

        # rotate around world-up
        r_yaw = R.from_rotvec(self.up * yaw)

        # rotate around camera's right axis
        front = self._get_front_vector()
        right = np.cross(front, self.up)
        right /= np.linalg.norm(right)
        r_pitch = R.from_rotvec(right * pitch)

        # apply new orientation
        r_current = R.from_quat(self.rotation)
        r_new = r_pitch * r_yaw * r_current
        self.rotation = r_new.as_quat()
        self.dirty_pose = True

    def _pan_camera(self, dx: float, dy: float) -> None:
        """
        Pan the camera parallel to the view plane.
        Drag right → camera moves right
        Drag up    → camera moves up
        """
        front = self._get_front_vector()
        front /= np.linalg.norm(front)
        right = np.cross(front, self.up)
        right /= np.linalg.norm(right)
        up_cam = np.cross(right, front)
        up_cam /= np.linalg.norm(up_cam)

        # camera moves in the opposite sense of the drag
        self.position += (
            -right * dx +
             up_cam * dy
        ) * self.trans_sensitivity
        self.dirty_pose = True

    def process_roll(self, direction: float, delta_time: float) -> None:
        """
        Roll the camera ccw/cw around its view (front) axis.

        :param direction: +1 for CCW (Q), -1 for CW (E)
        :param delta_time: time elapsed since last frame (in seconds)
        """
        # 1) Compute roll angle in radians
        angle_rad = math.radians(direction * self.roll_speed * delta_time)

        # 2) Get the camera’s forward vector in world space
        front = self._get_front_vector()
        front /= np.linalg.norm(front)

        # 3) Rotate the up vector around the front axis
        self.up = self._rotate_vector(self.up, front, angle_rad)
        self.up /= np.linalg.norm(self.up)

        # 4) Mark the view as needing an update
        self.dirty_pose = True


    def _rotate_vector(self, v: np.ndarray, axis: np.ndarray, θ: float) -> np.ndarray:
        """
        Rodrigues' rotation formula:
        v_rot = v*cosθ + (axis×v)*sinθ + axis*(axis·v)*(1−cosθ)
        """
        return (
            v * math.cos(θ) +
            np.cross(axis, v) * math.sin(θ) +
            axis * np.dot(axis, v) * (1 - math.cos(θ))
        )

    def _get_front_vector(self) -> np.ndarray:
        rot = R.from_quat(self.rotation)
        return rot.apply(np.array([0.0, 0.0, -1.0], dtype=np.float32))

    def get_view_matrix_glm(self) -> np.ndarray:
        pos = glm.vec3(*self.position)
        front = glm.vec3(*self._get_front_vector())
        up_vec = glm.vec3(*self.up)
        return np.array(glm.lookAt(pos, pos + front, up_vec), dtype=np.float32)

    def get_project_matrix(self) -> np.ndarray:
        proj = glm.perspective(self.fovy, self.w / self.h, self.znear, self.zfar)
        return np.array(proj, dtype=np.float32)

    def get_intrinsics_matrix(self) -> np.ndarray:
        f = self.w / (2 * math.tan(self.fovy/2))
        return np.array([
            [f, 0, self.w/2],
            [0, f, self.h/2],
            [0, 0, 1]
        ], dtype=np.float32)

    def get_view_matrix(self) -> np.ndarray:
        front = self._get_front_vector()
        front /= np.linalg.norm(front)
        up = self.up / np.linalg.norm(self.up)
        side = np.cross(front, up)
        side /= np.linalg.norm(side)
        up_cam = np.cross(side, front)

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = side
        view[1, :3] = up_cam
        view[2, :3] = -front
        view[:3, 3] = -view[:3, :3] @ self.position
        return view

    def get_htanfovxy_focal(self) -> list:
        htany = np.tan(self.fovy / 2)
        htanx = htany * (self.w / self.h)
        focal = self.w / (2 * htanx) if htanx != 0 else 0
        return [htanx, htany, focal]
