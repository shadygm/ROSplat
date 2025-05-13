# camera.py
import math
import numpy as np
import glm
import util

class Camera:
    """
    A camera class to manage viewing transformations for a 3D scene.
    """

    def __init__(self, height: int, width: int) -> None:
        self.h = height
        self.w = width
        self.znear = 0.01
        self.zfar = 100.0
        self.fovy = np.pi / 2

        # Position & orientation
        self.position = np.array([0.0, 0.0, 3.0], dtype=np.float32)
        self.target   = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.up       = np.array([0.0, -1.0, 0.0], dtype=np.float32)

        # Euler angles
        self.yaw   = -np.pi / 2
        self.pitch = 0.0

        # Mouse state
        self.first_mouse = True
        self.last_x = width  / 2
        self.last_y = height / 2
        self.is_leftmouse_pressed  = False
        self.is_rightmouse_pressed = False

        # “Dirty” flags
        self.dirty_pose      = True
        self.dirty_intrinsic = True

        # Sensitivities
        self.rot_sensitivity  = 0.005
        self.trans_sensitivity = 0.01
        self.zoom_sensitivity  = 0.08
        self.roll_sensitivity  = 0.03

        self.log = util.logger

    def process_translation(self, dx: float, dy: float, dz: float) -> None:
        front = self.target - self.position
        norm = np.linalg.norm(front)
        front = front / norm if norm>0 else np.array([0,0,1],dtype=np.float32)
        right = np.cross(front, self.up)
        move = front*dy + right*dx + self.up*dz
        norm = np.linalg.norm(move)
        if norm>0: move /= norm
        self.position += move * self.trans_sensitivity
        self.target   += move * self.trans_sensitivity
        self.dirty_pose = True

    def process_scroll(self, xoffset: float, yoffset: float) -> None:
        front = self.target - self.position
        front /= np.linalg.norm(front)
        delta = front * yoffset * self.zoom_sensitivity
        self.position += delta
        self.target   += delta
        self.dirty_pose = True

    def get_pose(self) -> np.ndarray:
        return self.position

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
        self.yaw   -= dx * self.rot_sensitivity
        self.pitch += dy * self.rot_sensitivity
        eps = 1e-3
        self.pitch = np.clip(self.pitch, -np.pi/2+eps, np.pi/2-eps)

        front = np.array([
            math.cos(self.pitch)*math.cos(self.yaw),
            math.sin(self.pitch),
            math.cos(self.pitch)*math.sin(self.yaw)
        ], dtype=np.float32)
        front /= np.linalg.norm(front)
        self.target = self.position + front
        self.dirty_pose = True

    def _pan_camera(self, dx: float, dy: float) -> None:
        front = self.target - self.position
        front /= np.linalg.norm(front)
        right = np.cross(self.up, front)
        hor = right * dx * self.trans_sensitivity
        vert = np.cross(right, front) * dy * self.trans_sensitivity
        self.position += hor + vert
        self.target   += hor + vert
        self.dirty_pose = True

    def process_roll(self, direction: float) -> None:
        """
        Rotate 'up' around the view axis by ±1 * roll_sensitivity.
        Positive direction -> CCW when looking from camera toward scene.
        """
        angle = direction * self.roll_sensitivity
        front = self.target - self.position
        front /= np.linalg.norm(front)
        u = self.up
        k = front
        cosA = math.cos(angle)
        sinA = math.sin(angle)
        # Rodrigues' rotation of u around axis k
        u_new = u*cosA + np.cross(k, u)*sinA + k*(np.dot(k, u))*(1-cosA)
        self.up = u_new / np.linalg.norm(u_new)
        self.dirty_pose = True

    def get_view_matrix_glm(self) -> np.ndarray:
        """
        Compute and return the view matrix using glm.
        """
        pos = glm.vec3(*self.position)
        tgt = glm.vec3(*self.target)
        up_vec = glm.vec3(*self.up)
        return np.array(glm.lookAt(pos, tgt, up_vec))

    def get_project_matrix(self) -> np.ndarray:
        proj = glm.perspective(self.fovy, self.w/self.h, self.znear, self.zfar)
        return np.array(proj, dtype=np.float32)

    def get_intrinsics_matrix(self) -> np.ndarray:
        f = self.w / (2 * math.tan(self.fovy/2))
        return np.array([
            [f, 0, self.w/2],
            [0, f, self.h/2],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def get_view_matrix(self) -> np.ndarray:
        """Returns the 4x4 view matrix using look-at."""
        front = self.target - self.position
        front = front / np.linalg.norm(front)
        up = self.up / np.linalg.norm(self.up)
        side = np.cross(front, up)
        side = side / np.linalg.norm(side)
        up = np.cross(side, front)

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = side
        view[1, :3] = up
        view[2, :3] = -front
        view[:3, 3] = -view[:3, :3] @ self.position
        return view

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


