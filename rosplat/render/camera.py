# camera.py
import math
import numpy as np
from rosplat.core import util
import glm

class Camera:
    """
    Free‐flight camera with full basis‐vector orientation:
      • front, up, right kept orthonormal
      • yaw/pitch/roll all performed via axis‐angle
      • auxiliary intrinsics + legacy getters preserved
    """

    def __init__(self, height: int, width: int) -> None:
        # viewport
        self.h = height
        self.w = width

        # clipping & FOV
        self.znear = 0.01
        self.zfar  = 100.0
        self.fovy  = np.pi / 2

        # position & orientation vectors
        self.position = np.array([0.0, 0.0, 3.0], dtype=np.float32)
        # look‐direction
        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        # inverted up‐vector
        self.up    = np.array([0.0, -1.0, 0.0], dtype=np.float32)

        # derive right from front × up
        self._reorthonormalize()

        # mouse‐look state
        self.first_mouse = True
        self.last_x = width / 2
        self.last_y = height / 2
        self.is_rotating = False
        self.is_panning  = False

        # “dirty” flags
        self.dirty_pose      = True
        self.dirty_intrinsic = True

        # sensitivities
        self.rot_sensitivity   = 0.002   # radians per pixel
        self.trans_sensitivity = 2.5     # units per second
        self.zoom_sensitivity  = 0.1
        self.roll_sensitivity  = 0.03    # radians per keypress

        self.log = util.logger

    def _reorthonormalize(self):
        """Ensure front, up, right remain an orthonormal basis."""
        self.front /= np.linalg.norm(self.front)
        self.up    -= self.front * np.dot(self.front, self.up)
        self.up    /= np.linalg.norm(self.up)
        self.right  = np.cross(self.front, self.up)
        self.right /= np.linalg.norm(self.right)

    def _rotate_vec(self, v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rodrigues’ rotation of v around axis by angle."""
        axis = axis / np.linalg.norm(axis)
        cosA = math.cos(angle)
        sinA = math.sin(angle)
        return (
            v * cosA
            + np.cross(axis, v) * sinA
            + axis * np.dot(axis, v) * (1 - cosA)
        )

    #
    # === Controls: translation, scroll, mouse‐drag ===
    #

    def process_translation(self, dx: float, dy: float, dz: float) -> None:
        """
        Move in local camera space:
          • forward/back = dy
          • right/left   = dx
          • up/down      = dz
        """
        move = self.front*dy + self.right*dx + self.up*dz
        self.position += move
        self.dirty_pose = True

    def process_scroll(self, xoffset: float, yoffset: float) -> None:
        """Zoom along the current look‐direction."""
        self.position += self.front * yoffset * self.zoom_sensitivity
        self.dirty_pose = True

    def process_mouse(self, xpos: float, ypos: float) -> None:
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False

        dx = xpos - self.last_x
        dy = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos

        if self.is_rotating:
            self._rotate_camera(dx, dy)
        elif self.is_panning:
            self._pan_camera(dx, dy)

    def _pan_camera(self, dx: float, dy: float) -> None:
        """
        Slide camera parallel to view plane:
          • horizontal (dx) along right
          • vertical   (dy) along up
        """
        self.position += (self.right*dx + self.up*dy) * self.trans_sensitivity * 0.01
        self.dirty_pose = True

    #
    # === Axis‐angle rotations ===
    #

    def _rotate_camera(self, dx: float, dy: float) -> None:
        """
        Yaw (around up) then pitch (around right), both respecting current roll.
        """
        # yaw: negative dx → turn right
        ang_yaw   = -dx * self.rot_sensitivity
        self.front = self._rotate_vec(self.front, self.up, ang_yaw)

        # pitch: positive dy → look up
        self._reorthonormalize()
        ang_pitch = +dy * self.rot_sensitivity
        self.front = self._rotate_vec(self.front, self.right, ang_pitch)
        self.up    = self._rotate_vec(self.up,    self.right, ang_pitch)

        self._reorthonormalize()
        self.dirty_pose = True

    def process_roll(self, direction: float) -> None:
        """
        Roll around the view axis:
          • +direction → CCW from camera POV
          • −direction → CW
        """
        angle = direction * self.roll_sensitivity
        self.up    = self._rotate_vec(self.up,    self.front, angle)
        self.right = self._rotate_vec(self.right, self.front, angle)
        self._reorthonormalize()
        self.dirty_pose = True

    #
    # === Matrices & auxiliary getters ===
    #

    def get_view_matrix(self) -> np.ndarray:
        """Look‐at using position + front/up."""
        pos   = glm.vec3(*self.position)
        tgt   = glm.vec3(*(self.position + self.front))
        upv   = glm.vec3(*self.up)
        return np.array(glm.lookAt(pos, tgt, upv), dtype=np.float32)

    def get_view_matrix_glm(self) -> np.ndarray:
        """Alias for backward compatibility."""
        return self.get_view_matrix()

    def get_project_matrix(self) -> np.ndarray:
        """Perspective projection."""
        return np.array(
            glm.perspective(self.fovy, self.w/self.h, self.znear, self.zfar),
            dtype=np.float32
        )

    def get_pose(self) -> np.ndarray:
        """Legacy: return camera position."""
        return self.position.copy()

    def get_intrinsics_matrix(self) -> np.ndarray:
        """3×3 pinhole intrinsics."""
        f = self.w / (2 * math.tan(self.fovy/2))
        return np.array([
            [f, 0, self.w/2],
            [0, f, self.h/2],
            [0, 0, 1]
        ], dtype=np.float32)

    def get_htanfovxy_focal(self) -> list:
        """
        [tan(fov_x/2), tan(fov_y/2), focal_length].
        """
        htany = math.tan(self.fovy/2)
        htanx = htany * (self.w / self.h)
        focal = self.w / (2 * htanx) if htanx != 0 else 0
        return [htanx, htany, focal]
