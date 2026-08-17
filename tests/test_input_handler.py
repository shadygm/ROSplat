import unittest

import glfw

from rosplat.input.input_handler import CAMERA_MOVE_DIRECTIONS


class InputHandlerTest(unittest.TestCase):
    def test_wasd_uses_local_camera_directions(self):
        self.assertEqual(CAMERA_MOVE_DIRECTIONS[glfw.KEY_W], (0, 1, 0))
        self.assertEqual(CAMERA_MOVE_DIRECTIONS[glfw.KEY_S], (0, -1, 0))
        self.assertEqual(CAMERA_MOVE_DIRECTIONS[glfw.KEY_A], (-1, 0, 0))
        self.assertEqual(CAMERA_MOVE_DIRECTIONS[glfw.KEY_D], (1, 0, 0))


if __name__ == "__main__":
    unittest.main()
