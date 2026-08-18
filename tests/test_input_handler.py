import unittest

from imgui_bundle import imgui

from rosplat.input.input_handler import CAMERA_MOVE_DIRECTIONS


class InputHandlerTest(unittest.TestCase):
    def test_wasd_uses_local_camera_directions(self):
        self.assertEqual(CAMERA_MOVE_DIRECTIONS[imgui.Key.w], (0, 1, 0))
        self.assertEqual(CAMERA_MOVE_DIRECTIONS[imgui.Key.s], (0, -1, 0))
        self.assertEqual(CAMERA_MOVE_DIRECTIONS[imgui.Key.a], (-1, 0, 0))
        self.assertEqual(CAMERA_MOVE_DIRECTIONS[imgui.Key.d], (1, 0, 0))


if __name__ == "__main__":
    unittest.main()
