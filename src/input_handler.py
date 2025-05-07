# input_handler.py
from imgui_bundle import glfw_utils, imgui
import time
import util

class InputHandler:
    def __init__(self, window, world_settings):
        """
        Initialize the InputHandler with a window and world settings.
        """
        self.window = window
        self.log = util.logger
        self.world_settings = world_settings
        self.last_time = time.time()
        self.init_callbacks()

    def init_callbacks(self):
        """
        Set up GLFW callbacks.
        """
        glfw_utils.glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw_utils.glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw_utils.glfw.set_window_size_callback(self.window, self.window_resize_callback)
        glfw_utils.glfw.set_scroll_callback(self.window, self.scroll_callback)
        self.log.info("Initializing callbacks")
    
    def mouse_button_callback(self, window, button, action, mods):
        io = imgui.get_io()
        pressed = action == glfw_utils.glfw.PRESS
        io.add_mouse_button_event(button, pressed)
        if not io.want_capture_mouse:
            cam = self.world_settings.world_camera
            if button == glfw_utils.glfw.MOUSE_BUTTON_LEFT:
                cam.is_leftmouse_pressed = pressed
            elif button == glfw_utils.glfw.MOUSE_BUTTON_RIGHT:
                cam.is_rightmouse_pressed = pressed

    def cursor_pos_callback(self, window, xpos, ypos):
        io = imgui.get_io()
        io.add_mouse_pos_event(xpos, ypos)
        if not io.want_capture_mouse:
            self.world_settings.world_camera.process_mouse(xpos, ypos)

    def scroll_callback(self, window, xoffset, yoffset):
        io = imgui.get_io()
        io.add_mouse_wheel_event(xoffset, yoffset)
        if not io.want_capture_mouse:
            self.world_settings.world_camera.process_scroll(xoffset, yoffset)

    def window_resize_callback(self, window, width, height):
        cam = self.world_settings.world_camera
        cam.w = width
        cam.h = height
        self.world_settings.update_window_size(width, height)

    def check_inputs(self):
        """
        Check and process inputs each frame.
        """
        curr_time = time.time()
        delta = curr_time - self.last_time
        self.last_time = curr_time

        self._process_keyboard_input(delta)

    def _process_keyboard_input(self, delta):
        io = imgui.get_io()
        if io.want_capture_keyboard:
            return

        self._process_view_translations(delta)
        self._process_view_rotations(delta)
        self._process_model_translations(delta)

    def _process_view_translations(self, delta):
        """
        WASD + space/shift for moving the camera.
        """
        keys = {
            glfw_utils.glfw.KEY_W: (0,  delta, 0),
            glfw_utils.glfw.KEY_S: (0, -delta, 0),
            glfw_utils.glfw.KEY_A: (-delta, 0,  0),
            glfw_utils.glfw.KEY_D: ( delta, 0,  0),
            glfw_utils.glfw.KEY_SPACE:      (0, 0,  delta),
            glfw_utils.glfw.KEY_LEFT_SHIFT: (0, 0, -delta),
        }
        for key, (dx, dy, dz) in keys.items():
            if glfw_utils.glfw.get_key(self.window, key) == glfw_utils.glfw.PRESS:
                self.world_settings.process_translation(dx, dy, dz)

    def _process_view_rotations(self, delta):
        """
        Q/E to roll the camera ccw/cw around its view axis.
        """
        win = self.window
        cam = self.world_settings.world_camera
        if glfw_utils.glfw.get_key(win, glfw_utils.glfw.KEY_Q) == glfw_utils.glfw.PRESS:
            cam.process_roll(+1)
        if glfw_utils.glfw.get_key(win, glfw_utils.glfw.KEY_E) == glfw_utils.glfw.PRESS:
            cam.process_roll(-1)

    def _process_model_translations(self, delta):
        """
        Stub for model movement (IJKL).
        """
        pass
