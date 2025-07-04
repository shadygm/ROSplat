from imgui_bundle import glfw_utils, imgui
import util
from worldsettings import RendererType
import time

class InputHandler:
    def __init__(self, window, world_settings):
        """
        Initialize the InputHandler with a window and world settings.
        Implements FPS‐style free‐flight controls:
          • Hold Right Mouse + drag to look around
          • Hold Middle Mouse + drag to pan
          • W/A/S/D to move forward/left/back/right
          • Space / Left Ctrl to move up/down
          • Left Shift to boost movement speed
        """
        self.window = window
        self.world_settings = world_settings
        self.cam = world_settings.world_camera

        self.last_time = time.time()
        self.init_callbacks()

    def init_callbacks(self):
        """Set up GLFW callbacks."""
        glfw = glfw_utils.glfw
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_window_size_callback(self.window, self.window_resize_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)

    def mouse_button_callback(self, window, button, action, mods):
        io = imgui.get_io()
        pressed = action == glfw_utils.glfw.PRESS
        io.add_mouse_button_event(button, pressed)

        # Right = look, Middle = pan
        if not io.want_capture_mouse:
            if button == glfw_utils.glfw.MOUSE_BUTTON_LEFT:
                self.cam.is_rotating = pressed
                # reset first_mouse on press so look jump is avoided
                if pressed:
                    self.cam.first_mouse = True
            elif button == glfw_utils.glfw.MOUSE_BUTTON_MIDDLE:
                self.cam.is_panning = pressed

    def cursor_pos_callback(self, window, xpos, ypos):
        io = imgui.get_io()
        invert = 1
        if self.world_settings.get_renderer_type() == RendererType.OPENGL:
            invert *= -1
        io.add_mouse_pos_event(xpos, ypos)
        if not io.want_capture_mouse:
            self.cam.process_mouse(xpos, invert * -ypos)

    def scroll_callback(self, window, xoffset, yoffset):
        io = imgui.get_io()
        io.add_mouse_wheel_event(xoffset, yoffset)
        if not io.want_capture_mouse:
            self.cam.process_scroll(xoffset, yoffset)

    def window_resize_callback(self, window, width, height):
        self.cam.w = width
        self.cam.h = height
        self.world_settings.update_window_size(width, height)

    def check_inputs(self):
        """Call each frame to update camera based on current input state."""
        curr = time.time()
        delta = curr - self.last_time
        self.last_time = curr

        io = imgui.get_io()
        if not io.want_capture_keyboard:
            self._process_keyboard(delta)

    def _process_keyboard(self, dt):
        """W/A/S/D move, Space/Ctrl up & down, Shift to boost speed."""
        glfw = glfw_utils.glfw
        # speed multiplier
        speed = self.cam.trans_sensitivity * dt
        if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            speed *= 3.0
        direction = 1
        if self.world_settings.get_renderer_type() ==  RendererType.OPENGL:
            direction = direction * (-1)

        # mapping: key -> (dx, dy, dz)
        moves = {
            glfw.KEY_W: (0, -direction * speed, 0),
            glfw.KEY_S: (0, direction * speed, 0),
            glfw.KEY_A: (direction * speed, 0, 0),
            glfw.KEY_D: (-direction * speed, 0, 0),
            glfw.KEY_SPACE:      (0, 0, speed),
            glfw.KEY_LEFT_CONTROL:(0, 0, -speed),
        }

        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            # Counter‐clockwise roll when looking forward
            self.cam.process_roll(direction)
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            # Clockwise roll when looking forward
            self.cam.process_roll(-direction)

        for key, (dx, dy, dz) in moves.items():
            if glfw.get_key(self.window, key) == glfw.PRESS:
                self.cam.process_translation(dx, dy, dz)
