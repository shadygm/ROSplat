import time
from imgui_bundle import imgui
from rosplat.config.world_settings import RendererType
from rosplat.core import util
import glfw

class InputHandler:
    def __init__(self, window, world_settings):
        """
        Handles free-flight camera movement using ImGui input API.
        Reimplements original behavior using new-style polling.
        """
        self.window = window
        self.world_settings = world_settings
        self.cam = world_settings.world_camera

        self.last_time = time.time()
        glfw.set_window_size_callback(window, self.window_resize_callback)

    def window_resize_callback(self, window, width, height):
        self.cam.w = width
        self.cam.h = height
        self.world_settings.update_window_size(width, height)

    def check_inputs(self):
        # Time delta
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        cam = self.cam
        invert_y = -1 if self.world_settings.get_renderer_type() == RendererType.OPENGL else 1
        direction = invert_y  # used for roll and movement directions

        # Get current mouse state
        x, y = glfw.get_cursor_pos(self.window)
        left_pressed   = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT)   == glfw.PRESS
        middle_pressed = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS

        dragging = left_pressed or middle_pressed

        # Handle mouse drag (rotation or panning)
        if dragging:
            if self.last_mouse_pos is not None:
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]

                # Apply movement based on which button is pressed
                if left_pressed:
                    cam.is_rotating = True
                    cam.process_mouse_delta(dx, invert_y * -dy)
                if middle_pressed:
                    cam.is_panning = True
                    cam.process_mouse_delta(dx, invert_y * -dy)
            self.last_mouse_pos = (x, y)
        else:
            cam.is_rotating = False
            cam.is_panning = False
            self.last_mouse_pos = None

        # Scroll (via ImGui)
        io = imgui.get_io()
        if io.mouse_wheel != 0.0:
            cam.process_scroll(0.0, io.mouse_wheel)

        # 6) Keyboard movement
        speed = cam.trans_sensitivity * dt
        if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            speed *= 3.0

        moves = {
            glfw.KEY_W: (0, -direction * speed, 0),
            glfw.KEY_S: (0,  direction * speed, 0),
            glfw.KEY_A: ( direction * speed, 0, 0),
            glfw.KEY_D: (-direction * speed, 0, 0),
            glfw.KEY_SPACE:       (0, 0, -speed),
            glfw.KEY_LEFT_CONTROL:(0, 0, speed),
        }

        for key, (dx, dy, dz) in moves.items():
            if glfw.get_key(self.window, key) == glfw.PRESS:
                cam.process_translation(dx, dy, dz)

        # Camera roll
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            cam.process_roll(-direction)
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            cam.process_roll(direction)