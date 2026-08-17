import time
from imgui_bundle import imgui
import glfw


CAMERA_MOVE_DIRECTIONS = {
    glfw.KEY_W: (0, 1, 0),
    glfw.KEY_S: (0, -1, 0),
    glfw.KEY_A: (-1, 0, 0),
    glfw.KEY_D: (1, 0, 0),
    glfw.KEY_SPACE: (0, 0, 1),
    glfw.KEY_LEFT_CONTROL: (0, 0, -1),
}


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
        self.last_mouse_pos = None

    def window_resize_callback(self, window, width, height):
        self.world_settings.update_window_size(width, height)

    def check_inputs(self):
        # Time delta
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        cam = self.cam
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
                    cam.process_mouse_delta(dx, -dy)
                if middle_pressed:
                    cam.is_panning = True
                    cam.process_mouse_delta(dx, -dy)
            self.last_mouse_pos = (x, y)
        else:
            cam.is_rotating = False
            cam.is_panning = False
            self.last_mouse_pos = None

        # Scroll (via ImGui)
        io = imgui.get_io()
        if io.mouse_wheel != 0.0:
            cam.process_scroll(0.0, io.mouse_wheel)

        if io.want_text_input:
            return

        # 6) Keyboard movement
        speed = cam.trans_sensitivity * dt
        if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            speed *= 3.0

        # Camera translation is expressed in local coordinates and is
        # renderer-independent: +X right, +Y forward, +Z up.
        for key, (dx, dy, dz) in CAMERA_MOVE_DIRECTIONS.items():
            if glfw.get_key(self.window, key) == glfw.PRESS:
                cam.process_translation(dx * speed, dy * speed, dz * speed)

        # Camera roll
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            cam.process_roll(-1)
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            cam.process_roll(1)
