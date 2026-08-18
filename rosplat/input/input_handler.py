import time
from imgui_bundle import imgui


CAMERA_MOVE_DIRECTIONS = {
    imgui.Key.w: (0, 1, 0),
    imgui.Key.s: (0, -1, 0),
    imgui.Key.a: (-1, 0, 0),
    imgui.Key.d: (1, 0, 0),
    imgui.Key.space: (0, 0, 1),
    imgui.Key.left_ctrl: (0, 0, -1),
}


class InputHandler:
    def __init__(self, world_settings):
        """
        Handles free-flight camera movement using ImGui input API.
        Reimplements original behavior using new-style polling.
        """
        self.world_settings = world_settings
        self.cam = world_settings.world_camera

        self.last_time = time.time()
        self.last_mouse_pos = None

    def check_inputs(self):
        # Time delta
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        cam = self.cam
        # Get current mouse state
        mouse = imgui.get_io().mouse_pos
        x, y = mouse.x, mouse.y
        left_pressed = imgui.is_mouse_down(imgui.MouseButton_.left)
        middle_pressed = imgui.is_mouse_down(imgui.MouseButton_.middle)

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
        if imgui.is_key_down(imgui.Key.left_shift):
            speed *= 3.0

        # Camera translation is expressed in local coordinates and is
        # renderer-independent: +X right, +Y forward, +Z up.
        for key, (dx, dy, dz) in CAMERA_MOVE_DIRECTIONS.items():
            if imgui.is_key_down(key):
                cam.process_translation(dx * speed, dy * speed, dz * speed)

        # Camera roll
        if imgui.is_key_down(imgui.Key.q):
            cam.process_roll(-1)
        if imgui.is_key_down(imgui.Key.e):
            cam.process_roll(1)
