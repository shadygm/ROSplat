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
        """
        Handle mouse button press events.
        """
        io = imgui.get_io()
        pressed = action == glfw_utils.glfw.PRESS
        io.add_mouse_button_event(button, pressed)
        
        # Process application logic only if ImGui does not capture the mouse.
        if not io.want_capture_mouse:
            if button == glfw_utils.glfw.MOUSE_BUTTON_LEFT:
                self.world_settings.world_camera.is_leftmouse_pressed = pressed
            elif button == glfw_utils.glfw.MOUSE_BUTTON_RIGHT:
                self.world_settings.world_camera.is_rightmouse_pressed = pressed

    def cursor_pos_callback(self, window, xpos, ypos):
        """
        Handle mouse move events.
        """
        io = imgui.get_io()
        io.add_mouse_pos_event(xpos, ypos)
        
        if not io.want_capture_mouse:
            self.world_settings.world_camera.process_mouse(xpos, ypos)

    def scroll_callback(self, window, xoffset, yoffset):
        """
        Handle scroll events.
        """
        io = imgui.get_io()
        io.add_mouse_wheel_event(xoffset, yoffset)
        
        if not io.want_capture_mouse:
            self.world_settings.world_camera.process_scroll(xoffset, yoffset)

    def window_resize_callback(self, window, width, height):
        """
        Handle window resize events.
        """
        self.world_settings.world_camera.w = width
        self.world_settings.world_camera.h = height
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
        """
        Process all keyboard inputs, splitting between view and model transformations.
        """
        io = imgui.get_io()
        if io.want_capture_keyboard:
            return

        self._process_view_translations(delta)
        self._process_model_translations(delta)

    def _process_view_translations(self, delta):
        """
        Process camera (world) translations.
        """
        view_translation_keys = {
            glfw_utils.glfw.KEY_W: (0, delta, 0),         # Forward
            glfw_utils.glfw.KEY_A: (-delta, 0, 0),         # Left
            glfw_utils.glfw.KEY_S: (0, -delta, 0),         # Backward
            glfw_utils.glfw.KEY_D: (delta, 0, 0),          # Right
            glfw_utils.glfw.KEY_SPACE: (0, 0, delta),       # Up
            glfw_utils.glfw.KEY_LEFT_SHIFT: (0, 0, -delta)  # Down
        }

        for key, translation in view_translation_keys.items():
            if glfw_utils.glfw.get_key(self.window, key) == glfw_utils.glfw.PRESS:
                self.world_settings.process_translation(*translation)

    def _process_model_translations(self, delta):
        """
        Process model translations.
        """
        # TODO: Re-enable when model translation is implemented
        
        # model_translation_keys = {
        #     glfw_utils.glfw.KEY_I: (0, -delta),  # Model forward
        #     glfw_utils.glfw.KEY_J: (delta, 0),   # Model left
        #     glfw_utils.glfw.KEY_K: (0, delta),   # Model backward
        #     glfw_utils.glfw.KEY_L: (-delta, 0)   # Model right
        # }

        # for key, translation in model_translation_keys.items():
        #     if glfw_utils.glfw.get_key(self.window, key) == glfw_utils.glfw.PRESS:
        #         self.world_settings.process_model_translation(*translation)
        pass