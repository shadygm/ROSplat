from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
from imgui_bundle import imgui, implot
import glfw
import OpenGL.GL as gl
import subprocess
import rclpy

from rosplat.input import InputHandler
from rosplat.core import util
from rosplat.gui import main_ui
from rosplat.config import WorldSettings

ENABLE_EXPERIMENTS=False

class App:
    def __init__(self):
        self.world_settings = WorldSettings()
        self.world_camera = self.world_settings.world_camera
        self.window = None
        self.glfw_renderer = None

    def init_glfw(self):
        if not glfw.init():
            log.error("Could not initialize GLFW")
            exit(1)

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        window_name = "ROSplat"
        self.window = glfw.create_window(
            self.world_camera.w, self.world_camera.h, window_name, None, None
        )

        if not self.window:
            log.error("Could not initialize Window")
            glfw.terminate()
            exit(1)

        glfw.make_context_current(self.window)

    def init_imgui(self):
        imgui.create_context()
        implot.create_context()

        # Enable docking
        io = imgui.get_io()

        io.config_flags |= (imgui.ConfigFlags_.nav_enable_keyboard.value)
        io.config_flags |= imgui.ConfigFlags_.docking_enable
        io.config_flags |= imgui.ConfigFlags_.viewports_enable
        self.io = io
        # Style
        imgui.style_colors_dark()

        style = imgui.get_style()
        if io.config_flags & imgui.ConfigFlags_.viewports_enable.value:
            style.window_rounding = 0.0
            window_bg_color = style.color_(imgui.Col_.window_bg.value)
            window_bg_color.w = 1.0
            style.set_color_(imgui.Col_.window_bg.value, window_bg_color)

        import ctypes
        window_address = ctypes.cast(self.window, ctypes.c_void_p).value
        assert window_address is not None
        imgui.backends.glfw_init_for_opengl(window_address, True)

        imgui.backends.opengl3_init("#version 430 core")

    def update_camera_pose_lazy(self):
        if self.world_camera.dirty_pose:
            self.world_settings.update_camera_pose()
            self.world_camera.dirty_pose = False

    def update_camera_intrin_lazy(self):
        if self.world_settings.world_camera.dirty_intrinsic:
            self.world_settings.update_camera_intrin()
            self.world_settings.world_camera.dirty_intrinsic = False

    @staticmethod
    def get_memory_usage_mb():
        """Returns resident memory (RAM) usage in MB"""
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # KB to MB
        except Exception as e:
            print("Warning: could not read RAM usage:", e)
        return -1

    @staticmethod
    def get_vram_usage_mb():
        """Returns used GPU memory in MB using nvidia-smi"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=used_memory",
                    "--format=csv,noheader,nounits"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            lines = result.stdout.strip().splitlines()
            usage = sum(int(line.strip()) for line in lines if line.strip().isdigit())
            return usage  # in MB
        except Exception as e:
            print("Warning: could not read VRAM usage:", e)
        return -1

    def process_frames(self):
        self.update_camera_pose_lazy()
        self.update_camera_intrin_lazy()

        if self.world_settings.auto_sort:
            self.world_settings.gauss_renderer.sort_and_update()

        if self.world_settings.have_new_gaussians:
            self.world_settings.update_activated_render_state()

            if ENABLE_EXPERIMENTS: # for testing purposes
                with open("memusage.csv", "a") as f:
                    f.write(f"{len(self.world_settings.gaussian_set)} {self.get_memory_usage_mb()}\n")
                with open("vramusage.csv", "a") as f:
                    f.write(f"{len(self.world_settings.gaussian_set)} {self.get_vram_usage_mb()}\n")


    def game_loop(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            imgui.backends.opengl3_new_frame()
            imgui.backends.glfw_new_frame()
            imgui.new_frame()

            imgui.dock_space_over_viewport(
                dockspace_id=0, viewport=imgui.get_main_viewport()
            )

            display_w, display_h = glfw.get_framebuffer_size(self.window)
            gl.glViewport(0,0,display_w, display_h)
            gl.glClearColor(0, 0, 0, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            main_ui(this_world_settings=self.world_settings)
            self.process_frames()

            imgui.render()


            imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

            if self.io.config_flags & imgui.ConfigFlags_.viewports_enable.value > 0:
                backup_current_context = glfw.get_current_context()
                imgui.update_platform_windows()
                imgui.render_platform_windows_default()
                glfw.make_context_current(backup_current_context)

            glfw.swap_buffers(self.window)

        glfw.terminate()

    def shutdown(self):
        glfw.terminate()

    def run(self):
        self.init_glfw()
        self.init_imgui()
        input_handler = InputHandler(self.window, self.world_settings)
        self.world_settings.input_handler = input_handler
        self.world_settings.create_gaussian_renderer()
        self.world_settings.update_activated_render_state()
        self.game_loop()


def main():
    app = App()
    if not rclpy.ok():
        rclpy.init()
    try:
        app.run()
    finally:
        app.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    global log
    log = util.logger
    main()