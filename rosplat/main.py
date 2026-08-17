from imgui_bundle import hello_imgui, imgui, immapp
import rclpy

from rosplat.config import WorldSettings
from rosplat.gui import main_ui, shutdown_ros
from rosplat.input import InputHandler


class App:
    """ROSplat's Python UI running on HelloImGui's Vulkan backend."""

    def __init__(self) -> None:
        self.world_settings = WorldSettings()
        self.world_camera = self.world_settings.world_camera

    def post_init(self) -> None:
        self.world_settings.input_handler = InputHandler(self.world_settings)
        self.world_settings.create_gaussian_renderer()

    def update_camera_lazy(self) -> None:
        if self.world_camera.dirty_pose:
            self.world_settings.update_camera_pose()
            self.world_camera.dirty_pose = False
        if self.world_camera.dirty_intrinsic:
            self.world_settings.update_camera_intrin()
            self.world_camera.dirty_intrinsic = False

    def show_gui(self) -> None:
        self.update_camera_lazy()
        if self.world_settings.have_new_gaussians:
            self.world_settings.update_activated_render_state()
        imgui.dock_space_over_viewport(
            dockspace_id=0,
            viewport=imgui.get_main_viewport(),
        )
        main_ui(this_world_settings=self.world_settings)

    def shutdown(self) -> None:
        self.world_settings.shutdown()
        shutdown_ros()
        if rclpy.ok():
            rclpy.shutdown()

    def run(self) -> None:
        params = hello_imgui.RunnerParams()
        params.app_window_params.window_title = "ROSplat"
        params.app_window_params.window_geometry.size = (1280, 720)
        params.app_window_params.restore_previous_geometry = True
        params.platform_backend_type = hello_imgui.PlatformBackendType.glfw
        params.renderer_backend_type = hello_imgui.RendererBackendType.vulkan
        params.imgui_window_params.default_imgui_window_type = (
            hello_imgui.DefaultImGuiWindowType.no_default_window
        )
        params.imgui_window_params.enable_viewports = False
        params.ini_filename = "imgui.ini"
        params.ini_filename_use_app_window_title = False
        params.fps_idling.enable_idling = False
        params.callbacks.post_init = self.post_init
        params.callbacks.show_gui = self.show_gui
        params.callbacks.before_exit = self.shutdown
        immapp.run(params)


def main() -> None:
    if not rclpy.ok():
        rclpy.init()
    App().run()


if __name__ == "__main__":
    main()
