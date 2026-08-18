import os

os.environ.setdefault("WGPU_BACKEND_TYPE", "Vulkan")

from imgui_bundle import imgui
import rclpy
import wgpu
from rendercanvas.auto import RenderCanvas, loop

from rosplat.config import WorldSettings
from rosplat.gui import main_ui, shutdown_ros
from rosplat.gui.vulkan_texture import VulkanTextureRegistry
from rosplat.gui.wgpu_backend import CANVAS_OPTIONS, RosplatImguiRenderer
from rosplat.input import InputHandler


class App:
    """ROSplat's Dear ImGui UI running through wgpu-py's Vulkan backend."""

    def __init__(self) -> None:
        self.world_settings = WorldSettings()
        self.world_camera = self.world_settings.world_camera
        self.canvas = None
        self.imgui_renderer = None
        self.texture_registry = None
        self._shutdown_done = False

    def post_init(self) -> None:
        self.world_settings.input_handler = InputHandler(self.world_settings)
        self.world_settings.create_gaussian_renderer(self.texture_registry)

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
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self.world_settings.shutdown()
        if self.texture_registry is not None:
            self.texture_registry.shutdown()
        if self.imgui_renderer is not None:
            self.imgui_renderer.shutdown()
        shutdown_ros()
        if rclpy.ok():
            rclpy.shutdown()

    def run(self) -> None:
        self.canvas = RenderCanvas(**CANVAS_OPTIONS)
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("No WebGPU adapter was available")
        if adapter.info["backend_type"].lower() != "vulkan":
            raise RuntimeError(
                "ROSplat requires wgpu's Vulkan backend, got "
                f"{adapter.info['backend_type']}"
            )
        device = adapter.request_device_sync()
        self.imgui_renderer = RosplatImguiRenderer(device, self.canvas)
        self.texture_registry = VulkanTextureRegistry(
            device, self.imgui_renderer.backend
        )

        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.docking_enable
        io.set_ini_filename("imgui.ini")
        self.post_init()
        self.imgui_renderer.set_gui(self.show_gui)
        self.canvas.request_draw(self.imgui_renderer.render)
        try:
            loop.run()
        finally:
            self.shutdown()


def main() -> None:
    if not rclpy.ok():
        rclpy.init()
    App().run()


if __name__ == "__main__":
    main()
