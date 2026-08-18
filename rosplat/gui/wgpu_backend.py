from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiRenderer


CANVAS_OPTIONS = {
    "title": "ROSplat",
    "size": (1280, 720),
    "update_mode": "fastest",
    "vsync": False,
}


class RosplatImguiRenderer(ImguiRenderer):
    """ImguiRenderer with the missing rendercanvas Space mapping restored."""

    KEY_MAP = {
        **ImguiRenderer.KEY_MAP,
        # rendercanvas's GLFW backend emits a literal space. Keep the named
        # spelling too so this remains correct with other canvas backends.
        " ": imgui.Key.space,
        "Space": imgui.Key.space,
    }
