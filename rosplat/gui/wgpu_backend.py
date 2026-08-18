from imgui_bundle import imgui, implot
from wgpu.utils.imgui import ImguiRenderer


CANVAS_OPTIONS = {
    "title": "ROSplat",
    "size": (1280, 720),
    "update_mode": "fastest",
    "vsync": False,
}


class RosplatImguiRenderer(ImguiRenderer):
    """WGPU renderer that owns ROSplat's Dear ImGui and ImPlot contexts."""

    KEY_MAP = {
        **ImguiRenderer.KEY_MAP,
        # rendercanvas's GLFW backend emits a literal space. Keep the named
        # spelling too so this remains correct with other canvas backends.
        " ": imgui.Key.space,
        "Space": imgui.Key.space,
    }

    def __init__(self, device, canvas, render_target_format=None):
        super().__init__(device, canvas, render_target_format)
        # WGPU creates the Dear ImGui context, but ImPlot has a separate
        # context that must be created afterwards.
        self._implot_context = implot.create_context()

    def render(self):
        imgui.set_current_context(self.imgui_context)
        implot.set_current_context(self._implot_context)
        return super().render()

    def shutdown(self) -> None:
        """Destroy ImPlot before its parent Dear ImGui context disappears."""
        if self._implot_context is None:
            return
        imgui.set_current_context(self.imgui_context)
        implot.destroy_context(self._implot_context)
        self._implot_context = None
