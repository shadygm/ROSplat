"""Open the full Vulkan UI briefly, then exit through its normal shutdown path."""

from imgui_bundle import imgui, implot
from rendercanvas.auto import loop

from rosplat.main import App
from rosplat.render.renderer import RenderOutputMode


class UiSmokeApp(App):
    """Exercise ImPlot as well as the default tab shown at startup."""

    def show_gui(self) -> None:
        super().show_gui()
        visible, _ = imgui.begin("ImPlot smoke test")
        try:
            if visible and implot.begin_plot("Context lifecycle"):
                implot.end_plot()
        finally:
            imgui.end()


def main() -> None:
    app = UiSmokeApp()
    loop.call_later(
        0.75,
        app.world_settings.update_render_output,
        RenderOutputMode.DEPTH,
    )
    loop.call_later(1.25, app.world_settings.update_sh_degree, 0)
    loop.call_later(1.5, app.world_settings.update_scale_modifier, 1.25)
    loop.call_later(
        1.75,
        app.world_settings.update_render_output,
        RenderOutputMode.OPACITY,
    )
    loop.call_later(3.0, loop.stop)
    app.run()
    print("backend=wgpu-vulkan imgui=ok implot=ok render-settings=ok")


if __name__ == "__main__":
    main()
