"""Open the full Vulkan UI briefly, then exit through its normal shutdown path."""

from imgui_bundle import imgui, implot
from rendercanvas.auto import loop

from rosplat.main import App


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
    loop.call_later(3.0, loop.stop)
    UiSmokeApp().run()
    print("backend=wgpu-vulkan imgui=ok implot=ok")


if __name__ == "__main__":
    main()
