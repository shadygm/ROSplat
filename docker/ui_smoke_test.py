"""Open the full Vulkan UI briefly, then exit through its normal shutdown path."""

from rendercanvas.auto import loop

from rosplat.main import App


def main() -> None:
    loop.call_later(3.0, loop.stop)
    App().run()
    print("backend=wgpu-vulkan ui=ok")


if __name__ == "__main__":
    main()
