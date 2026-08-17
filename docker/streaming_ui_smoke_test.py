#!/usr/bin/env python3
"""Verify that a real ROS GaussianArray stream reaches the Vulkan UI and GPU."""

from __future__ import annotations

import argparse
import time

from rendercanvas.auto import loop

from rosplat.gui.imgui_manager import ros_node_manager
from rosplat.main import App


class StreamingSmokeApp(App):
    def __init__(self, topic: str, expected_splats: int, timeout: float) -> None:
        super().__init__()
        self.topic = topic
        self.expected_splats = expected_splats
        self.timeout = timeout
        self.started_at = 0.0
        self.completed_seconds = None
        self.failure = None
        self._listener_added = False

    def post_init(self) -> None:
        super().post_init()
        self.started_at = time.perf_counter()
        loop.call_later(0.1, self._poll)

    def _poll(self) -> None:
        elapsed = time.perf_counter() - self.started_at
        if not self._listener_added:
            if ros_node_manager.get_msg_type(self.topic) is not None:
                ros_node_manager.add_listener(self.topic)
                self._listener_added = True
            elif elapsed >= self.timeout:
                self.failure = f"topic {self.topic!r} was not discovered"
                loop.stop()
                return
            else:
                loop.call_later(0.1, self._poll)
                return

        renderer = self.world_settings.gauss_renderer
        cpu_count = self.world_settings.get_num_gaussians()
        gpu_count = renderer.splat_count if renderer is not None else 0
        if cpu_count >= self.expected_splats and gpu_count >= self.expected_splats:
            self.completed_seconds = elapsed
            loop.call_later(0.25, loop.stop)
            return
        if elapsed >= self.timeout:
            self.failure = (
                f"timed out with CPU count {cpu_count:,} and GPU count {gpu_count:,}"
            )
            loop.stop()
            return
        loop.call_later(0.1, self._poll)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/gaussian_test")
    parser.add_argument("--expected-splats", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    app = StreamingSmokeApp(args.topic, args.expected_splats, args.timeout)
    app.run()
    if app.failure:
        raise RuntimeError(app.failure)
    if app.completed_seconds is None:
        raise RuntimeError("streaming smoke test stopped before the scene completed")
    print(
        f"backend=wgpu-vulkan topic={args.topic} splats={args.expected_splats} "
        f"seconds={app.completed_seconds:.3f} streaming_ui=ok"
    )


if __name__ == "__main__":
    main()
