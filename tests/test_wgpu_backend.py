from types import SimpleNamespace
from unittest.mock import Mock

from imgui_bundle import imgui

from rosplat.gui.wgpu_backend import CANVAS_OPTIONS, RosplatImguiRenderer


def test_space_key_is_forwarded_to_imgui():
    renderer = object.__new__(RosplatImguiRenderer)
    io = SimpleNamespace(
        add_key_event=Mock(),
        want_capture_keyboard=False,
    )
    renderer._backend = SimpleNamespace(io=io)
    event = {"event_type": "key_down", "key": " "}

    renderer._on_key(event)

    io.add_key_event.assert_called_once_with(imgui.Key.space, True)
    assert "stop_propagation" not in event


def test_canvas_runs_uncapped_without_vsync():
    assert CANVAS_OPTIONS["update_mode"] == "fastest"
    assert CANVAS_OPTIONS["vsync"] is False
    assert "max_fps" not in CANVAS_OPTIONS
