from types import SimpleNamespace
from unittest.mock import Mock, sentinel

from imgui_bundle import imgui, implot
from wgpu.utils.imgui import ImguiRenderer

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


def test_renderer_owns_implot_context(monkeypatch):
    def fake_init(self, device, canvas, render_target_format=None):
        self._imgui_context = sentinel.imgui_context

    create_context = Mock(return_value=sentinel.implot_context)
    monkeypatch.setattr(ImguiRenderer, "__init__", fake_init)
    monkeypatch.setattr(implot, "create_context", create_context)

    renderer = RosplatImguiRenderer(sentinel.device, sentinel.canvas)

    create_context.assert_called_once_with()
    assert renderer._implot_context is sentinel.implot_context


def test_renderer_selects_and_destroys_implot_context(monkeypatch):
    renderer = object.__new__(RosplatImguiRenderer)
    renderer._imgui_context = sentinel.imgui_context
    renderer._implot_context = sentinel.implot_context
    set_imgui_context = Mock()
    set_implot_context = Mock()
    destroy_context = Mock()
    parent_render = Mock(return_value=sentinel.draw_result)
    monkeypatch.setattr(imgui, "set_current_context", set_imgui_context)
    monkeypatch.setattr(implot, "set_current_context", set_implot_context)
    monkeypatch.setattr(implot, "destroy_context", destroy_context)
    monkeypatch.setattr(ImguiRenderer, "render", parent_render)

    assert renderer.render() is sentinel.draw_result
    renderer.shutdown()
    renderer.shutdown()

    set_imgui_context.assert_called_with(sentinel.imgui_context)
    set_implot_context.assert_called_once_with(sentinel.implot_context)
    parent_render.assert_called_once_with()
    destroy_context.assert_called_once_with(sentinel.implot_context)
    assert renderer._implot_context is None
