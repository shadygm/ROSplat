import queue
from datetime import datetime

import numpy as np
from PIL import Image

# ImGui Bundle
from imgui_bundle import (
    imgui,
    immapp,
    implot,
    portable_file_dialogs as pfd,
)
# Local modules
from rosplat.core import util
from rosplat.render.renderer import RenderOutputMode
from rosplat.ros import ROSNodeManager


# === Global State ===
world_settings = None
frame_queue = queue.Queue(maxsize=10)
latest_frame = [None]  # mutable container
imu_queue = queue.Queue(maxsize=1)
ros_node_manager = ROSNodeManager()

def shutdown_ros() -> None:
    """Stop ROS listeners and their executor before rclpy is shut down."""
    ros_node_manager.shutdown()

# IMU history
accel_x, accel_y, accel_z = [], [], []
gyro_x, gyro_y, gyro_z = [], [], []


def take_screenshot(filename: str = "screenshot.png") -> None:
    """Save the most recent Spirula render without reading a GL framebuffer."""
    renderer = world_settings.gauss_renderer
    if renderer is None:
        util.logger.warning("Cannot take a screenshot before the renderer starts")
        return
    Image.fromarray(renderer.latest_rgba, mode="RGBA").save(filename)
    util.logger.info(f"[Screenshot saved] {filename}")


class CircularBuffer:
    def __init__(self, max_size: int = 2000) -> None:
        self.max_size = max_size
        self.data = np.full(max_size, np.nan, dtype=np.float32)
        self.offset = 0
        self.size = 0

    def add_point(self, value: float) -> None:
        self.data[self.offset] = value
        self.offset = (self.offset + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_data(self) -> np.ndarray:
        return self.data[:self.size] if self.size < self.max_size else np.roll(self.data, -self.offset)


class ScrollingBuffer:
    def __init__(self, max_size: int = 2000) -> None:
        self.max_size = max_size
        self.data = np.full((max_size, 2), np.nan, dtype=np.float32)
        self.offset = 0
        self.size = 0

    def add_point(self, x: float, y: float) -> None:
        self.data[self.offset] = [x, y]
        self.offset = (self.offset + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_data(self) -> np.ndarray:
        return self.data[:self.size].T if self.size < self.max_size else np.roll(self.data, -self.offset).T

def set_image(image) -> None:
    latest_frame[0] = image

@immapp.static(open_file_dialog=None)
def load_file() -> None:
    """
    Display a file dialog to load a PLY file into the current world settings.
    """
    static = load_file
    if imgui.button("Open PLY"):
        static.open_file_dialog = pfd.open_file("Select a file")
    if static.open_file_dialog and static.open_file_dialog.ready():
        file = static.open_file_dialog.result()
        if file and file[0].lower().endswith(".ply"):
            try:
                world_settings.load_ply(file[0])
            except (OSError, ValueError) as error:
                util.logger.error(f"Could not load PLY '{file[0]}': {error}")
        elif file:
            util.logger.error(f"Selected file is not a PLY file: {file[0]}")
        static.open_file_dialog = None

RENDER_OUTPUT_LABELS = {
    RenderOutputMode.COLOR: "Color",
    RenderOutputMode.DEPTH: "Depth",
    RenderOutputMode.OPACITY: "Opacity",
}


def _status_row(label: str, value: str) -> None:
    imgui.table_next_row()
    imgui.table_set_column_index(0)
    imgui.text_disabled(label)
    imgui.table_set_column_index(1)
    imgui.text_wrapped(value)


def _item_tooltip(message: str) -> None:
    if imgui.is_item_hovered():
        imgui.set_tooltip(message)


@immapp.static(scale_modifier=None, settings_id=None)
def display_renderer_tab() -> None:
    """Display scene actions, renderer status, and visualization controls."""
    static = display_renderer_tab
    renderer = world_settings.gauss_renderer
    if static.settings_id != id(world_settings):
        static.settings_id = id(world_settings)
        static.scale_modifier = world_settings.scale_modifier

    imgui.separator_text("Status")
    if renderer is None:
        imgui.text_disabled("Renderer is initializing...")
    elif imgui.begin_table("RendererStatus", 2):
        _status_row("Device", renderer.device_name)
        _status_row("Frame rate", f"{imgui.get_io().framerate:.1f} FPS")
        _status_row("Splats", f"{world_settings.get_num_gaussians():,}")
        _status_row("Scene SH", f"Degree {renderer.scene_sh_degree}")
        imgui.end_table()

    imgui.separator_text("Scene")
    load_file()
    imgui.same_line()
    if imgui.button("Clear"):
        world_settings.reset_gaussians()
    imgui.same_line()
    if imgui.button("Capture"):
        take_screenshot(f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    imgui.separator_text("Appearance")
    output_modes = list(RenderOutputMode)
    current_output = output_modes.index(world_settings.render_output)
    imgui.text("Output")
    imgui.set_next_item_width(-1)
    changed, current_output = imgui.combo(
        "##RenderOutput",
        current_output,
        [RENDER_OUTPUT_LABELS[mode] for mode in output_modes],
    )
    _item_tooltip(
        "Color shows spherical-harmonic shading. Depth is auto-normalized "
        "per frame. Opacity shows accumulated Gaussian coverage."
    )
    if changed:
        world_settings.update_render_output(output_modes[current_output])

    scene_degree = renderer.scene_sh_degree if renderer is not None else 0
    sh_options = [f"Auto (scene degree {scene_degree})", "0 — DC only"]
    sh_options.extend(str(degree) for degree in range(1, scene_degree + 1))
    requested_degree = world_settings.active_sh_degree
    current_sh = 0 if requested_degree is None else min(requested_degree, scene_degree) + 1
    imgui.text("Spherical harmonics")
    imgui.set_next_item_width(-1)
    if renderer is None:
        imgui.begin_disabled()
    changed, current_sh = imgui.combo("##SphericalHarmonics", current_sh, sh_options)
    _item_tooltip(
        "Lower degrees remove view-dependent color detail. Auto uses the "
        "highest degree stored in the current scene."
    )
    if renderer is None:
        imgui.end_disabled()
    if changed:
        degree = None if current_sh == 0 else current_sh - 1
        world_settings.update_sh_degree(degree)

    imgui.text("Splat scale")
    imgui.set_next_item_width(-70)
    _, static.scale_modifier = imgui.slider_float(
        "##SplatScale",
        static.scale_modifier,
        0.05,
        3.0,
        "%.2f×",
    )
    scale_edit_finished = imgui.is_item_deactivated_after_edit()
    _item_tooltip(
        "Multiplies every Gaussian's scale. The value is applied when you "
        "release the control to keep large streamed scenes responsive."
    )
    imgui.same_line()
    if imgui.button("Reset##SplatScale"):
        static.scale_modifier = 1.0
        world_settings.update_scale_modifier(1.0)
    elif scale_edit_finished:
        world_settings.update_scale_modifier(static.scale_modifier)


@immapp.static(
    selected_topic="",
    available_topics=[],
    active_topics=[],
    prev_time=-1.0
)
def display_streams_tab() -> None:
    """
    Display ROS topic discovery and subscription management.
    """
    static = display_streams_tab
    imgui.text_wrapped("Discover ROS 2 topics and subscribe them to ROSplat.")

    current_time = imgui.get_time()
    if static.prev_time == -1.0 or current_time - static.prev_time > 1.0:
        static.available_topics = ros_node_manager._graph.get_topic_names_and_types()
        static.prev_time = current_time

    if imgui.button("Refresh topics"):
        static.available_topics = ros_node_manager._graph.get_topic_names_and_types()

    if imgui.begin_table("TopicsTable", 3):
        imgui.table_next_column()
        imgui.text("Available Topics")
        for topic in static.available_topics:
            topic_name = topic[0]
            selected = (static.selected_topic == topic_name)
            changed, selected = imgui.selectable(topic_name, selected)
            if changed and selected:
                static.selected_topic = topic_name

        imgui.table_next_column()
        imgui.text("Message Type")
        for topic in static.available_topics:
            imgui.text(topic[1][0])

        imgui.table_next_column()
        imgui.text("Subscriptions")
        for i, topic in enumerate(static.active_topics.copy()):
            changed, active = imgui.checkbox(f"{i}", True)
            imgui.same_line()
            imgui.text(topic)
            if topic not in [t[0] for t in static.available_topics] or not active:
                static.active_topics.remove(topic)
                ros_node_manager.kill_listener(topic)
        imgui.end_table()

    is_valid = static.selected_topic and static.selected_topic not in static.active_topics
    if not is_valid:
        imgui.begin_disabled()
    if imgui.button("Subscribe"):
        static.active_topics.append(static.selected_topic)
        static.active_topics.sort()
        ros_node_manager.add_listener(static.selected_topic)
        static.selected_topic = ""
    if not is_valid:
        imgui.end_disabled()



def display_camera_tab() -> None:
    """
    Placeholder for camera settings tab.
    """
    imgui.text("Camera settings go here.")


@immapp.static(image_texture=None, last_frame=None)
def display_frames_tab() -> None:
    """Display the latest ROS image using the active ImGui backend texture."""
    static = display_frames_tab

    # Grab the latest image from shared memory
    if latest_frame[0] is not None and latest_frame[0] is not static.last_frame:
        static.last_frame = latest_frame[0]
        frame = np.asarray(static.last_frame)
        if frame.shape[-1] == 3:
            alpha = np.full((*frame.shape[:2], 1), 255, dtype=np.uint8)
            frame = np.concatenate((frame, alpha), axis=-1)
        static.image_texture = world_settings.texture_registry.upload(
            "ros-image-frame", np.ascontiguousarray(frame, dtype=np.uint8)
        )

    frame = static.last_frame

    if frame is None:
        imgui.text("Waiting for frame...")
        return

    avail_w, avail_h = imgui.get_content_region_avail()
    if implot.begin_plot("Live Frame", size=(avail_w, avail_h)):
        implot.setup_axes(
            "X", "Y",
            implot.AxisFlags_.no_tick_labels,
            implot.AxisFlags_.no_tick_labels
        )
        implot.plot_image(
            "Frame",
            static.image_texture,
            (0, 0),
            (avail_w, avail_h)
        )
        implot.end_plot()



def update_imu_queue(lin_accel: np.ndarray, ang_vel: np.ndarray) -> None:
    """
    Thread-safe update to the IMU data queue.
    """
    try:
        imu_queue.get_nowait()
    except queue.Empty:
        pass
    imu_queue.put((lin_accel, ang_vel))


def update_imu(lin_accel: np.ndarray, ang_vel: np.ndarray) -> None:
    """
    Append IMU data to the global lists.
    """
    accel_x.extend(lin_accel)
    accel_y.extend(lin_accel[1:])
    accel_z.extend(lin_accel[2:])
    gyro_x.extend(ang_vel)
    gyro_y.extend(ang_vel[1:])
    gyro_z.extend(ang_vel[2:])


@immapp.static(auto_fit_accel=True, auto_fit_gyro=True)
def display_imu_tab() -> None:
    """
    Display IMU acceleration and gyro plots.
    """
    static = display_imu_tab
    _, static.auto_fit_accel = imgui.checkbox("Auto-Fit Acceleration Plot", static.auto_fit_accel)
    _, static.auto_fit_gyro = imgui.checkbox("Auto-Fit Gyro Plot", static.auto_fit_gyro)

    def _plot(name, x_data, y_data, fit_flag):
        if implot.begin_plot(name):
            flags = implot.AxisFlags_.auto_fit if fit_flag else 0
            implot.setup_axes("Sample Index", name, flags, flags)
            if len(x_data) > 0:
                xs = np.arange(len(x_data), dtype=np.float32)
                implot.plot_line(f"{name} X", xs, np.array(x_data, dtype=np.float32))
            implot.end_plot()

    _plot("Acceleration", accel_x, accel_y, static.auto_fit_accel)
    _plot("Gyroscope", gyro_x, gyro_y, static.auto_fit_gyro)


def set_gaussian(msg) -> None:
    """
    Append new Gaussian data from a message.
    """
    if msg is not None:
        world_settings.append_gaussians(msg)


def main_ui(this_world_settings) -> None:
    """
    Entry point for the main UI.
    """
    global world_settings
    if world_settings is None:
        world_settings = this_world_settings

    if imgui.begin("Main Application"):
        if imgui.begin_tab_bar("MainTabs"):
            if imgui.begin_tab_item("Renderer")[0]:
                display_renderer_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Streams")[0]:
                display_streams_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Camera")[0]:
                display_camera_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Frames")[0]:
                display_frames_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item("IMU")[0]:
                display_imu_tab()
                imgui.end_tab_item()
            imgui.end_tab_bar()
        imgui.end()



    if imgui.begin("Splat"):
        avail_w, avail_h = imgui.get_content_region_avail()
        
        # Cast both to int
        avail_w = int(avail_w)
        avail_h = int(avail_h)

        if avail_w > 0 and avail_h > 0:
            this_world_settings.update_window_size(avail_w, avail_h)
            tex = this_world_settings.gauss_renderer.draw()
            vec2 = imgui.ImVec2(avail_w, avail_h)
            imgui.image(tex, vec2)
            if imgui.is_window_hovered():
                this_world_settings.check_inputs()
        imgui.end()
