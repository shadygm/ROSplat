import queue
from datetime import datetime

import cv2
import numpy as np
import OpenGL.GL as gl
from PIL import Image
from rosplat.config import RendererType

# ImGui Bundle
from imgui_bundle import (
    imgui,
    immapp,
    implot,
    portable_file_dialogs as pfd,
)

# Local modules
from rosplat.core import util
from rosplat.ros import ROSNodeManager

try:
    from rosplat.render.renderer import CUDARenderer
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    util.logger.warning("CUDARenderer not available. CUDA rendering will be disabled.")
    from rosplat.render.renderer import OpenGLRenderer
    util.logger.info("Using OpenGL Renderer")


# === Global State ===
world_settings = None
frame_queue = queue.Queue(maxsize=10)
latest_frame = [None]  # mutable container
imu_queue = queue.Queue(maxsize=1)
ros_node_manager = ROSNodeManager()

type_visualization = [
    "Gaussian Ball", "Flat Ball", "Billboard",
    "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"
]

# IMU history
accel_x, accel_y, accel_z = [], [], []
gyro_x, gyro_y, gyro_z = [], [], []


def take_screenshot(filename: str = "screenshot.png") -> None:
    """
    Capture the current OpenGL framebuffer and save it as an image.
    """
    x, y, width, height = gl.glGetIntegerv(gl.GL_VIEWPORT)
    data = gl.glReadPixels(x, y, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(filename)
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
            world_settings.load_ply(file[0])
        elif file:
            util.logger.error(f"Selected file is not a PLY file: {file[0]}")
        static.open_file_dialog = None


def _renderer_settings() -> None:
    imgui.text("Renderer:")
    if HAS_TORCH and torch.cuda.is_available():
        options = [RendererType.CUDA.value, RendererType.OPENGL.value]
        current_idx = 0 if world_settings.get_renderer_type() == RendererType.CUDA else 1
        changed, current_idx = imgui.combo("Renderer", current_idx, options)
        if changed:
            world_settings.switch_renderer(RendererType.CUDA if current_idx == 0 else RendererType.OPENGL)
    if not isinstance(world_settings.gauss_renderer, CUDARenderer):
        imgui.text("OpenGL Renderer: Sorting needed.")
        if imgui.button("Sort and Update"):
            world_settings.gauss_renderer.sort_and_update()
        _, world_settings.auto_sort = imgui.checkbox("Auto-sort", world_settings.auto_sort)
        changed, mode = imgui.combo("Visualization Type", world_settings.render_mode, type_visualization)
        if changed:
            world_settings.update_render_mode(mode)
    else:
        imgui.text("CUDA Renderer: No sorting needed.")
        world_settings.auto_sort = False


def display_parameters_tab() -> None:
    """
    Display rendering and parameter control tab.
    """
    imgui.text(f"FPS: {imgui.get_io().framerate:.1f}")
    imgui.text(f"Num of Gauss: {world_settings.get_num_gaussians()}")
    load_file()
    imgui.same_line()
    if imgui.button("Reset Gaussians"):
        world_settings.reset_gaussians()
    imgui.same_line()
    if imgui.button("Screenshot"):
        take_screenshot(f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    imgui.text("Parameters:")
    # TODO: Add more parameters for the CUDA renderer. 
    # Currently, only the OpenGL has parameters.
    # _, world_settings.overwrite_gaussians = imgui.checkbox("Overwrite Gaussians", world_settings.overwrite_gaussians)
    _renderer_settings()


@immapp.static(
    selected_topic="",
    available_topics=[],
    active_topics=[],
    prev_time=-1.0
)
def display_ros_tab() -> None:
    """
    Display ROS topic selector and listener management.
    """
    static = display_ros_tab
    imgui.text("Add and Remove ROS nodes here.")
    imgui.same_line()

    current_time = imgui.get_time()
    if static.prev_time == -1.0 or current_time - static.prev_time > 1.0:
        static.available_topics = ros_node_manager._graph.get_topic_names_and_types()
        static.prev_time = current_time

    if imgui.button("Refresh"):
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
        imgui.text("Active Topics")
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
    if imgui.button("Add"):
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


def create_texture_from_frame(frame: np.ndarray) -> int:
    """
    Create an OpenGL texture from a NumPy image.
    """
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    h, w, c = frame.shape
    fmt = gl.GL_RGB if c == 3 else gl.GL_RGBA
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, fmt, w, h, 0, fmt, gl.GL_UNSIGNED_BYTE, frame)
    return tex


def update_texture(tex: int, frame: np.ndarray) -> None:
    """
    Update an OpenGL texture with a new frame.
    """
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    h, w, c = frame.shape
    fmt = gl.GL_RGB if c == 3 else gl.GL_RGBA
    gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w, h, fmt, gl.GL_UNSIGNED_BYTE, frame)
    gl.glFlush()

def refresh() -> None:
    """
    Refresh the current set of Gaussians.
    """
    world_settings.refresh_gaussians()


@immapp.static(image_texture=None, last_frame=None)
def display_frames_tab() -> None:
    """
    Display the most recent frame using OpenGL texture.
    """
    static = display_frames_tab

    # Grab the latest image from shared memory
    if latest_frame[0] is not None:
        static.last_frame = latest_frame[0]

    frame = static.last_frame

    if frame is None:
        imgui.text("Waiting for frame...")
        return

    h, w, _ = frame.shape
    if static.image_texture is None:
        static.image_texture = create_texture_from_frame(frame)
    else:
        update_texture(static.image_texture, frame)

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
            if imgui.begin_tab_item("ROSplat")[0]:
                display_parameters_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item("ROS Settings")[0]:
                display_ros_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Camera Settings")[0]:
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