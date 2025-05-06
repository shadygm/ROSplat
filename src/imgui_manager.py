import queue
from datetime import datetime

import cv2
import numpy as np
import OpenGL.GL as gl
from PIL import Image

from imgui_bundle import (
    imgui,
    immapp,
    implot,
    implot3d,
    portable_file_dialogs as pfd,
)
import ros_util
import util
from ros_util import ROSNodeManager
try:
    from CUDARenderer import CUDARenderer
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    util.logger.warning("CUDARenderer not available. CUDA rendering will be disabled.")
    from base_gaussian_renderer import OpenGLRenderer
    util.logger.info("Using OpenGL Renderer")


# Global variables and settings
world_settings = None
type_visualization = [
    "Gaussian Ball", "Flat Ball", "Billboard",
    "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"
]
frame_queue = queue.Queue(maxsize=1)
imu_queue = queue.Queue(maxsize=1)
ros_node_manager = ROSNodeManager()

accel_x, accel_y, accel_z = [], [], []
gyro_x, gyro_y, gyro_z = [], [], []

def take_screenshot(filename: str = "screenshot.png") -> None:
    """
    Captures the current OpenGL framebuffer and saves it as an image.
    """
    viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
    x, y, width, height = viewport

    # Read pixels from the framebuffer (bottom-up)
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
        if self.size < self.max_size:
            return self.data[:self.size]
        return np.roll(self.data, -self.offset)

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
        if self.size < self.max_size:
            return self.data[:self.size].T
        return np.roll(self.data, -self.offset).T

@immapp.static(open_file_dialog=None)
def load_file() -> None:
    """
    Show a file dialog to load a PLY file.
    """
    static = load_file
    if imgui.button("Open PLY"):
        static.open_file_dialog = pfd.open_file("Select a file")
    if static.open_file_dialog is not None and static.open_file_dialog.ready():
        file = static.open_file_dialog.result()
        if file and file[0].lower().endswith(".ply"):
            world_settings.load_ply(file[0])
        elif file:
            util.logger.error(f"Selected file is not a PLY file: {file[0]}")
        static.open_file_dialog = None

def _renderer_settings() -> None:
    imgui.text("Renderer:")
    if HAS_TORCH and torch.cuda.is_available():
        renderer_types = ["CUDA Renderer", "OpenGL Renderer"]
        current_renderer_idx = 0 if isinstance(world_settings.gauss_renderer, CUDARenderer) else 1
        changed, current_renderer_idx = imgui.combo("Renderer", current_renderer_idx, renderer_types)
        if changed:
            if renderer_types[current_renderer_idx] == "CUDA Renderer":
                world_settings.switch_renderer("CUDA")
            else:
                world_settings.switch_renderer("OpenGL")
    if not isinstance(world_settings.gauss_renderer, CUDARenderer):
        imgui.text("OpenGL Renderer: Sorting needed.")
        if imgui.button("Sort and Update"):
            world_settings.gauss_renderer.sort_and_update()
        _, world_settings.auto_sort = imgui.checkbox("Auto-sort", world_settings.auto_sort)
        changed, mode = imgui.combo("Visualization Type", world_settings.render_mode, type_visualization)
        if changed:
            world_settings.update_render_mode(mode)
        world_settings.update_render_mode(mode)
    else:
        imgui.text("CUDA Renderer: No sorting needed.")
        world_settings.auto_sort = False

def display_parameters_tab() -> None:
    """
    Display the ROSplat parameters tab.
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
    # make a check box that is for enabling/disabling overwriting of gaussians
    _, world_settings.overwrite_gaussians = imgui.checkbox(
        "Overwrite Gaussians", world_settings.overwrite_gaussians
    )
    _renderer_settings()

@immapp.static(
    selected_topic="",
    available_topics=ros_util.list_topics(),
    active_topics=[],
    prev_time=-1.0,
    curr_time=-1.0,
)
def display_ros_tab() -> None:
    """
    Display the ROS topics and node management tab.
    """
    static = display_ros_tab
    imgui.text("Add and Remove ROS nodes here.")
    imgui.same_line()

    # Auto-refresh available topics every second
    current_time = imgui.get_time()
    do_refresh = static.prev_time == -1.0 or current_time - static.prev_time > 1.0
    if do_refresh:
        static.available_topics = ros_util.list_topics()
        static.prev_time = current_time

    if imgui.button("Refresh") or do_refresh:
        static.available_topics = ros_util.list_topics()

    if imgui.begin_table("TopicsTable", 3):
        # Left Column: Available Topics
        imgui.table_next_column()
        imgui.text("Available Topics")
        for topic in static.available_topics:
            topic_name = topic[0]
            is_selected = (static.selected_topic == topic_name)
            changed, is_selected = imgui.selectable(topic_name, is_selected)
            if changed and is_selected:
                static.selected_topic = topic_name

        # Middle Column: Message Type
        imgui.table_next_column()
        imgui.text("Message Type")
        for topic in static.available_topics:
            imgui.text(topic[1][0])

        # Right Column: Active Topics
        imgui.table_next_column()
        imgui.text("Active Topics")
        # Copy active topics list since we might modify while iterating
        for i, topic in enumerate(static.active_topics.copy()):
            changed, active = imgui.checkbox(f"{i}", True)
            imgui.same_line()
            imgui.text(topic)
            if topic not in [t[0] for t in static.available_topics] or not active:
                static.active_topics.remove(topic)
                ros_node_manager.kill_listener(topic)

        imgui.end_table()

    is_valid_topic = static.selected_topic and static.selected_topic not in static.active_topics
    if not is_valid_topic:
        imgui.begin_disabled()

    if imgui.button("Add"):
        static.active_topics.append(static.selected_topic)
        static.active_topics.sort()
        ros_node_manager.add_listener(static.selected_topic)
        static.selected_topic = ""

    if not is_valid_topic:
        imgui.end_disabled()

def display_camera_tab() -> None:
    """
    Display the Camera Settings tab.
    """
    imgui.text("Camera settings go here.")

@immapp.static(
    camera_pose=None,
    t=0.0,
    last_t=-1.0,
    data_x=None,
    data_y=None,
    data_z=None,
)
def display_visualization_tab() -> None:
    """
    Display the 3D visualization tab (Camera Movement).
    """
    static = display_visualization_tab
    if static.camera_pose is None:
        static.camera_pose = np.eye(4)
        static.data_x = CircularBuffer(max_size=20000)
        static.data_y = CircularBuffer(max_size=20000)
        static.data_z = CircularBuffer(max_size=20000)

    if imgui.button("Reset Camera"):
        static.data_x = CircularBuffer(max_size=20000)
        static.data_y = CircularBuffer(max_size=20000)
        static.data_z = CircularBuffer(max_size=20000)
    
    if implot3d.begin_plot("Camera Plot", size=(-1, -1)):
        static.t += imgui.get_io().delta_time
        if static.t - static.last_t > 0.01:
            static.last_t = static.t
            cam_pose = world_settings.get_camera_pose()
            # Rearranging the axes for plotting
            x, y, z = -cam_pose[2], -cam_pose[0], -cam_pose[1]
            static.data_x.add_point(x)
            static.data_y.add_point(y)
            static.data_z.add_point(z)
        
        flags = implot3d.AxisFlags_.no_tick_labels.value | implot3d.AxisFlags_.auto_fit
        implot3d.setup_axes("Z", "X", "Y", flags, flags, flags)
        implot3d.setup_axis_limits(implot3d.ImAxis3D_.x.value, -1, 1, implot3d.Cond_.always.value)
        implot3d.setup_axis_limits(implot3d.ImAxis3D_.y.value, -1, 1, implot3d.Cond_.once.value)
        implot3d.setup_axis_limits(implot3d.ImAxis3D_.z.value, -1, 1, implot3d.Cond_.once.value)
        
        x_data = static.data_x.get_data()
        y_data = static.data_y.get_data()
        z_data = static.data_z.get_data()
        
        if len(x_data) > 0:
            implot3d.plot_line("Camera Pose", x_data, y_data, z_data)
        implot3d.end_plot()

def create_texture_from_frame(frame: np.ndarray) -> int:
    """
    Creates an OpenGL texture from a NumPy frame.
    """
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    height, width, channels = frame.shape
    if channels == 3:
        fmt = gl.GL_RGB
    elif channels == 4:
        fmt = gl.GL_RGBA
    else:
        raise ValueError("Frame must have 3 (RGB) or 4 (RGBA) channels")
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, fmt, width, height, 0, fmt, gl.GL_UNSIGNED_BYTE, frame)
    return tex

def update_texture(tex: int, frame: np.ndarray) -> None:
    """
    Updates an existing OpenGL texture with new frame data.
    """
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    
    height, width, channels = frame.shape
    fmt = gl.GL_RGB if channels == 3 else gl.GL_RGBA
    gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, width, height, fmt, gl.GL_UNSIGNED_BYTE, frame)
    gl.glFlush()

@immapp.static(
    image_texture=None, 
    texture_width=0, 
    texture_height=0, 
    auto_fit=True, 
    flags=implot.AxisFlags_.auto_fit | implot.AxisFlags_.no_tick_labels
)
def display_frames_tab() -> None:
    try:
        frame = frame_queue.get_nowait()
        # put it back so we keep showing it
        frame_queue.put(frame)
    except queue.Empty:
        frame = None

    if frame is None:
        imgui.text("Waiting for frame...")
        return

    h, w, _ = frame.shape

    if display_frames_tab.image_texture is None:
        # first time: create GL texture
        display_frames_tab.image_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, display_frames_tab.image_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        display_frames_tab.texture_width = w
        display_frames_tab.texture_height = h

    # Update texture with new frame
    gl.glBindTexture(gl.GL_TEXTURE_2D, display_frames_tab.image_texture)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D, 0, gl.GL_RGB,
        w, h, 0,
        gl.GL_RGB, gl.GL_UNSIGNED_BYTE,
        frame
    )

    avail_w, avail_h = imgui.get_content_region_avail()
    if implot.begin_plot("Live Frame", size=(avail_w, avail_h)):
        implot.setup_axes("X", "Y", implot.AxisFlags_.no_tick_labels, implot.AxisFlags_.no_tick_labels)
        implot.plot_image(
            "Frame", display_frames_tab.image_texture,
            (0,0), (avail_w, avail_h)
        )
        implot.end_plot()

def update_imu_queue(lin_accel: np.ndarray, ang_vel: np.ndarray) -> None:
    """
    Thread-safe update of IMU data in the imu_queue.
    """
    try:
        imu_queue.get_nowait()
    except queue.Empty:
        pass
    imu_queue.put((lin_accel, ang_vel))

def update_imu(lin_accel: np.ndarray, ang_vel: np.ndarray) -> None:
    """
    Receives IMU data and appends each axis to global lists.
    lin_accel and ang_vel are assumed to be 1D numpy arrays of shape [x, y, z].
    """
    global accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z

    accel_x.append(lin_accel[0])
    accel_y.append(lin_accel[1])
    accel_z.append(lin_accel[2])

    gyro_x.append(ang_vel[0])
    gyro_y.append(ang_vel[1])
    gyro_z.append(ang_vel[2])

@immapp.static(auto_fit_accel=True, auto_fit_gyro=True)
def display_imu_tab() -> None:
    """
    Display the IMU plots tab.
    """
    static = display_imu_tab

    changed, static.auto_fit_accel = imgui.checkbox("Auto-Fit Acceleration Plot", static.auto_fit_accel)
    changed, static.auto_fit_gyro = imgui.checkbox("Auto-Fit Gyro Plot", static.auto_fit_gyro)

    # Acceleration Plot
    if implot.begin_plot("Acceleration"):
        x_axis_flags = implot.AxisFlags_.auto_fit if static.auto_fit_accel else 0
        y_axis_flags = implot.AxisFlags_.auto_fit if static.auto_fit_accel else 0
        implot.setup_axes("Sample Index", "Acceleration (m/s^2)", x_axis_flags, y_axis_flags)

        num_points = len(accel_x)
        if num_points > 0:
            xs = np.arange(num_points, dtype=np.float32)
            implot.plot_line("Accel X", xs, np.array(accel_x, dtype=np.float32))
            implot.plot_line("Accel Y", xs, np.array(accel_y, dtype=np.float32))
            implot.plot_line("Accel Z", xs, np.array(accel_z, dtype=np.float32))
        implot.end_plot()

    # Gyroscope Plot
    if implot.begin_plot("Gyroscope"):
        x_axis_flags = implot.AxisFlags_.auto_fit if static.auto_fit_gyro else 0
        y_axis_flags = implot.AxisFlags_.auto_fit if static.auto_fit_gyro else 0
        implot.setup_axes("Sample Index", "Gyroscope (rad/s)", x_axis_flags, y_axis_flags)

        num_points = len(gyro_x)
        if num_points > 0:
            xs = np.arange(num_points, dtype=np.float32)
            implot.plot_line("Gyro X", xs, np.array(gyro_x, dtype=np.float32))
            implot.plot_line("Gyro Y", xs, np.array(gyro_y, dtype=np.float32))
            implot.plot_line("Gyro Z", xs, np.array(gyro_z, dtype=np.float32))
        implot.end_plot()

def set_gaussian(msg) -> None:
    """
    Update the Gaussian data in world settings.
    """
    if msg is not None:
        world_settings.append_gaussians(msg)


def main_ui(this_world_settings) -> None:
    """
    Main UI: initializes world_settings and displays UI tabs.
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
            if imgui.begin_tab_item("3D Visualization")[0]:
                display_visualization_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Frames")[0]:
                display_frames_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item("IMU")[0]:
                display_imu_tab()
                imgui.end_tab_item()
            imgui.end_tab_bar()
        imgui.end()