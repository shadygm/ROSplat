from imgui_bundle import hello_imgui, imgui
from . import imgui_manager

def create_dockable_windows(world_settings):
    imgui_manager.world_settings = world_settings
    
    return [
        hello_imgui.DockableWindow(
            label_="ROSplat",
            dock_space_name_="MainDockSpace",
            gui_function_=imgui_manager.display_parameters_tab,
        ),
        hello_imgui.DockableWindow(
            label_="ROS Settings",
            dock_space_name_="MainDockSpace",
            gui_function_=imgui_manager.display_ros_tab,
        ),
        hello_imgui.DockableWindow(
            label_="Camera Settings",
            dock_space_name_="MainDockSpace",
            gui_function_=imgui_manager.display_camera_tab,
        ),
        hello_imgui.DockableWindow(
            label_="Frames",
            dock_space_name_="MainDockSpace",
            gui_function_=imgui_manager.display_frames_tab,
        ),
        hello_imgui.DockableWindow(
            label_="IMU",
            dock_space_name_="MainDockSpace",
            gui_function_=imgui_manager.display_imu_tab,
        ),
    ]
