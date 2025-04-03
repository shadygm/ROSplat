import threading
from enum import Enum
from typing import Optional, Tuple, Dict, List

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu
from geometry_msgs.msg import PoseStamped
import cv_bridge

import util
import imgui_manager

try:
    from gaussian_interface.msg import SingleGaussian, GaussianArray
except ImportError:
    util.logger.error("Could not import SingleGaussian message. Make sure the package is built and sourced correctly.")
    SingleGaussian = None
    GaussianArray = None

class MsgType(Enum):
    IMAGE = ("sensor_msgs/msg/Image", Image)
    CAMERA_INFO = ("sensor_msgs/msg/CameraInfo", CameraInfo)
    POINT_CLOUD2 = ("sensor_msgs/msg/PointCloud2", PointCloud2)
    IMU = ("sensor_msgs/msg/Imu", Imu)
    POSE_STAMPED = ("geometry_msgs/msg/PoseStamped", PoseStamped)
    GAUSSIAN = ("gaussian_interface/msg/SingleGaussian", SingleGaussian)
    GAUSSIAN_ARRAY = ("gaussian_interface/msg/GaussianArray", GaussianArray)

def list_topics() -> List[Tuple[str, List[str]]]:
    """
    Initialize and create a temporary node to list available topics.
    Returns:
        A sorted list (by topic name) of topics with their message type strings.
    """
    should_shutdown = False
    if not rclpy.ok():
        rclpy.init(args=None)
        should_shutdown = True

    node = rclpy.create_node('list_topics')
    topics = node.get_topic_names_and_types()
    node.destroy_node()

    if should_shutdown:
        rclpy.shutdown()

    return sorted(topics, key=lambda x: x[0])

def get_message_class(message_type_str: str):
    """
    Retrieve the message class corresponding to the message type string.
    """
    util.logger.info(f"Getting message class for {message_type_str}")
    for msg_type in MsgType:
        if msg_type.value[0] == message_type_str:
            return msg_type.value[1]
    return None

class SingleNode(Node):
    """
    A node that subscribes to a specific topic and routes received messages to the appropriate handler.
    """
    def __init__(self, node_idx: int, topic_name: str, msg_type) -> None:
        # Create a valid node name by combining a counter and a sanitized topic name.
        node_name = f"listener_{node_idx}_{topic_name.replace('/', '_')}"
        super().__init__(node_name)
        self.topic_name = topic_name
        self.msg_type = msg_type
        self.bridge = cv_bridge.CvBridge()
        self.subscriber = self.create_subscription(msg_type, topic_name, self.callback, 10)

    def callback(self, msg) -> None:
        """
        Callback method that selects the proper update method based on the message type.
        """
        if self.msg_type == Image:
            self.update_image(msg)
        elif self.msg_type == Imu:
            self.update_imu(msg)
        elif SingleGaussian is not None and self.msg_type == SingleGaussian:
            self.update_gaussian(msg)
        elif GaussianArray is not None and self.msg_type == GaussianArray:
            self.update_gaussian(msg)

    def update_gaussian(self, msg) -> None:
        """
        Process and pass a gaussian message to the UI.
        """
        imgui_manager.set_gaussian(msg)

    def update_imu(self, msg) -> None:
        """
        Process and update IMU data in the UI.
        """
        try:
            lin_accel = np.array([msg.linear_acceleration.x,
                                  msg.linear_acceleration.y,
                                  msg.linear_acceleration.z])
            ang_vel = np.array([msg.angular_velocity.x,
                                msg.angular_velocity.y,
                                msg.angular_velocity.z])
            imgui_manager.update_imu(lin_accel, ang_vel)
        except Exception as e:
            util.logger.error(f"Error updating IMU: {e}")

    def update_image(self, msg) -> None:
        """
        Convert the ROS image message to OpenCV format and update the UI.
        """
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            imgui_manager.set_image(img)
        except Exception as e:
            util.logger.error(f"Error converting image: {e}")

    def destroy(self) -> None:
        """
        Clean up the node.
        """
        try:
            self.destroy_node()
        except Exception:
            pass

class ROSNodeManager:
    """
    Manager class that handles spinning ROS nodes in their dedicated threads.
    """
    def __init__(self) -> None:
        if not rclpy.ok():
            rclpy.init(args=None)
        self.nodes: Dict[str, Tuple[SingleNode, threading.Thread, threading.Event]] = {}
        self.node_idx_counter: int = 0

    def get_msg_type(self, topic_name: str):
        """
        Check the topic and return its corresponding message type.
        """
        topics = list_topics()
        for name, type_list in topics:
            if name == topic_name and type_list:
                message_class = get_message_class(type_list[0])
                return message_class
        return None

    def add_listener(self, topic_name: str) -> None:
        """
        Create and spin a node that listens to messages on the given topic.
        """
        if topic_name in self.nodes:
            util.logger.info(f"Node for topic {topic_name} already exists.")
            return

        msg_type = self.get_msg_type(topic_name)
        if msg_type is None:
            util.logger.error(f"Could not determine message type for topic {topic_name}")
            return

        node = SingleNode(self.node_idx_counter, topic_name, msg_type)
        self.node_idx_counter += 1

        stop_event = threading.Event()

        def spin_loop() -> None:
            executor = SingleThreadedExecutor()
            executor.add_node(node)
            while rclpy.ok() and not stop_event.is_set():
                executor.spin_once(timeout_sec=0.1)
            executor.shutdown()
            node.destroy()

        thread = threading.Thread(target=spin_loop, daemon=True)
        thread.start()
        self.nodes[topic_name] = (node, thread, stop_event)
        util.logger.info(f"Added listener for topic {topic_name}")

    def kill_listener(self, topic_name: str) -> None:
        """
        Stop the listener thread for the specified topic and clean it up.
        """
        if topic_name in self.nodes:
            node, thread, stop_event = self.nodes.pop(topic_name)
            stop_event.set()
            thread.join()
            util.logger.info(f"Killed listener for topic {topic_name}")
        else:
            util.logger.error(f"Node for topic {topic_name} does not exist.")
