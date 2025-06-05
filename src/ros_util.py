import threading
from enum import Enum
from typing import Optional, Tuple, Dict, List
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu
from geometry_msgs.msg import PoseStamped
import cv_bridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import cv2

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

def list_topics():
    if not rclpy.ok():
        rclpy.init()

    util.logger.error(rclpy.ok)
    
    node = rclpy.create_node('list_topics')
    exec = SingleThreadedExecutor()
    exec.add_node(node)
    exec.spin_once(timeout_sec=0.5)      # process discovery events
    topics = node.get_topic_names_and_types()
    exec.shutdown()
    node.destroy_node()
    rclpy.shutdown()
    return sorted(topics, key=lambda x: x[0])

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

        qos = QoSProfile(depth=1000)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.VOLATILE

        self.subscriber = self.create_subscription(
            msg_type,
            topic_name,
            self.callback,
            qos
        )

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
    Manager class that handles spinning ROS nodes in a single MultiThreadedExecutor.
    """
    def __init__(self) -> None:
        if not rclpy.ok():
            rclpy.init()

        self.executor = rclpy.executors.MultiThreadedExecutor()
        self._graph = GraphWatcher()
        self.executor.add_node(self._graph)

        self.nodes: Dict[str, SingleNode] = {}
        self.node_idx_counter: int = 0
        self._lock = threading.Lock()

        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        """Run the executor in a dedicated background thread."""
        self.executor.spin()

    def get_msg_type(self, topic_name: str):
        """
        Check the topic and return its corresponding message type.
        """
        topics = self._graph.get_topic_names_and_types()
        for name, type_list in topics:
            if name == topic_name and type_list:
                message_class = get_message_class(type_list[0])
                return message_class
        return None

    def add_listener(self, topic_name: str) -> None:
        """
        Create and add a node that listens to messages on the given topic.
        """
        with self._lock:
            if topic_name in self.nodes:
                util.logger.info(f"Node for topic {topic_name} already exists.")
                return

            msg_type = self.get_msg_type(topic_name)
            if msg_type is None:
                util.logger.error(f"Could not determine message type for topic {topic_name}")
                return

            node = SingleNode(self.node_idx_counter, topic_name, msg_type)
            self.node_idx_counter += 1

            self.executor.add_node(node)
            self.nodes[topic_name] = node
            util.logger.info(f"Added listener for topic {topic_name}")

    def kill_listener(self, topic_name: str) -> None:
        """
        Remove the listener node for the specified topic and clean it up.
        """
        with self._lock:
            if topic_name in self.nodes:
                node = self.nodes.pop(topic_name)
                self.executor.remove_node(node)
                node.destroy()
                util.logger.info(f"Killed listener for topic {topic_name}")
            else:
                util.logger.error(f"Node for topic {topic_name} does not exist.")

    def shutdown(self):
        """
        Shutdown the executor and clean up all nodes.
        """
        with self._lock:
            for topic_name, node in list(self.nodes.items()):
                self.kill_listener(topic_name)

        self.executor.remove_node(self._graph)
        self._graph.shutdown()
        self.executor.shutdown()
        util.logger.info("ROSNodeManager shutdown complete.")

class GraphWatcher(Node):
    """
    A node that can initialize ROS graph monitoring, but only starts its executor when explicitly requested.
    """
    def __init__(self):
        super().__init__('graph_watcher')
        self.executor = None
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        if self.executor is not None:
            util.logger.warning("GraphWatcher already started.")
            return

        if not rclpy.ok():
            rclpy.init()

        try:
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self)
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
            util.logger.info("GraphWatcher started.")
        except Exception as e:
            util.logger.error(f"Failed to start GraphWatcher: {e}")
            self.executor = None


    def _spin(self):
        while rclpy.ok() and not self._stop_event.is_set():
            self.executor.spin_once(timeout_sec=0.1)

    def get_topic_names_and_types(self):
        return super().get_topic_names_and_types()

    def get_node_names_and_namespaces(self):
        return super().get_node_names_and_namespaces()

    def shutdown(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        if self.executor is not None:
            self.executor.shutdown()
        self.destroy_node()
        util.logger.info("GraphWatcher shutdown complete.")
