#!/usr/bin/env python3
"""Stream a Gaussian-splat PLY over a ROS 2 ``GaussianArray`` topic."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rclpy
from gaussian_interface.msg import GaussianArray, SingleGaussian
from rclpy.duration import Duration
from rclpy.node import Node

from rosplat.core.gaussian_representation import GaussianData, from_ply


def make_gaussian_batch(
    gaussian_data: GaussianData,
    start: int,
    end: int,
    *,
    refresh: bool,
) -> GaussianArray:
    """Serialize the half-open range ``[start, end)`` into one ROS message."""
    message = GaussianArray()
    message.refresh = refresh
    message.gaussians = [SingleGaussian() for _ in range(end - start)]

    for output_index, source_index in enumerate(range(start, end)):
        gaussian = message.gaussians[output_index]
        gaussian.xyz = gaussian_data.xyz[source_index].tolist()
        gaussian.rotation = gaussian_data.rot[source_index].tolist()
        gaussian.scale = gaussian_data.scale[source_index].tolist()
        gaussian.opacity = int(
            np.clip(gaussian_data.opacity[source_index, 0] * 255.0, 0, 255)
        )
        gaussian.spherical_harmonics = gaussian_data.sh[source_index].tolist()

    return message


class GaussianPublisher(Node):
    """Publish one PLY scene in bounded batches, then stop."""

    def __init__(
        self,
        gaussian_data: GaussianData,
        *,
        topic: str,
        batch_size: int,
        rate: float,
        wait_for_subscriber: bool,
    ) -> None:
        super().__init__("gaussian_publisher")
        self.gaussian_data = gaussian_data
        self.publisher = self.create_publisher(GaussianArray, topic, 10)
        self.topic = topic
        self.batch_size = batch_size
        self.wait_for_subscriber = wait_for_subscriber
        self.next_index = 0
        self.finished = False
        self._waiting_logged = False
        self.timer = self.create_timer(1.0 / rate, self.publish_next_batch)

        sh_degree = int((gaussian_data.sh_dim // 3) ** 0.5 - 1)
        self.get_logger().info(
            f"Loaded {len(gaussian_data):,} Gaussians at SH degree {sh_degree}; "
            f"streaming batches of {batch_size:,} on {topic} at up to {rate:g} Hz."
        )

    def publish_next_batch(self) -> None:
        if self.wait_for_subscriber and self.publisher.get_subscription_count() == 0:
            if not self._waiting_logged:
                self.get_logger().info(f"Waiting for a subscriber on {self.topic}...")
                self._waiting_logged = True
            return

        start = self.next_index
        end = min(start + self.batch_size, len(self.gaussian_data))
        started_at = time.perf_counter()
        message = make_gaussian_batch(
            self.gaussian_data,
            start,
            end,
            refresh=start == 0,
        )
        self.publisher.publish(message)
        self.next_index = end

        self.get_logger().info(
            f"Published {start:,}:{end:,} ({end / len(self.gaussian_data):.1%}) "
            f"in {time.perf_counter() - started_at:.3f}s."
        )

        if end == len(self.gaussian_data):
            self.finished = True
            self.timer.cancel()
            self.get_logger().info("Finished streaming the PLY scene.")


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply-path", "--ply_path", required=True, dest="ply_path")
    parser.add_argument("--topic", default="/gaussian_test")
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Publish immediately instead of waiting for a subscriber.",
    )
    cli_args, ros_args = parser.parse_known_args(args)

    if cli_args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if cli_args.rate <= 0:
        parser.error("--rate must be positive")

    gaussian_data = from_ply(cli_args.ply_path)
    rclpy.init(args=ros_args)
    node = GaussianPublisher(
        gaussian_data,
        topic=cli_args.topic,
        batch_size=cli_args.batch_size,
        rate=cli_args.rate,
        wait_for_subscriber=not cli_args.no_wait,
    )

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.finished:
            acknowledged = node.publisher.wait_for_all_acked(Duration(seconds=30.0))
            if acknowledged:
                node.get_logger().info("All published batches were acknowledged.")
            else:
                node.get_logger().warning(
                    "Timed out waiting for all published batches to be acknowledged."
                )
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
