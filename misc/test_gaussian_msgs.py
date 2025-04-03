import rclpy
from rclpy.node import Node
from gaussian_interface.msg import SingleGaussian


class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')

        self.publisher = self.create_publisher(
            SingleGaussian,
            'gaussian',
            10
        )
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = SingleGaussian()
        msg.opacity = 1.0
        msg.scale.x = 0.0
        msg.scale.y = 0.0
        msg.scale.z = 0.0

        msg.xyz.x = 0.0
        msg.xyz.y = 0.0
        msg.xyz.z = 0.0

        msg.rotation.x = 0.0
        msg.rotation.y = 0.0
        msg.rotation.z = 0.0
        msg.rotation.w = 1.0

        # publish
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg)
    
def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    minimal_publisher.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()