#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('AMCL_Confidence_subscriber')
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_pose',
            self.listener_callback,
            10)
        self.subscription 

        self.get_logger().info("AMCL Confidence Subscriber Node has been started.")
    def listener_callback(self, msg):
        
        
        self.get_logger().info('Uncertainty in x axis  : "%s"' % msg.pose.covariance[0])
        self.get_logger().info('Uncertainty in y axis  : "%s"' % msg.pose.covariance[7])
        self.get_logger().info('Uncertainty in yaw axis: "%s"' % msg.pose.covariance[35])

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()