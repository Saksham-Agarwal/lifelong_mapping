#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseWithCovarianceStamped

class MinimalSubscriber(Node):

    def __init__(self):

        super().__init__('AMCL_Confidence_subscriber')
        
        self.declare_parameter('angle_relevance', 0.5)

        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_pose',
            self.listener_callback,
            10)
        
        self.pub = self.create_publisher(Float32, 'amcl_confidence', 10)
        self.subscription 
        self.confidence = 0.0
        self.get_logger().info("AMCL Confidence Subscriber Node has been started.")
    def listener_callback(self, msg):
        
        
        # self.get_logger().info('Uncertainty in x axis  : "%s"' % msg.pose.covariance[0])
        # self.get_logger().info('Uncertainty in y axis  : "%s"' % msg.pose.covariance[7])
        # self.get_logger().info('Uncertainty in yaw axis: "%s"' % msg.pose.covariance[35])
        
        k = self.get_parameter('angle_relevance').value

        pose_uncertainty = (msg.pose.covariance[0] + msg.pose.covariance[7])
        angle_uncertainty = msg.pose.covariance[35]
        total_uncertainty = pose_uncertainty + k * angle_uncertainty
                
        self.get_logger().info('pose : "%s"' % pose_uncertainty)
        self.get_logger().info('angle  : "%s"' % angle_uncertainty)
        self.get_logger().info('total: "%s"' % total_uncertainty)
        
        self.confidence = math.exp(-total_uncertainty)  
        self.pub.publish(Float32(data=self.confidence))


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()