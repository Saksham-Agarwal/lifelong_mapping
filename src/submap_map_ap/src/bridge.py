#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan

class QosBridge(Node):
    def __init__(self):
        super().__init__('qos_bridge')
        
        # 1. Define Best Effort QoS to match your LiDAR
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 2. Define Reliable QoS to satisfy slam_toolbox
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 3. Create the Publisher (Reliable)
        self.publisher_ = self.create_publisher(LaserScan, '/scan_reliable', reliable_qos)
        
        self.subscription = self.create_subscription(
            LaserScan, 
            '/scan', 
            self.scan_callback, 
            best_effort_qos
        )
        
        self.get_logger().info('QoS Bridge running: /scan (Best Effort) -> /scan_reliable (Reliable)')

    def scan_callback(self, msg):
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = QosBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
