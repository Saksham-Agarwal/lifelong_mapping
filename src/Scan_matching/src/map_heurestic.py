#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from std_msgs.msg import Float32

class map_heurestic(Node):
    def __init__(self):
        super().__init__('map_heuristic_node')
        
        # QoS Profile: Reliable and Transient Local (Latching)
        latching_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.perimeter_subscriber = self.create_subscription(
            Float32,
            '/wall_perimeter_length',
            self.perimeter_callback,
            latching_qos
        )

        self.perimeter_length = Float32()
    
    def perimeter_callback(self, msg):
        self.perimeter_length = msg.data
        # self.get_logger().info(f"Received wall perimeter length: {msg.data:.2f} meters")
    