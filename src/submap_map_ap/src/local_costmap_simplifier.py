#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


class LocalCostmapSimplifier(Node):

    def __init__(self):
        super().__init__('your_costmap_simplifier_node')
    
        # Declare the parameter with a default value of 98
        self.declare_parameter('obstacle_threshold', 98)
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.create_subscription(
            OccupancyGrid,
            '/local_costmap/costmap',
            self.costmap_callback,
            map_qos
        )

        self.region_pub = self.create_publisher(
            OccupancyGrid,
            '/simplified_local_costmap',
            10
        )
        
        self.get_logger().info("Local Costmap Simplifier started.")


    def costmap_callback(self, msg):
        l_data = msg.data
        # Fetch the latest parameter value dynamically
        current_thresh = self.get_parameter('obstacle_threshold').value
        
        simplified_local_data = []
        for value in l_data:
            if value == -1:
                simplified_local_data.append(-1)
            elif value > current_thresh:  # Use the dynamic variable here
                simplified_local_data.append(100)
            else:
                simplified_local_data.append(0)  

        # 1. Create the new OccupancyGrid message
        simplified_msg = OccupancyGrid()
        

        simplified_msg.header = msg.header
        simplified_msg.info = msg.info

        simplified_msg.info.origin.position.z = 0.05 
        
        simplified_msg.data = simplified_local_data
        
        self.region_pub.publish(simplified_msg)


def main():
    rclpy.init()
    node = LocalCostmapSimplifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()