#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2


class LocalMapInflater(Node):

    def __init__(self):
        super().__init__('local_map_inflater')

        # 1. Declare Parameters (Tune these for your specific rover chassis)
        self.declare_parameter('inscribed_radius_m', 0.3)  # Physical radius of the rover
        self.declare_parameter('inflation_radius_m', 1.0)  # Max distance to inflate
        self.declare_parameter('cost_scaling_factor', 3.0) # Steepness of the cost drop-off

        # 2. Setup Subscriber and Publisher
        self.sub = self.create_subscription(
            OccupancyGrid,
            '/robot_local_region',
            self.map_callback,
            10
        )
        
        self.pub = self.create_publisher(
            OccupancyGrid, 
            '/inflated_local_region', 
            10
        )

        self.get_logger().info("Local Map Inflater Node Initialized.")

    def apply_nav2_inflation(self, map_slice, resolution, inscribed_radius_m, inflation_radius_m, cost_scaling_factor):
        """
        Inflates a numpy-based occupancy grid using Nav2's exponential decay logic.
        """
        # Create a binary map for the Distance Transform (Obstacles = 0, Free/Unknown = 255)
        # Any value >= 80 is treated as a lethal obstacle.
        binary_map = np.where(map_slice >= 90, 0, 255).astype(np.uint8)

        # Calculate pixel distances to the nearest obstacle (0)
        dist_pixels = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)
        
        # Convert pixel distance to meters
        dist_meters = dist_pixels * resolution

        # Initialize the float costmap (0-255 scale)
        inflated_costmap = np.zeros_like(map_slice, dtype=np.float32)

        # Condition A: Inside inscribed radius (Lethal / Inscribed) -> Cost 253
        lethal_mask = dist_meters <= inscribed_radius_m
        inflated_costmap[lethal_mask] = 253.0

        # Condition B: Exponential decay area
        decay_mask = (dist_meters > inscribed_radius_m) & (dist_meters <= inflation_radius_m)
        
        # Apply the Nav2 equation to the decay area
        decay_costs = 252.0 * np.exp(-1.0 * cost_scaling_factor * (dist_meters[decay_mask] - inscribed_radius_m))
        inflated_costmap[decay_mask] = decay_costs

        # Preserve the original solid obstacles as 254 (Nav2 standard for lethal obstacles)
        inflated_costmap[map_slice >= 80] = 254.0

        # Convert back to standard ROS OccupancyGrid 0-100 scale
        # 254 gets mapped to 100, 253 gets mapped to ~99, etc.
        ros_100_map = (inflated_costmap / 254.0) * 100.0
        
        # Clip to ensure bounds and convert to int8
        np.clip(ros_100_map, 0, 100, out=ros_100_map)
        
        # Re-apply unknown space (-1) if it wasn't inflated over
        # If original was -1 and it didn't get inflated by a nearby wall, keep it -1
        final_map = ros_100_map.astype(np.int8)
        unknown_mask = (map_slice == -1) & (inflated_costmap == 0)
        final_map[unknown_mask] = -1

        return final_map

    def map_callback(self, msg):
        # Fetch latest parameters
        inscribed_radius = self.get_parameter('inscribed_radius_m').value
        inflation_radius = self.get_parameter('inflation_radius_m').value
        cost_scaling = self.get_parameter('cost_scaling_factor').value

        # Extract map metadata
        w = msg.info.width
        h = msg.info.height
        res = msg.info.resolution

        if w == 0 or h == 0:
            return

        # Convert OccupancyGrid data tuple to 2D NumPy array
        map_arr = np.array(msg.data, dtype=np.int8).reshape((h, w))

        # Apply inflation
        inflated_arr = self.apply_nav2_inflation(
            map_slice=map_arr,
            resolution=res,
            inscribed_radius_m=inscribed_radius,
            inflation_radius_m=inflation_radius,
            cost_scaling_factor=cost_scaling
        )

        # Create new OccupancyGrid message
        inflated_msg = OccupancyGrid()
        inflated_msg.header = msg.header
        inflated_msg.info = msg.info
        
        # Flatten the 2D array back to a 1D list for the ROS message
        inflated_msg.data = inflated_arr.flatten().tolist()

        # Publish
        self.pub.publish(inflated_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LocalMapInflater()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()