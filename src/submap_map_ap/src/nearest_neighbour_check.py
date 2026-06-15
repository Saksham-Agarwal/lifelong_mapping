#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2

class CostmapNeighbourFilter(Node):
    def __init__(self):
        super().__init__('costmap_neighbour_filter')

        # Subscribers
        self.inflated_sub = self.create_subscription(
            OccupancyGrid,
            '/inflated_local_region',
            self.inflated_callback,
            10
        )
        self.change_sub = self.create_subscription(
            OccupancyGrid,
            '/change/positive',
            self.change_callback,
            10
        )

        # Publisher for the filtered positive changes
        self.filtered_pub = self.create_publisher(
            OccupancyGrid,
            '/changes/positive_near_neighbour',
            10
        )

        # Cache for the latest inflated grid
        self.latest_inflated_msg = None
        
        # 40 cm radius at 5cm/block resolution = 8 blocks
        self.declare_parameter('search_radius_cells', 8.0)

        self.get_logger().info("Costmap Neighbour Filter Node has been started.")

    def inflated_callback(self, msg):
        """Cache the latest inflated local region."""
        self.latest_inflated_msg = msg

    def change_callback(self, change_msg):
        """Process filtering when the positive change map updates."""
        if self.latest_inflated_msg is None:
            self.get_logger().debug("Waiting for /inflated_local_region...")
            return

        inflated_msg = self.latest_inflated_msg
        radius_cells = self.get_parameter('search_radius_cells').value

        # --- Extract Grid A (Positive Change Map) ---
        res_A = change_msg.info.resolution
        orig_A_x = change_msg.info.origin.position.x
        orig_A_y = change_msg.info.origin.position.y
        w_A = change_msg.info.width
        h_A = change_msg.info.height
        data_A = np.array(change_msg.data, dtype=np.int8).reshape((h_A, w_A))

        # --- Extract Grid B (Inflated Map) ---
        res_B = inflated_msg.info.resolution
        orig_B_x = inflated_msg.info.origin.position.x
        orig_B_y = inflated_msg.info.origin.position.y
        w_B = inflated_msg.info.width
        h_B = inflated_msg.info.height
        data_B = np.array(inflated_msg.data, dtype=np.int8).reshape((h_B, w_B))

        # --- 1. Distance Transform on Inflated Map ---
        # Treat values >= 90 as lethal obstacles (0 for cv2, 255 for free space)
        binary_inflated = np.where(data_B >= 90, 0, 255).astype(np.uint8)
        
        # Calculate distance in pixels to the nearest 0 (obstacle)
        dist_map = cv2.distanceTransform(binary_inflated, cv2.DIST_L2, 5)

        # --- 2. Coordinate Mapping ---
        y_A_indices, x_A_indices = np.mgrid[0:h_A, 0:w_A]

        world_x = orig_A_x + (x_A_indices * res_A)
        world_y = orig_A_y + (y_A_indices * res_A)

        x_B_indices = np.floor((world_x - orig_B_x) / res_B).astype(int)
        y_B_indices = np.floor((world_y - orig_B_y) / res_B).astype(int)

        valid_mask = (
            (x_B_indices >= 0) & (x_B_indices < w_B) &
            (y_B_indices >= 0) & (y_B_indices < h_B)
        )

        # --- 3. Filter the Change Map ---
        # Initialize output grid as an exact copy of the positive changes
        out_data = data_A.copy()

        # Look up the distance to the nearest obstacle for every mapped point
        mapped_distances = dist_map[y_B_indices[valid_mask], x_B_indices[valid_mask]]

        # Mask where mapped points are within the 8-block radius (too close)
        too_close_mask = mapped_distances <= radius_cells

        # Create a full-size boolean mask for out_data
        clear_mask = np.zeros_like(data_A, dtype=bool)
        
        # Map the "too close" condition back into the valid overlap region
        overlap_too_close = np.zeros(np.count_nonzero(valid_mask), dtype=bool)
        overlap_too_close[too_close_mask] = True
        
        clear_mask[valid_mask] = overlap_too_close

        # Any positive change that was too close to an inflated obstacle becomes clear space (0)
        out_data[clear_mask] = 0

        # --- 4. Publish the Result ---
        out_msg = OccupancyGrid()
        out_msg.header = change_msg.header
        out_msg.info = change_msg.info  # Keeps the exact size of the change map
        out_msg.data = out_data.flatten().tolist()

        self.filtered_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CostmapNeighbourFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()