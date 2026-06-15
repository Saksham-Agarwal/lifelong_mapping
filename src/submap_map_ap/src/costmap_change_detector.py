#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np

class CostmapChangeDetector(Node):
    def __init__(self):
        super().__init__('costmap_change_detector')

        # Subscribers
        self.inflated_sub = self.create_subscription(
            OccupancyGrid,
            '/inflated_local_region',
            self.inflated_callback,
            10
        )
        self.simplified_sub = self.create_subscription(
            OccupancyGrid,
            '/simplified_local_costmap',
            self.simplified_callback,
            10
        )

        # Publishers for isolated changes
        self.pos_pub = self.create_publisher(
            OccupancyGrid,
            '/change/positive',
            10
        )
        self.neg_pub = self.create_publisher(
            OccupancyGrid,
            '/change/negative',
            10
        )

        # Cache for the latest inflated grid
        self.latest_inflated_msg = None

        self.get_logger().info("Costmap Change Detector Node has been started.")

    def inflated_callback(self, msg):
        """Cache the latest inflated local region."""
        self.latest_inflated_msg = msg

    def simplified_callback(self, simplified_msg):
        """Process subtraction and split when the primary map updates."""
        if self.latest_inflated_msg is None:
            self.get_logger().debug("Waiting for /inflated_local_region...")
            return

        inflated_msg = self.latest_inflated_msg

        # --- Extract Grid A (Simplified - The Target Grid) ---
        res_A = simplified_msg.info.resolution
        orig_A_x = simplified_msg.info.origin.position.x
        orig_A_y = simplified_msg.info.origin.position.y
        width_A = simplified_msg.info.width
        height_A = simplified_msg.info.height
        
        # Load A data into 2D numpy array
        data_A = np.array(simplified_msg.data, dtype=np.int8).reshape((height_A, width_A))

        # --- Extract Grid B (Inflated - The Subtrahend Grid) ---
        res_B = inflated_msg.info.resolution
        orig_B_x = inflated_msg.info.origin.position.x
        orig_B_y = inflated_msg.info.origin.position.y
        width_B = inflated_msg.info.width
        height_B = inflated_msg.info.height
        
        # Load B data into 2D numpy array
        data_B = np.array(inflated_msg.data, dtype=np.int8).reshape((height_B, width_B))

        # --- Coordinate Mapping ---
        y_A_indices, x_A_indices = np.mgrid[0:height_A, 0:width_A]

        world_x = orig_A_x + (x_A_indices * res_A)
        world_y = orig_A_y + (y_A_indices * res_A)

        x_B_indices = np.floor((world_x - orig_B_x) / res_B).astype(int)
        y_B_indices = np.floor((world_y - orig_B_y) / res_B).astype(int)

        valid_mask = (
            (x_B_indices >= 0) & (x_B_indices < width_B) &
            (y_B_indices >= 0) & (y_B_indices < height_B)
        )

        # --- Base Grids Setup ---
        # Initialize output grids with 0 (no change), but copy over -1 (unknown) from Grid A
        pos_out_data = np.zeros_like(data_A)
        pos_out_data[data_A == -1] = -1
        
        neg_out_data = np.zeros_like(data_A)
        neg_out_data[data_A == -1] = -1

        # Extract overlapping valid pixels from both grids
        valid_A_pixels = data_A[valid_mask]
        valid_B_pixels = data_B[y_B_indices[valid_mask], x_B_indices[valid_mask]]

        # Only process cells where BOTH grids have known data
        calc_mask = (valid_A_pixels != -1) & (valid_B_pixels != -1)

        # Convert to int16 to prevent overflow during subtraction
        calc_A = valid_A_pixels[calc_mask].astype(np.int16)
        calc_B = valid_B_pixels[calc_mask].astype(np.int16)

        # Perform the raw subtraction: Simplified - Inflated
        diff = calc_A - calc_B

        # --- Isolate Positive and Negative Changes ---
        # Positive change: diff > 0. Otherwise 0.
        pos_values = np.where(diff > 0, diff, 0).astype(np.int8)
        
        # Negative change: diff < 0. Convert to positive magnitude [0, 100].
        neg_values = np.where(diff < 0, -diff, 0).astype(np.int8)

        # Prepare sub-arrays for the overlapping region
        overlap_pos = np.zeros_like(valid_A_pixels)
        overlap_pos[valid_A_pixels == -1] = -1 # Keep unknowns
        overlap_pos[calc_mask] = pos_values
        
        overlap_neg = np.zeros_like(valid_A_pixels)
        overlap_neg[valid_A_pixels == -1] = -1 # Keep unknowns
        overlap_neg[calc_mask] = neg_values

        # Inject the isolated overlap data back into the full-sized grids
        pos_out_data[valid_mask] = overlap_pos
        neg_out_data[valid_mask] = overlap_neg

        # --- Publish Positive Changes ---
        pos_msg = OccupancyGrid()
        pos_msg.header = simplified_msg.header
        pos_msg.info = simplified_msg.info
        pos_msg.data = pos_out_data.flatten().tolist()
        self.pos_pub.publish(pos_msg)

        # --- Publish Negative Changes ---
        neg_msg = OccupancyGrid()
        neg_msg.header = simplified_msg.header
        neg_msg.info = simplified_msg.info
        neg_msg.data = neg_out_data.flatten().tolist()
        self.neg_pub.publish(neg_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CostmapChangeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()