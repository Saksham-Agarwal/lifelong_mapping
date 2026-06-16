#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import math
import cv2 

class CostmapComparator(Node):
    def __init__(self):
        super().__init__('costmap_comparator')
        
        # Subscribers
        self.sub_inflated = self.create_subscription(
            OccupancyGrid, '/inflated_local_region', self.inflated_cb, 10)
        self.sub_simplified = self.create_subscription(
            OccupancyGrid, '/simplified_local_costmap', self.simplified_cb, 10)
        self.sub_robot = self.create_subscription(
            OccupancyGrid, '/robot_local_region', self.robot_cb, 10)
            
        # Publisher
        self.pub_changes = self.create_publisher(
            OccupancyGrid, '/changes/negative_nearest_neighbour', 10)
            
        # State variables
        self.grid_inflated = None
        self.grid_simplified = None
        self.grid_robot = None
        
        # Timer to process the grids at a fixed rate
        self.timer = self.create_timer(0.2, self.process_grids)

    def inflated_cb(self, msg):
        self.grid_inflated = msg

    def simplified_cb(self, msg):
        self.grid_simplified = msg

    def robot_cb(self, msg):
        self.grid_robot = msg

    def world_to_grid(self, wx, wy, info):
        """Convert world coordinates to grid indices."""
        gx = np.floor((wx - info.origin.position.x) / info.resolution).astype(int)
        gy = np.floor((wy - info.origin.position.y) / info.resolution).astype(int)
        return gx, gy

    def grid_to_world(self, gx, gy, info):
        """Convert grid indices to world coordinates."""
        wx = (gx * info.resolution) + info.origin.position.x
        wy = (gy * info.resolution) + info.origin.position.y
        return wx, wy

    def process_grids(self):
        # Ensure we have all necessary data before processing
        if not all([self.grid_inflated, self.grid_simplified, self.grid_robot]):
            return

        # 1. Extract data and info
        info_inf = self.grid_inflated.info
        info_simp = self.grid_simplified.info
        info_rob = self.grid_robot.info

        data_inf = np.array(self.grid_inflated.data).reshape((info_inf.height, info_inf.width))
        data_simp = np.array(self.grid_simplified.data).reshape((info_simp.height, info_simp.width))
        data_rob = np.array(self.grid_robot.data).reshape((info_rob.height, info_rob.width))

        # Output grid (initialized to 0, matching the inflated region's dimensions)
        out_data = np.zeros_like(data_inf)

        # 2. Find unexplored cells (-1) in /inflated_local_region
        gy_inf, gx_inf = np.where(data_inf == -1)
        
        if len(gx_inf) == 0:
            self.publish_grid(out_data, info_inf)
            return

        # 3. Convert these indices to world coordinates
        wx, wy = self.grid_to_world(gx_inf, gy_inf, info_inf)

        # 4. Map world coordinates to /simplified_local_costmap indices
        gx_simp, gy_simp = self.world_to_grid(wx, wy, info_simp)

        # Filter out bounds for simplified grid
        valid_simp_mask = (gx_simp >= 0) & (gx_simp < info_simp.width) & \
                          (gy_simp >= 0) & (gy_simp < info_simp.height)
        
        valid_indices = np.where(valid_simp_mask)[0]
        free_in_simp_mask = data_simp[gy_simp[valid_indices], gx_simp[valid_indices]] == 0
        
        # Final candidate world coordinates
        candidate_indices = valid_indices[free_in_simp_mask]
        cand_wx = wx[candidate_indices]
        cand_wy = wy[candidate_indices]

        cand_gx_inf = gx_inf[candidate_indices]
        cand_gy_inf = gy_inf[candidate_indices]

        # 5. Use OpenCV distanceTransform for the 10 cm nearest-neighbor check
        radius_m = 0.05
        radius_cells = radius_m / info_rob.resolution

        # Create binary image: Unexplored (-1) = 0, Everything else = 255
        # distanceTransform calculates distance to the nearest 0
        binary_rob = np.full((info_rob.height, info_rob.width), 255, dtype=np.uint8)
        binary_rob[data_rob == -1] = 0

        # Calculate Euclidean distance (DIST_L2) to nearest unexplored cell for every pixel
        dist_transform = cv2.distanceTransform(binary_rob, cv2.DIST_L2, 5)

        # Map candidates to robot_local_region
        gx_rob, gy_rob = self.world_to_grid(cand_wx, cand_wy, info_rob)

        # Filter bounds to prevent indexing errors
        valid_rob_mask = (gx_rob >= 0) & (gx_rob < info_rob.width) & \
                         (gy_rob >= 0) & (gy_rob < info_rob.height)
        
        valid_cand_idx = np.where(valid_rob_mask)[0]
        
        # Retrieve the distance to nearest unexplored cell for our candidate pixels
        candidate_distances = dist_transform[gy_rob[valid_cand_idx], gx_rob[valid_cand_idx]]

        # 6. Evaluate condition: Is the nearest unexplored cell within 10 cm?
        condition_met_mask = candidate_distances <= radius_cells

        # Get final indices for the output grid
        final_indices = valid_cand_idx[condition_met_mask]
        final_gx_inf = cand_gx_inf[final_indices]
        final_gy_inf = cand_gy_inf[final_indices]

        # Mark as occupied (100) in output
        out_data[final_gy_inf, final_gx_inf] = 100

        # 7. Publish the resulting map
        self.publish_grid(out_data, info_inf)

    def publish_grid(self, data_array, info):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.grid_inflated.header.frame_id 
        msg.info = info
        msg.data = data_array.flatten().astype(np.int8).tolist()
        
        self.pub_changes.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CostmapComparator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()