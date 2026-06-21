#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32
import message_filters
import numpy as np
import cv2  


class ClusterChangeDetector(Node):

    def __init__(self):
        super().__init__('cluster_change_detector')

        self.global_sub = message_filters.Subscriber(
            self, 
            OccupancyGrid, 
            '/robot_local_region'
        )
        self.aligned_local_sub = message_filters.Subscriber(
            self, 
            OccupancyGrid, 
            '/aligned_local_costmap'
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.global_sub, self.aligned_local_sub], 
            queue_size=10, 
            slop=0.2
        )
        self.sync.registerCallback(self.detect_changes)

        # Map Publishers
        self.positive_pub = self.create_publisher(OccupancyGrid, '/changes/positive', 10)
        self.negative_pub = self.create_publisher(OccupancyGrid, '/changes/negative', 10)
        
        # New Confidence Publisher
        self.confidence_pub = self.create_publisher(Float32, '/confidence', 10)

        self.get_logger().info("Cluster Change Detector with Confidence Heuristic Initialized.")

    def detect_changes(self, global_msg, local_msg):
        
        g_res = global_msg.info.resolution
        g_w, g_h = global_msg.info.width, global_msg.info.height
        l_w, l_h = local_msg.info.width, local_msg.info.height

        g_arr = np.array(global_msg.data, dtype=np.int8).reshape((g_h, g_w))
        l_arr = np.array(local_msg.data, dtype=np.int8).reshape((l_h, l_w))

        col_offset = int(round((local_msg.info.origin.position.x - global_msg.info.origin.position.x) / g_res))
        row_offset = int(round((local_msg.info.origin.position.y - global_msg.info.origin.position.y) / g_res))

        if col_offset < 0 or row_offset < 0 or col_offset + l_w > g_w or row_offset + l_h > g_h:
            return

        g_slice = g_arr[row_offset : row_offset + l_h, col_offset : col_offset + l_w]

        # 1. Extract solid obstacles as binary images (0 or 255)
        g_obs = np.where(g_slice > 80, 255, 0).astype(np.uint8)
        l_obs = np.where(l_arr > 80, 255, 0).astype(np.uint8)

        # NEW: Extract explicitly free space from the local map (e.g., 0 to 50)
        # This prevents triggering negative changes in unobserved (-1) areas.
        l_free = np.where((l_arr >= 0) & (l_arr < 50), 255, 0).astype(np.uint8)

        # 2. Create Tolerance Bands (Inflation)
        # A 5x5 kernel creates a ~12.5cm tolerance radius around every wall
        # If the local map is within 12.5cm of the global map, we consider them a "match"
        kernel_dilate = np.ones((5, 5), np.uint8)
        g_dilated = cv2.dilate(g_obs, kernel_dilate)
        l_dilated = cv2.dilate(l_obs, kernel_dilate)

        # 3. Apply the Heuristic Rules
        # Positive: Local obstacle exists, but NO global tolerance band surrounds it
        pos_img = np.where((l_obs > 0) & (g_dilated == 0), 255, 0).astype(np.uint8)
        
        # Negative: Global obstacle exists, NO local tolerance band surrounds it, 
        # AND the local sensor explicitly confirms the space is FREE.
        neg_img = np.where((g_obs > 0) & (l_dilated == 0) & (l_free > 0), 255, 0).astype(np.uint8)

        # 4. Filter remaining tiny noise clusters (Morphological Opening)
        kernel_clean = np.ones((3, 3), np.uint8)
        pos_clean = cv2.morphologyEx(pos_img, cv2.MORPH_OPEN, kernel_clean)
        neg_clean = cv2.morphologyEx(neg_img, cv2.MORPH_OPEN, kernel_clean)

        # 5. Calculate Confidence Score (Proportional Overlap)
        # How many local pixels landed safely inside the global tolerance band?
        matched_local = np.sum((l_obs > 0) & (g_dilated > 0))
        # How many global pixels landed safely inside the local tolerance band?
        matched_global = np.sum((g_obs > 0) & (l_dilated > 0))
        
        total_local = np.sum(l_obs > 0)
        total_global = np.sum(g_obs > 0)

        # Prevent division by zero if the robot is in a completely empty space
        total_pixels = total_local + total_global + 1e-6 
        
        # Calculate proportional confidence (0.0 to 100.0)
        confidence_score = ((matched_local + matched_global) / total_pixels) * 100.0

        # 6. Convert final masks back to ROS 0-100 format
        pos_data = np.where(pos_clean > 0, 100, 0).astype(np.int8).flatten().tolist()
        neg_data = np.where(neg_clean > 0, 100, 0).astype(np.int8).flatten().tolist()

        # 7. Publish
        self.publish_map(self.positive_pub, local_msg, pos_data, 0.06)
        self.publish_map(self.negative_pub, local_msg, neg_data, 0.07)

        # Publish Confidence
        conf_msg = Float32()
        conf_msg.data = float(confidence_score)
        self.confidence_pub.publish(conf_msg)


    def publish_map(self, publisher, reference_msg, data_array, z_offset):
        msg = OccupancyGrid()
        msg.header = reference_msg.header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = reference_msg.info
        
        msg.info.origin.position.z = z_offset 
        msg.data = data_array
        publisher.publish(msg)


def main():
    rclpy.init()
    node = ClusterChangeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()