#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import message_filters
import numpy as np
import cv2  # <-- Don't forget to import OpenCV


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

        self.positive_pub = self.create_publisher(OccupancyGrid, '/changes/positive', 10)
        self.negative_pub = self.create_publisher(OccupancyGrid, '/changes/negative', 10)

        self.get_logger().info("Cluster Change Detector with Noise Filtering Initialized.")

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

        # 1. Calculate Raw Pixel Differences
        positive_mask = (l_arr > 80) & (g_slice < 20)
        negative_mask = (l_arr < 20) & (g_slice > 80)

        # 2. Convert to OpenCV Image Format (0 to 255)
        pos_img = np.where(positive_mask, 255, 0).astype(np.uint8)
        neg_img = np.where(negative_mask, 255, 0).astype(np.uint8)

        # 3. Cluster Noise Filtering (Morphological Opening)
        # A 3x3 kernel means "destroy any object that is thinner than 3 pixels".
        # Assuming a 0.05m resolution, 3 pixels = 15cm. 
        # This deletes misaligned wall edges but keeps objects larger than 15cm.
        kernel = np.ones((3, 3), np.uint8)
        
        pos_clean = cv2.morphologyEx(pos_img, cv2.MORPH_OPEN, kernel)
        neg_clean = cv2.morphologyEx(neg_img, cv2.MORPH_OPEN, kernel)

        # 4. Convert Cleaned Clusters back to ROS Data
        pos_data = np.where(pos_clean > 0, 100, 0).astype(np.int8).flatten().tolist()
        neg_data = np.where(neg_clean > 0, 100, 0).astype(np.int8).flatten().tolist()

        # 5. Publish
        self.publish_map(self.positive_pub, local_msg, pos_data, 0.06)
        self.publish_map(self.negative_pub, local_msg, neg_data, 0.07)


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