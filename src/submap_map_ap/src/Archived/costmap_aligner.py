#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import message_filters
import numpy as np
import cv2


class CostmapCrossCorrelator(Node):

    def __init__(self):
        super().__init__('costmap_cross_correlator')

        self.global_sub = message_filters.Subscriber(
            self, 
            OccupancyGrid, 
            '/inflated_local_region'
        )
        self.local_sub = message_filters.Subscriber(
            self, 
            OccupancyGrid, 
            '/simplified_local_costmap'
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.global_sub, self.local_sub], 
            queue_size=10, 
            slop=0.2
        )
        self.sync.registerCallback(self.align_callback)

        self.aligned_pub = self.create_publisher(
            OccupancyGrid, 
            '/aligned_local_costmap', 
            10
        )

        # Rotation search range: -15 to +15 degrees in 1-degree steps
        # Widen this if misalignment is larger
        self.angle_range = np.arange(-3, 3, 0.5)

        self.get_logger().info("Costmap Cross-Correlator (CCORR + Rotation) Initialized.")

    def rotate_image(self, img, angle_deg):
        """Rotate image around its center, keeping same canvas size."""
        h, w = img.shape
        cx, cy = w / 2.0, h / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)
        return rotated

    def align_callback(self, global_msg, local_msg):
        if global_msg.info.resolution != local_msg.info.resolution:
            return

        g_w, g_h = global_msg.info.width, global_msg.info.height
        l_w, l_h = local_msg.info.width, local_msg.info.height

        if l_h > g_h or l_w > g_w:
            return

        # 1. Convert to Numpy Arrays
        g_img = np.array(global_msg.data, dtype=np.float32).reshape((g_h, g_w))
        l_img = np.array(local_msg.data, dtype=np.float32).reshape((l_h, l_w))

        # 2. Binarize Maps
        g_bin = np.where(g_img >= 80, 1.0, 0.0).astype(np.float32)
        l_bin = np.where(l_img >= 80, 1.0, 0.0).astype(np.float32)

        # 3. Smooth global for gravity-well effect, keep local crisp
        g_smooth = cv2.GaussianBlur(g_bin, (15, 15), 0)
        l_crisp  = cv2.GaussianBlur(l_bin, (3, 3), 0)

        # 4. Search over rotation angles
        best_score = -np.inf
        best_angle = 0.0
        best_loc   = (0, 0)

        for angle in self.angle_range:
            # Rotate the local map candidate
            l_rotated = self.rotate_image(l_crisp, angle)

            # Skip if rotated template is too large for global
            if l_rotated.shape[0] > g_h or l_rotated.shape[1] > g_w:
                continue

            res = cv2.matchTemplate(g_smooth, l_rotated, cv2.TM_CCORR_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_score:
                best_score = max_val
                best_angle = angle
                best_loc   = max_loc

        best_x_idx, best_y_idx = best_loc

        self.get_logger().info(
            f"Best alignment — angle: {best_angle:.1f}°, "
            f"score: {best_score:.4f}, loc: {best_loc}"
        )

        # 5. Calculate new real-world origin based on best match location
        new_origin_x = global_msg.info.origin.position.x + (best_x_idx * global_msg.info.resolution)
        new_origin_y = global_msg.info.origin.position.y + (best_y_idx * global_msg.info.resolution)

        # 6. Convert best_angle (degrees, cv2 convention) to quaternion for ROS
        # cv2 rotates counter-clockwise for positive angles, which matches ROS convention
        angle_rad = np.deg2rad(best_angle)
        qz = np.sin(angle_rad / 2.0)
        qw = np.cos(angle_rad / 2.0)

        # 7. Publish aligned map
        aligned_msg = OccupancyGrid()
        aligned_msg.header = global_msg.header  # use map frame, not odom frame
        aligned_msg.info = local_msg.info

        aligned_msg.info.origin.position.x = new_origin_x
        aligned_msg.info.origin.position.y = new_origin_y
        aligned_msg.info.origin.position.z = 0.1
        aligned_msg.info.origin.orientation.x = 0.0
        aligned_msg.info.origin.orientation.y = 0.0
        aligned_msg.info.origin.orientation.z = qz
        aligned_msg.info.origin.orientation.w = qw

        aligned_msg.data = local_msg.data

        self.aligned_pub.publish(aligned_msg)


def main():
    rclpy.init()
    node = CostmapCrossCorrelator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()