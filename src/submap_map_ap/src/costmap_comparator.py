#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped


class CostmapComparator(Node):

    def __init__(self):
        super().__init__('costmap_comparator')

        self.costmap = None
        self.robot_pose = None

        self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.costmap_callback,
            10
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )

        self.region_pub = self.create_publisher(
            OccupancyGrid,
            '/robot_local_region',
            10
        )

        self.side_length = 1.0  # meters

    def pose_callback(self, msg):
        self.robot_pose = msg.pose.pose
        self.get_logger().info(f"Received robot pose: ({self.robot_pose.position.x}, "
                         f"{self.robot_pose.position.y})")


    def costmap_callback(self, msg):
        self.costmap = msg
        self.get_logger().info(f"Received costmap: {msg.info.width}x{msg.info.height} "
                         f"resolution: {msg.info.resolution}")
        if self.robot_pose is not None:
            self.extract_region(self.side_length)

    def extract_region(self, side_length_m):

        if self.costmap is None or self.robot_pose is None:
            return

        resolution = self.costmap.info.resolution
        origin_x = self.costmap.info.origin.position.x
        origin_y = self.costmap.info.origin.position.y

        robot_x = self.robot_pose.position.x
        robot_y = self.robot_pose.position.y

        center_col = int((robot_x - origin_x) / resolution)
        center_row = int((robot_y - origin_y) / resolution)

        half_cells = int((side_length_m / 2.0) / resolution)

        min_col = center_col - half_cells
        max_col = center_col + half_cells

        min_row = center_row - half_cells
        max_row = center_row + half_cells

        width = self.costmap.info.width
        height = self.costmap.info.height
        data = self.costmap.data

        region = []

        for row in range(min_row, max_row + 1):
            current_row = []

            for col in range(min_col, max_col + 1):

                if 0 <= row < height and 0 <= col < width:
                    idx = row * width + col
                    current_row.append(data[idx])
                else:
                    # Unknown space
                    current_row.append(255)

            region.append(current_row)

        region_height = len(region)
        region_width = len(region[0])

        flattened_region = [
            cell
            for row_data in region
            for cell in row_data
        ]

        region_msg = OccupancyGrid()

        region_msg.header.stamp = self.get_clock().now().to_msg()
        region_msg.header.frame_id = "map"

        region_msg.info.resolution = resolution
        region_msg.info.width = region_width
        region_msg.info.height = region_height

        region_msg.info.origin.position.x = (
            robot_x - side_length_m / 2.0
        )
        region_msg.info.origin.position.y = (
            robot_y - side_length_m / 2.0
        )
        region_msg.info.origin.position.z = 0.0

        region_msg.info.origin.orientation.x = 0.0
        region_msg.info.origin.orientation.y = 0.0
        region_msg.info.origin.orientation.z = 0.0
        region_msg.info.origin.orientation.w = 1.0

        region_msg.data = flattened_region
        self.get_logger().info(
           f"Publishing region {region_width}x{region_height}"
        )
        self.region_pub.publish(region_msg)

        self.get_logger().info(
            f"Robot cell: ({center_row}, {center_col})"
        )

        self.get_logger().info(
            f"Published region: "
            f"{region_width}x{region_height}"
        )


def main():
    rclpy.init()

    node = CostmapComparator()
    node.get_logger().info("Costmap Comparator node started 2.")  
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()