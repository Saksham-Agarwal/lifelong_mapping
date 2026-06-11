#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from tf2_ros import Buffer, TransformListener, TransformException


class CostmapComparator(Node):

    def __init__(self):
        super().__init__('costmap_comparator')

        self.costmap = None
        self.side_length = 1.0  # meters

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.costmap_callback,
            map_qos
        )

        self.region_pub = self.create_publisher(
            OccupancyGrid,
            '/robot_local_region',
            10
        )

        # TF listener — tracks the smooth combination of AMCL + Odometry
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer drives the update — 20Hz is plenty fast for a real-time feel
        self.timer = self.create_timer(0.05, self.process_region) 

    def costmap_callback(self, msg):
        self.costmap = msg
        self.get_logger().info(
            f"Received global costmap: {msg.info.width}x{msg.info.height}"
        )

    def process_region(self):
        if self.costmap is None:
            return

        # Get the latest smooth robot pose from the TF tree
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_footprint',
                rclpy.time.Time() # rclpy.time.Time() correctly requests the latest available transform
            )
        except TransformException as ex:
            self.get_logger().debug(f"TF not ready yet: {ex}")
            return
        except Exception as e:
            self.get_logger().error(f"Unexpected error during TF lookup: {e}")
            return

        robot_x = transform.transform.translation.x
        robot_y = transform.transform.translation.y

        self.extract_region(robot_x, robot_y)

    def extract_region(self, robot_x, robot_y):
        resolution = self.costmap.info.resolution
        origin_x = self.costmap.info.origin.position.x
        origin_y = self.costmap.info.origin.position.y

        # Find where the robot is in the global array
        center_col = int((robot_x - origin_x) / resolution)
        center_row = int((robot_y - origin_y) / resolution)

        half_cells = int((self.side_length / 2.0) / resolution)

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
                    current_row.append(data[row * width + col])
                else:
                    current_row.append(-1) # Unknown space for out of bounds
            region.append(current_row)

        region_height = len(region)
        region_width = len(region[0])

        region_msg = OccupancyGrid()
        region_msg.header.stamp = self.get_clock().now().to_msg()
        region_msg.header.frame_id = 'map'
        region_msg.info.resolution = resolution
        region_msg.info.width = region_width
        region_msg.info.height = region_height
        

        region_msg.info.origin.position.x = origin_x + (min_col * resolution)
        region_msg.info.origin.position.y = origin_y + (min_row * resolution)
        region_msg.info.origin.position.z = 0.0
        region_msg.info.origin.orientation.w = 1.0
        
        # Flatten the 2D array back to 1D for publishing
        region_msg.data = [cell for row_data in region for cell in row_data]

        self.region_pub.publish(region_msg)


def main():
    rclpy.init()
    node = CostmapComparator()
    node.get_logger().info("Costmap Comparator started.")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()