#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from submap_map_ap.msg import MapGridDefinition
from std_msgs.msg import String

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class RobotGridTracker(Node):
    def __init__(self):
        super().__init__('robot_grid_tracker')

        latch_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Subscribe to the grid boundaries
        self.grid_def_sub = self.create_subscription(MapGridDefinition, '/map_grid_definitions', self.grid_def_callback, latch_qos)
        
        # Publish the current grid the robot is in
        self.current_grid_pub = self.create_publisher(String, '/robot_current_grid', 10)

        # TF2 Setup for finding the robot's position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to check location every 1 second
        self.timer = self.create_timer(1.0, self.check_robot_location)
        
        self.grid_definitions = []
        self.current_grid = -1

    def grid_def_callback(self, msg):
        self.grid_definitions = msg.grids
        self.get_logger().info("Received Grid Boundaries.")

    def check_robot_location(self):
        if not self.grid_definitions:
            return

        try:
            # Look up the transform from 'map' to 'base_link' (or 'base_footprint')
            # Adjust 'base_link' to match your robot's actual frame ID if necessary
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time())
                
            bot_x = t.transform.translation.x
            bot_y = t.transform.translation.y

            found_grid = -1

            # Check which grid bounding box contains the robot's point
            for grid in self.grid_definitions:
                if (grid.min_x <= bot_x <= grid.max_x) and (grid.min_y <= bot_y <= grid.max_y):
                    found_grid = grid.grid_id
                    break
            
            if found_grid != -1 and found_grid != self.current_grid:
                self.current_grid = found_grid
                self.get_logger().info(f"Robot moved into Grid {self.current_grid}")
                
            # Publish current grid
            msg = String()
            msg.data = f"Grid {found_grid}" if found_grid != -1 else "Unknown"
            self.current_grid_pub.publish(msg)

        except TransformException as ex:
            self.get_logger().warn(f"Could not find robot position: {ex}")

def main(args=None):
    rclpy.init(args=args)
    node = RobotGridTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()