#!/usr/bin/env python3

import rclpy
from rclpy.lifecycle import Node, State, TransitionCallbackReturn
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import OccupancyGrid
import numpy as np

class SubmapGenerator(Node):
    def __init__(self):
        super().__init__('submap_generator_node')
        
        # Store the latest map
        self.base_map_msg = None
        
        # Initialize component variables to None until configured
        self.submap_pub = None
        self.map_sub = None
        self.changes_sub = None

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring Submap Generator Node...")
        
        # QoS Profile: Reliable and Transient Local (Latching)
        latching_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # Publisher for the new submap
        self.submap_pub = self.create_lifecycle_publisher(
            OccupancyGrid, 
            '/submap_map', 
            latching_qos
        )
        
        # Subscriber to the static map
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            latching_qos
        )
        
        # Subscriber to the overall changes 
        self.changes_sub = self.create_subscription(
            OccupancyGrid,
            '/overall_changes',
            self.changes_callback,
            10
        )
        
        self.get_logger().info("Submap Generator Node configured. Waiting for /map and /overall_changes...")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Activating Submap Generator Node.')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Deactivating Submap Generator Node.')
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Cleaning up Submap Generator Node.')
        if self.submap_pub: self.destroy_publisher(self.submap_pub)
        if self.map_sub: self.destroy_subscription(self.map_sub)
        if self.changes_sub: self.destroy_subscription(self.changes_sub)
        
        self.base_map_msg = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Shutting down Submap Generator Node.')
        return TransitionCallbackReturn.SUCCESS

    def map_callback(self, msg):
        """Stores the base map when it arrives."""
        self.base_map_msg = msg
        self.get_logger().info("Base /map received.")

    def changes_callback(self, msg):
        """Processes the changes and publishes the new submap."""
        if self.base_map_msg is None:
            self.get_logger().warn("Received /overall_changes, but /map hasn't been received yet.")
            return
            
        # Verify sizes match to prevent index errors
        if len(self.base_map_msg.data) != len(msg.data):
            self.get_logger().error("Size mismatch between /map and /overall_changes! Cannot process.")
            return

        # Convert ROS messages to NumPy arrays for fast vectorized operations
        base_map_arr = np.array(self.base_map_msg.data, dtype=np.int8)
        changes_arr = np.array(msg.data, dtype=np.int8)
        
        # Initialize the submap with the base map data
        submap_arr = np.copy(base_map_arr)
        
        # --- Rule 1: If occupied (100) on /overall_changes -> map as occupied (100) in submap
        occupied_in_changes = (changes_arr == 100)
        submap_arr[occupied_in_changes] = 100
        
        # --- Rule 2: If unexplored (-1) on /overall_changes AND occupied (100) on /map -> map as free (0) in submap
        unexplored_in_changes = (changes_arr == -1)
        occupied_in_base = (base_map_arr == 100)
        condition_to_free = unexplored_in_changes & occupied_in_base
        submap_arr[condition_to_free] = 0
        
        # --- Rule 3: If marked as cleared unexplored (50) on /overall_changes -> map as free (0) in submap
        cleared_unexplored = (changes_arr == 50)
        submap_arr[cleared_unexplored] = 0
        
        # Construct the new OccupancyGrid message
        submap_msg = OccupancyGrid()
        
        # Inherit metadata from the base map (origin, resolution, dimensions)
        submap_msg.header = self.base_map_msg.header
        # Update the timestamp to now
        submap_msg.header.stamp = self.get_clock().now().to_msg() 
        submap_msg.info = self.base_map_msg.info
        
        # Convert the NumPy array back to a standard Python list for the ROS message
        submap_msg.data = submap_arr.tolist()
        
        # Publish the new map
        self.submap_pub.publish(submap_msg)
        self.get_logger().debug("Published updated /submap_map.")

def main(args=None):
    rclpy.init(args=args)
    node = SubmapGenerator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()