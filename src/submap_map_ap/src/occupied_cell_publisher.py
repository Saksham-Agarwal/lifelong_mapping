#!/usr/bin/env python3

import rclpy
from rclpy.lifecycle import Node, State, TransitionCallbackReturn
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Int32
import numpy as np
import cv2

class MapAnalyticsPublisher(Node):
    def __init__(self):
        # Updated to match the YAML config root key
        super().__init__('map_analytics_publisher')
        
        # 1. Declare Parameter
        self.declare_parameter('wall_inclusion', False)
        
        # Initialize component variables to None until configured
        self.sub_map = None
        self.pub_internal_occupied = None
        
        self.published = False

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring Map Analytics Publisher...')
        
        # QoS Profile: Reliable and Transient Local (Latching)
        latching_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # Subscriber for the global map
        self.sub_map = self.create_subscription(
            OccupancyGrid, 
            '/map', 
            self.map_callback, 
            latching_qos
        )
        
        # Publisher for the occupied points count
        self.pub_internal_occupied = self.create_lifecycle_publisher(
            Int32, 
            '/internal_occupied_count', 
            latching_qos
        )
        
        self.get_logger().info('Waiting for /map to calculate occupied points...')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Activating Map Analytics Publisher.')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Deactivating Map Analytics Publisher.')
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Cleaning up Map Analytics Publisher.')
        if self.sub_map: self.destroy_subscription(self.sub_map)
        if self.pub_internal_occupied: self.destroy_publisher(self.pub_internal_occupied)
        
        # Allow the node to calculate and publish again if re-configured
        self.published = False
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Shutting down Map Analytics Publisher.')
        return TransitionCallbackReturn.SUCCESS

    def map_callback(self, msg):
        # Process and publish this once
        if not self.published:
            # Dynamically fetch the parameter value
            wall_inclusion = self.get_parameter('wall_inclusion').value
            
            width = msg.info.width
            height = msg.info.height
            
            # Reshape the 1D flat array into a 2D grid
            grid_2d = np.array(msg.data, dtype=np.int8).reshape((height, width))
            
            # Create a mask of ALL occupied cells in the map (> 50 probability)
            occupied_cells = np.where(grid_2d > 50, 255, 0).astype(np.uint8)
            
            if wall_inclusion:
                # If true, simply count all occupied cells (including walls)
                final_count = int(np.count_nonzero(occupied_cells))
            else:
                # If false, isolate the room interior and exclude walls
                free_space = np.where(grid_2d == 0, 255, 0).astype(np.uint8)
                
                # Clean up sensor noise with a morphological OPEN operation
                kernel = np.ones((3, 3), np.uint8)
                free_space = cv2.morphologyEx(free_space, cv2.MORPH_OPEN, kernel)
                
                # Find the outermost contour to define the room bounds
                contours, _ = cv2.findContours(free_space, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                if not contours:
                    self.get_logger().error("No free space found in the map!")
                    return
                    
                # Isolate the main room (largest contour area)
                main_room_contour = max(contours, key=cv2.contourArea)
                
                # Create a solid mask of the entire room interior
                room_interior_mask = np.zeros_like(free_space)
                cv2.drawContours(room_interior_mask, [main_room_contour], -1, 255, thickness=-1)
                
                # Find the intersection: Occupied cells strictly INSIDE the room
                internal_objects_mask = cv2.bitwise_and(occupied_cells, room_interior_mask)
                
                # Count the points
                final_count = int(np.count_nonzero(internal_objects_mask))
            
            # Publish the count
            count_msg = Int32()
            count_msg.data = final_count
            self.pub_internal_occupied.publish(count_msg)
            
            self.published = True
            
            # Log output based on parameter state
            mode_str = "INCLUDING Walls" if wall_inclusion else "EXCLUDING Walls"
            self.get_logger().info(f'Success! Occupied points published ({mode_str}).')
            self.get_logger().info(f' - Occupied points count: {final_count}')

def main(args=None):
    rclpy.init(args=args)
    node = MapAnalyticsPublisher()
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