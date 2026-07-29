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
        super().__init__('map_analytics_publisher')
        
        self.declare_parameter('wall_inclusion', False)
        
        self.sub_map = None
        self.pub_internal_occupied = None
        
        # State flags
        self.published = False
        self.is_active_state = False
        self.saved_map_msg = None

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring Map Analytics Publisher...')
        
        latching_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        self.sub_map = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, latching_qos
        )
        
        self.pub_internal_occupied = self.create_lifecycle_publisher(
            Int32, '/internal_occupied_count', latching_qos
        )
        
        self.get_logger().info('Waiting for /map to calculate occupied points...')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Activating Map Analytics Publisher.')
        # 1. Call super() first so the lifecycle publisher becomes active
        ret = super().on_activate(state)
        self.is_active_state = True
        
        # 2. Force processing if a map was already cached during configuration
        if self.saved_map_msg is not None and not self.published:
            self.get_logger().info('Processing cached /map upon activation...')
            self.try_process_map()
            
        return ret

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Deactivating Map Analytics Publisher.')
        self.is_active_state = False
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Cleaning up Map Analytics Publisher.')
        if self.sub_map: 
            self.destroy_subscription(self.sub_map)
            self.sub_map = None
        if self.pub_internal_occupied: 
            self.destroy_publisher(self.pub_internal_occupied)
            self.pub_internal_occupied = None
        
        # Reset ALL flags so re-configuration allows recalculating on a new map
        self.published = False
        self.is_active_state = False
        self.saved_map_msg = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Shutting down Map Analytics Publisher.')
        return TransitionCallbackReturn.SUCCESS

    def map_callback(self, msg):
        # Save map message whenever it arrives
        self.saved_map_msg = msg
        self.try_process_map()

    def try_process_map(self):
        # Only process if active, not yet published, and map data is available
        if not self.is_active_state or self.published or self.saved_map_msg is None:
            return
            
        wall_inclusion = self.get_parameter('wall_inclusion').value
        
        width = self.saved_map_msg.info.width
        height = self.saved_map_msg.info.height
        grid_2d = np.array(self.saved_map_msg.data, dtype=np.int8).reshape((height, width))
        occupied_cells = np.where(grid_2d > 50, 255, 0).astype(np.uint8)
        
        if wall_inclusion:
            final_count = int(np.count_nonzero(occupied_cells))
        else:
            free_space = np.where(grid_2d == 0, 255, 0).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            free_space = cv2.morphologyEx(free_space, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(free_space, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if not contours:
                self.get_logger().error("No free space found in the map!")
                return
                
            main_room_contour = max(contours, key=cv2.contourArea)
            room_interior_mask = np.zeros_like(free_space)
            cv2.drawContours(room_interior_mask, [main_room_contour], -1, 255, thickness=-1)
            
            internal_objects_mask = cv2.bitwise_and(occupied_cells, room_interior_mask)
            final_count = int(np.count_nonzero(internal_objects_mask))
        
        count_msg = Int32()
        count_msg.data = final_count
        
        # Publish to the latched topic
        self.pub_internal_occupied.publish(count_msg)
        self.published = True
        
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