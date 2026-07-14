#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Int32
import numpy as np
import cv2

class MapAnalyticsPublisher(Node):
    def __init__(self):
        super().__init__('map_analytics_publisher')
        
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
        
        # Publisher ONLY for the internal occupied points
        self.pub_internal_occupied = self.create_publisher(Int32, '/internal_occupied_count', latching_qos)
        
        self.published = False
        self.get_logger().info('Waiting for /map to calculate internal occupied points...')

    def map_callback(self, msg):
        # Process and publish this once
        if not self.published:
            width = msg.info.width
            height = msg.info.height
            
            # 1. Reshape the 1D flat array into a 2D grid
            grid_2d = np.array(msg.data, dtype=np.int8).reshape((height, width))
            
            # 2. Isolate free space (0 = free space in standard ROS grids)
            free_space = np.where(grid_2d == 0, 255, 0).astype(np.uint8)
            
            # 3. Clean up sensor noise with a morphological OPEN operation
            kernel = np.ones((3, 3), np.uint8)
            free_space = cv2.morphologyEx(free_space, cv2.MORPH_OPEN, kernel)
            
            # 4. Find the outermost contour to define the room bounds
            contours, _ = cv2.findContours(free_space, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if not contours:
                self.get_logger().error("No free space found in the map!")
                return
                
            # 5. Isolate the main room (largest contour area)
            main_room_contour = max(contours, key=cv2.contourArea)
            
            # 6. Create a solid mask of the entire room interior
            room_interior_mask = np.zeros_like(free_space)
            cv2.drawContours(room_interior_mask, [main_room_contour], -1, 255, thickness=-1)
            
            # 7. Create a mask of ALL occupied cells in the map (> 50 probability)
            occupied_cells = np.where(grid_2d > 50, 255, 0).astype(np.uint8)
            
            # 8. Find the intersection: Occupied cells strictly INSIDE the room
            internal_objects_mask = cv2.bitwise_and(occupied_cells, room_interior_mask)
            
            # 9. Count the points and explicitly cast to Python int for the ROS message
            internal_occupied_count = int(np.count_nonzero(internal_objects_mask))
            
            # Publish the count
            count_msg = Int32()
            count_msg.data = internal_occupied_count
            self.pub_internal_occupied.publish(count_msg)
            
            self.published = True
            self.get_logger().info('Success! Internal occupied points published.')
            self.get_logger().info(f' - Internal occupied points: {internal_occupied_count}')

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