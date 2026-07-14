#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32
import numpy as np
import cv2

class WallPerimeterLengthPublisher(Node):
    def __init__(self):
        super().__init__('wall_perimeter_publisher')
        
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
        
        # Publisher for the perimeter length (in meters)
        self.pub_perimeter_length = self.create_publisher(Float32, '/wall_perimeter_length', latching_qos)
        
        self.published = False
        self.get_logger().info('Waiting for /map to calculate wall perimeter length...')

    def map_callback(self, msg):
        # We only need to process and publish this once
        if not self.published:
            width = msg.info.width
            height = msg.info.height
            resolution = msg.info.resolution
            
            # 1. Reshape the 1D flat array into a 2D grid
            grid_2d = np.array(msg.data, dtype=np.int8).reshape((height, width))
            
            # 2. Isolate free space (0 = free space in standard ROS grids)
            free_space = np.where(grid_2d == 0, 255, 0).astype(np.uint8)
            
            # 3. Clean up sensor noise with a morphological OPEN operation
            kernel = np.ones((3, 3), np.uint8)
            free_space = cv2.morphologyEx(free_space, cv2.MORPH_OPEN, kernel)
            
            # 4. Find the outermost contour
            contours, _ = cv2.findContours(free_space, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if not contours:
                self.get_logger().error("No free space found in the map!")
                return
                
            # 5. Isolate the main room (largest contour area)
            main_room_contour = max(contours, key=cv2.contourArea)
            
            # 6. Calculate the perimeter length in pixels
            # The 'True' flag specifies that the contour is a closed loop
            perimeter_pixels = cv2.arcLength(main_room_contour, True)
            
            # 7. Convert pixel length to real-world meters
            perimeter_meters = perimeter_pixels * resolution
            
            # 8. Create the Float32 message and publish
            length_msg = Float32()
            length_msg.data = float(perimeter_meters)
            
            self.pub_perimeter_length.publish(length_msg)
            self.published = True
            
            self.get_logger().info(f'Success! Room perimeter length published to /wall_perimeter_length.')
            self.get_logger().info(f'Length: {perimeter_meters:.3f} meters')

def main(args=None):
    rclpy.init(args=args)
    node = WallPerimeterLengthPublisher()
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