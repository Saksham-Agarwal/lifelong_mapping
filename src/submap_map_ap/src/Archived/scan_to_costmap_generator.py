#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid

import math

class ScanToGridNode(Node):
    def __init__(self):
        super().__init__('scan_to_grid_node')
        
        self.resolution = 0.05  
        self.grid_width_m = 5.0  
        self.grid_height_m = 5.0
        
        self.width_cells = int(self.grid_width_m / self.resolution)
        self.height_cells = int(self.grid_height_m / self.resolution)
        
        self.origin_x = - (self.grid_width_m / 2.0)
        self.origin_y = - (self.grid_height_m / 2.0)
        
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        
        self.publisher = self.create_publisher(
            OccupancyGrid,
            '/scan_grid',
            10)
            
        self.get_logger().info(f"Scan-to-Grid node initialized. Grid size: {self.width_cells}x{self.height_cells}")

    def scan_callback(self, msg: LaserScan):
        grid = OccupancyGrid()
        
        grid.header = msg.header
        
        grid.info.resolution = self.resolution
        grid.info.width = self.width_cells
        grid.info.height = self.height_cells
        
        grid.info.origin.position.x = self.origin_x
        grid.info.origin.position.y = self.origin_y
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0  # No rotation relative to sensor
        
        grid_data = [0] * (self.width_cells * self.height_cells)
        
        for i, scan_range in enumerate(msg.ranges):
            if math.isnan(scan_range) or math.isinf(scan_range) or not (msg.range_min <= scan_range <= msg.range_max):
                continue
                
            angle = msg.angle_min + (i * msg.angle_increment)
            
            x = scan_range * math.cos(angle)
            y = scan_range * math.sin(angle)
            
            cell_x = int((x - self.origin_x) / self.resolution)
            cell_y = int((y - self.origin_y) / self.resolution)
            
            if (0 <= cell_x < self.width_cells) and (0 <= cell_y < self.height_cells):
                index = cell_y * self.width_cells + cell_x
                grid_data[index] = 100  
                
        grid.data = grid_data
        self.publisher.publish(grid)

def main(args=None):
    rclpy.init(args=args)
    node = ScanToGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()