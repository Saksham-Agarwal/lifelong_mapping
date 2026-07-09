import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import pandas as pd
import os

class OverallChangesPublisher(Node):
    def __init__(self):
        super().__init__('overall_changes_publisher')
        
        # Publisher for the new changes map
        self.publisher_ = self.create_publisher(OccupancyGrid, '/overall_changes', 10)
        
        # Timer to broadcast the map every 2 seconds
        self.timer = self.create_timer(2.0, self.publish_map)
        
        self.get_logger().info('Publishing isolated obstacles to /overall_changes...')

    def publish_map(self):
        if not os.path.exists('map_bounds.txt') or not os.path.exists('lifelong_obstacles.csv'):
            self.get_logger().warn('Waiting for map_bounds.txt and lifelong_obstacles.csv...')
            return

        # 1. Read the physical bounds and grid properties
        with open('map_bounds.txt', 'r') as f:
            lines = f.readlines()
            origin_x = float(lines[0].split(':')[1].strip())
            origin_y = float(lines[2].split(':')[1].strip())
            width = int(lines[4].split(':')[1].strip())
            height = int(lines[5].split(':')[1].strip())
            resolution = float(lines[6].split(':')[1].strip())

        # 2. Read the historical database of new obstacles
        try:
            df = pd.read_csv('lifelong_obstacles.csv')
        except pd.errors.EmptyDataError:
            self.get_logger().warn('Obstacle CSV is empty.')
            return

        # 3. Construct the Blank ROS OccupancyGrid
        grid = OccupancyGrid()
        grid.header.frame_id = 'map'
        grid.header.stamp = self.get_clock().now().to_msg()
        
        grid.info.resolution = resolution
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = origin_x
        grid.info.origin.position.y = origin_y
        grid.info.origin.position.z = 0.0
        
        # Initialize a completely blank map (0 = Free Space)
        # Using a 1D list representing the 2D grid
        grid_data = [0] * (width * height)

        # 4. Overlay the positive changes onto the blank map
        for _, row in df.iterrows():
            x = row['x']
            y = row['y']
            
            # Convert physical (x, y) into grid column and row indices
            col = int((x - origin_x) / resolution)
            row_idx = int((y - origin_y) / resolution)
            
            # Ensure the point actually fits inside the grid dimensions
            if 0 <= col < width and 0 <= row_idx < height:
                # Calculate the 1D array index
                index = (row_idx * width) + col
                grid_data[index] = 100  # Mark as 100% Occupied/Lethal Obstacle

        # 5. Attach data and publish
        grid.data = grid_data
        self.publisher_.publish(grid)

def main(args=None):
    rclpy.init(args=args)
    node = OverallChangesPublisher()
    
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