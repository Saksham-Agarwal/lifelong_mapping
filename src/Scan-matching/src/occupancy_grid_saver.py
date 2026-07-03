import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import csv

class DualMapSaver(Node):
    def __init__(self):
        super().__init__('dual_map_saver')
        
        # Subscriptions
        self.sub_local = self.create_subscription(
            OccupancyGrid, '/robot_local_region', self.local_callback, 10)
        self.sub_costmap = self.create_subscription(
            OccupancyGrid, '/costmap/costmap', self.costmap_callback, 10)
        
        self.local_data = None
        self.costmap_data = None
        self.saved = False

        self.get_logger().info('Waiting for /robot_local_region and /costmap/costmap...')

    def local_callback(self, msg):
        if not self.local_data:
            self.get_logger().info('Received /robot_local_region')
            self.local_data = self.extract_occupied(msg)
            self.check_and_save()

    def costmap_callback(self, msg):
        if not self.costmap_data:
            self.get_logger().info('Received /costmap/costmap')
            self.costmap_data = self.extract_occupied(msg)
            self.check_and_save()

    def extract_occupied(self, msg):
        resolution = msg.info.resolution
        width = msg.info.width
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        coords = []
        # Filter for unoccupied cells (only keep > 50 probability)
        for i, cell_value in enumerate(msg.data):
            if cell_value > 30:
                col = i % width
                row = i // width
                
                # Real-world coordinates intrinsically align them to the TF frame
                world_x = (col * resolution) + origin_x
                world_y = (row * resolution) + origin_y
                coords.append((world_x, world_y))
                
        return {'coords': coords, 'origin': (origin_x, origin_y)}

    def check_and_save(self):
        # Proceed only if both grids have been received
        if self.local_data and self.costmap_data and not self.saved:
            self.saved = True
            
            local_coords = self.local_data['coords']
            costmap_coords = self.costmap_data['coords']
            
            local_orig = self.local_data['origin']
            costmap_orig = self.costmap_data['origin']
            
            # The alignment vector (difference between their origins in the world frame)
            align_x = costmap_orig[0] - local_orig[0]
            align_y = costmap_orig[1] - local_orig[1]
            
            # Save CSVs
            self.save_csv('local_region.csv', local_coords)
            self.save_csv('costmap.csv', costmap_coords)
            
            # Save the alignment vector
            with open('alignment_vector.txt', 'w') as f:
                f.write(f"Alignment Vector (Offset from Local Region to Costmap Origin):\n")
                f.write(f"Delta X: {align_x} meters\n")
                f.write(f"Delta Y: {align_y} meters\n")
                
            self.get_logger().info('Success! Saved CSVs and alignment vector. Shutting down.')
            raise SystemExit

    def save_csv(self, filename, coords):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            writer.writerows(coords)

def main(args=None):
    rclpy.init(args=args)
    node = DualMapSaver()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
        # rclpy.get_logger('dual_map_saver').info('Process complete.')
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()