import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid
import csv
import math

# TF2 Imports
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class DualMapSaver(Node):
    def __init__(self):
        super().__init__('dual_map_saver')
        
        # TF Listener Setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Subscriptions
        self.sub_local = self.create_subscription(
            OccupancyGrid, '/robot_local_region', self.local_callback, 10)
        self.sub_costmap = self.create_subscription(
            OccupancyGrid, '/costmap', self.costmap_callback, 10)
        
        # Store raw messages until we have both and a valid TF
        self.local_msg = None
        self.costmap_msg = None
        self.saved = False

        self.get_logger().info('Waiting for maps and TF Tree data...')

    def local_callback(self, msg):
        if not self.local_msg:
            self.get_logger().info(f'Received /robot_local_region in frame: {msg.header.frame_id}')
            self.local_msg = msg
            self.check_and_save()

    def costmap_callback(self, msg):
        if not self.costmap_msg:
            self.get_logger().info(f'Received /costmap in frame: {msg.header.frame_id}')
            self.costmap_msg = msg
            self.check_and_save()

    def check_and_save(self):
        if self.local_msg and self.costmap_msg and not self.saved:
            target_frame = self.local_msg.header.frame_id
            source_frame = self.costmap_msg.header.frame_id
            
            try:
                # Ask TF: "How do I get from the costmap's frame to the local region's frame?"
                trans = self.tf_buffer.lookup_transform(
                    target_frame, 
                    source_frame, 
                    rclpy.time.Time(),
                    timeout=Duration(seconds=2.0)
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f'Waiting for TF transform: {e}')
                return # Don't save yet, wait for the next callback to try again

            self.saved = True
            self.get_logger().info('Transform found! Aligning maps...')
            
            # Extract Global points (Already in target_frame)
            local_coords = self.extract_occupied(self.local_msg)
            
            # Extract Local points and transform them to match Global
            costmap_coords = self.extract_and_transform(self.costmap_msg, trans)
            
            # Save CSVs
            self.save_csv('local_region.csv', local_coords)
            self.save_csv('costmap.csv', costmap_coords)
                
            self.get_logger().info('Success! Saved TF-aligned CSVs. Shutting down.')
            raise SystemExit

    def extract_occupied(self, msg):
        """Extracts X/Y coordinates in the message's native frame."""
        resolution = msg.info.resolution
        width = msg.info.width
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        coords = []
        for i, cell_value in enumerate(msg.data):
            if cell_value > 30:
                col = i % width
                row = i // width
                world_x = (col * resolution) + origin_x
                world_y = (row * resolution) + origin_y
                coords.append((world_x, world_y))
                
        return coords

    def extract_and_transform(self, msg, trans):
        """Extracts points and applies the TF rotation/translation to match the global frame."""
        raw_coords = self.extract_occupied(msg)
        
        # 1. Extract Translation
        tx = trans.transform.translation.x
        ty = trans.transform.translation.y
        
        # 2. Extract Rotation (Convert Quaternion to Yaw)
        q = trans.transform.rotation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        
        transformed_coords = []
        for x, y in raw_coords:
            # Standard 2D Transformation Math
            new_x = (x * cos_y) - (y * sin_y) + tx
            new_y = (x * sin_y) + (y * cos_y) + ty
            transformed_coords.append((new_x, new_y))
            
        return transformed_coords

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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()