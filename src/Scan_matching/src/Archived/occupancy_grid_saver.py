import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseWithCovarianceStamped
import csv
import math

import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class DualMapSaver(Node):
    def __init__(self):
        super().__init__('dual_map_saver')
        
        # TF Setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # --- MISSING QOS & MAP SUBSCRIBER ADDED BACK ---
        map_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # Subscriptions
        self.sub_local = self.create_subscription(OccupancyGrid, '/robot_local_region', self.local_callback, 10)
        self.sub_costmap = self.create_subscription(OccupancyGrid, '/costmap', self.costmap_callback, 10)
        self.sub_amcl = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_callback, 10)
        self.sub_map = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.snap = self.create_subscription(Bool, '/take_snapshot', self.snap_callback, 10)
        
        # State variables
        self.local_msg = None
        self.costmap_msg = None
        self.amcl_msg = None
        self.map_msg = None  
        self.take_snapshot = False
        
        # Counter to prevent overwriting files on multiple snapshots
        self.snapshot_count = 0
        
        self.get_logger().info('Waiting for maps, AMCL, and global /map bounds. Ready for snapshots!')

    def local_callback(self, msg):
        self.local_msg = msg
    
    def costmap_callback(self, msg):
        self.costmap_msg = msg

    def amcl_callback(self, msg):
        self.amcl_msg = msg

    # --- MISSING CALLBACK ADDED BACK ---
    def map_callback(self, msg):
        self.map_msg = msg

    def snap_callback(self, msg):  
        self.take_snapshot = bool(msg.data)
        # Immediately evaluate when a trigger is received
        if self.take_snapshot:
            self.check_and_save()

    def check_and_save(self):
        # --- MISSING MAP_MSG GUARDRAIL ADDED BACK ---
        # If the map hasn't loaded yet, warn the user and abort the save attempt
        if not self.map_msg:
            self.get_logger().warn('Snapshot triggered, but the global /map has not been received yet. Waiting...')
            self.take_snapshot = False
            return

        if self.local_msg and self.costmap_msg and self.amcl_msg and self.take_snapshot:
            target_frame = self.local_msg.header.frame_id
            source_frame = self.costmap_msg.header.frame_id
            
            try:
                # Let TF perfectly align the costmap to the global map based on AMCL
                trans = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, rclpy.time.Time(), timeout=Duration(seconds=2.0))
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f'Waiting for TF: {e}')
                # Reset trigger so it can try again on the next callback if TF fails
                self.take_snapshot = False 
                return 

            self.snapshot_count += 1
            
            # Extract and transform points
            local_coords = self.extract_occupied(self.local_msg)
            costmap_coords = self.extract_and_transform(self.costmap_msg, trans)
            
            # Save files with the counter appended
            self.save_csv(f'local_region_{self.snapshot_count}.csv', local_coords)
            self.save_csv(f'costmap_{self.snapshot_count}.csv', costmap_coords)
            
            # Save AMCL absolute position for drift calculations in Python
            amcl_x = self.amcl_msg.pose.pose.position.x
            amcl_y = self.amcl_msg.pose.pose.position.y
            q = self.amcl_msg.pose.pose.orientation
            amcl_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            
            with open(f'amcl_guess_{self.snapshot_count}.txt', 'w') as f:
                f.write(f"amcl_x:{amcl_x}\n")
                f.write(f"amcl_y:{amcl_y}\n")
                f.write(f"amcl_yaw:{amcl_yaw}\n")

            # 3. Save Global Map Boundaries for the "Blank Map"
            origin_x = self.map_msg.info.origin.position.x
            origin_y = self.map_msg.info.origin.position.y
            width_cells = self.map_msg.info.width
            height_cells = self.map_msg.info.height
            resolution = self.map_msg.info.resolution
            
            width_meters = width_cells * resolution
            height_meters = height_cells * resolution
            
            with open('map_bounds.txt', 'w') as f:
                f.write(f"min_x:{origin_x}\n")
                f.write(f"max_x:{origin_x + width_meters}\n")
                f.write(f"min_y:{origin_y}\n")
                f.write(f"max_y:{origin_y + height_meters}\n")
                f.write(f"width:{width_cells}\n")
                f.write(f"height:{height_cells}\n")
                f.write(f"resolution:{resolution}\n")
            
            # Reset the trigger so it waits for the next True message
            self.take_snapshot = False
            self.get_logger().info(f'Success! Saved snapshot #{self.snapshot_count}.')

    def extract_occupied(self, msg):
        resolution = msg.info.resolution
        width = msg.info.width
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        coords = []
        for i, cell_value in enumerate(msg.data):
            if cell_value > 30:
                col = i % width
                row = i // width
                coords.append(((col * resolution) + origin_x, (row * resolution) + origin_y))
        return coords

    def extract_and_transform(self, msg, trans):
        raw_coords = self.extract_occupied(msg)
        tx = trans.transform.translation.x
        ty = trans.transform.translation.y
        q = trans.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        
        transformed = []
        for x, y in raw_coords:
            new_x = (x * cos_y) - (y * sin_y) + tx
            new_y = (x * sin_y) + (y * cos_y) + ty
            transformed.append((new_x, new_y))
        return transformed

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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()