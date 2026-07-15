#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PoseWithCovarianceStamped, Point
import math
import time  # --- NEW: Added for cooldown tracking ---

import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from submap_map_ap.msg import MapSnapshot 

class SnapshotPublisher(Node):
    def __init__(self):
        super().__init__('snapshot_publisher')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.sub_local = self.create_subscription(OccupancyGrid, '/robot_local_region', self.local_callback, 10)
        self.sub_submap = self.create_subscription(OccupancyGrid, '/submap_local_region', self.submap_callback, 10)
        self.sub_amcl = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_conf = self.create_subscription(Float32, '/amcl_confidence', self.conf_callback, 10)
        self.snap = self.create_subscription(Bool, '/take_snapshot', self.snap_callback, 10)
        
        self.pub_snapshot = self.create_publisher(MapSnapshot, '/map_snapshot_data', 10)
        self.declare_parameter('cooldown_seconds', 5)
        self.local_msg = None
        self.submap_msg = None
        self.scan_msg = None
        self.amcl_msg = None
        self.conf_msg = None
        
        # --- NEW: Cooldown tracking variables ---
        self.last_snap_time = 0.0
        self.cooldown_seconds = self.get_parameter('cooldown_seconds').value  # Prevents spamming triggers for 5 seconds
        
        self.get_logger().info('Waiting for map, scan, AMCL, and confidence topics...')

    def local_callback(self, msg): self.local_msg = msg
    def submap_callback(self, msg): self.submap_msg = msg
    def scan_callback(self, msg): self.scan_msg = msg
    def amcl_callback(self, msg): self.amcl_msg = msg
    def conf_callback(self, msg): self.conf_msg = msg

    def snap_callback(self, msg):  
        # --- NEW: Debounce logic using a cooldown timer ---
        if bool(msg.data):
            current_time = time.time()
            if (current_time - self.last_snap_time) > self.cooldown_seconds:
                self.get_logger().info('Snapshot requested! Processing...')
                self.last_snap_time = current_time
                self.check_and_publish()
            else:
                # Silently ignore the spam triggers while in cooldown
                pass

    def check_and_publish(self):
        if not self.local_msg or not self.scan_msg or not self.amcl_msg or not self.conf_msg:
            self.get_logger().warn('Waiting for core topics (global map, scan, amcl, conf) to arrive...')
            return

        target_frame = self.local_msg.header.frame_id
        source_frame = self.scan_msg.header.frame_id
        
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time(), timeout=Duration(seconds=2.0))
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'Waiting for TF: {e}')
            return 

        snap_msg = MapSnapshot()

        # 1. Populate AMCL Guess & Confidence Score
        snap_msg.amcl_x = self.amcl_msg.pose.pose.position.x
        snap_msg.amcl_y = self.amcl_msg.pose.pose.position.y
        q = self.amcl_msg.pose.pose.orientation
        snap_msg.amcl_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        
        snap_msg.amcl_confidence = float(self.conf_msg.data)

        # 2. Extract Data
        local_coords = self.extract_occupied(self.local_msg)
        scan_coords = self.extract_and_transform_scan(self.scan_msg, trans)

        if self.submap_msg is not None:
            submap_coords = self.extract_occupied(self.submap_msg)
        else:
            self.get_logger().info('Submap not yet published. Falling back to global map.')
            submap_coords = local_coords

        # 3. Populate Arrays
        for x, y in local_coords:
            snap_msg.global_points.append(Point(x=float(x), y=float(y), z=0.0))

        for x, y in submap_coords:
            snap_msg.submap_points.append(Point(x=float(x), y=float(y), z=0.0))

        for x, y in scan_coords:
            snap_msg.local_points.append(Point(x=float(x), y=float(y), z=0.0))

        # 4. Publish
        self.pub_snapshot.publish(snap_msg)
        self.get_logger().info('Success! Snapshot published.')

    def extract_occupied(self, msg):
        res, w, ox, oy = msg.info.resolution, msg.info.width, msg.info.origin.position.x, msg.info.origin.position.y
        return [((i % w * res) + ox, (i // w * res) + oy) for i, val in enumerate(msg.data) if val > 30]

    def extract_and_transform_scan(self, msg, trans):
        raw_coords = []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max and not math.isinf(r) and not math.isnan(r):
                raw_coords.append((r * math.cos(angle), r * math.sin(angle)))
            angle += msg.angle_increment

        tx, ty = trans.transform.translation.x, trans.transform.translation.y
        q = trans.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        
        return [((x * cos_y) - (y * sin_y) + tx, (x * sin_y) + (y * cos_y) + ty) for x, y in raw_coords]

def main(args=None):
    rclpy.init(args=args)
    node = SnapshotPublisher()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()