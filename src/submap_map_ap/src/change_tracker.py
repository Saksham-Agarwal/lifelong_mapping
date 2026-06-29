#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException

class MapClusterAggregator(Node):
    def __init__(self):
        super().__init__('map_cluster_aggregator')

        map_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.blank_map = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.agg_map_pub = self.create_publisher(OccupancyGrid, '/aggregated_map', map_qos)

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, map_qos)
        
        self.pos_cluster_sub = self.create_subscription(
            OccupancyGrid, '/changes/positive_near_neighbour', self.positive_cluster_callback, 10)
        
        self.neg_cluster_sub = self.create_subscription(
            OccupancyGrid, '/changes/negative_nearest_neighbour', self.negative_cluster_callback, 10)

        self.get_logger().info("Map Cluster Aggregator Started: Smart Continuous Updates.")

    def map_callback(self, msg: OccupancyGrid):
        if self.blank_map is None:
            self.get_logger().info("Received initial /map. Creating blank map replica...")
            self.blank_map = OccupancyGrid()
            self.blank_map.header = msg.header
            self.blank_map.info = msg.info
            
            grid_size = msg.info.width * msg.info.height
            self.blank_map.data = [0] * grid_size
            
            self.agg_map_pub.publish(self.blank_map)

    def positive_cluster_callback(self, msg: OccupancyGrid):
        self.process_cluster(msg, is_positive=True)

    def negative_cluster_callback(self, msg: OccupancyGrid):
        self.process_cluster(msg, is_positive=False)

    def process_cluster(self, cluster_msg: OccupancyGrid, is_positive: bool):
        if self.blank_map is None:
            return

        target_frame = self.blank_map.header.frame_id
        source_frame = cluster_msg.header.frame_id

        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
        except TransformException:
            return

        tx = trans.transform.translation.x
        ty = trans.transform.translation.y
        q = trans.transform.rotation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        cluster_res = cluster_msg.info.resolution
        cluster_w = cluster_msg.info.width
        cluster_origin_x = cluster_msg.info.origin.position.x
        cluster_origin_y = cluster_msg.info.origin.position.y

        map_res = self.blank_map.info.resolution
        map_w = self.blank_map.info.width
        map_h = self.blank_map.info.height
        map_origin_x = self.blank_map.info.origin.position.x
        map_origin_y = self.blank_map.info.origin.position.y

        map_updated = False

        for i, cell_value in enumerate(cluster_msg.data):
            # Ignore unknown local space
            if cell_value == -1:
                continue 

            # 1. Convert 1D -> Local 2D -> Global 2D FIRST
            col = i % cluster_w
            row = i // cluster_w
            local_x = cluster_origin_x + (col + 0.5) * cluster_res
            local_y = cluster_origin_y + (row + 0.5) * cluster_res

            global_x = tx + (local_x * math.cos(yaw)) - (local_y * math.sin(yaw))
            global_y = ty + (local_x * math.sin(yaw)) + (local_y * math.cos(yaw))

            map_col = int((global_x - map_origin_x) / map_res)
            map_row = int((global_y - map_origin_y) / map_res)

            # 2. Check bounds and apply Smart Overwriting Logic
            if 0 <= map_col < map_w and 0 <= map_row < map_h:
                map_index = map_row * map_w + map_col
                current_map_val = self.blank_map.data[map_index]
                
                write_value = current_map_val # Default to keeping whatever is already there

                if is_positive:
                    if cell_value > 0:
                        write_value = cell_value # Add new positive change
                    elif cell_value == 0 and current_map_val > 0:
                        write_value = 0 # Revert positive change ONLY (Protects negative cells)
                else:
                    if cell_value > 0:
                        write_value = -1 # Add new negative change
                    elif cell_value == 0 and current_map_val == -1:
                        write_value = 0 # Revert negative change ONLY (Protects positive cells)

                # 3. Update only if a valid change occurred
                if current_map_val != write_value:
                    self.blank_map.data[map_index] = write_value
                    map_updated = True

        if map_updated:
            self.blank_map.header.stamp = self.get_clock().now().to_msg()
            self.agg_map_pub.publish(self.blank_map)

def main(args=None):
    rclpy.init(args=args)
    node = MapClusterAggregator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()