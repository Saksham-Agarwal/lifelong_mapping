#!/usr/bin/env python3

import rclpy
from rclpy.lifecycle import Node, State, TransitionCallbackReturn
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import OccupancyGrid

from submap_map_ap.msg import MapBounds, MapUpdate, ClusterChange

class GlobalChangesUpdater(Node):
    def __init__(self):
        super().__init__('global_changes_updater')
        
        self.declare_parameter('min_cluster_size', 10)
        
        # Initialize component variables to None until configured
        self.sub_bounds = None
        self.sub_changes = None
        self.pub_grid = None
        self.grid_msg = None
        self.min_cluster_size = None

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring Global Changes Updater...')
        
        latching_qos = QoSProfile(
            depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        self.sub_bounds = self.create_subscription(MapBounds, '/map_bounds', self.bounds_callback, latching_qos)
        self.sub_changes = self.create_subscription(MapUpdate, '/map_changes', self.changes_callback, 10)
        
        self.pub_grid = self.create_lifecycle_publisher(OccupancyGrid, '/overall_changes', latching_qos)
        
        # --- Noise Filtering Threshold ---
        # Ignore any change cluster with fewer than this many points
        self.min_cluster_size = self.get_parameter('min_cluster_size').value
        
        self.get_logger().info('Global Changes Updater configured. Waiting for /map_bounds...')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Activating Global Changes Updater.')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Deactivating Global Changes Updater.')
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Cleaning up Global Changes Updater.')
        if self.sub_bounds: self.destroy_subscription(self.sub_bounds)
        if self.sub_changes: self.destroy_subscription(self.sub_changes)
        if self.pub_grid: self.destroy_publisher(self.pub_grid)
        
        self.grid_msg = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Shutting down Global Changes Updater.')
        return TransitionCallbackReturn.SUCCESS

    def bounds_callback(self, msg):
        if not self.grid_msg:
            self.grid_msg = OccupancyGrid()
            self.grid_msg.header.frame_id = 'map'
            
            self.grid_msg.info.resolution = msg.resolution
            self.grid_msg.info.width = msg.width
            self.grid_msg.info.height = msg.height
            self.grid_msg.info.origin.position.x = msg.min_x
            self.grid_msg.info.origin.position.y = msg.min_y
            
            # Start with an entirely empty grid (0 = free space)
            self.grid_msg.data = [0] * (msg.width * msg.height)
            self.get_logger().info('Blank global map initialized. Waiting for changes...')

    def changes_callback(self, msg):
        if not self.grid_msg:
            self.get_logger().warn('Received changes, but global map is not initialized yet!')
            return
            
        self.get_logger().info('Processing new map updates...')
        
        res = self.grid_msg.info.resolution
        w = self.grid_msg.info.width
        h = self.grid_msg.info.height
        ox = self.grid_msg.info.origin.position.x
        oy = self.grid_msg.info.origin.position.y
        
        current_data = list(self.grid_msg.data)
        
        for cluster in msg.clusters:
            # --- Filter out noise ---
            if len(cluster.points) < self.min_cluster_size:
                continue
                
            # Determine grid value based on change type
            if cluster.change_type == ClusterChange.POSITIVE_CHANGE:
                grid_value = 100  # Lethal Obstacle
            elif cluster.change_type == ClusterChange.NEGATIVE_CHANGE:
                grid_value = -1   # Unknown / Unexplored (Negative Space)
            elif cluster.change_type == ClusterChange.NEGATIVE_TO_POSITIVE:
                grid_value = 100  # Lethal Obstacle
            elif cluster.change_type == ClusterChange.POSITIVE_TO_NEGATIVE:
                grid_value = -1   # Unknown / Unexplored (Negative Space)   
                
            # --- NEW: Process the 50 tags directly ---
            elif cluster.change_type == 50:
                grid_value = 50   # Cleared Unexplored Free Space
                
            else:
                continue

            # Map the points
            for pt in cluster.points:
                col = int((pt.x - ox) / res)
                row = int((pt.y - oy) / res)
                
                if 0 <= col < w and 0 <= row < h:
                    index = (row * w) + col
                    current_data[index] = grid_value
                        
        self.grid_msg.data = current_data
        self.grid_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_grid.publish(self.grid_msg)
        self.get_logger().info('Updated /overall_changes map published!')

def main(args=None):
    rclpy.init(args=args)
    node = GlobalChangesUpdater()
    try: 
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__': main()