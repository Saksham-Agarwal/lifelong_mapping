#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from vision_msgs.msg import Detection2DArray
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from submap_map_ap.msg import MapGridSummary, GridObjectList, MapGridDefinition, GridBounds

class GridSpatialAssigner(Node):
    def __init__(self):
        super().__init__('grid_spatial_assigner')

        # Transient Local QoS (Latching)
        self.latch_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, self.latch_qos)
        self.detection_sub = self.create_subscription(Detection2DArray, '/detected_map_objects', self.detections_callback, self.latch_qos)
        
        # Publishers
        self.grid_summary_publisher = self.create_publisher(MapGridSummary, '/map_grid_summary', self.latch_qos)
        self.grid_def_publisher = self.create_publisher(MapGridDefinition, '/map_grid_definitions', self.latch_qos)

        self.map_info = None
        self.get_logger().info("Grid Spatial Assigner ready. Waiting for map and detections...")

    def map_callback(self, msg):
        self.map_info = msg.info
        self.get_logger().info("Map metadata received. Publishing grid definitions...")
        
        # Calculate and Publish Grid Definitions immediately upon receiving the map
        width = self.map_info.width
        height = self.map_info.height
        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y

        w_cell = (width * resolution) / 3.0
        h_cell = (height * resolution) / 3.0

        def_msg = MapGridDefinition()
        def_msg.header = msg.header

        for row in range(3):
            for col in range(3):
                cell_id = row * 3 + col + 1
                bounds = GridBounds()
                bounds.grid_id = cell_id
                bounds.min_x = origin_x + (col * w_cell)
                bounds.max_x = origin_x + ((col + 1) * w_cell)
                bounds.min_y = origin_y + ((2 - row) * h_cell)
                bounds.max_y = origin_y + ((3 - row) * h_cell)
                def_msg.grids.append(bounds)
                
        self.grid_def_publisher.publish(def_msg)

    def detections_callback(self, msg):
        if self.map_info is None:
            return

        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        w_cell = (self.map_info.width * resolution) / 3.0
        h_cell = (self.map_info.height * resolution) / 3.0

        grid_contents = {i: [] for i in range(1, 10)}

        for det in msg.detections:
            if not det.results: continue
            obj_name = det.results[0].hypothesis.class_id
            
            world_cx = det.bbox.center.position.x
            world_cy = det.bbox.center.position.y
            size_x = det.bbox.size_x
            size_y = det.bbox.size_y

            obj_min_x = world_cx - (size_x / 2.0)
            obj_max_x = world_cx + (size_x / 2.0)
            obj_min_y = world_cy - (size_y / 2.0)
            obj_max_y = world_cy + (size_y / 2.0)

            for row in range(3):
                for col in range(3):
                    cell_id = row * 3 + col + 1
                    cell_min_x = origin_x + (col * w_cell)
                    cell_max_x = origin_x + ((col + 1) * w_cell)
                    cell_min_y = origin_y + ((2 - row) * h_cell)
                    cell_max_y = origin_y + ((3 - row) * h_cell)
                    
                    if (obj_min_x <= cell_max_x) and (obj_max_x >= cell_min_x) and (obj_min_y <= cell_max_y) and (obj_max_y >= cell_min_y):
                        grid_contents[cell_id].append(obj_name)

        summary_msg = MapGridSummary()
        summary_msg.header = msg.header
        for i in range(1, 10):
            grid_list = GridObjectList()
            grid_list.grid_id = i
            grid_list.object_ids = grid_contents[i]
            summary_msg.grids.append(grid_list)
            
        self.grid_summary_publisher.publish(summary_msg)
        self.get_logger().info("Processed objects into grids and published summary.")

def main(args=None):
    rclpy.init(args=args)
    node = GridSpatialAssigner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()