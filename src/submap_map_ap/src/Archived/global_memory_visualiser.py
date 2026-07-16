#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
import json
import os

class GlobalMemoryVisualizer(Node):
    def __init__(self):
        super().__init__('global_memory_visualizer')
        
        # Publisher for the RViz Markers
        self.marker_pub = self.create_publisher(MarkerArray, '/global_memory_markers', 10)
        
        # Path to the JSON memory file
        self.save_path = os.path.expanduser('~/robot_grid_memory.json')
        
        # Timer to read the file and update RViz every 1 second
        self.timer = self.create_timer(1.0, self.publish_markers)
        
        self.get_logger().info("Global Memory Visualizer started. Reading from JSON...")

    def publish_markers(self):
        # 1. Check if the file exists
        if not os.path.exists(self.save_path):
            return

        # 2. Safely read the JSON file
        try:
            with open(self.save_path, 'r') as f:
                memory = json.load(f)
        except Exception as e:
            self.get_logger().warn(f"Could not read JSON file: {e}")
            return

        marker_array = MarkerArray()
        
        # 3. Add a DELETEALL marker to clear the previous frame's ghost boxes
        delete_all = Marker()
        delete_all.action = 3  # Marker.DELETEALL
        marker_array.markers.append(delete_all)
        
        marker_id = 0
        current_time = self.get_clock().now().to_msg()

        # 4. Loop through all 9 grids in the JSON
        for grid_id, grid_data in memory.items():
            
            # If the robot hasn't explored this grid yet, skip it
            if not grid_data.get("explored", False):
                continue
                
            # --- Draw POSITIVE Boxes (Green) ---
            for box in grid_data.get("positive_boxes", []):
                m = self.create_cube_marker(box, marker_id, current_time, is_positive=True)
                marker_array.markers.append(m)
                marker_id += 1
                
            # --- Draw NEGATIVE Boxes (Red) ---
            for box in grid_data.get("negative_boxes", []):
                m = self.create_cube_marker(box, marker_id, current_time, is_positive=False)
                marker_array.markers.append(m)
                marker_id += 1

        # 5. Publish to RViz
        self.marker_pub.publish(marker_array)

    def create_cube_marker(self, box, m_id, timestamp, is_positive):
        """Helper function to generate a standardized RViz Cube Marker"""
        m = Marker()
        # Set frame to 'map' since these are global coordinates
        m.header.frame_id = 'map'
        m.header.stamp = timestamp
        
        m.ns = "global_positive" if is_positive else "global_negative"
        m.id = m_id
        m.type = Marker.CUBE
        m.action = Marker.ADD
        
        # Position
        m.pose.position.x = box['cx']
        m.pose.position.y = box['cy']
        m.pose.position.z = 0.2  # Float slightly above the floor
        m.pose.orientation.w = 1.0
        
        # Size (Calculate width and height from min/max)
        # Using max(val, 0.1) ensures the box never accidentally becomes completely flat/invisible
        m.scale.x = max(box['max_x'] - box['min_x'], 0.1)
        m.scale.y = max(box['max_y'] - box['min_y'], 0.1)
        m.scale.z = 0.4
        
        # Color
        if is_positive:
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
        else:
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            
        m.color.a = 0.6  # Semi-transparent
        
        return m

def main(args=None):
    rclpy.init(args=args)
    node = GlobalMemoryVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()