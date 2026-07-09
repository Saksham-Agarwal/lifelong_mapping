#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import numpy as np
import cv2

# Data and Visualization Messages
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Quaternion

class MapObjectVisualizer(Node):
    def __init__(self):
        super().__init__('map_object_visualizer')

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )

        self.subscription = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.data_publisher = self.create_publisher(Detection2DArray, '/detected_map_objects', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/detected_map_markers', 10)
        
        self.get_logger().info("Node ready. Publishing data and markers based on Unexplored Islands.")

    def map_callback(self, msg):
        self.get_logger().info("Processing new map matrix...")
        
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        grid = np.array(msg.data).reshape((height, width))

        # Binary Masks
        occupied_mask = np.where(grid > 50, 255, 0).astype(np.uint8)
        unexplored_mask = np.where(grid == -1, 255, 0).astype(np.uint8)

        # REVERSE LOGIC: Find contours of the Unexplored space instead!
        contours, _ = cv2.findContours(unexplored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detection_array = Detection2DArray()
        detection_array.header = msg.header 
        marker_array = MarkerArray()

        object_id = 1
        
        for contour in contours:
            # 1. Filter out tiny artifacts and the massive "outside world"
            area = cv2.contourArea(contour)
            if area < 4.0 or area > (width * height * 0.05): # Reject if larger than 5% of total map
                continue

            # 2. Filter out anything touching the very edge of the map array
            x, y, w, h = cv2.boundingRect(contour)
            if x <= 1 or y <= 1 or (x + w) >= (width - 1) or (y + h) >= (height - 1):
                continue

            # 3. Verify this unexplored island is bordered by walls
            island_mask = np.zeros_like(unexplored_mask)
            cv2.drawContours(island_mask, [contour], -1, 255, thickness=cv2.FILLED)

            # Dilate the island slightly to create an outer search ring
            kernel = np.ones((5,5), np.uint8)
            dilated_island = cv2.dilate(island_mask, kernel, iterations=1)
            border_ring = cv2.bitwise_xor(dilated_island, island_mask)

            # Check if the border ring overlaps with occupied walls
            occupied_in_ring = cv2.bitwise_and(border_ring, occupied_mask)
            
            # If the ring contains occupied pixels, it is an enclosed object
            if np.count_nonzero(occupied_in_ring) > 5:
                
                # Use the dilated mask to calculate the bounding box (so it includes the walls)
                dilated_contours, _ = cv2.findContours(dilated_island, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not dilated_contours:
                    continue
                
                best_contour = max(dilated_contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(best_contour)
                (cx_pixel, cy_pixel), (w_pixel, h_pixel), angle = rect

                # Convert to real-world metric space
                size_x = w_pixel * resolution
                size_y = h_pixel * resolution
                world_cx = (cx_pixel * resolution) + origin_x
                world_cy = (cy_pixel * resolution) + origin_y

                yaw_rad = np.radians(angle)
                q_z = float(np.sin(yaw_rad / 2.0))
                q_w = float(np.cos(yaw_rad / 2.0))

                # ==========================================
                # 1. BUILD THE VISION_MSGS DATA PAYLOAD
                # ==========================================
                detection = Detection2D()
                detection.header = msg.header
                
                # ROS 2 Humble formatting
                detection.bbox.center.position.x = float(world_cx)
                detection.bbox.center.position.y = float(world_cy)
                detection.bbox.center.theta = float(yaw_rad)
                detection.bbox.size_x = float(size_x)
                detection.bbox.size_y = float(size_y)

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = f"object_{object_id}"
                hypothesis.hypothesis.score = 1.0
                detection.results.append(hypothesis)
                detection_array.detections.append(detection)

                # ==========================================
                # 2. BUILD THE RVIZ MARKER VISUALS
                # ==========================================
                box_marker = Marker()
                box_marker.header = msg.header
                box_marker.ns = "object_boxes"
                box_marker.id = object_id
                box_marker.type = Marker.CUBE
                box_marker.action = Marker.ADD
                
                box_marker.pose.position.x = float(world_cx)
                box_marker.pose.position.y = float(world_cy)
                box_marker.pose.position.z = 0.1 
                box_marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=q_z, w=q_w)
                
                box_marker.scale.x = float(size_x)
                box_marker.scale.y = float(size_y)
                box_marker.scale.z = 0.2 
                
                box_marker.color.r = 0.0
                box_marker.color.g = 1.0
                box_marker.color.b = 0.0
                box_marker.color.a = 0.4
                marker_array.markers.append(box_marker)

                text_marker = Marker()
                text_marker.header = msg.header
                text_marker.ns = "object_labels"
                text_marker.id = object_id
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                
                text_marker.pose.position.x = float(world_cx)
                text_marker.pose.position.y = float(world_cy)
                text_marker.pose.position.z = 0.4 
                
                text_marker.text = f"Object {object_id}"
                text_marker.scale.z = 0.25 
                
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                marker_array.markers.append(text_marker)

                object_id += 1

        self.data_publisher.publish(detection_array)
        self.marker_publisher.publish(marker_array)
        self.get_logger().info(f"Successfully processed and published details for {object_id - 1} objects.")

def main(args=None):
    rclpy.init(args=args)
    node = MapObjectVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()