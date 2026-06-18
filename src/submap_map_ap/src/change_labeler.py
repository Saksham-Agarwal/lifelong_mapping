#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import numpy as np
import cv2

from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Quaternion

class ChangeClusterLabeler(Node):
    def __init__(self):
        super().__init__('change_cluster_labeler')

        # Standard QoS for OccupancyGrids
        map_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.VOLATILE, 
            reliability=ReliabilityPolicy.RELIABLE
        )

        # 1. Subscriptions
        self.sub_positive = self.create_subscription(
            OccupancyGrid, '/changes/positive_near_neighbour', self.positive_callback, map_qos)
            
        self.sub_negative = self.create_subscription(
            OccupancyGrid, '/changes/negative_nearest_neighbour', self.negative_callback, map_qos)

        # 2. Publishers
        self.pub_pos_det = self.create_publisher(Detection2DArray, '/cluster_positive', 10)
        self.pub_pos_mark = self.create_publisher(MarkerArray, '/cluster_positive_markers', 10)
        
        self.pub_neg_det = self.create_publisher(Detection2DArray, '/cluster_negative', 10)
        self.pub_neg_mark = self.create_publisher(MarkerArray, '/cluster_negative_markers', 10)
        
        self.get_logger().info("Change Labeler initialized and waiting for grids...")

    def positive_callback(self, msg):
        self.get_logger().info("Processing positive near neighbour grid...")
        # Process all clusters (min 1 cell to avoid purely empty arrays)
        self.process_clusters(msg, min_cells=10, det_pub=self.pub_pos_det, 
                              mark_pub=self.pub_pos_mark, ns="pos_cluster", 
                              color=(0.0, 1.0, 0.0, 0.4)) # Green for Positive

    def negative_callback(self, msg):
        self.get_logger().info("Processing negative nearest neighbour grid...")
        # Process only clusters > 25 cells
        self.process_clusters(msg, min_cells=25, det_pub=self.pub_neg_det, 
                              mark_pub=self.pub_neg_mark, ns="neg_cluster", 
                              color=(1.0, 0.0, 0.0, 0.4)) # Red for Negative

    def process_clusters(self, msg, min_cells, det_pub, mark_pub, ns, color):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # Convert msg.data to a numpy array
        grid = np.array(msg.data).reshape((height, width))

        # Create mask of occupied spaces (assuming standard 0-100 probabilities, > 50 is occupied)
        occupied_mask = np.where(grid > 50, 255, 0).astype(np.uint8)

        # Find external contours
        contours, _ = cv2.findContours(occupied_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detection_array = Detection2DArray()
        detection_array.header = msg.header 
        marker_array = MarkerArray()

        # Add a DELETEALL marker first to prevent ghosting of old boxes in RViz when clusters disappear
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        object_id = 1
        
        for contour in contours:
            # Area in raw pixel/cell count
            area_cells = cv2.contourArea(contour)

            # Apply size threshold
            if area_cells < min_cells:
                continue

            rect = cv2.minAreaRect(contour)
            (cx_pixel, cy_pixel), (w_pixel, h_pixel), angle = rect

            size_x = w_pixel * resolution
            size_y = h_pixel * resolution
            world_cx = (cx_pixel * resolution) + origin_x
            world_cy = (cy_pixel * resolution) + origin_y

            yaw_rad = np.radians(angle)
            q_z = float(np.sin(yaw_rad / 2.0))
            q_w = float(np.cos(yaw_rad / 2.0))

            # ---------------------------
            # 1. Publish Vision Data
            # ---------------------------
            detection = Detection2D()
            detection.header = msg.header
            detection.bbox.center.position.x = float(world_cx)
            detection.bbox.center.position.y = float(world_cy)
            detection.bbox.center.theta = float(yaw_rad)
            detection.bbox.size_x = float(size_x)
            detection.bbox.size_y = float(size_y)

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = f"{ns}_{object_id}"
            hypothesis.hypothesis.score = 1.0
            detection.results.append(hypothesis)
            detection_array.detections.append(detection)

            # ---------------------------
            # 2. Publish RViz Box
            # ---------------------------
            box_marker = Marker()
            box_marker.header = msg.header
            box_marker.ns = f"{ns}_boxes"
            box_marker.id = object_id
            box_marker.type = Marker.CUBE
            box_marker.action = Marker.ADD
            box_marker.pose.position.x = float(world_cx)
            box_marker.pose.position.y = float(world_cy)
            box_marker.pose.position.z = 0.1 
            box_marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=q_z, w=q_w)
            box_marker.scale.x = max(float(size_x), 0.1) # Ensure min thickness
            box_marker.scale.y = max(float(size_y), 0.1)
            box_marker.scale.z = 0.2 
            box_marker.color.r = color[0]
            box_marker.color.g = color[1]
            box_marker.color.b = color[2]
            box_marker.color.a = color[3]
            marker_array.markers.append(box_marker)

            # ---------------------------
            # 3. Publish RViz Text
            # ---------------------------
            text_marker = Marker()
            text_marker.header = msg.header
            text_marker.ns = f"{ns}_labels"
            text_marker.id = object_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(world_cx)
            text_marker.pose.position.y = float(world_cy)
            text_marker.pose.position.z = 0.4 
            text_marker.text = f"{ns.split('_')[0].capitalize()} Obj {object_id}"
            text_marker.scale.z = 0.25 
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            marker_array.markers.append(text_marker)

            object_id += 1

        det_pub.publish(detection_array)
        mark_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = ChangeClusterLabeler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()