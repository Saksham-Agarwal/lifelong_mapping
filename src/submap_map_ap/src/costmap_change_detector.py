#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import SetParametersResult
import numpy as np
import math

import tf2_ros
from tf2_ros import TransformException

def get_yaw_from_quaternion(q):
    """Convert a ROS geometry_msgs Quaternion to Euler Yaw."""
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)

class CostmapChangeDetector(Node):
    def __init__(self):
        super().__init__('costmap_change_detector')

        # --- Parameters ---
        self.declare_parameter('positive_noise_threshold', 10)
        self.noise_threshold = self.get_parameter('positive_noise_threshold').value

        self.declare_parameter('inflated_obstacle_threshold', 50) 
        self.inflated_obs_threshold = self.get_parameter('inflated_obstacle_threshold').value

        self.add_on_set_parameters_callback(self.parameter_callback)

        # --- TF2 Setup ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- Subscribers ---
        self.inflated_sub = self.create_subscription(
            OccupancyGrid,
            '/inflated_local_region',
            self.inflated_callback,
            10
        )
        self.simplified_sub = self.create_subscription(
            OccupancyGrid,
            '/simplified_local_costmap',
            self.simplified_callback,
            10
        )

        # --- Publishers ---
        self.pos_pub = self.create_publisher(OccupancyGrid, '/change/positive', 10)
        self.neg_pub = self.create_publisher(OccupancyGrid, '/change/negative', 10)

        self.latest_inflated_msg = None
        self.get_logger().info("TF-Aware Costmap Change Detector started.")

    def parameter_callback(self, params):
        for param in params:
            if param.type_ != param.Type.INTEGER:
                continue
            if param.name == 'positive_noise_threshold':
                self.noise_threshold = param.value
            elif param.name == 'inflated_obstacle_threshold':
                self.inflated_obs_threshold = param.value
        return SetParametersResult(successful=True)

    def inflated_callback(self, msg):
        self.latest_inflated_msg = msg

    def simplified_callback(self, simplified_msg):
        if self.latest_inflated_msg is None:
            return

        inflated_msg = self.latest_inflated_msg

        # --- Get the Transform ---
        frame_A = simplified_msg.header.frame_id # likely 'odom'
        frame_B = inflated_msg.header.frame_id   # likely 'map'

        try:
            # We want the transform from Grid A's frame TO Grid B's frame
            t = self.tf_buffer.lookup_transform(
                frame_B, 
                frame_A, 
                rclpy.time.Time() # Get the latest available transform
            )
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform {frame_A} to {frame_B}: {ex}')
            return

        # Extract translation and rotation
        tx = t.transform.translation.x
        ty = t.transform.translation.y
        yaw = get_yaw_from_quaternion(t.transform.rotation)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # --- Extract Grid A Metadata ---
        res_A = simplified_msg.info.resolution
        orig_A_x = simplified_msg.info.origin.position.x
        orig_A_y = simplified_msg.info.origin.position.y
        width_A = simplified_msg.info.width
        height_A = simplified_msg.info.height
        data_A = np.array(simplified_msg.data, dtype=np.int8).reshape((height_A, width_A))

        # --- Extract Grid B Metadata ---
        res_B = inflated_msg.info.resolution
        orig_B_x = inflated_msg.info.origin.position.x
        orig_B_y = inflated_msg.info.origin.position.y
        width_B = inflated_msg.info.width
        height_B = inflated_msg.info.height
        data_B = np.array(inflated_msg.data, dtype=np.int8).reshape((height_B, width_B))

        # --- Coordinate Math ---
        y_A_indices, x_A_indices = np.mgrid[0:height_A, 0:width_A]
        
        # 1. Calculate world coordinates in Frame A (odom)
        world_A_x = orig_A_x + (x_A_indices * res_A)
        world_A_y = orig_A_y + (y_A_indices * res_A)

        # 2. Transform these coordinates into Frame B (map) using 2D Affine Rotation/Translation
        world_B_x = (world_A_x * cos_yaw) - (world_A_y * sin_yaw) + tx
        world_B_y = (world_A_x * sin_yaw) + (world_A_y * cos_yaw) + ty

        # 3. Find the indices in Grid B based on the transformed coordinates
        x_B_indices = np.round((world_B_x - orig_B_x) / res_B).astype(int)
        y_B_indices = np.round((world_B_y - orig_B_y) / res_B).astype(int)

        valid_mask = (
            (x_B_indices >= 0) & (x_B_indices < width_B) &
            (y_B_indices >= 0) & (y_B_indices < height_B)
        )

        pos_out_data = np.zeros_like(data_A)
        pos_out_data[data_A == -1] = -1
        
        neg_out_data = np.zeros_like(data_A)
        neg_out_data[data_A == -1] = -1

        valid_A_pixels = data_A[valid_mask]
        valid_B_pixels = data_B[y_B_indices[valid_mask], x_B_indices[valid_mask]]

        calc_mask = (valid_A_pixels != -1) & (valid_B_pixels != -1)

        calc_A = valid_A_pixels[calc_mask].astype(np.int16)
        calc_B_raw = valid_B_pixels[calc_mask].astype(np.int16)

        calc_B = np.where(calc_B_raw >= self.inflated_obs_threshold, 100, 0).astype(np.int16)

        diff = calc_A - calc_B

        pos_values = np.where(diff > self.noise_threshold, diff, 0).astype(np.int8)
        neg_values = np.where(diff < -self.noise_threshold, -diff, 0).astype(np.int8)

        overlap_pos = np.zeros_like(valid_A_pixels)
        overlap_pos[valid_A_pixels == -1] = -1
        overlap_pos[calc_mask] = pos_values
        
        overlap_neg = np.zeros_like(valid_A_pixels)
        overlap_neg[valid_A_pixels == -1] = -1
        overlap_neg[calc_mask] = neg_values

        pos_out_data[valid_mask] = overlap_pos
        neg_out_data[valid_mask] = overlap_neg

        pos_msg = OccupancyGrid()
        pos_msg.header = simplified_msg.header
        pos_msg.info = simplified_msg.info
        pos_msg.data = pos_out_data.flatten().tolist()
        self.pos_pub.publish(pos_msg)

        neg_msg = OccupancyGrid()
        neg_msg.header = simplified_msg.header
        neg_msg.info = simplified_msg.info
        neg_msg.data = neg_out_data.flatten().tolist()
        self.neg_pub.publish(neg_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CostmapChangeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()