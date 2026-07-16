#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import math

import tf2_ros
from tf2_ros import TransformException

def get_yaw_from_quaternion(q):
    """Convert a ROS geometry_msgs Quaternion to Euler Yaw."""
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)

class CostmapComparator(Node):
    def __init__(self):
        super().__init__('costmap_comparator')
        
        # --- TF2 Setup ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.sub_simplified = self.create_subscription(
            OccupancyGrid, '/simplified_local_costmap', self.simplified_cb, 10)
        self.sub_robot = self.create_subscription(
            OccupancyGrid, '/robot_local_region', self.robot_cb, 10)
            
        # Publisher
        self.pub_changes = self.create_publisher(
            OccupancyGrid, '/changes/negative_nearest_neighbour', 10)
            
        # State variables
        self.grid_simplified = None
        self.grid_robot = None
        
        # Timer to process the grids at a fixed rate
        self.timer = self.create_timer(0.2, self.process_grids)

    def simplified_cb(self, msg):
        # Numpy optimization for instant matrix math
        data_arr = np.array(msg.data)
        simplified_arr = np.zeros_like(data_arr)
        
        simplified_arr[data_arr == -1] = -1
        simplified_arr[data_arr > 99] = 100

        self.grid_simplified = OccupancyGrid()
        self.grid_simplified.header = msg.header
        self.grid_simplified.info = msg.info
        self.grid_simplified.info.origin.position.z = 0.05   
        self.grid_simplified.data = simplified_arr.tolist()

    def robot_cb(self, msg):
        self.grid_robot = msg

    def world_to_grid(self, wx, wy, info):
        """Convert world coordinates to grid indices. 
        Using np.round to prevent floating point grid aliasing."""
        gx = np.round((wx - info.origin.position.x) / info.resolution).astype(int)
        gy = np.round((wy - info.origin.position.y) / info.resolution).astype(int)
        return gx, gy

    def grid_to_world(self, gx, gy, info):
        """Convert grid indices to world coordinates."""
        wx = (gx * info.resolution) + info.origin.position.x
        wy = (gy * info.resolution) + info.origin.position.y
        return wx, wy

    def process_grids(self):
        if not self.grid_simplified or not self.grid_robot:
            return

        # --- Get the Transform ---
        frame_simp = self.grid_simplified.header.frame_id
        frame_rob = self.grid_robot.header.frame_id

        try:
            # Look up transform FROM robot region frame TO simplified costmap frame
            t = self.tf_buffer.lookup_transform(
                frame_simp, 
                frame_rob, 
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform {frame_rob} to {frame_simp}: {ex}')
            return

        # Extract translation and rotation
        tx = t.transform.translation.x
        ty = t.transform.translation.y
        yaw = get_yaw_from_quaternion(t.transform.rotation)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # 1. Extract data and info
        info_simp = self.grid_simplified.info
        info_rob = self.grid_robot.info

        data_simp = np.array(self.grid_simplified.data).reshape((info_simp.height, info_simp.width))
        data_rob = np.array(self.grid_robot.data).reshape((info_rob.height, info_rob.width))

        # Output grid now matches the SMALLER simplified region dimensions!
        out_data = np.zeros_like(data_simp)

        # 2. Find unexplored cells (-1) strictly in /robot_local_region
        gy_rob, gx_rob = np.where(data_rob == -1)
        
        if len(gx_rob) == 0:
            self.publish_grid(out_data, info_simp, self.grid_simplified.header.frame_id)
            return

        # 3. Convert these indices to world coordinates (Relative to frame_rob)
        wx_rob, wy_rob = self.grid_to_world(gx_rob, gy_rob, info_rob)

        # --- NEW: Apply TF Rotation and Translation ---
        # 3.5 Convert world coords from frame_rob to frame_simp
        wx_simp = (wx_rob * cos_yaw) - (wy_rob * sin_yaw) + tx
        wy_simp = (wx_rob * sin_yaw) + (wy_rob * cos_yaw) + ty

        # 4. Map the newly transformed world coordinates to the smaller /simplified_local_costmap indices
        gx_simp, gy_simp = self.world_to_grid(wx_simp, wy_simp, info_simp)

        # 5. Filter bounds to ensure we only look at points that actually fall inside the smaller map
        valid_simp_mask = (gx_simp >= 0) & (gx_simp < info_simp.width) & \
                          (gy_simp >= 0) & (gy_simp < info_simp.height)
        
        valid_gx_simp = gx_simp[valid_simp_mask]
        valid_gy_simp = gy_simp[valid_simp_mask]
        
        # 6. Evaluate condition: Are these mapped cells FREE (0) in the simplified costmap?
        free_mask = data_simp[valid_gy_simp, valid_gx_simp] == 0
        
        # 7. Get the final coordinates in the simplified grid that met all rules
        final_gx = valid_gx_simp[free_mask]
        final_gy = valid_gy_simp[free_mask]

        # Mark as occupied (100) in output map
        out_data[final_gy, final_gx] = 100

        # Publish the resulting map using the simplified grid's metadata
        self.publish_grid(out_data, info_simp, self.grid_simplified.header.frame_id)

    def publish_grid(self, data_array, info, frame_id):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.info = info
        msg.data = data_array.flatten().astype(np.int8).tolist()
        
        self.pub_changes.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CostmapComparator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()