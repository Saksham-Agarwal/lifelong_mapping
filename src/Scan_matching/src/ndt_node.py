#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
import numpy as np
import math
import os
import time
from scipy.spatial import KDTree

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from submap_map_ap.msg import MapSnapshot, MapBounds, MapUpdate, ClusterChange

def skewd(p): return np.array([-p[1], p[0]])
def expmap(delta):
    x, y, theta = delta[0], delta[1], delta[2]
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

def compute_target_covariances(points, k=5):
    tree = KDTree(points)
    covs = []
    for pt in points:
        _, idxs = tree.query(pt, k=k)
        neighbors = points[idxs]
        cov = np.cov(neighbors[:, :2], rowvar=False) if len(neighbors) > 1 else np.eye(2) * 1e-3
        cov += np.eye(2) * 1e-5  
        covs.append(cov)
    return np.array(covs)

def transform_points(trans_mat, points):
    homogenous_pts = np.hstack((points, np.ones((points.shape[0], 1))))
    return np.dot(trans_mat, homogenous_pts.T).T[:, :2]

def filter_occluded_points(global_points, local_points, rx, ry, ryaw, angular_res_deg=0.5, margin=0.3):
    if len(global_points) == 0:
        return global_points, np.array([])
        
    angular_res = math.radians(angular_res_deg)
    
    local_shifted = local_points - np.array([rx, ry])
    global_shifted = global_points - np.array([rx, ry])
    
    local_angles = np.arctan2(local_shifted[:, 1], local_shifted[:, 0])
    local_radii = np.linalg.norm(local_shifted, axis=1)
    
    global_angles = np.arctan2(global_shifted[:, 1], global_shifted[:, 0])
    global_radii = np.linalg.norm(global_shifted, axis=1)
    
    # --- UPDATED: 100 DEGREE DEADZONE (Active FOV is now +/- 130 deg) ---
    rel_global_angles = (global_angles - ryaw + np.pi) % (2 * np.pi) - np.pi
    deadzone_mask = np.abs(rel_global_angles) > math.radians(130)
    
    local_bins = np.floor((local_angles + np.pi) / angular_res).astype(int)
    global_bins = np.floor((global_angles + np.pi) / angular_res).astype(int)
    
    num_bins = int(np.ceil(2 * np.pi / angular_res))
    bin_hit_dist = np.full(num_bins, np.inf)
    np.minimum.at(bin_hit_dist, local_bins, local_radii)
    
    visible_mask = global_radii <= (bin_hit_dist[global_bins] + margin)
    final_keep_mask = visible_mask & ~deadzone_mask
    
    visible_global = global_points[final_keep_mask]
    occluded_global = global_points[~final_keep_mask]
    
    return visible_global, occluded_global

def plot_robot_and_deadspace(ax, x, y, yaw):
    ax.plot(x, y, 'go', markersize=10, zorder=10)
    line_length = 0.5 
    ax.plot([x, x + line_length * math.cos(yaw)], [y, y + line_length * math.sin(yaw)], 'g-', linewidth=2.5, zorder=10)
    
    # --- UPDATED: 100 DEGREE DEADZONE VISUALS (+/- 130 deg) ---
    angle_pos = yaw + math.radians(130)
    angle_neg = yaw - math.radians(130)
    ax.plot([x, x + line_length * math.cos(angle_pos)], [y, y + line_length * math.sin(angle_pos)], 'k--', linewidth=1.5, zorder=9)
    ax.plot([x, x + line_length * math.cos(angle_neg)], [y, y + line_length * math.sin(angle_neg)], 'k--', linewidth=1.5, zorder=9)

class NDTNode(Node):
    def __init__(self):
        super().__init__('ndt_node')
        self.save_dir = os.path.join(os.getcwd(), 'saves')
        os.makedirs(self.save_dir, exist_ok=True)
        
        latching_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.map_bounds = None
        
        self.sub_bounds = self.create_subscription(MapBounds, '/map_bounds', self.bounds_callback, latching_qos)
        self.sub_snapshot = self.create_subscription(MapSnapshot, '/map_snapshot_data', self.snapshot_callback, 10)
        
        self.pub_update = self.create_publisher(MapUpdate, '/map_changes', 10)
        self.pub_sanity = self.create_publisher(Bool, '/sanity_ndt', 10)
        
        self.sanity_dist_thresh = 0.6  
        self.sanity_yaw_thresh = 0.5   
        
        self.get_logger().info('NDT Node running. Waiting for snapshots...')

    def bounds_callback(self, msg):
        self.map_bounds = msg

    def snapshot_callback(self, msg):
        if not self.map_bounds:
            self.get_logger().warn('No map bounds yet. Ignoring snapshot.')
            return
            
        self.get_logger().info('Snapshot received! Cropping scan data and aligning...')
        
        target_points = np.array([[p.x, p.y] for p in msg.global_points])
        raw_source_points = np.array([[p.x, p.y] for p in msg.local_points])
        
        if len(target_points) == 0 or len(raw_source_points) == 0: 
            return

        amcl_x, amcl_y = msg.amcl_x, msg.amcl_y
        bbox_mask_x = (raw_source_points[:, 0] >= amcl_x - 3.0) & (raw_source_points[:, 0] <= amcl_x + 3.0)
        bbox_mask_y = (raw_source_points[:, 1] >= amcl_y - 3.0) & (raw_source_points[:, 1] <= amcl_y + 3.0)
        source_points = raw_source_points[bbox_mask_x & bbox_mask_y]

        if len(source_points) == 0:
            return

        target_covs = compute_target_covariances(target_points)
        initial_trans_mat = np.eye(3)
        drift_mat = self.ndt_scan_matching(initial_trans_mat, source_points, target_points, target_covs)
        
        aligned_source_points = transform_points(drift_mat, source_points)

        amcl_mat = expmap([msg.amcl_x, msg.amcl_y, msg.amcl_yaw])
        true_mat = np.dot(drift_mat, amcl_mat)
        
        true_x, true_y = true_mat[0, 2], true_mat[1, 2]
        true_yaw = math.atan2(true_mat[1, 0], true_mat[0, 0])
        
        dx = true_x - msg.amcl_x
        dy = true_y - msg.amcl_y
        dyaw = math.atan2(math.sin(true_yaw - msg.amcl_yaw), math.cos(true_yaw - msg.amcl_yaw))
        
        drift_distance = math.hypot(dx, dy)

        sanity_msg = Bool()
        
        if drift_distance > self.sanity_dist_thresh or abs(dyaw) > self.sanity_yaw_thresh:
            self.get_logger().warn(f"NDT drifted too far! (Dist: {drift_distance:.2f}m). /sanity_ndt = False. Falling back to AMCL.")
            sanity_msg.data = False
            
            aligned_source_points = source_points
            true_x, true_y, true_yaw = msg.amcl_x, msg.amcl_y, msg.amcl_yaw
            dx, dy, dyaw = 0.0, 0.0, 0.0
        else:
            self.get_logger().info("/sanity_ndt = True.")
            sanity_msg.data = True
            
        self.pub_sanity.publish(sanity_msg)

        visible_global, occluded_global = filter_occluded_points(
            target_points, aligned_source_points, true_x, true_y, true_yaw
        )
        
        if len(visible_global) > 0:
            global_tree = KDTree(target_points) 
            distances_pos, _ = global_tree.query(aligned_source_points)
            unmatched_local_points = aligned_source_points[distances_pos > 0.2]
            
            local_tree = KDTree(aligned_source_points)
            distances_neg, _ = local_tree.query(visible_global)
            missing_global_points = visible_global[distances_neg > 0.2]
        else:
            unmatched_local_points = aligned_source_points
            missing_global_points = np.array([])

        self.publish_changes(unmatched_local_points)

        self.save_plot(target_points, source_points, aligned_source_points, 
                       visible_global, occluded_global, unmatched_local_points, missing_global_points,
                       msg, true_x, true_y, true_yaw, dx, dy, dyaw)

    def publish_changes(self, new_obstacles):
        if len(new_obstacles) == 0: return
        update_msg = MapUpdate()
        update_msg.header.stamp = self.get_clock().now().to_msg()
        update_msg.header.frame_id = 'map'
        
        cluster = ClusterChange()
        cluster.change_type = ClusterChange.POSITIVE_CHANGE
        for pt in new_obstacles:
            cluster.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=0.0))
            
        update_msg.clusters.append(cluster)
        self.pub_update.publish(update_msg)

    def save_plot(self, target, source, aligned, visible_global, occluded_global, positive, negative, msg, tx, ty, tyaw, dx, dy, dyaw):
        fig = plt.figure(figsize=(24, 7))
        
        plt_min_x, plt_max_x = msg.amcl_x - 3.0, msg.amcl_x + 3.0
        plt_min_y, plt_max_y = msg.amcl_y - 3.0, msg.amcl_y + 3.0

        def format_ax(ax, title):
            ax.set_title(title)
            ax.set_xlim(plt_min_x, plt_max_x)
            ax.set_ylim(plt_min_y, plt_max_y)
            ax.set_aspect('equal')

        ax1 = fig.add_subplot(141)
        format_ax(ax1, "1. Initial Guess (AMCL)")
        ax1.scatter(target[:, 0], target[:, 1], c='gray', s=5, alpha=0.5)
        ax1.scatter(source[:, 0], source[:, 1], c='red', s=5, alpha=0.8)
        plot_robot_and_deadspace(ax1, msg.amcl_x, msg.amcl_y, msg.amcl_yaw)

        ax2 = fig.add_subplot(142)
        format_ax(ax2, "2. Corrected (Post-NDT)")
        ax2.scatter(target[:, 0], target[:, 1], c='gray', s=5, alpha=0.5)
        ax2.scatter(aligned[:, 0], aligned[:, 1], c='blue', s=5, alpha=0.8)
        plot_robot_and_deadspace(ax2, tx, ty, tyaw)

        ax3 = fig.add_subplot(143)
        format_ax(ax3, "3. Classified Map Changes")
        if len(visible_global) > 0:
            ax3.scatter(visible_global[:, 0], visible_global[:, 1], c='gray', s=5, alpha=0.3, label="Visible Map")
        if len(positive) > 0:
            ax3.scatter(positive[:, 0], positive[:, 1], c='blue', s=15, marker='x', label="Positive (New)")
        if len(negative) > 0:
            ax3.scatter(negative[:, 0], negative[:, 1], c='orange', s=15, marker='x', label="Negative (Removed)")
        plot_robot_and_deadspace(ax3, tx, ty, tyaw)
        ax3.legend(loc='upper right')

        ax4 = fig.add_subplot(144)
        format_ax(ax4, "4. Filter Masking (Shadows + Deadzone)")
        if len(occluded_global) > 0:
            ax4.scatter(occluded_global[:, 0], occluded_global[:, 1], c='black', s=5, alpha=0.15, label="Discarded")
        if len(visible_global) > 0:
            ax4.scatter(visible_global[:, 0], visible_global[:, 1], c='green', s=5, alpha=0.6, label="Kept")
        plot_robot_and_deadspace(ax4, tx, ty, tyaw)
        ax4.legend(loc='upper right')

        diag_text = (
            f"AMCL Confidence Score     |  {msg.amcl_confidence:.4f}\n"
            f"Initial Pose (AMCL)       |  X: {msg.amcl_x:.4f}m   |   Y: {msg.amcl_y:.4f}m   |   Yaw: {msg.amcl_yaw:.4f}rad\n"
            f"Corrected Pose (NDT)      |  X: {tx:.4f}m   |   Y: {ty:.4f}m   |   Yaw: {tyaw:.4f}rad\n"
            f"Detected Drift            |  ΔX: {dx:+.4f}m   |   ΔY: {dy:+.4f}m   |   ΔYaw: {dyaw:+.4f}rad"
        )
        fig.text(0.5, 0.05, diag_text, ha='center', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))
        plt.subplots_adjust(bottom=0.2)
        
        filename = os.path.join(self.save_dir, f"snapshot_{int(time.time())}.png")
        plt.savefig(filename)
        plt.close(fig)
        self.get_logger().info(f'Diagnostic plot saved to {filename}')

    def ndt_scan_matching(self, trans_mat, source_points, target_points, target_covs):
        max_iter_num = 15
        damping = 1e-5  
        kdtree = KDTree(target_points)
        
        for iter_num in range(max_iter_num):
            H = np.zeros((3, 3))
            b = np.zeros(3)
            R = trans_mat[:2, :2]
            corresponding_points_num = 0
            
            for i in range(len(source_points)):
                pt = np.array([source_points[i][0], source_points[i][1], 1.0])
                query = np.dot(trans_mat, pt)[:2]
                dist, idx = kdtree.query(query)
                
                if dist > 0.3: continue
                
                tuning_constant = 0.15
                weight = 1.0 if dist <= tuning_constant else tuning_constant / dist
                target = target_points[idx]
                C = np.eye(3)
                C[0:2, 0:2] = target_covs[idx]
                
                try: IM = np.linalg.inv(C)
                except: continue
                    
                error = np.array([target[0] - query[0], target[1] - query[1], 0.0])
                v = np.dot(R, skewd(source_points[i]))
                J = np.zeros((3, 3))
                J[0:2, 0:2] = -R
                J[0, 2], J[1, 2] = -v[0], -v[1]
                
                H += weight * np.dot(J.T, np.dot(IM, J))
                b += weight * np.dot(J.T, np.dot(IM, error))
                corresponding_points_num += 1

            if corresponding_points_num < 5: break
            
            H += np.eye(3) * damping
            try: delta = np.linalg.solve(H, -b)
            except: break
                
            update = np.dot(delta, delta)
            trans_mat = np.dot(trans_mat, expmap(delta))
            if update < 1e-4: break
            
        return trans_mat

def main(args=None):
    rclpy.init(args=args)
    node = NDTNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__': main()