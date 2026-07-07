#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Point
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
        cov += np.eye(2) * 1e-5  # Your strictly tuned parameter
        covs.append(cov)
    return np.array(covs)

def transform_points(trans_mat, points):
    homogenous_pts = np.hstack((points, np.ones((points.shape[0], 1))))
    return np.dot(trans_mat, homogenous_pts.T).T[:, :2]

def plot_robot_and_deadspace(ax, x, y, yaw):
    ax.plot(x, y, 'go', markersize=10, zorder=10)
    line_length = 1.2
    ax.plot([x, x + line_length * math.cos(yaw)], [y, y + line_length * math.sin(yaw)], 'g-', linewidth=2.5, zorder=10)
    angle_pos = yaw + math.radians(140)
    angle_neg = yaw - math.radians(140)
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
        
        self.get_logger().info('NDT Node running. Waiting for snapshots...')

    def bounds_callback(self, msg):
        self.map_bounds = msg

    def snapshot_callback(self, msg):
        if not self.map_bounds:
            self.get_logger().warn('No map bounds yet. Ignoring snapshot.')
            return
            
        self.get_logger().info('Snapshot received! Running NDT Alignment...')
        
        target_points = np.array([[p.x, p.y] for p in msg.global_points])
        source_points = np.array([[p.x, p.y] for p in msg.local_points])
        
        if len(target_points) == 0 or len(source_points) == 0: return

        target_covs = compute_target_covariances(target_points)
        initial_trans_mat = np.eye(3)
        drift_mat = self.ndt_scan_matching(initial_trans_mat, source_points, target_points, target_covs)
        
        aligned_source_points = transform_points(drift_mat, source_points)

        # Pose & Drift Calculations
        amcl_mat = expmap([msg.amcl_x, msg.amcl_y, msg.amcl_yaw])
        true_mat = np.dot(drift_mat, amcl_mat)
        
        true_x, true_y = true_mat[0, 2], true_mat[1, 2]
        true_yaw = math.atan2(true_mat[1, 0], true_mat[0, 0])
        
        drift_x = drift_mat[0, 2]
        drift_y = drift_mat[1, 2]
        drift_yaw = math.atan2(drift_mat[1, 0], drift_mat[0, 0])

        # New Obstacles
        global_tree = KDTree(target_points)
        distances, _ = global_tree.query(aligned_source_points)
        unmatched_mask = distances > 0.2
        unmatched_local_points = aligned_source_points[unmatched_mask]
        matched_local_points = aligned_source_points[~unmatched_mask]

        self.publish_changes(unmatched_local_points)

        self.save_plot(target_points, source_points, aligned_source_points, 
                       matched_local_points, unmatched_local_points, 
                       msg, true_x, true_y, true_yaw, drift_x, drift_y, drift_yaw)

    def publish_changes(self, new_obstacles):
        update_msg = MapUpdate()
        update_msg.header.stamp = self.get_clock().now().to_msg()
        update_msg.header.frame_id = 'map'
        
        cluster = ClusterChange()
        cluster.change_type = ClusterChange.POSITIVE_CHANGE
        for pt in new_obstacles:
            cluster.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=0.0))
            
        update_msg.clusters.append(cluster)
        self.pub_update.publish(update_msg)

    def save_plot(self, target, source, aligned, matched, unmatched, msg, tx, ty, tyaw, dx, dy, dyaw):
        # We make the figure slightly taller (8 inches instead of 6) to fit the text block at the bottom
        fig = plt.figure(figsize=(18, 8))
        
        ax1 = fig.add_subplot(131)
        ax1.set_title("1. Initial Guess (AMCL)")
        ax1.scatter(target[:, 0], target[:, 1], c='gray', s=5, alpha=0.5)
        ax1.scatter(source[:, 0], source[:, 1], c='red', s=5, alpha=0.8)
        plot_robot_and_deadspace(ax1, msg.amcl_x, msg.amcl_y, msg.amcl_yaw)
        ax1.axis('equal')

        ax2 = fig.add_subplot(132)
        ax2.set_title("2. Corrected (Post-NDT)")
        ax2.scatter(target[:, 0], target[:, 1], c='gray', s=5, alpha=0.5)
        ax2.scatter(aligned[:, 0], aligned[:, 1], c='blue', s=5, alpha=0.8)
        plot_robot_and_deadspace(ax2, tx, ty, tyaw)
        ax2.axis('equal')

        ax3 = fig.add_subplot(133)
        ax3.set_title("3. Positive Change Marking")
        ax3.set_xlim(self.map_bounds.min_x, self.map_bounds.max_x)
        ax3.set_ylim(self.map_bounds.min_y, self.map_bounds.max_y)
        ax3.scatter(target[:, 0], target[:, 1], c='gray', s=5, alpha=0.3)
        ax3.scatter(unmatched[:, 0], unmatched[:, 1], c='blue', s=15, marker='x')
        plot_robot_and_deadspace(ax3, tx, ty, tyaw)
        ax3.set_aspect('equal')

        # --- DIAGNOSTICS TEXT BLOCK ---
        diag_text = (
            f"AMCL Confidence Score     |  {msg.amcl_confidence:.4f}\n"
            f"Initial Pose (AMCL)       |  X: {msg.amcl_x:.4f}m   |   Y: {msg.amcl_y:.4f}m   |   Yaw: {msg.amcl_yaw:.4f}rad\n"
            f"Corrected Pose (NDT)      |  X: {tx:.4f}m   |   Y: {ty:.4f}m   |   Yaw: {tyaw:.4f}rad\n"
            f"Detected Drift            |  ΔX: {dx:+.4f}m   |   ΔY: {dy:+.4f}m   |   ΔYaw: {dyaw:+.4f}rad"
        )
        
        # Add text to the bottom of the figure
        fig.text(0.5, 0.05, diag_text, ha='center', va='bottom', fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))
        # Adjust layout to make room for the text box
        plt.subplots_adjust(bottom=0.2)
        
        filename = os.path.join(self.save_dir, f"snapshot_{int(time.time())}.png")
        plt.savefig(filename)
        plt.close(fig)
        self.get_logger().info(f'Diagnostic plot saved to {filename}')

    def ndt_scan_matching(self, trans_mat, source_points, target_points, target_covs):
        max_iter_num = 15
        damping = 1e-5  # Your tuned damping parameter
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
                
                # Your strict distance cutoff
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
