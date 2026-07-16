#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, String
import numpy as np
import math
import json
from scipy.spatial import KDTree
from scipy.ndimage import minimum_filter1d

# Replace 'submap_map_ap' with your actual package name if different
from submap_map_ap.msg import MapSnapshot, MapBounds
from submap_map_ap.msg import AlignedMapChanges 

def skewd(p): return np.array([-p[1], p[0]])
def expmap(delta):
    x, y, theta = delta[0], delta[1], delta[2]
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

# Updated to accept config parameters
def compute_target_covariances(points, k_neighbors, fallback_scale, regularization):
    tree = KDTree(points)
    covs = []
    for pt in points:
        _, idxs = tree.query(pt, k=k_neighbors)
        neighbors = points[idxs]
        cov = np.cov(neighbors[:, :2], rowvar=False) if len(neighbors) > 1 else np.eye(2) * fallback_scale
        cov += np.eye(2) * regularization  
        covs.append(cov)
    return np.array(covs)

def transform_points(trans_mat, points):
    homogenous_pts = np.hstack((points, np.ones((points.shape[0], 1))))
    return np.dot(trans_mat, homogenous_pts.T).T[:, :2]

# Updated to accept deadzone_deg parameter
def filter_occluded_points(global_points, local_points, rx, ry, ryaw, deadzone_deg, angular_res_deg=1.5, margin=0.2):
    if len(global_points) == 0:
        return global_points, np.array([])
    angular_res = math.radians(angular_res_deg)
    local_shifted = local_points - np.array([rx, ry])
    global_shifted = global_points - np.array([rx, ry])
    local_angles = np.arctan2(local_shifted[:, 1], local_shifted[:, 0])
    local_radii = np.linalg.norm(local_shifted, axis=1)
    global_angles = np.arctan2(global_shifted[:, 1], global_shifted[:, 0])
    global_radii = np.linalg.norm(global_shifted, axis=1)
    rel_global_angles = (global_angles - ryaw + np.pi) % (2 * np.pi) - np.pi
    
    # Using deadzone parameter here
    deadzone_mask = np.abs(rel_global_angles) > math.radians(deadzone_deg)
    
    local_bins = np.floor((local_angles + np.pi) / angular_res).astype(int)
    global_bins = np.floor((global_angles + np.pi) / angular_res).astype(int)
    num_bins = int(np.ceil(2 * np.pi / angular_res))
    bin_hit_dist = np.full(num_bins, np.inf)
    np.minimum.at(bin_hit_dist, local_bins, local_radii)
    bin_hit_dist = minimum_filter1d(bin_hit_dist, size=3, mode='wrap')
    bin_hit_dist[bin_hit_dist == np.inf] = 3.0
    visible_mask = global_radii <= (bin_hit_dist[global_bins] + margin)
    final_keep_mask = visible_mask & ~deadzone_mask
    visible_global = global_points[final_keep_mask]
    occluded_global = global_points[~final_keep_mask]
    return visible_global, occluded_global

class NDTAlignerNode(Node):
    def __init__(self):
        super().__init__('ndt_aligner_node')
        latching_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.map_bounds = None
        
        # 1. Declare Parameters
        self.declare_parameter('sanity_dist_threshold', 0.5)
        self.declare_parameter('sanity_yaw_threshold', 0.5)
        self.declare_parameter('sanity_change_threshold', 50.0)
        self.declare_parameter('deadzone_of_bot', 70.0)
        self.declare_parameter('grid_comparison_size', 3.0)

        self.declare_parameter('ndt_aligner.covariance.k_neighbors', 5)
        self.declare_parameter('ndt_aligner.covariance.regularization', 1e-5)
        self.declare_parameter('ndt_aligner.covariance.fallback_scale', 1e-3)
        
        self.declare_parameter('ndt_aligner.optimization.max_iterations', 15)
        self.declare_parameter('ndt_aligner.optimization.damping_factor', 1e-5)
        self.declare_parameter('ndt_aligner.optimization.convergence_threshold', 1e-4)
        self.declare_parameter('ndt_aligner.optimization.min_corresponding_points', 5)
        
        self.declare_parameter('ndt_aligner.association.max_correspondence_distance', 0.3)
        self.declare_parameter('ndt_aligner.association.weight_distance_threshold', 0.15)

        # 2. Retrieve Parameters
        self.sanity_dist_thresh = self.get_parameter('sanity_dist_threshold').value
        self.sanity_yaw_thresh = self.get_parameter('sanity_yaw_threshold').value
        self.sanity_change_thresh = self.get_parameter('sanity_change_threshold').value
        self.deadzone_of_bot = self.get_parameter('deadzone_of_bot').value
        self.grid_comparison_size = self.get_parameter('grid_comparison_size').value

        self.k_neighbors = self.get_parameter('ndt_aligner.covariance.k_neighbors').value
        self.regularization = self.get_parameter('ndt_aligner.covariance.regularization').value
        self.fallback_scale = self.get_parameter('ndt_aligner.covariance.fallback_scale').value
        
        self.max_iterations = self.get_parameter('ndt_aligner.optimization.max_iterations').value
        self.damping_factor = self.get_parameter('ndt_aligner.optimization.damping_factor').value
        self.convergence_threshold = self.get_parameter('ndt_aligner.optimization.convergence_threshold').value
        self.min_corresponding_points = self.get_parameter('ndt_aligner.optimization.min_corresponding_points').value
        
        self.max_correspondence_distance = self.get_parameter('ndt_aligner.association.max_correspondence_distance').value
        self.weight_distance_threshold = self.get_parameter('ndt_aligner.association.weight_distance_threshold').value

        # Subscribers and Publishers
        self.sub_bounds = self.create_subscription(MapBounds, '/map_bounds', self.bounds_callback, latching_qos)
        self.sub_snapshot = self.create_subscription(MapSnapshot, '/map_snapshot_data', self.snapshot_callback, 10)
        
        self.pub_changes = self.create_publisher(AlignedMapChanges, '/aligned_map_changes', 10)
        self.pub_sanity = self.create_publisher(Bool, '/sanity_ndt', 10)
        self.pub_debug = self.create_publisher(String, '/ndt_debug_data', 10)
        
        self.get_logger().info('NDT Aligner Node running with Config Parameters.')

    def bounds_callback(self, msg):
        self.map_bounds = msg

    def snapshot_callback(self, msg):
        if not self.map_bounds:
            self.get_logger().warn('No map bounds yet. Ignoring snapshot.')
            return
            
        raw_global_points = np.array([[p.x, p.y] for p in msg.global_points])
        raw_submap_points = np.array([[p.x, p.y] for p in msg.submap_points]) 
        raw_source_points = np.array([[p.x, p.y] for p in msg.local_points])
        
        if len(raw_global_points) == 0 or len(raw_source_points) == 0: return
        
        raw_scan_count = len(raw_source_points)
        amcl_x, amcl_y, amcl_yaw = msg.amcl_x, msg.amcl_y, msg.amcl_yaw
        
        def crop_to_bbox(pts, cx, cy, margin):
            if len(pts) == 0: return pts
            mask_x = (pts[:, 0] >= cx - margin) & (pts[:, 0] <= cx + margin)
            mask_y = (pts[:, 1] >= cy - margin) & (pts[:, 1] <= cy + margin)
            return pts[mask_x & mask_y]

        # Use config parameter for margin
        source_points = crop_to_bbox(raw_source_points, amcl_x, amcl_y, self.grid_comparison_size)
        global_points = crop_to_bbox(raw_global_points, amcl_x, amcl_y, self.grid_comparison_size)
        submap_points = crop_to_bbox(raw_submap_points, amcl_x, amcl_y, self.grid_comparison_size)

        if len(source_points) == 0: return

        # Phase 1: Independent NDT alignment
        global_res = self.process_ndt_pipeline(global_points, source_points, amcl_x, amcl_y, amcl_yaw, raw_scan_count, is_submap=False)
        submap_res = self.process_ndt_pipeline(submap_points, source_points, amcl_x, amcl_y, amcl_yaw, raw_scan_count, is_submap=True)

        # Phase 2: Conflict Resolution
        if global_res and submap_res and global_res['sanity_ok'] and submap_res['sanity_ok']:
            dist_diff = math.hypot(submap_res['tx'] - global_res['tx'], submap_res['ty'] - global_res['ty'])
            yaw_diff = abs(math.atan2(math.sin(submap_res['tyaw'] - global_res['tyaw']), math.cos(submap_res['tyaw'] - global_res['tyaw'])))
            dist_thresh, yaw_thresh_rad = 0.01, math.radians(0.1)

            if dist_diff > dist_thresh or yaw_diff > yaw_thresh_rad:
                global_err = global_res['pos_ratio'] + global_res['neg_ratio']
                submap_err = submap_res['pos_ratio'] + submap_res['neg_ratio']
                if global_err <= submap_err:
                    submap_res = self.process_ndt_pipeline(submap_points, source_points, amcl_x, amcl_y, amcl_yaw, raw_scan_count, is_submap=True, override_drift_mat=global_res['drift_mat'])
                    global_res['status'] = "Winner (Independent)"
                    submap_res['status'] = "Forced (Aligned to Global)"
                else:
                    global_res = self.process_ndt_pipeline(global_points, source_points, amcl_x, amcl_y, amcl_yaw, raw_scan_count, is_submap=False, override_drift_mat=submap_res['drift_mat'])
                    submap_res['status'] = "Winner (Independent)"
                    global_res['status'] = "Forced (Aligned to Submap)"
            else:
                global_res['status'] = "Independent (Matched Submap)"
                submap_res['status'] = "Independent (Matched Global)"

        # Phase 3: Publish changes
        if submap_res and global_res:
            sanity_msg = Bool()
            if not submap_res['sanity_ok']:
                sanity_msg.data = False
            else:
                sanity_msg.data = True
                self.publish_changes(
                    global_res['positive'], global_res['negative'], 
                    submap_res['positive'], submap_res['negative']
                )
            self.pub_sanity.publish(sanity_msg)
        elif global_res:
            sanity_msg = Bool(data=False)
            self.pub_sanity.publish(sanity_msg)

        self.publish_debug_data(msg, source_points, global_points, submap_points, global_res, submap_res)

    def process_ndt_pipeline(self, target_points, source_points, amcl_x, amcl_y, amcl_yaw, raw_scan_count, is_submap=False, override_drift_mat=None):
        if len(target_points) == 0: return None
        center = np.array([amcl_x, amcl_y])
        centered_target = target_points - center
        centered_source = source_points - center
        
        if override_drift_mat is not None:
            drift_mat = override_drift_mat
        else:
            # Pass config params to global function
            target_covs = compute_target_covariances(centered_target, self.k_neighbors, self.fallback_scale, self.regularization)
            drift_mat = self.ndt_scan_matching(np.eye(3), centered_source, centered_target, target_covs)
        
        aligned_source_points = transform_points(drift_mat, centered_source) + center
        dx, dy = drift_mat[0, 2], drift_mat[1, 2]
        dyaw = math.atan2(drift_mat[1, 0], drift_mat[0, 0])
        true_x, true_y, true_yaw = amcl_x + dx, amcl_y + dy, amcl_yaw + dyaw
        drift_distance = math.hypot(dx, dy)

        sanity_ok = not (drift_distance > self.sanity_dist_thresh or abs(dyaw) > self.sanity_yaw_thresh)

        # Pass config param to global function
        visible_target, occluded_target = filter_occluded_points(
            target_points, aligned_source_points, true_x, true_y, true_yaw, self.deadzone_of_bot
        )
        
        if len(visible_target) > 0:
            target_tree = KDTree(target_points) 
            distances_pos, _ = target_tree.query(aligned_source_points)
            unmatched_local_points = aligned_source_points[distances_pos > 0.2]
            local_tree = KDTree(aligned_source_points)
            distances_neg, _ = local_tree.query(visible_target)
            missing_target_points = visible_target[distances_neg > 0.2]
        else:
            unmatched_local_points, missing_target_points = aligned_source_points, np.array([])

        def strict_crop(pts):
            if len(pts) == 0: return pts
            # Use config parameter for margin
            margin = self.grid_comparison_size
            mask = (pts[:, 0] >= true_x - margin) & (pts[:, 0] <= true_x + margin) & (pts[:, 1] >= true_y - margin) & (pts[:, 1] <= true_y + margin)
            return pts[mask]

        if sanity_ok:
            unmatched_local_points = strict_crop(unmatched_local_points)
            missing_target_points = strict_crop(missing_target_points)
            if len(unmatched_local_points) > 0:
                u_shifted = unmatched_local_points - np.array([true_x, true_y])
                # Using deadzone config here
                u_in_fov = np.abs((np.arctan2(u_shifted[:, 1], u_shifted[:, 0]) - true_yaw + np.pi) % (2 * np.pi) - np.pi) <= math.radians(self.deadzone_of_bot)
                unmatched_local_points = unmatched_local_points[u_in_fov]
        
        pos_count = len(unmatched_local_points)
        neg_count = len(missing_target_points)
        
        pos_ratio = (pos_count / raw_scan_count * 100) if raw_scan_count > 0 else 0.0
        
        expected_visible_cropped = strict_crop(visible_target)
        total_visible_map_points = len(expected_visible_cropped)
        
        neg_ratio = (neg_count / total_visible_map_points * 100) if total_visible_map_points > 0 else 0.0

        if is_submap and sanity_ok and (pos_ratio > self.sanity_change_thresh or neg_ratio > self.sanity_change_thresh):
            sanity_ok = False
            self.get_logger().warn(f"NDT Sanity Failed! Massive SUBMAP changes detected (Pos: {pos_ratio:.1f}%, Neg: {neg_ratio:.1f}%).")

        if not sanity_ok:
            aligned_source_points, true_x, true_y, true_yaw = source_points, amcl_x, amcl_y, amcl_yaw
            dx, dy, dyaw = 0.0, 0.0, 0.0
            unmatched_local_points = np.array([])
            missing_target_points = np.array([])
            pos_count, neg_count = 0, 0
            pos_ratio, neg_ratio = 0.0, 0.0
            status_msg = 'Rejected (Sanity Failed)'
        else:
            status_msg = 'Independent'

        return {
            'drift_mat': drift_mat, 
            'aligned': aligned_source_points,
            'tx': true_x, 'ty': true_y, 'tyaw': true_yaw,
            'dx': dx, 'dy': dy, 'dyaw': dyaw,
            'drift_distance': drift_distance,
            'sanity_ok': sanity_ok,
            'visible': visible_target,
            'occluded': occluded_target,
            'positive': unmatched_local_points,
            'negative': missing_target_points,
            'total_scan_freezone': raw_scan_count, 
            'pos_count': pos_count,
            'neg_count': neg_count,
            'pos_ratio': pos_ratio,
            'neg_ratio': neg_ratio,
            'status': status_msg
        }

    def publish_changes(self, g_pos, g_neg, s_pos, s_neg):
        msg = AlignedMapChanges()
        def to_pts(arr): return [Point(x=float(p[0]), y=float(p[1]), z=0.0) for p in arr]
        
        msg.global_positive = to_pts(g_pos)
        msg.global_negative = to_pts(g_neg)
        msg.submap_positive = to_pts(s_pos)
        msg.submap_negative = to_pts(s_neg)
        self.pub_changes.publish(msg)

    def publish_debug_data(self, snap_msg, source, global_tgt, submap_tgt, global_res, submap_res):
        def serialize_res(r):
            if not r: return None
            return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in r.items()}
            
        data = {
            'amcl': {'x': snap_msg.amcl_x, 'y': snap_msg.amcl_y, 'yaw': snap_msg.amcl_yaw, 'conf': snap_msg.amcl_confidence},
            'source': source.tolist() if len(source) else [],
            'global_target': global_tgt.tolist() if len(global_tgt) else [],
            'submap_target': submap_tgt.tolist() if len(submap_tgt) else [],
            'global_res': serialize_res(global_res),
            'submap_res': serialize_res(submap_res)
        }
        self.pub_debug.publish(String(data=json.dumps(data)))

    def ndt_scan_matching(self, trans_mat, source_points, target_points, target_covs):
        kdtree = KDTree(target_points)
        for iter_num in range(self.max_iterations):
            H, b = np.zeros((3, 3)), np.zeros(3)
            R, corresponding_points_num = trans_mat[:2, :2], 0
            for i in range(len(source_points)):
                pt = np.array([source_points[i][0], source_points[i][1], 1.0])
                query = np.dot(trans_mat, pt)[:2]
                dist, idx = kdtree.query(query)
                
                # Replaced hardcoded limits with Config limits
                if dist > self.max_correspondence_distance: continue
                weight = 1.0 if dist <= self.weight_distance_threshold else self.weight_distance_threshold / dist
                
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
                
            # Replaced hardcoded variables with Config 
            if corresponding_points_num < self.min_corresponding_points: break
            H += np.eye(3) * self.damping_factor
            
            try: delta = np.linalg.solve(H, -b)
            except: break
            update = np.dot(delta, delta)
            trans_mat = np.dot(trans_mat, expmap(delta))
            
            # Replaced hardcoded threshold with Config
            if update < self.convergence_threshold: break
        return trans_mat

def main(args=None):
    rclpy.init(args=args)
    try: rclpy.spin(NDTAlignerNode())
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()