#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import numpy as np
from scipy.spatial import KDTree

from submap_map_ap.msg import MapUpdate, ClusterChange
from submap_map_ap.msg import AlignedMapChanges

class ClusterCreatorNode(Node):
    def __init__(self):
        super().__init__('cluster_creator_node')
        
        self.sub_changes = self.create_subscription(
            AlignedMapChanges, 
            '/aligned_map_changes', 
            self.changes_callback, 
            10
        )
        self.pub_update = self.create_publisher(MapUpdate, '/map_changes', 10)
        
        self.get_logger().info('Cluster Creator Node running (Direct Mapping for state transitions).')

    def extract_pts(self, point_array):
        if not point_array: return np.array([])
        return np.array([[p.x, p.y] for p in point_array])

    def get_intersection_and_diff(self, pts_A, pts_B, threshold=0.2):
        """Returns (Points in both, Points in A ONLY)"""
        if len(pts_A) == 0: return np.array([]), np.array([])
        if len(pts_B) == 0: return np.array([]), pts_A
        
        tree_B = KDTree(pts_B)
        dists, _ = tree_B.query(pts_A)
        
        intersect_mask = dists <= threshold
        return pts_A[intersect_mask], pts_A[~intersect_mask]

    def changes_callback(self, msg):
        g_pos = self.extract_pts(msg.global_positive)
        g_neg = self.extract_pts(msg.global_negative)
        s_pos = self.extract_pts(msg.submap_positive)
        s_neg = self.extract_pts(msg.submap_negative)
        
        # 1. Standard Intersections
        both_pos, _ = self.get_intersection_and_diff(g_pos, s_pos)
        both_neg, _ = self.get_intersection_and_diff(g_neg, s_neg)
        
        # 2. Submap Unique Differences (State Transitions)
        _, submap_only_neg = self.get_intersection_and_diff(s_neg, g_neg)
        _, submap_only_pos = self.get_intersection_and_diff(s_pos, g_pos)

        update_msg = MapUpdate()
        update_msg.header.stamp = self.get_clock().now().to_msg()
        update_msg.header.frame_id = 'map'

        def add_clusters(points, change_type, use_clustering=True):
            if len(points) == 0: return
            
            # --- NEW: Bypass Nearest Neighbor Clustering entirely ---
            # If false, dumps all raw points directly into a single cluster package
            if not use_clustering:
                cc = ClusterChange()
                cc.change_type = change_type
                for pt in points:
                    cc.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=0.0))
                update_msg.clusters.append(cc)
                return

            # --- STANDARD: Spatial Clustering for basic Positive/Negative ---
            tree = KDTree(points)
            pairs = tree.query_pairs(r=0.3)
            adj = {i: [] for i in range(len(points))}
            for i, j in pairs:
                adj[i].append(j)
                adj[j].append(i)
                
            visited = set()
            for i in range(len(points)):
                if i not in visited:
                    cluster_indices = []
                    q = [i]
                    visited.add(i)
                    while q:
                        curr = q.pop(0)
                        cluster_indices.append(curr)
                        for neighbor in adj[curr]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                q.append(neighbor)
                    
                    if len(cluster_indices) < 5: 
                        continue
                    
                    cc = ClusterChange()
                    cc.change_type = change_type
                    for idx in cluster_indices:
                        pt = points[idx]
                        cc.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=0.0))
                    update_msg.clusters.append(cc)

        # Standard changes use nearest-neighbor grouping to filter out floating noise
        add_clusters(both_pos, ClusterChange.POSITIVE_CHANGE, use_clustering=True)
        add_clusters(both_neg, ClusterChange.NEGATIVE_CHANGE, use_clustering=True)
        
        # State transitions (Pos->Neg, Neg->Pos) are applied raw, bypassing eraser/clustering
        add_clusters(submap_only_pos, ClusterChange.NEGATIVE_TO_POSITIVE, use_clustering=False)
        add_clusters(submap_only_neg, ClusterChange.POSITIVE_TO_NEGATIVE, use_clustering=False)

        if len(update_msg.clusters) > 0:
            self.pub_update.publish(update_msg)

def main(args=None):
    rclpy.init(args=args)
    try: rclpy.spin(ClusterCreatorNode())
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()