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
        
        # Grid resolution for dilation (assume 5cm occupancy grid)
        self.grid_res = 0.05 
        
        self.get_logger().info('Cluster Creator Node running with Cross-Referencing & Dilation.')

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

    def dilate_points(self, points):
        """Creates a 3x3 square around each point to act as a fat eraser"""
        if len(points) == 0: return points
        
        dilated = []
        offsets = [-self.grid_res, 0, self.grid_res]
        
        for p in points:
            for dx in offsets:
                for dy in offsets:
                    dilated.append([p[0] + dx, p[1] + dy])
                    
        # Round to nearest millimeter and filter duplicates for performance
        dilated_arr = np.round(np.array(dilated), decimals=3)
        return np.unique(dilated_arr, axis=0)

    def changes_callback(self, msg):
        g_pos = self.extract_pts(msg.global_positive)
        g_neg = self.extract_pts(msg.global_negative)
        s_pos = self.extract_pts(msg.submap_positive)
        s_neg = self.extract_pts(msg.submap_negative)
        
        # --- USER LOGIC IMPLEMENTATION ---
        
        # 1. Exists in BOTH maps -> Standard Positive/Negative
        both_pos, _ = self.get_intersection_and_diff(g_pos, s_pos)
        both_neg, _ = self.get_intersection_and_diff(g_neg, s_neg)
        
        # 2. Exists in Submap but NOT Global
        _, submap_only_neg = self.get_intersection_and_diff(s_neg, g_neg)
        _, submap_only_pos = self.get_intersection_and_diff(s_pos, g_pos)
        
        # 3. Apply Dilation to the dynamic removed objects
        raw_dilated = self.dilate_points(submap_only_neg)
        
        # 4. THE WALL SHIELD (Prevent erasing real walls)
        # Assuming g_pos (global positive) represents your permanent map structures
        if len(raw_dilated) > 0 and len(g_pos) > 0:
            wall_tree = KDTree(g_pos)
            # Find which eraser points are dangerously close to a wall (e.g., within 5cm)
            dists, _ = wall_tree.query(raw_dilated)
            
            # Only keep the eraser points that are safely away from the walls
            safe_mask = dists > 0.05 
            dilated_pos_to_neg = raw_dilated[safe_mask]
        else:
            dilated_pos_to_neg = raw_dilated
        # --- CLUSTERING AND PACKAGING ---
        update_msg = MapUpdate()
        update_msg.header.stamp = self.get_clock().now().to_msg()
        update_msg.header.frame_id = 'map'

        def add_clusters(points, change_type):
            if len(points) == 0: return
            
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
                    
                    cc = ClusterChange()
                    cc.change_type = change_type
                    for idx in cluster_indices:
                        pt = points[idx]
                        cc.points.append(Point(x=float(pt[0]), y=float(pt[1]), z=0.0))
                    
                    update_msg.clusters.append(cc) # [cite: 2]

        # Package them up using your custom constants
        add_clusters(both_pos, ClusterChange.POSITIVE_CHANGE) # 
        add_clusters(both_neg, ClusterChange.NEGATIVE_CHANGE) # 
        add_clusters(submap_only_pos, ClusterChange.NEGATIVE_TO_POSITIVE) # 
        add_clusters(dilated_pos_to_neg, ClusterChange.POSITIVE_TO_NEGATIVE) # 

        if len(update_msg.clusters) > 0: # [cite: 2]
            self.pub_update.publish(update_msg)

def main(args=None):
    rclpy.init(args=args)
    try: rclpy.spin(ClusterCreatorNode())
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()