import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from scipy.spatial import KDTree

# ==========================================
# HELPER FUNCTIONS & NDT
# ==========================================
def skewd(p):
    return np.array([-p[1], p[0]])

def expmap(delta):
    x, y, theta = delta[0], delta[1], delta[2]
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

def compute_target_covariances(points, k=15):
    tree = KDTree(points)
    covs = []
    for pt in points:
        _, idxs = tree.query(pt, k=k)
        neighbors = points[idxs]
        if len(neighbors) > 1:
            cov = np.cov(neighbors[:, :2], rowvar=False)
        else:
            cov = np.eye(2) * 1e-3
        cov += np.eye(2) * 1e-5  # Prevent singular matrices
        covs.append(cov)
    return np.array(covs)

def transform_points(trans_mat, points):
    homogenous_pts = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed = np.dot(trans_mat, homogenous_pts.T).T
    return transformed[:, :2]

def ndt_scan_matching(trans_mat, source_points, target_points, target_covs):
    max_iter_num = 20
    scan_step = 1       
    tuning_constant = 0.15 # Huber loss threshold for new obstacles
    epsilon = 1e-5
    damping = 1e-5
    
    kdtree = KDTree(target_points)
    
    for iter_num in range(max_iter_num):
        H = np.zeros((3, 3))
        b = np.zeros(3)
        R = trans_mat[:2, :2]
        corresponding_points_num = 0
        error_sum = 0.0

        for i in range(0, len(source_points), scan_step):
            point = np.array([source_points[i][0], source_points[i][1], 1.0])
            transformed_point = np.dot(trans_mat, point)
            query = [transformed_point[0], transformed_point[1]]
            
            dist, idx = kdtree.query(query)
            
            # Soft weighting so new obstacles don't ruin alignment
            weight = 1.0 if dist <= tuning_constant else (tuning_constant / dist)

            target = target_points[idx]
            
            C = np.eye(3)
            C[0:2, 0:2] = target_covs[idx]
            
            try:
                IM = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                continue
                
            error = np.array([target[0] - query[0], target[1] - query[1], 0.0])
            error_sum += math.sqrt(error[0]*error[0] + error[1]*error[1] + error[2]*error[2])
            
            v = np.dot(R, skewd(source_points[i]))
            
            J = np.zeros((3, 3))
            J[0:2, 0:2] = -R
            J[0, 2] = -v[0] 
            J[1, 2] = -v[1] 
            
            H += weight * np.dot(J.T, np.dot(IM, J))
            b += weight * np.dot(J.T, np.dot(IM, error))
            corresponding_points_num += 1

        if corresponding_points_num < 5:
            break

        H += np.eye(3) * damping
        
        try:
            delta = np.linalg.solve(H, -b)
        except np.linalg.LinAlgError:
            break
            
        update = np.dot(delta, delta)
        trans_mat = np.dot(trans_mat, expmap(delta))
        
        if update < epsilon:
            break
            
    return trans_mat

# ==========================================
# MAIN EXECUTION & CROPPING
# ==========================================
def main():
    if not os.path.exists('local_region.csv') or not os.path.exists('costmap.csv'):
        print("CSV files not found.")
        return

    global_df = pd.read_csv('local_region.csv')
    local_df = pd.read_csv('costmap.csv')

    target_points = global_df[['x', 'y']].values  # Full Global Map
    source_points = local_df[['x', 'y']].values   # Full Local Map

    print("1. Running NDT on full maps for accurate alignment...")
    target_covs = compute_target_covariances(target_points)
    initial_trans_mat = np.eye(3)
    final_trans_mat = ndt_scan_matching(initial_trans_mat, source_points, target_points, target_covs)

    print("2. Projecting Global Map into Local Frame...")
    inv_trans_mat = np.linalg.inv(final_trans_mat)
    projected_global_points = transform_points(inv_trans_mat, target_points)

    print("3. Cropping Global Map to Local Dimensions...")
    # Get the exact physical boundaries of the local costmap
    min_x, max_x = np.min(source_points[:, 0]), np.max(source_points[:, 0])
    min_y, max_y = np.min(source_points[:, 1]), np.max(source_points[:, 1])

    # Create a boolean mask to physically delete global points outside the local box
    mask_x = (projected_global_points[:, 0] >= min_x) & (projected_global_points[:, 0] <= max_x)
    mask_y = (projected_global_points[:, 1] >= min_y) & (projected_global_points[:, 1] <= max_y)
    
    # Apply the mask
    cropped_global_points = projected_global_points[mask_x & mask_y]
    print("4. Identifying unmatched local points (new obstacles)...")
    
    # Build a KDTree using the cropped global points
    global_tree = KDTree(cropped_global_points)

    # Query the distance to the nearest global point for EVERY local point
    distances, _ = global_tree.query(source_points)

    # Set a threshold distance (you may need to tune this based on your map scale)
    # Anything further than this is considered a "new" or "unmatched" point
    distance_threshold = 0.2
    
    # Create a boolean mask to separate matched vs unmatched points
    unmatched_mask = distances > distance_threshold

    unmatched_local_points = source_points[unmatched_mask]
    matched_local_points = source_points[~unmatched_mask]

    print(f"Identified {len(unmatched_local_points)} unmatched points.")

    print(f"Done! Reduced {len(projected_global_points)} global points down to {len(cropped_global_points)} local-relevant points.")

    # ==========================================
    # VISUALIZATION
    # ==========================================
    plt.figure(figsize=(8, 8))
    plt.title("Post-NDT Unmatched Cluster Identification")

    # Plot the cropped global map
    plt.scatter(
        cropped_global_points[:, 0], cropped_global_points[:, 1], 
        c='gray', s=8, alpha=0.5, label='Aligned & Cropped Global Map'
    )

    # Plot the normal, matched local map
    plt.scatter(
        matched_local_points[:, 0], matched_local_points[:, 1], 
        c='red', s=8, alpha=0.9, label='Matched Local Map'
    )

    # Highlight the newly identified isolated cluster
    plt.scatter(
        unmatched_local_points[:, 0], unmatched_local_points[:, 1], 
        c='blue', s=20, marker='x', label='Unmatched Cluster (New Obstacle)'
    )

    plt.gca().set_aspect('equal', adjustable='box') 
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()