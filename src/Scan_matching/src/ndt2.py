import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from scipy.spatial import KDTree

# ==========================================
# 1. MATHEMATICAL HELPER FUNCTIONS
# ==========================================
def skewd(p):
    return np.array([-p[1], p[0]])

def expmap(delta):
    x, y, theta = delta[0], delta[1], delta[2]
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

def compute_target_covariances(points, k=5):
    print("Computing covariances for global map...")
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

# ==========================================
# 2. THE STRICT NDT ALGORITHM 
# ==========================================
def ndt_scan_matching(trans_mat, source_points, target_points, target_covs):
    max_iter_num = 15
    scan_step = 1       
    epsilon = 1e-4
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
            
            # --- THE MAGIC CUTOFF ---
            if dist > 0.3:
                continue
            
            tuning_constant = 0.15
            if dist <= tuning_constant:
                weight = 1.0
            else:
                weight = tuning_constant / dist

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
            print("Warning: Not enough correspondences within 0.3m. Aborting.")
            break

        H += np.eye(3) * damping
        error_ave = error_sum / float(corresponding_points_num)
        
        try:
            delta = np.linalg.solve(H, -b)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered. Stopping iterations.")
            break
            
        update = np.dot(delta, delta)
        trans_mat = np.dot(trans_mat, expmap(delta))
        
        print(f"Iteration {iter_num+1}: Error = {error_ave:.4f}, Update = {update:.6f}")
        
        if update < epsilon:
            print('NDT scan matching has converged!')
            break
            
    return trans_mat

# ==========================================
# 3. MAIN EXECUTION & 3-COLUMN VISUALIZATION
# ==========================================
def main():
    if not os.path.exists('local_region.csv') or not os.path.exists('costmap.csv'):
        print("CSV files not found. Ensure they are in the same directory.")
        return

    global_df = pd.read_csv('local_region.csv')
    local_df = pd.read_csv('costmap.csv')

    target_points = global_df[['x', 'y']].values  # Global Map
    source_points = local_df[['x', 'y']].values   # Local Map

    print(f"Loaded {len(target_points)} global points and {len(source_points)} local points.")

    # --- Step 1: Align ---
    print("\n1. Running NDT for alignment...")
    target_covs = compute_target_covariances(target_points)
    initial_trans_mat = np.eye(3) 
    final_trans_mat = ndt_scan_matching(initial_trans_mat, source_points, target_points, target_covs)
    
    # Calculate the newly aligned source points for visualization
    aligned_source_points = transform_points(final_trans_mat, source_points)

    # --- Step 2: Crop Global Map (in Aligned Frame) ---
    print("\n2. Cropping Global Map to Aligned Local Dimensions...")
    
    # Get bounding box from the ALIGNED source points
    min_x, max_x = np.min(aligned_source_points[:, 0]), np.max(aligned_source_points[:, 0])
    min_y, max_y = np.min(aligned_source_points[:, 1]), np.max(aligned_source_points[:, 1])

    # Crop the target_points (Global Map) directly using a small 0.5m buffer
    buffer = 0.5
    mask_x = (target_points[:, 0] >= min_x - buffer) & (target_points[:, 0] <= max_x + buffer)
    mask_y = (target_points[:, 1] >= min_y - buffer) & (target_points[:, 1] <= max_y + buffer)
    cropped_global_points = target_points[mask_x & mask_y]

    # --- Step 3: Difference/Positive Change Check ---
    print("3. Identifying unmatched local points (Positive Change/New Obstacles)...")
    
    # Run the KDTree query safely since both clouds are now in the global frame!
    global_tree = KDTree(cropped_global_points)
    distances, _ = global_tree.query(aligned_source_points)

    distance_threshold = 0.2
    unmatched_mask = distances > distance_threshold

    unmatched_local_points = aligned_source_points[unmatched_mask]
    matched_local_points = aligned_source_points[~unmatched_mask]
    print(f"Identified {len(unmatched_local_points)} new obstacle points.")

    # ==========================================
    # VISUALIZATION (3 COLUMNS)
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Column 1: Original
    axes[0].set_title("1. Original (Drifted State)")
    axes[0].scatter(target_points[:, 0], target_points[:, 1], c='gray', s=5, alpha=0.5, label='Global Target')
    axes[0].scatter(source_points[:, 0], source_points[:, 1], c='red', s=5, alpha=0.8, label='Local Source')
    axes[0].axis('equal')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Column 2: Aligned
    axes[1].set_title("2. Aligned (Post-NDT)")
    axes[1].scatter(target_points[:, 0], target_points[:, 1], c='gray', s=5, alpha=0.5, label='Global Target')
    axes[1].scatter(aligned_source_points[:, 0], aligned_source_points[:, 1], c='blue', s=5, alpha=0.8, label='Aligned Local')
    axes[1].axis('equal')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Column 3: Positive Change Marking
    axes[2].set_title("3. Positive Change Marking (Cropped Grid)")
    axes[2].scatter(cropped_global_points[:, 0], cropped_global_points[:, 1], c='gray', s=5, alpha=0.5, label='Cropped Global')
    axes[2].scatter(matched_local_points[:, 0], matched_local_points[:, 1], c='red', s=5, alpha=0.5, label='Matched Local')
    axes[2].scatter(unmatched_local_points[:, 0], unmatched_local_points[:, 1], c='blue', s=15, marker='x', label='Positive Change (New)')
    axes[2].axis('equal')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()