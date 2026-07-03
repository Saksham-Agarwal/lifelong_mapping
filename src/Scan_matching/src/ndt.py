import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.spatial import KDTree
import os

# ==========================================
# 1. MATHEMATICAL HELPER FUNCTIONS
# ==========================================

def skewd(p):
    """Returns the skew-symmetric vector for 2D rotation derivative"""
    return np.array([-p[1], p[0]])

def expmap(delta):
    """Converts a [dx, dy, dtheta] vector into a 3x3 transformation matrix"""
    x, y, theta = delta[0], delta[1], delta[2]
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, -s, x],
        [s,  c, y],
        [0,  0, 1]
    ])

def compute_target_covariances(points, k=5):
    """
    NDT requires a covariance matrix for the target points.
    This calculates the 2x2 covariance for the local neighborhood of each point.
    """
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
            
        # Add a tiny value to diagonal to prevent singular/non-invertible matrices
        cov += np.eye(2) * 1e-5 
        covs.append(cov)
    return np.array(covs)

def transform_points(trans_mat, points):
    """Applies the 3x3 transformation matrix to an N x 2 array of points"""
    # Convert to homogeneous coordinates (x, y, 1)
    homogenous_pts = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed = np.dot(trans_mat, homogenous_pts.T).T
    return transformed[:, :2]

# ==========================================
# 2. THE CORE NDT ALGORITHM
# ==========================================

def ndt_scan_matching(trans_mat, source_points, target_points, target_covs):
    # Tuned for your specific case: catching small features to fix slight drift
    max_iter_num = 15
    scan_step = 1       
    max_dist = 0.5      
    epsilon = 1e-4
    damping = 1e-5
    
    kdtree = KDTree(target_points)
    
    for iter_num in range(max_iter_num):
        H = np.zeros((3, 3))
        b = np.zeros(3)
        R = trans_mat[:2, :2]
        corresponding_points_num = 0
        error_sum = 0.0

        # Define the threshold where we start trusting points less (e.g., 0.3 meters)
        tuning_constant = 0.5

        for i in range(0, len(source_points), scan_step):
            point = np.array([source_points[i][0], source_points[i][1], 1.0])
            transformed_point = np.dot(trans_mat, point)
            query = [transformed_point[0], transformed_point[1]]
            
            dist, idx = kdtree.query(query)
            
            # --- THE FIX: HARD CUTOFF ---
            # Ignore points that are too far from the global map walls (e.g., dynamic obstacles/noise)
            if dist > 0.3:
                continue
            
            # Calculate the Robust Weight (Huber style)
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
            
            # --- APPLY THE WEIGHT TO THE COMPUTATION ---
            # The matrix math now respects how "trustworthy" the point is
            H += weight * np.dot(J.T, np.dot(IM, J))
            b += weight * np.dot(J.T, np.dot(IM, error))
            corresponding_points_num += 1
        
        if corresponding_points_num < 5:
            print("Warning: Not enough correspondences. Aborting to prevent wild drift.")
            break

        # Hessian Regularization
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
# 3. DATA LOADING AND VISUALIZATION
# ==========================================

def main():
    # Load the CSV files saved from the ROS 2 node
    if not os.path.exists('local_region.csv') or not os.path.exists('costmap.csv'):
        print("CSV files not found. Ensure they are in the same directory.")
        return

    # Reminder: local_region = Global Target, costmap = Local Source
    global_df = pd.read_csv('local_region.csv')
    local_df = pd.read_csv('costmap.csv')

    target_points = global_df[['x', 'y']].values
    source_points = local_df[['x', 'y']].values

    print(f"Loaded {len(target_points)} global points and {len(source_points)} local points.")

    # Compute covariances for the global map
    target_covs = compute_target_covariances(target_points)

    #    ---------------------------------------------------------
    # NEW: Load the initial offset from your saver node's output
    # ---------------------------------------------------------
    initial_dx, initial_dy = 0.0, 0.0
    if os.path.exists('alignment_vector.txt'):
        with open('alignment_vector.txt', 'r') as f:
            lines = f.readlines()
            # Parses lines like "Delta X: -5.1 meters"
            initial_dx = float(lines[1].split(':')[1].replace('meters', '').strip())
            initial_dy = float(lines[2].split(':')[1].replace('meters', '').strip())
            print(f"Loaded initial origin offset: dx={initial_dx:.3f}, dy={initial_dy:.3f}")
    else:
        print("Warning: alignment_vector.txt not found. Using identity matrix.")

   # Compute covariances for the global map
    target_covs = compute_target_covariances(target_points)

    # ---------------------------------------------------------
    # Initial Transformation Matrix 
    # Because the CSVs are pre-aligned via TF in the saver node, 
    # our initial guess is simply zero offset (the Identity matrix).
    # ---------------------------------------------------------
    initial_trans_mat = np.eye(3)

    print("\nStarting NDT Alignment...")
    final_trans_mat = ndt_scan_matching(initial_trans_mat, source_points, target_points, target_covs)
    
    # Calculate the aligned points for visualization
    aligned_source_points = transform_points(final_trans_mat, source_points)

    # Calculate final translation and rotation
    dx = final_trans_mat[0, 2]
    dy = final_trans_mat[1, 2]
    dtheta = math.atan2(final_trans_mat[1, 0], final_trans_mat[0, 0])
    print(f"\nFinal Correction Offset -> X: {dx:.3f}m, Y: {dy:.3f}m, Yaw: {dtheta:.3f} rad")

    # Plotting
    plt.figure(figsize=(12, 6))

    # Subplot 1: Before Alignment (Drifted)
    plt.subplot(1, 2, 1)
    plt.title("Before NDT (Drifted State)")
    plt.scatter(target_points[:, 0], target_points[:, 1], c='gray', s=5, alpha=0.5, label='Global Map (Target)')
    plt.scatter(source_points[:, 0], source_points[:, 1], c='red', s=5, alpha=0.8, label='Local Map (Source)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: After Alignment (Corrected)
    plt.subplot(1, 2, 2)
    plt.title("After NDT (Aligned State)")
    plt.scatter(target_points[:, 0], target_points[:, 1], c='gray', s=5, alpha=0.5, label='Global Map (Target)')
    plt.scatter(aligned_source_points[:, 0], aligned_source_points[:, 1], c='blue', s=5, alpha=0.8, label='Aligned Local Map')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()