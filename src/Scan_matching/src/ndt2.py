import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from scipy.spatial import KDTree

# ==========================================
# CONFIGURATION PARAMETERS
# Change these filenames as needed for different test runs
# ==========================================
GLOBAL_MAP_FILE = 'local_region_10.csv'
LOCAL_MAP_FILE = 'costmap_10.csv'
AMCL_GUESS_FILE = 'amcl_guess.txt'

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
        cov = np.cov(neighbors[:, :2], rowvar=False) if len(neighbors) > 1 else np.eye(2) * 1e-3
        cov += np.eye(2) * 1e-5
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
            
            # Strict cutoff parameter
            if dist > 0.3:
                continue
            
            tuning_constant = 0.15
            weight = 1.0 if dist <= tuning_constant else tuning_constant / dist
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
            break
            
        update = np.dot(delta, delta)
        trans_mat = np.dot(trans_mat, expmap(delta))
        
        if update < epsilon:
            print(f'NDT scan matching converged at iteration {iter_num+1}!')
            break
            
    return trans_mat

# ==========================================
# 3. VISUALIZATION HELPER
# ==========================================
def plot_robot_and_deadspace(ax, x, y, yaw, label="Robot"):
    """
    Plots the robot as a circle, a solid line for its forward heading, 
    and dashed lines representing the boundaries of its 80-degree blind spot.
    """
    # 1. Plot Robot Body
    ax.plot(x, y, 'go', markersize=10, label=label, zorder=10)
    
    # 2. Plot Forward Heading (Front of the robot)
    line_length = 1.2
    ax.plot([x, x + line_length * math.cos(yaw)], 
            [y, y + line_length * math.sin(yaw)], 
            'g-', linewidth=2.5, zorder=10)
    
    # 3. Plot Deadspace Boundaries (+140 and -140 degrees)
    angle_pos = yaw + math.radians(140)
    angle_neg = yaw - math.radians(140)
    
    ax.plot([x, x + line_length * math.cos(angle_pos)], 
            [y, y + line_length * math.sin(angle_pos)], 
            'k--', linewidth=1.5, zorder=9, label='Sensor Deadspace Edge')
    ax.plot([x, x + line_length * math.cos(angle_neg)], 
            [y, y + line_length * math.sin(angle_neg)], 
            'k--', linewidth=1.5, zorder=9)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    if not os.path.exists(GLOBAL_MAP_FILE) or not os.path.exists(LOCAL_MAP_FILE):
        print(f"CSV files not found. Ensure {GLOBAL_MAP_FILE} and {LOCAL_MAP_FILE} are in the directory.")
        return

    target_points = pd.read_csv(GLOBAL_MAP_FILE)[['x', 'y']].values
    source_points = pd.read_csv(LOCAL_MAP_FILE)[['x', 'y']].values

    print(f"Loaded {len(target_points)} global points and {len(source_points)} local points.")

    # Initialize variables for the visualization fallback
    amcl_x, amcl_y, amcl_yaw = 0.0, 0.0, 0.0
    true_x, true_y, true_yaw = 0.0, 0.0, 0.0

    # --- 1. Align Maps ---
    target_covs = compute_target_covariances(target_points)
    
    # The TF listener pre-aligned the CSVs to AMCL's guess. We start at zero-offset.
    initial_trans_mat = np.eye(3)
    
    print("\nRunning NDT for alignment...")
    drift_mat = ndt_scan_matching(initial_trans_mat, source_points, target_points, target_covs)
    
    aligned_source_points = transform_points(drift_mat, source_points)

    # --- 2. Calculate Drift & Corrected Pose ---
    drift_x = drift_mat[0, 2]
    drift_y = drift_mat[1, 2]
    drift_yaw = math.atan2(drift_mat[1, 0], drift_mat[0, 0])

    if os.path.exists(AMCL_GUESS_FILE):
        with open(AMCL_GUESS_FILE, 'r') as f:
            lines = f.readlines()
            amcl_x = float(lines[0].split(':')[1].strip())
            amcl_y = float(lines[1].split(':')[1].strip())
            amcl_yaw = float(lines[2].split(':')[1].strip())
            
        # Matrix math to find true absolute pose
        amcl_mat = expmap([amcl_x, amcl_y, amcl_yaw])
        true_mat = np.dot(drift_mat, amcl_mat)
        
        true_x = true_mat[0, 2]
        true_y = true_mat[1, 2]
        true_yaw = math.atan2(true_mat[1, 0], true_mat[0, 0])

        print("\n==================================================")
        print(f"AMCL POSE DRIFT DETECTED:")
        print(f"  ΔX   : {drift_x:+.4f} meters")
        print(f"  ΔY   : {drift_y:+.4f} meters")
        print(f"  ΔYaw : {drift_yaw:+.4f} radians")
        print(f"--------------------------------------------------")
        print(f"RAW AMCL POSE (Initial Guess):")
        print(f"  X: {amcl_x:.4f}, Y: {amcl_y:.4f}, Yaw: {amcl_yaw:.4f}")
        print(f"TRUE LOCALIZED ROBOT POSE (Corrected):")
        print(f"  X: {true_x:.4f}, Y: {true_y:.4f}, Yaw: {true_yaw:.4f}")
        print("==================================================\n")
    else:
        print(f"Warning: {AMCL_GUESS_FILE} not found. Visualizer will plot robot at 0,0.")

    # --- 3. Identify New Obstacles ---
    print("Identifying unmatched local points...")
    min_x, max_x = np.min(aligned_source_points[:, 0]), np.max(aligned_source_points[:, 0])
    min_y, max_y = np.min(aligned_source_points[:, 1]), np.max(aligned_source_points[:, 1])

    buffer = 0.5
    mask_x = (target_points[:, 0] >= min_x - buffer) & (target_points[:, 0] <= max_x + buffer)
    mask_y = (target_points[:, 1] >= min_y - buffer) & (target_points[:, 1] <= max_y + buffer)
    cropped_global_points = target_points[mask_x & mask_y]

    global_tree = KDTree(cropped_global_points)
    distances, _ = global_tree.query(aligned_source_points)

    distance_threshold = 0.2
    unmatched_mask = distances > distance_threshold

    unmatched_local_points = aligned_source_points[unmatched_mask]
    matched_local_points = aligned_source_points[~unmatched_mask]

    # --- 4. Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Column 1: AMCL Guess (Pre-aligned by TF saver)
    axes[0].set_title("1. Initial Guess (AMCL Pose Only)")
    axes[0].scatter(target_points[:, 0], target_points[:, 1], c='gray', s=5, alpha=0.5, label='Global Target')
    axes[0].scatter(source_points[:, 0], source_points[:, 1], c='red', s=5, alpha=0.8, label='AMCL Local Map')
    plot_robot_and_deadspace(axes[0], amcl_x, amcl_y, amcl_yaw, label='AMCL Robot Pose')
    axes[0].axis('equal')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Column 2: Aligned via NDT
    axes[1].set_title("2. Corrected (Post-NDT)")
    axes[1].scatter(target_points[:, 0], target_points[:, 1], c='gray', s=5, alpha=0.5, label='Global Target')
    axes[1].scatter(aligned_source_points[:, 0], aligned_source_points[:, 1], c='blue', s=5, alpha=0.8, label='Corrected Local')
    plot_robot_and_deadspace(axes[1], true_x, true_y, true_yaw, label='True Robot Pose')
    axes[1].axis('equal')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Column 3: Positive Change Marking
    axes[2].set_title("3. Positive Change Marking (Cropped Grid)")
    axes[2].scatter(cropped_global_points[:, 0], cropped_global_points[:, 1], c='gray', s=5, alpha=0.5, label='Cropped Global')
    axes[2].scatter(matched_local_points[:, 0], matched_local_points[:, 1], c='red', s=5, alpha=0.5, label='Matched Local')
    axes[2].scatter(unmatched_local_points[:, 0], unmatched_local_points[:, 1], c='blue', s=15, marker='x', label='Positive Change (New)')
    plot_robot_and_deadspace(axes[2], true_x, true_y, true_yaw, label='True Robot Pose')
    axes[2].axis('equal')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()