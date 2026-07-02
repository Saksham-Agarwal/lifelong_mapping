#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from nav_msgs.msg import OccupancyGrid
import message_filters

# ---------------------------------------------------------
# NDT & Math Core Functions 
# ---------------------------------------------------------

def compute_mean_and_covariance(points, indices):
    x = 0.0
    y = 0.0
    num = len(indices)
    if num == 0:
        return [0.0, 0.0], np.eye(2)
        
    for i in range(num):
        x += points[indices[i]][0]
        y += points[indices[i]][1]

    mean = [x / float(num), y / float(num)]
    vxx = 0.0
    vxy = 0.0
    vyy = 0.0
    for i in range(num):
        dx = points[indices[i]][0] - mean[0]
        dy = points[indices[i]][1] - mean[1]
        vxx += dx * dx
        vxy += dx * dy
        vyy += dy * dy
        
    cov = np.array([[vxx / float(num), vxy / float(num)], 
                    [vxy / float(num), vyy / float(num)]])
    
    # Regularize covariance to prevent singular matrices
    cov += np.eye(2) * 1e-4
    return mean, cov

def compute_ndt_points(points):
    N = 10
    covs = []
    ndt_means = [row[:] for row in points] 
    
    if len(points) < N:
        N = len(points)
        
    tree = KDTree(points)
    for i in range(len(points)):
        query = np.array([points[i][0], points[i][1]])
        dists, indices = tree.query(query, k=N)
        mean, cov = compute_mean_and_covariance(points, indices)
        ndt_means[i][0] = mean[0]
        ndt_means[i][1] = mean[1]
        covs.append(cov)
    return ndt_means, covs

def make_transformation_matrix(tx, ty, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), tx],
        [np.sin(theta), np.cos(theta),  ty],
        [0.0,           0.0,            1.0]
    ])

def transform_points(mat, points):
    transformed = []
    for i in range(len(points)):
        point = np.array([points[i][0], points[i][1], 1.0])
        transformed_point = np.dot(mat, point)
        transformed.append([transformed_point[0], transformed_point[1]])
    return transformed

def skewd(v):
    return np.array([v[1], -v[0]])

def expmap(v):
    t = v[2]
    c = np.cos(t)
    s = np.sin(t)
    if np.abs(t) < 1e-10:
        V = np.eye(2)
    else:
        a = (1.0 - c) / t
        V = np.array([[s / t, -a], [a, s / t]])
    R = np.array([[c, -s], [s, c]])
    u = np.array([v[0], v[1]])
    t_val = np.dot(V, u)
    T = np.eye(3)
    T[:2, :2] = R
    T[0, 2] = t_val[0]
    T[1, 2] = t_val[1]
    return T

def plot_points(points1, points2, title):
    if not points1 or not points2:
        return
    x1, y1 = zip(*points1)
    x2, y2 = zip(*points2)
    plt.clf()
    plt.scatter(x2, y2, color='blue', label='Target (Global)', s=20)
    plt.scatter(x1, y1, color='red', label='Source (Aligned Scan)', s=10)
    plt.grid(True)
    plt.title(title)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend(loc='upper left', fontsize=12)
    plt.draw()
    plt.pause(0.01)

def ndt_scan_matching(trans_mat, source_points, target_points, target_covs, logger):
    max_iter_num = 30
    scan_step = 2 
    max_dist = 3.0
    epsilon = 1e-4
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
            
            if dist > max_dist:
                continue

            target = target_points[idx]
            C = np.eye(3)
            C[0:2, 0:2] = target_covs[idx]
            
            try:
                IM = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                continue 

            error = np.array([target[0] - query[0], target[1] - query[1], 0.0])
            error_sum += math.sqrt(np.dot(error, error))
            v = np.dot(R, skewd(source_points[i]))
            
            J = np.zeros((3, 3))
            J[0:2, 0:2] = -R
            J[0, 2] = v[0]
            J[1, 2] = v[1]
            
            H += np.dot(J.T, np.dot(IM, J))
            b += np.dot(J.T, np.dot(IM, error))
            corresponding_points_num += 1

        if corresponding_points_num == 0:
            logger.warn("NDT: No corresponding points found in this iteration.")
            break

        error_ave = error_sum / float(corresponding_points_num)
        
        try:
            delta = np.linalg.solve(H, -b)
        except np.linalg.LinAlgError:
            logger.error("NDT: Singular matrix encountered. Aborting iteration.")
            break
            
        update = np.dot(delta, delta)
        trans_mat = np.dot(trans_mat, expmap(delta))
        
        logger.info(f"NDT Iteration {iter_num}: Error = {error_ave:.4f}, Update = {update:.6f}")
        
        if update < epsilon:
            logger.info('NDT scan matching has converged.')
            break

    title = f'NDT Alignment Converged ({iter_num + 1} iterations)'
    plot_points(transform_points(trans_mat, source_points), target_points, title)
    return trans_mat


# ---------------------------------------------------------
# ROS 2 Node Class
# ---------------------------------------------------------

class NDTGridMatcher(Node):
    def __init__(self):
        super().__init__('ndt_grid_matcher')
        
        # ROS 2 Message filters pass 'self' as the first argument
        self.sub_scan = message_filters.Subscriber(self, OccupancyGrid, '/robot_local_region')
        self.sub_costmap = message_filters.Subscriber(self, OccupancyGrid, '/simplified_local_costmap')
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_scan, self.sub_costmap], 10, 5.0)
        self.ts.registerCallback(self.sync_callback)
        
        self.processing = False
        plt.ion() 
        self.get_logger().info("Waiting for OccupancyGrids on both topics to perform NDT...")

    def grid_to_pointcloud(self, grid_msg):
        points = []
        res = grid_msg.info.resolution
        ox = grid_msg.info.origin.position.x
        oy = grid_msg.info.origin.position.y
        w = grid_msg.info.width
        h = grid_msg.info.height
        
        data = np.array(grid_msg.data, dtype=np.int8).reshape((h, w))
        ys, xs = np.where(data > 50) 
        
        for x, y in zip(xs, ys):
            px = ox + (x * res) + (res / 2.0)
            py = oy + (y * res) + (res / 2.0)
            points.append([px, py])
            
        return points

    def sync_callback(self, scan_msg, costmap_msg):
        if self.processing:
            return 
        self.get_logger().info("Synchronized OccupancyGrids received. Starting NDT alignment...")    
        self.processing = True
        self.get_logger().info("Received synchronized grids. Extracting points...")

        source_points = self.grid_to_pointcloud(scan_msg)    
        target_points = self.grid_to_pointcloud(costmap_msg) 
        
        if len(source_points) < 10 or len(target_points) < 10:
            self.get_logger().warn("Not enough occupied points in grids to run NDT.")
            self.processing = False
            return

        self.get_logger().info(f"Extracted {len(source_points)} source points and {len(target_points)} target points.")
        self.get_logger().info("Computing NDT representations for target grid...")
        
        ndt_means, ndt_covs = compute_ndt_points(target_points)
        trans_mat = make_transformation_matrix(0.0, 0.0, 0.0)

        self.get_logger().info("Starting NDT Optimization...")
        
        # Notice we pass self.get_logger() down into the pure math function so it logs correctly
        final_trans_mat = ndt_scan_matching(trans_mat, source_points, ndt_means, ndt_covs, self.get_logger())
        
        dx = final_trans_mat[0, 2]
        dy = final_trans_mat[1, 2]
        dtheta = math.degrees(math.atan2(final_trans_mat[1, 0], final_trans_mat[0, 0]))
        
        self.get_logger().info(f"--- Alignment Complete ---")
        self.get_logger().info(f"Delta X: {dx:.3f}m | Delta Y: {dy:.3f}m | Delta Yaw: {dtheta:.2f} deg")
        
        self.processing = False

def main(args=None):
    rclpy.init(args=args)
    matcher = NDTGridMatcher()
    
    try:
        rclpy.spin(matcher)
    except KeyboardInterrupt:
        pass
    finally:
        matcher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()