import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import numpy as np
import os

# 1. Load the saved coordinate data
# We know 'local_region.csv' is actually our BIGGER Global map
# We know 'costmap.csv' is actually our SMALLER Local map
if not os.path.exists('local_region.csv') or not os.path.exists('costmap.csv'):
    print("Error: CSV files not found. Run the ROS 2 node first.")
    exit()

# Renamed variables to reflect reality, not the bad topic names
global_df = pd.read_csv('local_region.csv') 
local_df = pd.read_csv('costmap.csv')

# Convert to numpy arrays
global_pts = global_df[['x', 'y']].values
local_pts = local_df[['x', 'y']].values

print(f"Loaded {len(global_pts)} global costmap points and {len(local_pts)} local costmap points.")

# 2. KDTree Spatial Filtering
# Build a spatial tree of the LARGER global costmap
print("Building KDTree on the larger global map...")
tree = KDTree(global_pts)

# Define a search radius (in meters) around the bot's local costmap
search_radius = 2.0 

# For every point in the small local map, find points in the big global map within the radius
indices_list = tree.query_ball_point(local_pts, r=search_radius)

# Flatten the list and remove duplicates
unique_indices = np.unique(np.concatenate(indices_list))

# Extract the filtered points from the larger map
filtered_global_pts = global_pts[unique_indices]
print(f"Filtered the global map down to {len(filtered_global_pts)} relevant points.")

# 3. Plotting
plt.figure(figsize=(10, 10))

# Plot the KDTree-filtered Global Costmap (Blue)
plt.scatter(
    filtered_global_pts[:, 0], filtered_global_pts[:, 1], 
    c='blue', s=8, alpha=0.5, label='Filtered Global Costmap (/robot_local_region)'
)

# Plot the smaller Local Costmap over it (Red)
plt.scatter(
    local_pts[:, 0], local_pts[:, 1], 
    c='red', s=8, alpha=0.9, label='Local Costmap (/costmap/costmap)'
)

# Formatting the graph
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title('KDTree Filtered Occupancy Grids (Aligned)')
plt.legend()
plt.axis('equal') # Ensures 1:1 scale
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()