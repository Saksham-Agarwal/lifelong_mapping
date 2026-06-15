# Nav2 Parameters Documentation

This document contains the Nav2 local costmap configuration used for the lifelong mapping project and a summary of the modifications made to support map change detection.

---

# Current Local Costmap Configuration

```yaml
local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 5
      height: 5
      resolution: 0.05
      robot_radius: 0.22
      plugins: ["obstacle_layer", "inflation_layer"]

      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 5.0
        inflation_radius: 0.45

      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan

        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
          inf_is_valid: True

      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True

      always_send_full_costmap: True
```

---

# Previous Configuration

```yaml
local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.22
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.05
        z_voxels: 16
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      always_send_full_costmap: True
```

---

# Changes Made

The following modifications were introduced to improve compatibility between the Nav2 local costmap and the saved occupancy map used for lifelong mapping:

## 1. Increased Local Costmap Size

* Width increased from **3 m** to **5 m**
* Height increased from **3 m** to **5 m**

This allows a larger surrounding area to be observed and compared with the extracted local map region.

## 2. Replaced Voxel Layer with Obstacle Layer

Previous implementation used a voxel-based obstacle representation.

Updated configuration uses:

```yaml
plugins: ["obstacle_layer", "inflation_layer"]
```

This provides a cleaner 2D obstacle representation that is easier to align and compare with occupancy-grid maps generated from the saved SLAM map.

## 3. Inflation Tuning

Inflation parameters were adjusted as follows:

```yaml
cost_scaling_factor: 5.0
inflation_radius: 0.45
```

The resulting inflated costmap is later processed by `/simplified_local_costmap` to obtain a representation that closely matches `/inflated_local_region`.

## 4. Laser Scan Based Obstacle Detection (The most important change)

Obstacle observations are generated directly from:

```yaml
topic: /scan
data_type: "LaserScan"
```

with:

```yaml
obstacle_max_range: 2.5
raytrace_max_range: 3.0
```

This provides the real-time obstacle information that is compared against the expected environment stored in the saved map.

---

