

# Lifelong Mapping Parameter Configuration

## Overview

This document outlines the ROS 2 configuration parameters for the various nodes handling localization, scan matching (NDT), snapshotting, and map analytics.

---

## Global Parameters (`/`)

These parameters apply globally to the nodes within the namespace.

* **`deadzone_of_bot`**: Set to **70.0**.Defines the radius or area around the robot where sensor readings or map updates are ignored.


* **`use_sim_time`**: Set to **true**. Indicates the system is running in simulation or using bag files.


* **`map_path`**: Points to `/path/to/your/file/map.yaml`. The absolute path to the base map file.



---

## Scan Matching & Alignment (`ndt_node`)

Controls the Normal Distributions Transform (NDT) algorithm used for scan matching and localizing the robot.

### Sanity Checks

* **`sanity_dist_threshold`**: **0.5**. Maximum allowed distance translation before an alignment is flagged as anomalous.


* **`sanity_yaw_threshold`**: **0.5**. Maximum allowed rotational change (yaw) before an alignment is flagged.


* **`sanity_change_threshold`**: **50.0**. The overall threshold for detected changes.


* **`grid_comparison_size`**: **3.0**. Half the size of the grid used when comparing the current scan against the map.



### NDT Aligner Configurations

* **Covariance:**
* `k_neighbors`: **5**. Number of nearest neighbors used to calculate covariance.


* `regularization`: **1e-5**. Prevents singularities in the covariance matrix.


* `fallback_scale`: **1e-3**. Scale used if standard covariance calculations fail.




* **Optimization:**
* `max_iterations`: **15**. The maximum number of optimization steps per alignment.


* `damping_factor`: **1e-5**. Controls the step size during the optimization process.


* `convergence_threshold`: **1e-4**. The threshold at which the optimization stops, assuming a match is found.


* `min_corresponding_points`: **5**. Minimum points required to accept a match.




* **Association:**
* `max_correspondence_distance`: **0.3**. Maximum distance between points to be considered a pair.


* `weight_distance_threshold`: **0.15**. The distance threshold used when applying weights to point correspondences.





---

## Localization Confidence

Nodes responsible for tracking how well the robot knows its position on the map.

* **`AMCL_Confidence`** -> `k`: **0.5**. A tuning constant for calculating the Adaptive Monte Carlo Localization confidence score.


* **`bot_position_tracker`** -> `confidence_threshold_lost`: **0.25**. If the confidence score drops below this value, the robot is officially considered "lost."



---

## Map Analytics & Updating

Parameters handling the dynamic updates and analysis of the map.

* **`global_change_updater`** -> `min_cluster_size`: **10**. The minimum number of points or cells required to classify a cluster as a valid change in the environment.


* **`local_costmap_generator`** -> `grid_size_length`: **8**. The size/dimension of the local costmap grid around the robot.


* **`submap_local_region`** -> `grid_size`: **8**. The size of the local submap region being tracked.


* **`map_analytics_publisher`** -> `wall_inclusion`: **False**. Dictates whether walls are included in the published analytics.



---

## Snapshots & Reporting

Controls when the system saves its current state and how it generates reports.

### Snapshot Triggers (`snapshot_trigger_node`)

Conditions that must be met to trigger a snapshot of the map/state.

* **`distance_threshold`**: **0.5**. Triggers if the robot moves this distance.


* **`angle_threshold`**: **45**. Triggers if the robot rotates this amount (in degrees).


* **`confidence_threshold`**: **0.55**. Triggers based on the localization confidence level.



### Snapshot Publishing (`snapshot_publisher` & `ndt_visualizer_node`)

* **`cooldown_seconds`**: **5**. The minimum time to wait between publishing snapshots to prevent spamming.


* **`ndt_visualise`**: **False**. Toggles the visualization of the NDT alignments.


* **`save_dir_path`**: `/path/to/your/folder`. Directory where snapshot data is saved.



### Report Generation (`bot_report_generator`)

* **`Report_Save_Location`**: `/path/to/your/folder`. Directory where the final bot reports are saved.


* **`occupied_cells_diluter`**: **2.5**. A dilation factor applied to occupied cells to make obstacles thicker/safer in the generated report.