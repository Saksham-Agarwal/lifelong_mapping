# Topics Documentation

This document describes the important topics used in the lifelong mapping project and their respective roles within the processing pipeline.

---

## 1. `/robot_local_region`

### Purpose

Generates a localized occupancy grid around the robot using the previously generated global map (`/map`).

Unlike a standard local costmap, this topic directly extracts a subset of the occupancy grid from the saved map produced by `slam_toolbox`.

### Details

* Source map: `/map`
* Map origin: Previously generated and saved SLAM map.
* Coverage area: 8 meters around the robot in form of square.
* Map resolution: 0.05 m per cell.
* Localized region size:

[
8 / 0.05 = 160
]

Approximately 160 cells are represented per direction from the robot position.

### Importance

This topic provides the reference map segment against which current observations can later be compared for change detection.

---

## 2. `/inflated_local_region`

### Purpose

Publishes an inflated version of `/robot_local_region`.

### Details

Obstacle cells within the localized occupancy grid are expanded (inflated) to account for:

* Sensor uncertainty
* Localization inaccuracies
* Safety margins around obstacles

### Importance

Inflation improves robustness when comparing occupancy maps by reducing sensitivity to minor alignment errors and sensor noise.

---

## 3. `/simplified_local_costmap`

### Purpose

Creates a simplified version of the robot's local costmap.

### Details

The inflation characteristics of the navigation costmap are reduced and processed so that the resulting representation closely resembles the occupancy structure of `/inflated_local_region`.

### Importance

Direct comparison between a navigation costmap and an occupancy grid is difficult due to differing inflation and representation methods.

This topic provides a normalized representation suitable for map comparison.

---

## 4. `/aligned_local_costmap`

### Purpose

Aligns the simplified local costmap with the coordinate frame and structure of `/robot_local_region`.

### Details

The topic performs spatial alignment so that corresponding cells in both maps refer to the same physical locations.

### Importance

Accurate alignment is a prerequisite for reliable change detection.

Without alignment, differences caused by coordinate offsets could be incorrectly interpreted as environmental changes.

---

# Processing Pipeline

```text
/map
  │
  ▼
/robot_local_region
  │
  ▼
/inflated_local_region

local_costmap
  │
  ▼
/simplified_local_costmap
  │
  ▼
/aligned_local_costmap

/inflated_local_region
           +
/aligned_local_costmap
           │
           ▼
     Change Detection
```

