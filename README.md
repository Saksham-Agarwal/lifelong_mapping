#  TurtleBot Navigation and Change Detection

This repository contains the workflow for mapping, localization, navigation, and change detection in a simulation environment.

---

## 1. Launch the Simulation

```bash
ros2 launch submap_map_ap warehouse_turtlebot.launch.py
```

---

## 2. Create a SLAM Map

Start SLAM Toolbox:

```bash
ros2 launch slam_toolbox online_async_launch.py 
```

Drive the robot around the environment to build a map.

### Save the Map

```bash
ros2 run nav2_map_server map_saver_cli -f my_map_name
```

This will generate:

- `my_map_name.yaml`
- `my_map_name.pgm`

---

## 3. Start the Navigation Stack

### Localization

```bash
ros2 launch nav2_bringup localization_launch.py map:=src/submap_map_ap/map/Training/map_2.yaml
```

### Navigation

```bash
ros2 launch nav2_bringup navigation_launch.py map:=src/submap_map_ap/map/Training/map_2.yaml
```


---

## 4. Launch RViz

```bash
ros2 run rviz2 rviz2 -d /opt/ros/jazzy/share/nav2_bringup/rviz/nav2_default_view.rviz
```

---

## 5. Set the Initial Robot Pose

Update the position values to approximately match the robot's current location on the map.

```bash
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "
header:
  frame_id: 'map'
pose:
  pose:
    position:
      x: 0.0
      y: 0.0
      z: 0.0
    orientation:
      z: 0.0
      w: 1.0
"
```

Alternatively, use the **2D Pose Estimate** tool in RViz.

---

## 6. Run the Change Detector

```bash
ros2 launch submap_map_ap change_detector.launch.py
```

This launches the change detection pipeline and compares the current environment against the reference map.

---
