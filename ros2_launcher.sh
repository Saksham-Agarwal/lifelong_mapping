#!/bin/bash
# A wrapper script to route commands to Terminator panes

# Uncomment and adjust these if your Terminator panes don't source ROS automatically:
source /opt/ros/humble/setup.bash
source /usr/share/colcon_cd/function/colcon_cd.sh
export _colcon_cd_root=/opt/ros/humble/
#alias spidey='ssh -x spiderbot@erc.local'
#source ~/microros_ws/install/local_setup.bash
#source ~/ws_moveit2/install/setup.bash
source ~/lifelong_mapping/install/setup.bash
export TURTLEBOT3_MODEL=burger
source ~/turtlebot3_ws/install/setup.bash
export ROS_DOMAIN_ID=30 #TURTLEBOT3
source /usr/share/gazebo/setup.sh
source /opt/ros/humble/setup.bash
#source ~/kratos/install/setup.bash

case "$1" in
    "warehouse")
        ros2 launch submap_map_ap bookstore_turtlebot.launch.py
        ;;
    "localization")
        cd ~/lifelong_mapping || exit
        ros2 launch nav2_bringup localization_launch.py map:=src/submap_map_ap/map/bookstore/map.yaml use_sim_time:=true
        ;;
    "navigation")
        cd ~/lifelong_mapping || exit
        ros2 launch nav2_bringup navigation_launch.py map:=src/submap_map_ap/map/bookstore/map.yaml use_sim_time:=true
        ;;
    "rviz")
        ros2 run rviz2 rviz2 -d /opt/ros/humble/share/nav2_bringup/rviz/nav2_default_view.rviz --ros-args -p use_sim_time:=true
        ;;
    "cluster_grid_saved")
    	ros2 launch submap_map_ap cluster_tracker.launch.py
    	;;
    "initialpose")
        # Adding a 5-second delay to ensure localization is fully up before publishing
        echo "Waiting for localization nodes to initialize..."
        sleep 5
		ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: 'map'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {z: 0.0, w: 1.0}}}}"
        ;;
    "change_detector")
        ros2 launch submap_map_ap change_detector.launch.py
        ;;
    *)
        echo "Invalid node argument."
        ;;
esac

# Drop into an interactive shell so the pane doesn't close immediately if a node crashes/exits
exec bash
