 

 ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

 ros2 launch nav2_bringup navigation_launch.py   map:=my_map.yaml
ros2 launch nav2_bringup localization_launch.py  map:=my_map.yaml
 ros2 launch slam_toolbox online_async_launch.py use_sim_time:=True

ros2 run rviz2 rviz2 -d /opt/ros/humble/share/nav2_bringup/rviz/nav2_default_view.rviz



ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "
   header:
     frame_id: 'map'
   pose:
     pose:
       position:
         x: -2.0
         y: -0.5
         z: 0.0
       orientation:
         z: 0.0
         w: 1.0
  "


