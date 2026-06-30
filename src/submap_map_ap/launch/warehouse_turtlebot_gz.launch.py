#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # ---------------------------------------------------------
    # Update this path to wherever you saved the modified SDF
    # ---------------------------------------------------------
    world_file_path = os.path.join(
        get_package_share_directory('submap_map_ap'), 'world', 'warehouse_turtlebot.sdf'
    )
    turtlebot3_gazebo_path = get_package_share_directory('turtlebot3_gazebo')
    burger_model_file = os.path.join(
        turtlebot3_gazebo_path, 'models', 'turtlebot3_burger', 'model.sdf'
    )
    # Launch Argument for the world file
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=world_file_path,
        description='Path to the SDF world file'
    )

    # 1. Start Gazebo Sim (Ignition) with the custom world
    # This uses the standard `gz_sim.launch.py` from `ros_gz_sim`
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r ', LaunchConfiguration('world')]}.items()
    )

    spawn_turtlebot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'turtlebot3_burger',
            '-x', '13.9', '-y', '-10.6', '-z', '0.1', 
            '-file', burger_model_file
        ],
        output='screen'
    )

    # 3. Optional: Bridge /cmd_vel between ROS 2 and Gazebo
    # Allows you to use standard `ros2 run teleop_twist_keyboard teleop_twist_keyboard`
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist'
        ],
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        gazebo,
        spawn_turtlebot,
        ros_gz_bridge
    ])