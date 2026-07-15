#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
 
    tb3_launch_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    aws_warehouse_dir = get_package_share_directory('aws_robomaker_small_warehouse_world')

    # Launch Configurations
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')

    # 1. Launch the AWS Small Warehouse (No Roof) world
    warehouse_world_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(aws_warehouse_dir, 'launch', 'small_warehouse.launch.py')
        ),
        launch_arguments={
            'world': os.path.join(
                aws_warehouse_dir, 
                'worlds', 
                'no_roof_small_warehouse', 
                'no_roof_small_warehouse.world'
            )
        }.items()
    )

    # 2. Start the TurtleBot3 Robot State Publisher
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_launch_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # 3. Spawn the TurtleBot3 in the Gazebo world
    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_launch_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose
        }.items()
    )

    # Create Launch Description and add actions
    ld = LaunchDescription()

    ld.add_action(warehouse_world_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)

    return ld