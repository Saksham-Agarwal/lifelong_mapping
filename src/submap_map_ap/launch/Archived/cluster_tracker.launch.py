import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    pkg_name = 'submap_map_ap'


    return LaunchDescription([

        Node(
            package=pkg_name,
            executable='cluster_labelers.py',
            name='general_map_object_detector',
            output='screen'
        ),
        
        
        Node(
            package=pkg_name,
            executable='grid_spatial.py',
            name='grid_spatial_assigner',
            output='screen'
        ),
        
        Node(
            package=pkg_name,
            executable='robot_grid_tracker.py',
            name='robot_grid_tracker',
            output='screen',
        ),

        Node(
            package=pkg_name,
            executable='tracked_map_changes.py',
            name='dynamic_change_tracker',
            output='screen'
        )
    ])