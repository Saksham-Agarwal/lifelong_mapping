import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Resolve the path to the centralized configuration file
    submap_map_ap_dir = get_package_share_directory('submap_map_ap')
    config_file_path = os.path.join(submap_map_ap_dir, 'config', 'submap_map.yaml')

    return LaunchDescription([
        # 1. Map Analytics Node
        Node(
            package='submap_map_ap',
            executable='occupied_cell_publisher.py', 
            name='map_analytics_publisher',
            parameters=[config_file_path],
            output='screen'
        ),
        
        # 2. Position Tracker Node
        Node(
            package='submap_map_ap',
            executable='bot_pos_stat_tracker.py', 
            name='bot_position_tracker',
            parameters=[config_file_path],
            output='screen'
        ),
        
        # 3. State Verifier Node
        Node(
            package='submap_map_ap',
            executable='bot_report_generator.py', 
            name='bot_report_generator',
            output='screen',
            parameters=[config_file_path]
        )
    ])