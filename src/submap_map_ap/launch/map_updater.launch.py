from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Map Analytics Node
        Node(
            package='submap_map_ap',
            executable='occupied_cell_publisher.py', # Must match the exact filename in scripts/
            name='map_analytics_publisher',
            output='screen'
        ),
        
        # 2. Position Tracker Node
        Node(
            package='submap_map_ap',
            executable='bot_pos_stat_tracker.py', # Must match the exact filename in scripts/
            name='bot_position_tracker',
            output='screen'
        ),
        
        # 3. State Verifier Node
        Node(
            package='submap_map_ap',
            executable='bot_stat_verifier.py', # Must match the exact filename in scripts/
            name='bot_state_verifier',
            output='screen',
            parameters=[
                {'ratio_threshold': 0.45}
            ]
        )
    ])