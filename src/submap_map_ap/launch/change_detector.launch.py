from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # Replace this if your package name in package.xml is different
    pkg_name = 'submap_map_ap'

    return LaunchDescription([
        # 1. Generates the global map crop
        Node(
            package=pkg_name,
            executable='costmap_generator.py',
            name='costmap_generator',
            output='screen'
        ),
        
        # 2. Simplifies the local costmap into 0/100/-1
        Node(
            package=pkg_name,
            executable='local_costmap_simplifier.py',
            name='local_costmap_simplifier',
            output='screen'
        ),
        
        # 3. Cross-correlates and aligns the maps
        Node(
            package=pkg_name,
            executable='costmap_aligner.py',
            name='costmap_cross_correlator',
            output='screen'
        ),
        
        # # 4. Detects the positive/negative cluster changes
        # Node(
        #     package=pkg_name,
        #     executable='costmap_change_detector.py',
        #     name='cluster_change_detector',
        #     output='screen'
        # )
    ])