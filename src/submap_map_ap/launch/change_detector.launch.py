import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # Replace this if your package name in package.xml is different
    pkg_name = 'submap_map_ap'

    # Path to your saved rqt dashboard. 
    # Update 'my_dashboard.perspective' to the exact name of your saved file!
    # os.path.expanduser('~') automatically translates to '/home/saksham-22'
    rqt_perspective_path = os.path.join(
        os.path.expanduser('~'), 
        'rqt', 
        'my_dashboard.perspective'
    )

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
        # ),

        # 5. Your Custom RQT Dashboard (replaces the standard rqt_reconfigure)
        Node(
            package='rqt_gui',
            executable='rqt_gui',
            name='custom_rqt_dashboard',
            output='screen',
            arguments=['--perspective-file', rqt_perspective_path]
        )
    ])