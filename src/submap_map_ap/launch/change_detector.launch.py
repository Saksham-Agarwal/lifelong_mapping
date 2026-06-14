import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    pkg_name = 'submap_map_ap'

    # This dynamically finds your package's install directory.
    # Change 'rqt' to whatever folder name you actually put it in (e.g., 'config')
    rqt_perspective_path = os.path.join(
        get_package_share_directory(pkg_name),
        'rqt', 
        'my_dashboard.perspective' # Make sure this matches your exact file name
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
            output='screen',
            parameters=[{'use_sim_time': True}] # <-- Add this line to your 3 custom nodes
        ),
        
        # 4. Detects the positive/negative cluster changes
        Node(
            package=pkg_name,
            executable='costmap_change_detector.py',
            name='cluster_change_detector',
            output='screen'
        ),

        # 5. Your Custom RQT Dashboard
        Node(
            package='rqt_gui',
            executable='rqt_gui',
            name='custom_rqt_dashboard',
            output='screen',
            arguments=['--perspective-file', rqt_perspective_path]
        )
    ])