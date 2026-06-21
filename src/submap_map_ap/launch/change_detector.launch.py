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

    # for launching custom_costmap.yaml parameters with the costmap_generator node
    params_file = os.path.join(
        get_package_share_directory('submap_map_ap'),
        'params',
        'custom_costmap.yaml'
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
        
        # # 3. Cross-correlates and aligns the maps
        # Node(
        #     package=pkg_name,
        #     executable='costmap_aligner.py',
        #     name='costmap_cross_correlator',
        #     output='screen',
        #     parameters=[{'use_sim_time': True}] # <-- Add this line to your 3 custom nodes
        # ),

        # 4. Inflated the local_region map of the bot using cv2
        Node(
            package=pkg_name,
            executable='local_region_inflated_map.py',
            name='local_map_inflater',
            output='screen'
        ),
        
        # 5. Detects the positive/negative cluster changes
        Node(
            package=pkg_name,
            executable='costmap_change_detector.py',
            name='costmap_change_detector',
            output='screen'
        ),

        Node(
            package=pkg_name,
            executable='nearest_neighbour_check.py',
            name='costmap_neighbour_filter',
            output='screen'
        ),
        
        Node(
            package=pkg_name,
            executable='unexplored_to_free_space.py',
            name='costmap_comparator',
            output='screen'
        ),
        
        Node(
            package=pkg_name,
            executable='change_labeler.py',
            name='change_cluster_labeler',
            output='screen'
        ),

        # --- THE CUSTOM COSTMAP ---
        Node (
            package='nav2_costmap_2d',
            executable='nav2_costmap_2d', 
            name='custom_costmap',
            output='screen',
            parameters=[params_file]
        ),

        # --- THE DEDICATED LIFECYCLE MANAGER ---
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_custom_costmap',
            output='screen',
            parameters=[
                {'autostart': True},
                {'node_names': ['costmap/costmap']},
                {'bond_timeout': 0.0}
            ]
        ),

        # 6. Your Custom RQT Dashboard
        Node(
            package='rqt_gui',
            executable='rqt_gui',
            name='custom_rqt_dashboard',
            output='screen',
            arguments=['--perspective-file', rqt_perspective_path]
        )
    ])