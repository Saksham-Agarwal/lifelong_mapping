import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # ---------------------------------------------------------
    # 1. Path Resolutions
    # ---------------------------------------------------------
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    submap_map_ap_dir = get_package_share_directory('submap_map_ap')
    
    # Resolving your specific relative workspace paths to absolute paths
    home_dir = os.path.expanduser('~')
    map_path = os.path.join(home_dir, 'lifelong_mapping/src/submap_map_ap/map/Training/map_3.yaml')
    laser_filter_config = os.path.join(home_dir, 'lifelong_mapping/src/submap_map_ap/config/laser_filter_config.yaml')
    # nav2_bringup_dir is already defined at the top of your script!
    rviz_config_path = os.path.join(nav2_bringup_dir, 'rviz', 'nav2_default_view.rviz')
    # ---------------------------------------------------------
    # 2. Define Launch Actions & Nodes
    # ---------------------------------------------------------

    # Localization
    localization_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'localization_launch.py')
        ),
        launch_arguments={'map': map_path}.items()
    )

    # Navigation
    navigation_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={'map': map_path}.items()
    )

    # RViz2
    rviz2_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )

    # Change Detector Launch
    change_detector_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(submap_map_ap_dir, 'launch', 'change_detector.launch.py')
        )
    )

    # Laser Filters Node
    laser_filter_cmd = Node(
        package='laser_filters',
        executable='scan_to_scan_filter_chain',
        name='laser_filter',
        parameters=[laser_filter_config],
        remappings=[
            ('/scan', '/scan_raw'),
            ('/scan_filtered', '/scan_reliable')
        ],
        output='screen'
    )

    # AMCL Confidence Node
    amcl_confidence_cmd = Node(
        package='submap_map_ap',
        executable='amcl_confidence.py', 
        name='AMCL_Confidence',
        output='screen'
    )

    # ---------------------------------------------------------
    # 3. Create and Populate Launch Description
    # ---------------------------------------------------------
    ld = LaunchDescription()

    ld.add_action(localization_cmd)
    ld.add_action(navigation_cmd)
    ld.add_action(rviz2_cmd)
    ld.add_action(change_detector_cmd)
    ld.add_action(laser_filter_cmd)
    ld.add_action(amcl_confidence_cmd)

    return ld 