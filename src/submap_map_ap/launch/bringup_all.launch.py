import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    submap_pkg_dir = get_package_share_directory('submap_map_ap')

    map_yaml_file = os.path.join(submap_pkg_dir, 'map', 'map.yaml')    
    rviz_config_file = os.path.join(submap_pkg_dir, 'rviz', 'nav2_default_view.rviz')
    
    # REQUIRED FIX: Explicitly define the default Nav2 parameter file
    nav2_params_file = os.path.join(nav2_bringup_dir, 'params', 'nav2_params.yaml')

    # A. Launch the Turtlebot simulation
    warehouse_turtlebot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(submap_pkg_dir, 'launch', 'warehouse_turtlebot.launch.py')
        )
    )

    # B. Launch Nav2 Navigation
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'map': map_yaml_file,
            'params_file': nav2_params_file,
            'use_sim_time': 'True'
        }.items()
    )

    # C. Launch Nav2 Localization (AMCL)
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'localization_launch.py')
        ),
        launch_arguments={
            'map': map_yaml_file,
            'params_file': nav2_params_file,
            'use_sim_time': 'True'
        }.items()
    )

    # D. Start RViz2 with the Nav2 configuration
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        # REQUIRED FIX: Ensure RViz syncs with simulation time
        parameters=[{'use_sim_time': True}] 
    )

    # Return the LaunchDescription
    return LaunchDescription([
        warehouse_turtlebot_launch,
        navigation_launch,
        localization_launch,
        rviz_node
    ])