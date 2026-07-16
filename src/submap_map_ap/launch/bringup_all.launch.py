import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # ---------------------------------------------------------
    # 1. Path Resolutions & Configurations
    # ---------------------------------------------------------
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    submap_map_ap_dir = get_package_share_directory('submap_map_ap')
    
    config_file_path = os.path.join(submap_map_ap_dir, 'config', 'submap_map.yaml')
    rviz_config_path = os.path.join(nav2_bringup_dir, 'rviz', 'nav2_default_view.rviz')
    
    # ---------------------------------------------------------
    # 2. Parse Parameters Natively From YAML File
    # ---------------------------------------------------------
    try:
        with open(config_file_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Isolate global parameters from the '/**' wildcard block
        global_params = yaml_data.get('/**', {}).get('ros__parameters', {})
        
        # Extract values with safe defaults if keys are missing
        map_path_val = global_params.get('map_path', os.path.join(submap_map_ap_dir, 'map', 'Training', 'map_4.yaml'))
        use_sim_time_val = str(global_params.get('use_sim_time', True)).lower() # Nav2 arguments expect 'true'/'false' strings
    except Exception as e:
        # Print the error so it doesn't fail silently!
        print(f"\n[WARNING] Failed to load YAML config: {e}")
        print("[WARNING] Falling back to default map_4.yaml!\n")
        
        map_path_val = os.path.join(submap_map_ap_dir, 'map', 'Training', 'map_4.yaml')
        use_sim_time_val = 'true'

    # ---------------------------------------------------------
    # 3. Define Launch Actions & Nodes
    # ---------------------------------------------------------

    # Localization Stack
    localization_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'localization_launch.py')
        ),
        launch_arguments={
            'map': map_path_val, 
            'use_sim_time': use_sim_time_val
        }.items()
    )

    # Navigation Stack
    navigation_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'map': map_path_val, 
            'use_sim_time': use_sim_time_val
        }.items()
    )

    # RViz2 Visualization
    rviz2_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        parameters=[{'use_sim_time': use_sim_time_val == 'true'}],
        output='screen'
    )

    # Costmap Generator Node
    costmap_generator_cmd = Node(
        package='submap_map_ap',
        executable='costmap_generator.py',
        name='costmap_generator',
        parameters=[config_file_path],
        output='screen'
    )

    # AMCL Confidence Node
    amcl_confidence_cmd = Node(
        package='submap_map_ap',
        executable='amcl_confidence.py', 
        name='AMCL_Confidence',
        parameters=[config_file_path],
        output='screen'
    )

    # ---------------------------------------------------------
    # 4. Create and Populate Launch Description
    # ---------------------------------------------------------
    ld = LaunchDescription()

    ld.add_action(localization_cmd)
    ld.add_action(navigation_cmd)
    ld.add_action(rviz2_cmd)
    ld.add_action(costmap_generator_cmd)
    ld.add_action(amcl_confidence_cmd)

    return ld