import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Resolve the path to the centralized configuration file
    submap_map_ap_dir = get_package_share_directory('submap_map_ap')
    config_file_path = os.path.join(submap_map_ap_dir, 'config', 'submap_map.yaml')

    # ---------------------------------------------------------
    # Define Nodes
    # ---------------------------------------------------------
    
    map_bounds_cmd = Node(
        package='submap_map_ap',
        executable='map_bounds_publisher.py',
        name='map_bounds_publisher',
        parameters=[config_file_path],
        output='screen'
    )

    snapshot_tracker_cmd = Node(
        package='submap_map_ap',
        executable='snapshot_tracker.py',
        name='snapshot_trigger_node',
        parameters=[config_file_path],
        output='screen'
    )

    snapshot_publisher_cmd = Node(
        package='submap_map_ap',
        executable='snapshot_publisher.py',
        name='snapshot_publisher',
        parameters=[config_file_path],
        output='screen'
    )

    ndt_node_cmd = Node(
        package='Scan_matching',
        executable='ndt_node.py',
        name='ndt_node',
        parameters=[config_file_path],
        output='screen'
    )

    submap_node_cmd = Node(
        package='submap_map_ap',
        executable='submap.py',
        name='submap_generator_node',
        parameters=[config_file_path],
        output='screen'
    )

    submap_local_region_cmd = Node(
        package='submap_map_ap',
        executable='submap_local_region.py',
        name='submap_local_region',
        parameters=[config_file_path],
        output='screen'
    )

    ndt_visualizer_cmd = Node(
        package='Scan_matching',
        executable='ndt_visualiser.py',
        name='ndt_visualizer_node',
        parameters=[config_file_path],
        output='screen'
    )

    cluster_creator_cmd = Node(
        package='Scan_matching',
        executable='cluster_creator.py',
        name='cluster_creator_node',
        parameters=[config_file_path],
        output='screen'
    )

    global_change_update_cmd = Node(
        package='Scan_matching',
        executable='global_changes_updater.py',
        name='global_changes_updater',
        parameters=[config_file_path],
        output='screen'
    )

    # ---------------------------------------------------------
    # Create and Populate Launch Description
    # ---------------------------------------------------------
    ld = LaunchDescription()

    ld.add_action(map_bounds_cmd)
    ld.add_action(snapshot_tracker_cmd)
    ld.add_action(snapshot_publisher_cmd)
    ld.add_action(ndt_node_cmd)
    ld.add_action(submap_node_cmd)
    ld.add_action(submap_local_region_cmd)
    ld.add_action(global_change_update_cmd)
    ld.add_action(ndt_visualizer_cmd)
    ld.add_action(cluster_creator_cmd)

    return ld