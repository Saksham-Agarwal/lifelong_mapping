from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # ---------------------------------------------------------
    # Define Nodes
    # ---------------------------------------------------------
    
    map_bounds_cmd = Node(
        package='submap_map_ap',
        executable='map_bounds_publisher.py',
        name='map_bounds_publisher',
        output='screen'
    )

    snapshot_tracker_cmd = Node(
        package='submap_map_ap',
        executable='snapshot_tracker.py',
        name='snapshot_trigger_node',
        output='screen'
    )

    snapshot_publisher_cmd = Node(
        package='submap_map_ap',
        executable='snapshot_publisher.py',
        name='snapshot_publisher',
        output='screen'
    )

    ndt_node_cmd = Node(
        package='Scan_matching',
        executable='ndt_node.py',
        name='ndt_node',
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

    return ld