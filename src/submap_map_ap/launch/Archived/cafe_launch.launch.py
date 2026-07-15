import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, AppendEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # ---------------------------------------------------------
    # 1. FIX THE MESH PATHS
    # ---------------------------------------------------------
    map_parent_dir = os.path.abspath('map')
    
    set_env_vars = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        map_parent_dir
    )

    cafe_world_path = os.path.abspath('src/submap_map_ap/map/cafe/model.sdf')
    
    # ---------------------------------------------------------
    # 2. CALL THE OFFICIAL TURTLEBOT 4 SIMULATOR
    # ---------------------------------------------------------
    tb4_pkg = get_package_share_directory('turtlebot4_gz_bringup')
    
    tb4_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb4_pkg, 'launch', 'turtlebot4_gz.launch.py')
        ),
        launch_arguments={
            'world': 'empty',
            'x': '0.0',
            'y': '0.0',
            # Your floor is 0.19m thick. Spawning at 0.22m means a tiny 3cm drop!
            'z': '0.22', 
            # Disable the dock so it doesn't get stuck in a docking state
            'spawn_dock': 'false' 
        }.items()
    )

    # ---------------------------------------------------------
    # 3. SPAWN YOUR CAFE MAP
    # ---------------------------------------------------------
    # Now we drop the cafe in alongside the robot
    spawn_cafe = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'cafe',
            '-file', cafe_world_path,
            '-x', '0.0', '-y', '0.0', '-z', '0.0'
        ],
        output='screen'
    )

    return LaunchDescription([
        set_env_vars,
        tb4_launch,
        spawn_cafe
    ])