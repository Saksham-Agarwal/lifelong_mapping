#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition
from nav2_msgs.srv import LoadMap, ClearEntireCostmap
from std_srvs.srv import Empty

import os
import glob
import time
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

class RecoveryManager(Node):
    def __init__(self):
        super().__init__('recovery_manager_node')
        
        # Safe default path; overridable by submap_map.yaml when using launch files
        default_dir = '/home/saksham-22/map/Training'
        self.declare_parameter('Report_Save_Location', default_dir)
        
        self.recovering = False
        
        # 1. Create a Reentrant Callback Group
        self.cb_group = ReentrantCallbackGroup()
        
        # Trigger Subscriber
        self.sub_is_lost = self.create_subscription(
            Bool, '/is_bot_lost', self.lost_callback, 10, callback_group=self.cb_group)
            
        # List of the exact lifecycle node names
        self.nodes_to_reset = [
            'ndt_node',
            'global_changes_updater',
            'submap_generator_node',
            'map_analytics_publisher',
            'bot_position_tracker',
            'bot_report_generator'
        ]
        
        self.get_logger().info('System Recovery Manager ready. Monitoring /is_bot_lost...')

    def get_latest_map_yaml(self):
        """Scans the directory to find the most recently generated map YAML file."""
        try:
            raw_dir = self.get_parameter('Report_Save_Location').value
        except Exception as e:
            self.get_logger().error(f"Failed to fetch Report_Save_Location parameter: {e}")
            return None

        if not raw_dir:
            self.get_logger().error("Report_Save_Location parameter is empty!")
            return None
            
        save_dir = os.path.expanduser(raw_dir)
        search_pattern = os.path.join(save_dir, 'map_*.yaml')
        existing_maps = glob.glob(search_pattern)
        
        highest_num = -1
        latest_map_path = None
        
        for filepath in existing_maps:
            filename = os.path.basename(filepath)
            try:
                num_str = filename.replace('map_', '').replace('.yaml', '')
                num = int(num_str)
                if num > highest_num:
                    highest_num = num
                    latest_map_path = filepath
            except ValueError:
                pass
                
        return latest_map_path

    def call_nav2_service(self, service_name, service_type, request_obj):
        """Helper to call Nav2 services synchronously via the multi-threaded executor."""
        client = self.create_client(service_type, service_name, callback_group=self.cb_group)
        
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f'Service {service_name} not available!')
            return False
            
        client.call(request_obj)
        return True

    def lost_callback(self, msg):
        if msg.data and not self.recovering:
            self.get_logger().warn('CRITICAL: Bot is lost! Initiating full system Nav2 recovery sequence...')
            self.recovering = True
            
            # 1. Wait for map generator to save files
            self.get_logger().info('Waiting for map generator to save the new environment files...')
            time.sleep(4.0)
            
            # 2. Locate the new map
            latest_map = self.get_latest_map_yaml()
            if not latest_map:
                self.get_logger().error('Could not find a saved map to load! Aborting recovery.')
                self.recovering = False
                return
                
            self.get_logger().info(f'Loading new map into Nav2 Map Server: {latest_map}')
            
            # 3. Swap the map in Nav2's map_server officially
            load_req = LoadMap.Request()
            load_req.map_url = latest_map
            self.call_nav2_service('/map_server/load_map', LoadMap, load_req)
            
            # 4. Clear Global and Local Costmaps
            self.get_logger().info('Purging stale data from costmaps...')
            self.call_nav2_service('/global_costmap/clear_entirely_global_costmap', ClearEntireCostmap, ClearEntireCostmap.Request())
            self.call_nav2_service('/local_costmap/clear_entirely_local_costmap', ClearEntireCostmap, ClearEntireCostmap.Request())
            
            # 5. Reinitialize AMCL
            self.get_logger().info('Reinitializing AMCL Global Localization...')
            self.call_nav2_service('/reinitialize_global_localization', Empty, Empty.Request())
            
            # 6. Cycle custom lifecycle nodes
            self.get_logger().info('Resetting custom lifecycle nodes...')
            for node_name in self.nodes_to_reset:
                self.reset_lifecycle_node(node_name)
                
            self.get_logger().info('==================================================')
            self.get_logger().info('RECOVERY COMPLETE: System is back online with new map!')
            self.get_logger().info('==================================================')
            
            time.sleep(5.0)
            self.recovering = False

    def reset_lifecycle_node(self, node_name):
        self.get_logger().info(f'   -> Cycling node: {node_name}')
        
        client = self.create_client(ChangeState, f'/{node_name}/change_state', callback_group=self.cb_group)
        
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f'   -> {node_name} lifecycle service unresponsive. Skipping.')
            return
            
        transitions = [
            Transition.TRANSITION_DEACTIVATE,
            Transition.TRANSITION_CLEANUP,
            Transition.TRANSITION_CONFIGURE,
            Transition.TRANSITION_ACTIVATE
        ]
        
        for transition_id in transitions:
            req = ChangeState.Request()
            req.transition.id = transition_id
            client.call(req)
            
def main(args=None):
    rclpy.init(args=args)
    node = RecoveryManager()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()