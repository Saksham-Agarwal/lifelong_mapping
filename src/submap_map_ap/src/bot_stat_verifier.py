#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32, Float32
from nav_msgs.msg import OccupancyGrid
import numpy as np

class BotStateVerifier(Node):
    def __init__(self):
        super().__init__('bot_state_verifier')
        
        # Declare the threshold parameter (default is 0.45)
        self.declare_parameter('ratio_threshold', 0.45)
        
        # Caches for the data we need before doing math
        self.latest_occupied_count = None
        self.latest_changes_map = None
        
        # 1. Passive Subscribers (Caching data)
        self.sub_occupied_count = self.create_subscription(
            Int32, '/internal_occupied_count', self.count_callback, 10)
            
        self.sub_overall_changes = self.create_subscription(
            OccupancyGrid, '/overall_changes', self.map_callback, 10)
            
        # 2. Trigger Subscriber (Fires the calculation)
        self.sub_is_lost = self.create_subscription(
            Bool, '/is_bot_lost', self.is_lost_callback, 10)
            
        # 3. Publishers for individual metrics
        self.pub_pos_ratio = self.create_publisher(Float32, '/ratio_positive_change', 10)
        self.pub_neg_ratio = self.create_publisher(Float32, '/ratio_negative_change', 10)
        self.pub_total_ratio = self.create_publisher(Float32, '/ratio_total_change', 10)
        
        # 4. Publishers for final decision / client return
        self.pub_confirmed_lost = self.create_publisher(Bool, '/confirmed_bot_lost', 10)
        self.pub_checkpoint = self.create_publisher(Bool, '/Bot_Save_checkpoint', 10)

        self.get_logger().info('Verifier Node started. Waiting for /is_bot_lost trigger...')

    def count_callback(self, msg):
        self.latest_occupied_count = msg.data

    def map_callback(self, msg):
        self.latest_changes_map = msg

    def is_lost_callback(self, msg):
        # We only run the heavy calculation if the bot suspects it is lost
        if not msg.data:
            return
            
        # Ensure we have the required data to do the math
        if self.latest_occupied_count is None or self.latest_changes_map is None:
            self.get_logger().warn('Trigger received, but missing map or count data. Cannot verify.')
            return
            
        if self.latest_occupied_count == 0:
            self.get_logger().error('Internal occupied count is 0. Cannot divide by zero!')
            return

        # Fetch parameter threshold
        threshold = self.get_parameter('ratio_threshold').get_parameter_value().double_value
        
        # Convert map to numpy array
        grid = np.array(self.latest_changes_map.data, dtype=np.int8)
        
        # Calculate Positive Change: Occupied cells (Standard ROS assumes >50 is occupied)
        positive_change_cells = np.count_nonzero(grid > 50)
        
        # Calculate Negative Change: Unexplored cells (Standard ROS assumes -1 is unknown)
        negative_change_cells = np.count_nonzero(grid == -1)
        
        # Calculate Ratios
        pos_ratio = positive_change_cells / self.latest_occupied_count
        neg_ratio = negative_change_cells / self.latest_occupied_count
        total_ratio = pos_ratio + neg_ratio
        
        # Publish Ratios
        self.pub_pos_ratio.publish(Float32(data=float(pos_ratio)))
        self.pub_neg_ratio.publish(Float32(data=float(neg_ratio)))
        self.pub_total_ratio.publish(Float32(data=float(total_ratio)))
        
        # Evaluate against threshold
        is_actually_lost = bool(total_ratio > threshold)
        
        # Publish final decisions
        self.pub_confirmed_lost.publish(Bool(data=is_actually_lost))
        
        if is_actually_lost:
            self.get_logger().warn(f'CONFIRMED LOST! Total ratio {total_ratio:.2f} > {threshold:.2f}')
            # Trigger the checkpoint save
            self.pub_checkpoint.publish(Bool(data=True))
        else:
            self.get_logger().info(f'FALSE ALARM. Total ratio {total_ratio:.2f} is below {threshold:.2f}')
            # Explicitly tell the system we do not need to save checkpoint
            self.pub_checkpoint.publish(Bool(data=False))


def main(args=None):
    rclpy.init(args=args)
    node = BotStateVerifier()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()