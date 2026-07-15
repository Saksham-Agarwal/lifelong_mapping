#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from std_msgs.msg import Bool, Int32
from nav_msgs.msg import OccupancyGrid

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

class BotReportGenerator(Node):
    def __init__(self):
        super().__init__('bot_report_generator')
        
        # 1. Parameter for the save directory 
        # (Default assumes you are running from a standard ROS2 workspace)
        default_dir = '~/lifelong_mapping/src/submap_map_ap/map/Training'
        self.declare_parameter('Report_Save_Location', default_dir)
        self.declare_parameter('occupied_cells_diluter', 2.5)
        self.diluter = self.get_parameter('occupied_cells_diluter').value

        # 2. Caches for the data
        self.latest_occupied_count = None
        self.latest_changes_map = None
        self.latest_submap = None
        
        # 3. Latching QoS (Required for grabbing maps that are published infrequently)
        latching_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # 4. Subscribers
        self.sub_occupied_count = self.create_subscription(
            Int32, '/internal_occupied_count', self.count_callback, latching_qos)
            
        self.sub_overall_changes = self.create_subscription(
            OccupancyGrid, '/overall_changes', self.changes_callback, latching_qos)
            
        # Subscribe to submap with latching QoS to ensure we get the latest map
        self.sub_submap = self.create_subscription(
            OccupancyGrid, '/submap_map', self.submap_callback, latching_qos)
            
        # Trigger Subscriber
        self.sub_is_lost = self.create_subscription(
            Bool, '/is_bot_lost', self.is_lost_callback, 10)

        self.get_logger().info('Report Generator started. Waiting for /is_bot_lost trigger...')

    def count_callback(self, msg):
        self.latest_occupied_count = msg.data

    def changes_callback(self, msg):
        self.latest_changes_map = msg
        
    def submap_callback(self, msg):
        self.latest_submap = msg

    def get_next_map_number(self, save_dir):
        """Scans the directory and returns the next highest map_X number."""
        search_pattern = os.path.join(save_dir, 'map_*.pgm')
        existing_maps = glob.glob(search_pattern)
        
        highest_num = 0
        for filepath in existing_maps:
            filename = os.path.basename(filepath)
            try:
                # Extract the number X from 'map_X.pgm'
                num_str = filename.replace('map_', '').replace('.pgm', '')
                num = int(num_str)
                if num > highest_num:
                    highest_num = num
            except ValueError:
                pass # Ignore files that don't match exactly
                
        return highest_num + 1

    def is_lost_callback(self, msg):
        # We only act if the bot is actually reporting lost
        if not msg.data:
            return
            
        self.get_logger().warn('Bot lost triggered! Generating diagnostic report...')
            
        if self.latest_occupied_count is None or self.latest_changes_map is None or self.latest_submap is None:
            self.get_logger().error('Missing necessary map or count data. Cannot generate report.')
            return
            
        if self.latest_occupied_count == 0:
            self.get_logger().error('Internal occupied count is 0. Cannot divide by zero!')
            return

        # 1. Setup Directory
        raw_dir = self.get_parameter('Report_Save_Location').get_parameter_value().string_value
        save_dir = os.path.expanduser(raw_dir) # Expands '~' to /home/username
        os.makedirs(save_dir, exist_ok=True)
        
        next_id = self.get_next_map_number(save_dir)
        
        # 2. Calculate Percentages with Deflation Heuristic
        changes_grid = np.array(self.latest_changes_map.data, dtype=np.int8)
        
        raw_pos_cells = np.count_nonzero(changes_grid > 50)
        raw_neg_cells = np.count_nonzero(changes_grid == -1)
        
        # Deflation Math: A 3x3 inflation turns 1 cell into 9. 
        # Dividing by 9 gives us a much more accurate estimate of the actual points.
        deflation_factor = 6.0
        est_pos_cells = raw_pos_cells
        est_neg_cells = raw_neg_cells / deflation_factor
        occupied_cells = self.latest_occupied_count/ self.diluter
        
        pos_pct = (est_pos_cells / occupied_cells) * 100
        neg_pct = (est_neg_cells / occupied_cells) * 100
        tot_pct = pos_pct + neg_pct
        
        # 3. Save Submap (PGM + YAML)
        self.save_map_files(save_dir, next_id)

        # 4. Generate & Display Matplotlib Report
        # FIX: Pass the RAW cell counts so the text matches the visual pixels
        self.generate_visual_report(save_dir, next_id, raw_pos_cells, raw_neg_cells, pos_pct, neg_pct, tot_pct)
        
        self.get_logger().info(f'Report #{next_id} successfully saved to {save_dir}')


    def save_map_files(self, save_dir, map_id):
        """Converts the ROS OccupancyGrid to a PGM image and generates the YAML."""
        submap = self.latest_submap
        width = submap.info.width
        height = submap.info.height
        
        grid_2d = np.array(submap.data, dtype=np.int8).reshape((height, width))
        
        # ROS standard to Image standard mapping
        # 0 (Free) -> 254 (White)
        # 100 (Occupied) -> 0 (Black)
        # -1 (Unknown) -> 205 (Gray)
        img = np.full((height, width), 205, dtype=np.uint8)
        img[grid_2d == 0] = 254
        img[grid_2d > 50] = 0
        
        # Flip vertically (ROS origin is bottom-left, images are top-left)
        img = cv2.flip(img, 0)
        
        # Save PGM
        pgm_path = os.path.join(save_dir, f'map_{map_id}.pgm')
        cv2.imwrite(pgm_path, img)
        
        # Save YAML
        yaml_path = os.path.join(save_dir, f'map_{map_id}.yaml')
        yaml_content = f"""image: map_{map_id}.pgm
resolution: {submap.info.resolution}
origin: [{submap.info.origin.position.x}, {submap.info.origin.position.y}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)


    def generate_visual_report(self, save_dir, map_id, pos_cells, neg_cells, pos_pct, neg_pct, tot_pct):
        """Creates the Matplotlib diagnostic plot and saves it."""
        width = self.latest_changes_map.info.width
        height = self.latest_changes_map.info.height
        
        changes_2d = np.array(self.latest_changes_map.data, dtype=np.int8).reshape((height, width))
        changes_2d = np.rot90(changes_2d)

        # --- FIX: Create an RGB image so the map is actually readable ---
        # Unknown (-1) -> Red (Negative Change)
        # Free (0) -> White
        # Occupied (>50) -> Green (Positive Change)
        display_img = np.zeros((height, width, 3), dtype=np.uint8)
        display_img[changes_2d == -1] = [255, 0, 0]
        display_img[changes_2d == 0] = [255, 255, 255]
        display_img[changes_2d > 50] = [0, 255, 0]
        
        plt.figure(figsize=(8, 8))
        
        # origin='lower' correctly aligns ROS maps in matplotlib
        plt.imshow(display_img, origin='lower')
        plt.title(f"Diagnostic Report #{map_id}: /overall_changes")
        
        # Create the report text block (Updated to clarify raw pixels vs impact)
        report_text = (
            f"Base Internal Occupied Points: {self.latest_occupied_count}\n"
            f"Positive Change (Green): {pos_cells} raw pixels (~{pos_pct:.2f}% impact)\n"
            f"Negative Change (Red): {neg_cells} raw pixels (~{neg_pct:.2f}% impact)\n"
            f"Total Aggregate Change: {tot_pct:.2f}%"
        )
        
        # Place text at the bottom
        plt.figtext(0.5, 0.02, report_text, wrap=True, horizontalalignment='center', 
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))
        
        # Squeeze the plot up slightly so the text doesn't overlap the image
        plt.subplots_adjust(bottom=0.2)
        
        # Save the report image
        report_path = os.path.join(save_dir, f'report_{map_id}.png')
        plt.savefig(report_path)
        
        # Pop up the window for 3 seconds without freezing the ROS node execution
        plt.show(block=False)
        plt.pause(3)
        plt.close()


def main(args=None):
    rclpy.init(args=args)
    node = BotReportGenerator()
    
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