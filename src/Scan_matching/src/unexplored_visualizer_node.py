#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math
import os
import time
import json
from std_msgs.msg import String

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def plot_robot_and_deadspace(ax, x, y, yaw, deadzone):
    ax.plot(x, y, 'go', markersize=10, zorder=10)
    line_length = 0.5 
    ax.plot([x, x + line_length * math.cos(yaw)], [y, y + line_length * math.sin(yaw)], 'g-', linewidth=2.5, zorder=10)
    angle_pos = yaw + math.radians(deadzone)
    angle_neg = yaw - math.radians(deadzone)
    ax.plot([x, x + line_length * math.cos(angle_pos)], [y, y + line_length * math.sin(angle_pos)], 'k--', linewidth=1.5, zorder=9)
    ax.plot([x, x + line_length * math.cos(angle_neg)], [y, y + line_length * math.sin(angle_neg)], 'k--', linewidth=1.5, zorder=9)

class UnexploredPlotterNode(Node):
    def __init__(self):
        super().__init__('unexplored_plotter_node')
        self.declare_parameter('deadzone_of_bot', 70.0)
        self.declare_parameter('visualise', True)
        self.declare_parameter('save_dir_path', os.path.join(os.getcwd(), 'saves', 'unexplored'))

        self.deadzone_of_bot = self.get_parameter('deadzone_of_bot').value
        self.visualise = self.get_parameter('visualise').value
        self.save_dir = self.get_parameter('save_dir_path').value
        os.makedirs(self.save_dir, exist_ok=True)

        if not self.visualise:
            self.get_logger().info('Unexplored Visualizer is DISABLED via config. Node will sit idle.')
            return

        self.sub_unexp_debug = self.create_subscription(String, '/unexplored_debug_data', self.debug_callback, 10)
        self.get_logger().info(f'Dumb Visualizer Node running. Storing plots to: {self.save_dir}')

    def debug_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse JSON: {e}")
            return
            
        def to_np(arr): return np.array(arr) if arr is not None else np.array([])

        pose = data['true_pose']
        aligned_source = to_np(data['source'])
        raw_unexplored = to_np(data['raw_unexplored'])
        cleared_clusters = to_np(data['cleared_clusters'])

        self.save_plot(raw_unexplored, cleared_clusters, aligned_source, pose)

    def save_plot(self, all_unexplored, cleared_unexplored, source_scan, pose):
        fig = plt.figure(figsize=(16, 8)) 
        plt_min_x, plt_max_x = pose['x'] - 3.0, pose['x'] + 3.0
        plt_min_y, plt_max_y = pose['y'] - 3.0, pose['y'] + 3.0

        def format_ax(ax, title):
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlim(plt_min_x, plt_max_x)
            ax.set_ylim(plt_min_y, plt_max_y)
            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.6)

        ax1 = fig.add_subplot(1, 2, 1)
        format_ax(ax1, "1. Whole Unexplored Map (Submap)")
        if len(source_scan) > 0:
            ax1.scatter(source_scan[:, 0], source_scan[:, 1], c='gray', s=2, alpha=0.3, label="Aligned Scan")
        if len(all_unexplored) > 0:
            ax1.scatter(all_unexplored[:, 0], all_unexplored[:, 1], c='red', s=15, marker='s', alpha=0.7, label="All Unexplored")
        plot_robot_and_deadspace(ax1, pose['x'], pose['y'], pose['yaw'], self.deadzone_of_bot)
        ax1.legend(loc='upper right')

        ax2 = fig.add_subplot(1, 2, 2)
        format_ax(ax2, "2. NDT Computed Negative Space")
        if len(source_scan) > 0:
            ax2.scatter(source_scan[:, 0], source_scan[:, 1], c='gray', s=2, alpha=0.3)
        if len(cleared_unexplored) > 0:
            ax2.scatter(cleared_unexplored[:, 0], cleared_unexplored[:, 1], c='green', s=15, marker='s', alpha=0.9, label="Computed Free Space")
        plot_robot_and_deadspace(ax2, pose['x'], pose['y'], pose['yaw'], self.deadzone_of_bot)
        ax2.legend(loc='upper right')

        diag = f"True Pose: X: {pose['x']:.4f}m, Y: {pose['y']:.4f}m, Yaw: {pose['yaw']:.4f}rad\n[STATS] Total Unexplored: {len(all_unexplored)} | Filtered & Clustered Free Space: {len(cleared_unexplored)}"
        fig.text(0.5, 0.05, diag, ha='center', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))
        plt.subplots_adjust(bottom=0.25, wspace=0.2) 
        
        filename = os.path.join(self.save_dir, f"unexplored_cleared_{int(time.time())}.png")
        plt.savefig(filename)
        plt.close(fig)

def main(args=None):
    rclpy.init(args=args)
    try: rclpy.spin(UnexploredPlotterNode())
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()