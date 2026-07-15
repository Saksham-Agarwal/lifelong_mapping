#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import math
import os
import time
import json

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def plot_robot_and_deadspace(ax, x, y, yaw, deadzone):
    ax.plot(x, y, 'go', markersize=10, zorder=10)
    line_length = 0.5 
    ax.plot([x, x + line_length * math.cos(yaw)], [y, y + line_length * math.sin(yaw)], 'g-', linewidth=2.5, zorder=10)
    angle_pos = yaw +math.radians(deadzone)
    angle_neg = yaw - math.radians(deadzone)
    ax.plot([x, x + line_length * math.cos(angle_pos)], [y, y + line_length * math.sin(angle_pos)], 'k--', linewidth=1.5, zorder=9)
    ax.plot([x, x + line_length * math.cos(angle_neg)], [y, y + line_length * math.sin(angle_neg)], 'k--', linewidth=1.5, zorder=9)

class NDTVisualizerNode(Node):
    def __init__(self):
        super().__init__('ndt_visualizer_node')
        self.save_dir = os.path.join(os.getcwd(), 'saves')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 1. Declare parameters
        self.declare_parameter('deadzone_of_bot', 70.0)
        self.declare_parameter('ndt_visualise', True)

        # 2. Retrieve parameters
        self.deadzone_of_bot = self.get_parameter('deadzone_of_bot').value
        self.ndt_visualise = self.get_parameter('ndt_visualise').value

        # 3. Toggle logic: Only run if ndt_visualise is True
        if not self.ndt_visualise:
            self.get_logger().info('NDT Visualizer is DISABLED via config. Node will sit idle.')
            return

        self.sub_debug = self.create_subscription(String, '/ndt_debug_data', self.debug_callback, 10)
        self.get_logger().info('NDT Visualizer Node running. Awaiting debug payloads...')

    def debug_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse debug JSON: {e}")
            return
            
        def to_np(arr): return np.array(arr) if arr is not None else np.array([])
        
        # Deserialize arrays inside results
        def parse_res(res):
            if not res: return None
            for key in ['drift_mat', 'aligned', 'visible', 'occluded', 'positive', 'negative']:
                if key in res: res[key] = to_np(res[key])
            return res

        amcl = data['amcl']
        source = to_np(data['source'])
        global_tgt = to_np(data['global_target'])
        submap_tgt = to_np(data['submap_target'])
        global_res = parse_res(data['global_res'])
        submap_res = parse_res(data['submap_res'])

        self.save_plot(source, global_tgt, submap_tgt, global_res, submap_res, amcl)

    def save_plot(self, source, global_target, submap_target, global_res, submap_res, amcl):
        fig = plt.figure(figsize=(24, 14)) 
        
        plt_min_x, plt_max_x = amcl['x'] - 3.0, amcl['x'] + 3.0
        plt_min_y, plt_max_y = amcl['y'] - 3.0, amcl['y'] + 3.0

        def format_ax(ax, title):
            ax.set_title(title)
            ax.set_xlim(plt_min_x, plt_max_x)
            ax.set_ylim(plt_min_y, plt_max_y)
            ax.set_aspect('equal')

        def plot_row(row_offset, target, res, title_prefix):
            if not res: return

            ax1 = fig.add_subplot(2, 4, row_offset + 1)
            format_ax(ax1, f"{title_prefix} - 1. Initial (AMCL)")
            ax1.scatter(target[:, 0], target[:, 1], c='gray', s=5, alpha=0.5)
            ax1.scatter(source[:, 0], source[:, 1], c='red', s=5, alpha=0.8)
            plot_robot_and_deadspace(ax1, amcl['x'], amcl['y'], amcl['yaw'], self.deadzone_of_bot)

            ax2 = fig.add_subplot(2, 4, row_offset + 2)
            format_ax(ax2, f"{title_prefix} - 2. Corrected (NDT)")
            ax2.scatter(target[:, 0], target[:, 1], c='gray', s=5, alpha=0.5)
            ax2.scatter(res['aligned'][:, 0], res['aligned'][:, 1], c='blue', s=5, alpha=0.8)
            # FIXED: Added self.deadzone_of_bot here
            plot_robot_and_deadspace(ax2, res['tx'], res['ty'], res['tyaw'], self.deadzone_of_bot)

            ax3 = fig.add_subplot(2, 4, row_offset + 3)
            format_ax(ax3, f"{title_prefix} - 3. Freezone Map Changes")
            if len(res['visible']) > 0:
                ax3.scatter(res['visible'][:, 0], res['visible'][:, 1], c='gray', s=5, alpha=0.3, label="Visible Base Map")
            if res['pos_count'] > 0:
                ax3.scatter(res['positive'][:, 0], res['positive'][:, 1], c='blue', s=15, marker='x', label="New (Pos)")
            if res['neg_count'] > 0:
                ax3.scatter(res['negative'][:, 0], res['negative'][:, 1], c='orange', s=15, marker='x', label="Removed (Neg)")
            # FIXED: Added self.deadzone_of_bot here
            plot_robot_and_deadspace(ax3, res['tx'], res['ty'], res['tyaw'], self.deadzone_of_bot)
            ax3.legend(loc='upper right')

            ax4 = fig.add_subplot(2, 4, row_offset + 4)
            format_ax(ax4, f"{title_prefix} - 4. Masks")
            if len(res['occluded']) > 0:
                ax4.scatter(res['occluded'][:, 0], res['occluded'][:, 1], c='black', s=5, alpha=0.15, label="Discarded")
            if len(res['visible']) > 0:
                ax4.scatter(res['visible'][:, 0], res['visible'][:, 1], c='green', s=5, alpha=0.6, label="Kept Freezone")
            # FIXED: Added self.deadzone_of_bot here
            plot_robot_and_deadspace(ax4, res['tx'], res['ty'], res['tyaw'], self.deadzone_of_bot)
            ax4.legend(loc='upper right')

        plot_row(0, global_target, global_res, "GLOBAL")
        plot_row(4, submap_target, submap_res, "SUBMAP")

        diag_text = f"AMCL Score: {amcl['conf']:.4f}  |  Initial Pose: X: {amcl['x']:.4f}m, Y: {amcl['y']:.4f}m, Yaw: {amcl['yaw']:.4f}rad\n\n"
        
        if global_res:
            diag_text += (
                f"[GLOBAL] Alignment Status |  {global_res.get('status', 'N/A')}\n"
                f"[GLOBAL] Corrected Pose   |  X: {global_res['tx']:.4f}m   |   Y: {global_res['ty']:.4f}m   |   Yaw: {global_res['tyaw']:.4f}rad\n"
                f"[GLOBAL] Drift            |  ΔX: {global_res['dx']:+.4f}m   |   ΔY: {global_res['dy']:+.4f}m   |   ΔYaw: {global_res['dyaw']:+.4f}rad\n"
                f"[GLOBAL] Freezone Stats   |  Total Valid Scan Points: {global_res['total_scan_freezone']}   |   Positive Updates: {global_res['pos_count']} ({global_res['pos_ratio']:.1f}%)   |   Negative Updates: {global_res['neg_count']} ({global_res['neg_ratio']:.1f}%)\n\n"
            )
        if submap_res:
            diag_text += (
                f"[SUBMAP] Alignment Status |  {submap_res.get('status', 'N/A')}\n"
                f"[SUBMAP] Corrected Pose   |  X: {submap_res['tx']:.4f}m   |   Y: {submap_res['ty']:.4f}m   |   Yaw: {submap_res['tyaw']:.4f}rad\n"
                f"[SUBMAP] Drift            |  ΔX: {submap_res['dx']:+.4f}m   |   ΔY: {submap_res['dy']:+.4f}m   |   ΔYaw: {submap_res['dyaw']:+.4f}rad\n"
                f"[SUBMAP] Freezone Stats   |  Total Valid Scan Points: {submap_res['total_scan_freezone']}   |   Positive Updates: {submap_res['pos_count']} ({submap_res['pos_ratio']:.1f}%)   |   Negative Updates: {submap_res['neg_count']} ({submap_res['neg_ratio']:.1f}%)"
            )

        if global_res and submap_res:
            diff_x = submap_res['tx'] - global_res['tx']
            diff_y = submap_res['ty'] - global_res['ty']
            raw_dyaw = submap_res['tyaw'] - global_res['tyaw']
            diff_yaw = math.atan2(math.sin(raw_dyaw), math.cos(raw_dyaw)) 
            diag_text += (
                f"\n\n[FINAL ALIGNMENT DIFFERENCE] ΔX: {diff_x:+.4f}m   |   ΔY: {diff_y:+.4f}m   |   ΔYaw: {diff_yaw:+.4f}rad"
            )

        fig.text(0.5, 0.05, diag_text, ha='center', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))
        plt.subplots_adjust(bottom=0.28) 
        
        filename = os.path.join(self.save_dir, f"snapshot_{int(time.time())}.png")
        plt.savefig(filename)
        plt.close(fig)
        self.get_logger().info(f'Diagnostic plot saved to {filename}')

def main(args=None):
    rclpy.init(args=args)
    try: rclpy.spin(NDTVisualizerNode())
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()