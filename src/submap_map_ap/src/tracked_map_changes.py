#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
import numpy as np
import cv2
import math
import json
import os

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from submap_map_ap.msg import ChangeObject, ChangeList

class DynamicChangeTracker(Node):
    def __init__(self):
        super().__init__('dynamic_change_tracker')

        # Subscribers
        self.sub_grid = self.create_subscription(String, '/robot_current_grid', self.grid_cb, 10)
        self.sub_pos_change = self.create_subscription(OccupancyGrid, '/changes/positive_near_neighbour', self.pos_cb, 10)
        self.sub_neg_change = self.create_subscription(OccupancyGrid, '/changes/negative_nearest_neighbour', self.neg_cb, 10)
        
        # Publishers
        self.pub_changes = self.create_publisher(ChangeList, '/tracked_map_changes', 10)
        self.pub_memory = self.create_publisher(ChangeList, '/active_grid_memory', 10) # NEW TOPIC

        # TF2 Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- MEMORY BANK SETUP ---
        self.save_path = os.path.expanduser('~/robot_grid_memory.json')
        self.current_grid = -1
        
        # Initialize 9 empty memory banks (No IDs, just lists of bounding boxes)
        self.global_memory = {
            str(i): {"positive_boxes": [], "negative_boxes": []} for i in range(1, 10)
        }
        
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    self.global_memory = json.load(f)
                self.get_logger().info(f"Loaded spatial memory from {self.save_path}")
            except Exception as e:
                self.get_logger().warn(f"Could not load previous memory: {e}")

        # ACTIVE Spatial Variables for the current grid
        self.active_pos_boxes = [] 
        self.active_neg_boxes = []

        self.latest_pos_grid = None
        self.latest_neg_grid = None
        
        self.observation_radius = 3.0 

        self.timer = self.create_timer(0.1, self.process_state)
        self.get_logger().info("ID-less Spatial Tracker initialized.")

    def grid_cb(self, msg):
        grid_str = msg.data
        if grid_str == "Unknown":
            return

        try:
            new_grid = int(grid_str.split(" ")[1])
        except (IndexError, ValueError):
            return

        if new_grid != self.current_grid:
            grid_key = str(self.current_grid)
            new_key = str(new_grid)

            # SAVE current active state to old grid memory
            if self.current_grid != -1:
                self.global_memory[grid_key]["positive_boxes"] = self.active_pos_boxes
                self.global_memory[grid_key]["negative_boxes"] = self.active_neg_boxes
                
                with open(self.save_path, 'w') as f:
                    json.dump(self.global_memory, f, indent=4)

            # LOAD state for the new grid
            self.current_grid = new_grid
            self.active_pos_boxes = self.global_memory[new_key]["positive_boxes"]
            self.active_neg_boxes = self.global_memory[new_key]["negative_boxes"]
            
            self.get_logger().info(f"Grid {new_key} Loaded: {len(self.active_pos_boxes)} POS, {len(self.active_neg_boxes)} NEG boxes.")

    def pos_cb(self, msg): self.latest_pos_grid = msg
    def neg_cb(self, msg): self.latest_neg_grid = msg

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except TransformException:
            return None, None

    def extract_clusters_from_grid(self, msg):
        if not msg: return []
        w, h, res = msg.info.width, msg.info.height, msg.info.resolution
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
        
        data = np.array(msg.data, dtype=np.int8).reshape((h, w))
        binary_mask = np.where(data > 50, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        clusters = []
        for contour in contours:
            if cv2.contourArea(contour) < 2: continue
            x_pix, y_pix, w_pix, h_pix = cv2.boundingRect(contour)
            min_x, max_x = ox + (x_pix * res), ox + ((x_pix + w_pix) * res)
            min_y, max_y = oy + (y_pix * res), oy + ((y_pix + h_pix) * res)
            
            clusters.append({
                'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y,
                'cx': (min_x + max_x) / 2.0, 'cy': (min_y + max_y) / 2.0
            })
        return clusters

    def check_overlap(self, boxA, boxB):
        intersect_x = (boxA['min_x'] <= boxB['max_x']) and (boxA['max_x'] >= boxB['min_x'])
        intersect_y = (boxA['min_y'] <= boxB['max_y']) and (boxA['max_y'] >= boxB['min_y'])
        return intersect_x and intersect_y

    def update_boxes(self, current_clusters, active_memory_boxes, bot_x, bot_y):
        """Generic spatial logic to update memories based purely on bounding boxes."""
        matched_old_indices = set()
        new_boxes = []

        # 1. Match current visual clusters to our active memory
        for cluster in current_clusters:
            matched = False
            for i, mem_box in enumerate(active_memory_boxes):
                if self.check_overlap(cluster, mem_box):
                    # Expand the memory box to encompass the new cluster
                    active_memory_boxes[i]['min_x'] = min(mem_box['min_x'], cluster['min_x'])
                    active_memory_boxes[i]['max_x'] = max(mem_box['max_x'], cluster['max_x'])
                    active_memory_boxes[i]['min_y'] = min(mem_box['min_y'], cluster['min_y'])
                    active_memory_boxes[i]['max_y'] = max(mem_box['max_y'], cluster['max_y'])
                    active_memory_boxes[i]['cx'] = (active_memory_boxes[i]['min_x'] + active_memory_boxes[i]['max_x']) / 2.0
                    active_memory_boxes[i]['cy'] = (active_memory_boxes[i]['min_y'] + active_memory_boxes[i]['max_y']) / 2.0
                    
                    matched_old_indices.add(i)
                    matched = True
                    break
            
            if not matched:
                new_boxes.append(cluster)

        # Add any completely new boxes we discovered
        active_memory_boxes.extend(new_boxes)

        # 2. Check for missing boxes
        boxes_to_keep = []
        num_old_boxes = len(active_memory_boxes) - len(new_boxes)
        
        for i, box in enumerate(active_memory_boxes):
            # If we just saw it (either matched or brand new), definitely keep it
            if i in matched_old_indices or i >= num_old_boxes:
                boxes_to_keep.append(box)
            else:
                # We DID NOT see it this frame. Is it because it's gone, or because the robot is far away?
                dist_to_bot = math.hypot(box['cx'] - bot_x, box['cy'] - bot_y)
                if dist_to_bot > self.observation_radius:
                    # Too far away to confirm it's gone. Keep it in memory.
                    boxes_to_keep.append(box)
                # Else: It's close, but we don't see a cluster here anymore. It's actually gone. Drop it.

        return boxes_to_keep

    def process_state(self):
        if self.current_grid == -1: return

        bot_x, bot_y = self.get_robot_pose()
        if bot_x is None: return 

        # Extract current visual data
        pos_clusters = self.extract_clusters_from_grid(self.latest_pos_grid)
        neg_clusters = self.extract_clusters_from_grid(self.latest_neg_grid)

        # Pass through our generic spatial overlap logic
        self.active_pos_boxes = self.update_boxes(pos_clusters, self.active_pos_boxes, bot_x, bot_y)
        self.active_neg_boxes = self.update_boxes(neg_clusters, self.active_neg_boxes, bot_x, bot_y)

        # --- PUBLISH DATA ---
        out_msg = ChangeList()
        if self.latest_pos_grid: out_msg.header = self.latest_pos_grid.header
        elif self.latest_neg_grid: out_msg.header = self.latest_neg_grid.header
        
        # We assign temporary IDs purely to satisfy the ChangeObject.msg structure.
        # They are not used for tracking.
        for i, box in enumerate(self.active_neg_boxes):
            out_msg.changes.append(ChangeObject(
                object_id=f"neg_space_{i}", change_type="NEGATIVE",
                min_x=box['min_x'], max_x=box['max_x'], min_y=box['min_y'], max_y=box['max_y']
            ))
            
        for i, box in enumerate(self.active_pos_boxes):
            out_msg.changes.append(ChangeObject(
                object_id=f"pos_space_{i}", change_type="POSITIVE",
                min_x=box['min_x'], max_x=box['max_x'], min_y=box['min_y'], max_y=box['max_y']
            ))
            
        # Publish to both the standard tracker topic and the new memory topic
        self.pub_changes.publish(out_msg)
        self.pub_memory.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DynamicChangeTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Save one final time when shutting down
        if node.current_grid != -1:
            node.global_memory[str(node.current_grid)]["positive_boxes"] = node.active_pos_boxes
            node.global_memory[str(node.current_grid)]["negative_boxes"] = node.active_neg_boxes
            with open(node.save_path, 'w') as f:
                json.dump(node.global_memory, f, indent=4)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()