#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String, Bool, Float32
from visualization_msgs.msg import Marker, MarkerArray
import math
import json
import os
import time

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from submap_map_ap.msg import ChangeObject, ChangeList

class DynamicChangeTracker(Node):
    def __init__(self):
        super().__init__('dynamic_change_tracker')

        # 1. Subscribers
        self.sub_grid = self.create_subscription(String, '/robot_current_grid', self.grid_cb, 10)
        self.sub_pos_change = self.create_subscription(Detection2DArray, '/cluster_positive', self.pos_cb, 10)
        self.sub_neg_change = self.create_subscription(Detection2DArray, '/cluster_negative', self.neg_cb, 10)
        self.sub_loc_fail = self.create_subscription(Bool, '/localisation_failed', self.loc_fail_cb, 10)
        
        # --- NEW: AMCL Confidence Subscriber ---
        self.sub_amcl = self.create_subscription(Float32, '/amcl_confidence', self.amcl_cb, 10)
        
        # 2. Publishers
        self.pub_changes = self.create_publisher(ChangeList, '/tracked_map_changes', 10)
        self.pub_memory = self.create_publisher(ChangeList, '/active_grid_memory', 10) 
        self.pub_change_markers = self.create_publisher(MarkerArray, '/tracked_changes_markers', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 3. MEMORY BANK SETUP
        self.save_path = os.path.expanduser('~/robot_grid_memory.json')
        self.current_grid = -1
        
        self.global_memory = {
            str(i): {"explored": False, "timestamp": None, "positive_boxes": [], "negative_boxes": []} 
            for i in range(1, 10)
        }
        
        # Load whatever good state we had on startup
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    self.global_memory = json.load(f)
            except Exception:
                pass
        self.save_to_disk()

        # ACTIVE RAM Variables
        self.active_pos_boxes = [] 
        self.active_neg_boxes = []
        self.candidate_pos_boxes = []
        self.candidate_neg_boxes = []

        self.latest_pos_msg = None
        self.latest_neg_msg = None
        
        self.observation_radius = 1.5 

        # --- NEW: AMCL Safety Variables ---
        self.current_amcl_confidence = 1
        self.tracking_threshold = 0.7
        self.lost_threshold = 0.4
        self.is_lost_lockdown = False

        self.timer = self.create_timer(0.1, self.process_state)
        self.sync_timer = self.create_timer(1.0, self.live_sync_to_disk)
        
        self.get_logger().info("AMCL-Protected Fresh-Scan Tracker initialized...")

    def save_to_disk(self):
        try:
            with open(self.save_path, 'w') as f:
                json.dump(self.global_memory, f, indent=4)
        except Exception as e:
            self.get_logger().error(f"Failed to save JSON: {e}")

    def live_sync_to_disk(self):
        # Don't overwrite the hard drive if the robot is currently lost and hallucinating!
        if self.current_grid != -1 and not self.is_lost_lockdown:
            grid_key = str(self.current_grid)
            self.global_memory[grid_key]["positive_boxes"] = self.active_pos_boxes
            self.global_memory[grid_key]["negative_boxes"] = self.active_neg_boxes
            self.save_to_disk()

    def amcl_cb(self, msg):
        self.current_amcl_confidence = msg.data
        
        # The ultimate safety net: If confidence drops below 40, initiate Lockdown
        if self.current_amcl_confidence < self.lost_threshold and not self.is_lost_lockdown:
            self.get_logger().error(f"AMCL dropped below {self.lost_threshold}! Robot is LOST. Trashing RAM and restoring safe JSON.")
            self.is_lost_lockdown = True
            
            # Throw away any hallucinated changes in RAM
            self.active_pos_boxes = []
            self.active_neg_boxes = []
            self.candidate_pos_boxes = []
            self.candidate_neg_boxes = []
            
            # Reload the last known good state from the Hard Drive
            if os.path.exists(self.save_path):
                with open(self.save_path, 'r') as f:
                    self.global_memory = json.load(f)
                    
        elif self.current_amcl_confidence >= self.lost_threshold and self.is_lost_lockdown:
            self.get_logger().info("AMCL recovered! Lifting lockdown.")
            self.is_lost_lockdown = False

    def loc_fail_cb(self, msg):
        if msg.data == True and self.current_grid != -1:
            self.get_logger().warn(f"Localisation Failed Topic triggered! Emergency lockdown...")
            # We treat a direct localization failure the exact same as dropping below 40%
            self.is_lost_lockdown = True 

    def grid_cb(self, msg):
        grid_str = msg.data
        if grid_str == "Unknown": return

        try:
            new_grid = int(grid_str.split(" ")[1])
        except (IndexError, ValueError):
            return

        if new_grid != self.current_grid:
            grid_key = str(new_grid)
            
            # 1. We are entering a new grid! 
            self.current_grid = new_grid
            
            # 2. THE NEW RULE: Wipe the grid completely clean upon entry.
            self.get_logger().info(f"Entering Grid {new_grid}. Wiping previous memory to start a fresh scan.")
            self.global_memory[grid_key]["positive_boxes"] = []
            self.global_memory[grid_key]["negative_boxes"] = []
            
            # 3. Wipe the active RAM
            self.active_pos_boxes = []
            self.active_neg_boxes = []
            self.candidate_pos_boxes = []
            self.candidate_neg_boxes = []

            self.global_memory[grid_key]["explored"] = True
            self.global_memory[grid_key]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.save_to_disk()

    def pos_cb(self, msg): self.latest_pos_msg = msg
    def neg_cb(self, msg): self.latest_neg_msg = msg

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except TransformException:
            return None, None

    def extract_boxes_from_detections(self, msg, is_positive):
        if not msg: return []
        boxes = []
        for det in msg.detections:
            w = det.bbox.size_x
            h = det.bbox.size_y
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            theta = det.bbox.center.theta  
            
            if not is_positive:
                if w < 0.15 or h < 0.15: continue
                if (w * h) < 0.1: continue

            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            hw = w / 2.0
            hh = h / 2.0
            corners = [ (hw, hh), (hw, -hh), (-hw, hh), (-hw, -hh) ]
            
            x_coords = []
            y_coords = []
            for rx, ry in corners:
                x_coords.append(cx + (rx * cos_t - ry * sin_t))
                y_coords.append(cy + (rx * sin_t + ry * cos_t))
                
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            boxes.append({
                'min_x': min_x, 'max_x': max_x,
                'min_y': min_y, 'max_y': max_y,
                'cx': (min_x + max_x) / 2.0, 
                'cy': (min_y + max_y) / 2.0
            })
        return boxes

    def check_overlap(self, incoming, memory_box, is_positive):
        if not is_positive:
            # NEGATIVE SPACE: Standard edge-touching overlap is fine for shadows
            intersect_x = (incoming['min_x'] <= memory_box['max_x']) and (incoming['max_x'] >= memory_box['min_x'])
            intersect_y = (incoming['min_y'] <= memory_box['max_y']) and (incoming['max_y'] >= memory_box['min_y'])
            return intersect_x and intersect_y
        else:
            # POSITIVE SPACE: The center of the new incoming box MUST fall inside the saved memory box.
            # We add a 0.3m (30cm) pad to allow for slight LiDAR jitter.
            pad = 0.3 
            in_x = (memory_box['min_x'] - pad <= incoming['cx'] <= memory_box['max_x'] + pad)
            in_y = (memory_box['min_y'] - pad <= incoming['cy'] <= memory_box['max_y'] + pad)
            return in_x and in_y
        
    def update_spatial_memory_with_persistence(self, incoming_boxes, active_boxes, candidate_boxes, bot_x, bot_y, is_positive):
        matched_active = set()
        matched_candidate = set()
        new_candidates = []
        required_frames = 3 if is_positive else 8

        # 1. Process incoming live boxes
        for incoming in incoming_boxes:
            
            # Step A: Match Active Memory
            matched_to_active = False
            for i, active in enumerate(active_boxes):
                # Pass the 'is_positive' flag to our new strict overlap check!
                if self.check_overlap(incoming, active, is_positive):
                    if is_positive:
                        # RESTORED: Your max() logic to prevent shrinking!
                        active['min_x'] = min(active['min_x'], incoming['min_x'])
                        active['max_x'] = max(active['max_x'], incoming['max_x'])
                        active['min_y'] = min(active['min_y'], incoming['min_y'])
                        active['max_y'] = max(active['max_y'], incoming['max_y'])
                        active['cx'] = (active['min_x'] + active['max_x']) / 2.0
                        active['cy'] = (active['min_y'] + active['max_y']) / 2.0
                    
                    matched_active.add(i)
                    matched_to_active = True
                    break 
            
            if matched_to_active: continue

            # Step B: Match Candidate
            matched_to_candidate = False
            for i, candidate in enumerate(candidate_boxes):
                # Pass the 'is_positive' flag here too!
                if self.check_overlap(incoming, candidate, is_positive):
                    if is_positive:
                        # RESTORED: Candidate max() logic
                        candidate['min_x'] = min(candidate['min_x'], incoming['min_x'])
                        candidate['max_x'] = max(candidate['max_x'], incoming['max_x'])
                        candidate['min_y'] = min(candidate['min_y'], incoming['min_y'])
                        candidate['max_y'] = max(candidate['max_y'], incoming['max_y'])
                        candidate['cx'] = (candidate['min_x'] + candidate['max_x']) / 2.0
                        candidate['cy'] = (candidate['min_y'] + candidate['max_y']) / 2.0
                        
                    candidate['hits'] += 1
                    matched_candidate.add(i)
                    matched_to_candidate = True
                    break
                    
            if matched_to_candidate: continue

            incoming['hits'] = 1
            new_candidates.append(incoming)

        # 2. Handle Disappearances
        next_active = []
        for i, active in enumerate(active_boxes):
            if i in matched_active:
                next_active.append(active)
            else:
                dist_to_bot = math.hypot(active['cx'] - bot_x, active['cy'] - bot_y)
                if dist_to_bot > self.observation_radius:
                    next_active.append(active)

        # 3. Handle Graduations
        next_candidates = []
        for i, candidate in enumerate(candidate_boxes):
            if i in matched_candidate:
                if candidate['hits'] >= required_frames:
                    del candidate['hits']
                    next_active.append(candidate)
                else:
                    next_candidates.append(candidate)

        next_candidates.extend(new_candidates)
        return next_active, next_candidates

    def process_state(self):
        if self.current_grid == -1: return

        # --- THE SAFETY GUARD ---
        # If the robot is in total lockdown (<40%), do absolutely nothing. 
        if self.is_lost_lockdown:
            return

        # If confidence is sketchy (<70%), skip updating the boxes but STILL publish the markers 
        # so you can see what the robot remembered before it got confused.
        if self.current_amcl_confidence < self.tracking_threshold:
            self.get_logger().info(f"AMCL {self.current_amcl_confidence}% is too low. Pausing tracking.", throttle_duration_sec=2.0)
        else:
            # We are healthy! Run the tracking math.
            bot_x, bot_y = self.get_robot_pose()
            if bot_x is not None: 
                incoming_pos = self.extract_boxes_from_detections(self.latest_pos_msg, is_positive=True)
                incoming_neg = self.extract_boxes_from_detections(self.latest_neg_msg, is_positive=False)

                # Update the active RAM specifically for the grid we are standing in
                self.active_pos_boxes, self.candidate_pos_boxes = self.update_spatial_memory_with_persistence(
                    incoming_pos, self.active_pos_boxes, self.candidate_pos_boxes, bot_x, bot_y, is_positive=True)
                    
                self.active_neg_boxes, self.candidate_neg_boxes = self.update_spatial_memory_with_persistence(
                    incoming_neg, self.active_neg_boxes, self.candidate_neg_boxes, bot_x, bot_y, is_positive=False)

        # Extract ALL boxes from ALL grids for the live RViz publishing
        all_pos_boxes = []
        all_neg_boxes = []
        for grid_data in self.global_memory.values():
            all_pos_boxes.extend(grid_data["positive_boxes"])
            all_neg_boxes.extend(grid_data["negative_boxes"])

        # --- PUBLISH DATA ---
        out_msg = ChangeList()
        if self.latest_pos_msg: out_msg.header = self.latest_pos_msg.header
        elif self.latest_neg_msg: out_msg.header = self.latest_neg_msg.header
        
        for i, box in enumerate(all_neg_boxes):
            out_msg.changes.append(ChangeObject(
                object_id=f"neg_space_{i}", change_type="NEGATIVE",
                min_x=box['min_x'], max_x=box['max_x'], min_y=box['min_y'], max_y=box['max_y']
            ))
            
        for i, box in enumerate(all_pos_boxes):
            out_msg.changes.append(ChangeObject(
                object_id=f"pos_space_{i}", change_type="POSITIVE",
                min_x=box['min_x'], max_x=box['max_x'], min_y=box['min_y'], max_y=box['max_y']
            ))
            
        self.pub_changes.publish(out_msg)
        self.pub_memory.publish(out_msg)

        # --- RViz VISUALIZATION ---
        marker_array = MarkerArray()
        
        delete_all = Marker()
        delete_all.action = 3 
        marker_array.markers.append(delete_all)
        
        marker_id = 0
        
        for box in all_pos_boxes:
            m = Marker()
            if out_msg.header.stamp: m.header = out_msg.header
            m.ns = "tracked_changes"
            m.id = marker_id
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = box['cx']
            m.pose.position.y = box['cy']
            m.pose.position.z = 0.2 
            m.pose.orientation.w = 1.0
            m.scale.x = box['max_x'] - box['min_x']
            m.scale.y = box['max_y'] - box['min_y']
            m.scale.z = 0.4
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.6  
            marker_array.markers.append(m)
            marker_id += 1
            
        for box in all_neg_boxes:
            m = Marker()
            if out_msg.header.stamp: m.header = out_msg.header
            m.ns = "tracked_changes"
            m.id = marker_id
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = box['cx']
            m.pose.position.y = box['cy']
            m.pose.position.z = 0.2
            m.pose.orientation.w = 1.0
            m.scale.x = box['max_x'] - box['min_x']
            m.scale.y = box['max_y'] - box['min_y']
            m.scale.z = 0.4
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.6 
            marker_array.markers.append(m)
            marker_id += 1
            
        self.pub_change_markers.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = DynamicChangeTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.live_sync_to_disk()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()