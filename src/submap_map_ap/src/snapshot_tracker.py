#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32, Bool
def get_yaw_from_quaternion(q):
    """
    Convert a ROS geometry_msgs Quaternion to Euler yaw (Z-axis rotation).
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def get_shortest_angle_diff(angle1, angle2):
    """
    Calculate the shortest absolute distance between two angles in radians.
    """
    diff = (angle1 - angle2) % (2 * math.pi)
    if diff > math.pi:
        diff -= 2 * math.pi
    return abs(diff)

class SnapshotTriggerNode(Node):
    def __init__(self):
        super().__init__('snapshot_trigger_node')
        
        # Subscriber for AMCL Pose
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )
        
        # Subscriber for AMCL Confidence
        self.confidence_sub = self.create_subscription(
            Float32,
            '/amcl_confidence',
            self.confidence_callback,
            10
        )
        
        # Publisher for the snapshot trigger
        self.publisher_ = self.create_publisher(Bool, '/take_snapshot', 10)
        
        # State variables
        self.current_confidence = 0.0
        self.last_recorded_pose = None
        
        # Thresholds
        self.distance_threshold = 0.75 # meters
        self.angle_threshold = math.radians(90.0) # Convert 90 degrees to radians
        
        self.get_logger().info('AMCL Distance & Angle Snapshot trigger node started.')

    def confidence_callback(self, msg):
        # Update the latest confidence value
        self.current_confidence = msg.data

    def pose_callback(self, msg):
        current_pose = msg.pose.pose
        
        # 1. Check if the confidence threshold is met
        if self.current_confidence > 0.65:
            
            # 2. If this is the first valid pose, record it and wait for movement
            if self.last_recorded_pose is None:
                self.last_recorded_pose = current_pose
                self.get_logger().info('Initial high-confidence pose recorded. Tracking distance and angle.')
                return
                
            # 3. Calculate 2D Euclidean distance from the last recorded pose
            dx = current_pose.position.x - self.last_recorded_pose.position.x
            dy = current_pose.position.y - self.last_recorded_pose.position.y
            distance = math.hypot(dx, dy)
            
            # 4. Calculate the angular distance (delta yaw)
            current_yaw = get_yaw_from_quaternion(current_pose.orientation)
            last_yaw = get_yaw_from_quaternion(self.last_recorded_pose.orientation)
            angle_diff = get_shortest_angle_diff(current_yaw, last_yaw)
            
            # 5. Trigger if the robot has moved >= 1 meter OR turned >= 120 degrees
            triggered = False
            if distance >= self.distance_threshold:
                self.get_logger().info(f'Moved {distance:.2f} meters. Sending snapshot trigger.')
                triggered = True
            elif angle_diff >= self.angle_threshold:
                self.get_logger().info(f'Turned {math.degrees(angle_diff):.2f} degrees. Sending snapshot trigger.')
                triggered = True
                
            if triggered:
                trigger_msg = Bool()
                trigger_msg.data = True
                self.publisher_.publish(trigger_msg)
                
                # Update the recorded pose to start tracking the next segment
                self.last_recorded_pose = current_pose

def main(args=None):
    rclpy.init(args=args)
    node = SnapshotTriggerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()