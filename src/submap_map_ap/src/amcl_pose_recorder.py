#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Bool, Float64

class PoseRecoveryNode(Node):
    def __init__(self):
        super().__init__('pose_recovery_node')

        # Subscriptions
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )
        self.confidence_sub = self.create_subscription(
            Float64,
            '/amcl_confidence',
            self.confidence_callback,
            10
        )
        self.lost_sub = self.create_subscription(
            Bool,
            '/is_bot_lost',
            self.lost_callback,
            10
        )

        # Publisher
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        # State Variables
        self.last_confident_pose = None
        self.previous_pose = None
        self.current_confidence = 1.0
        self.has_teleported = False
        self.recovery_in_progress = False
        
        # Configuration parameters
        self.confidence_threshold = 0.5
        self.jump_distance_threshold = 2.5  # Max meters bot can move between AMCL updates without it being a "teleport"
        self.recovery_delay_seconds = 2.5   # 2-3 seconds delay

        self.get_logger().info("Pose Recovery Node has been started.")

    def pose_callback(self, msg):
        # Check for teleportation by calculating distance from the last known pose
        if self.previous_pose is not None:
            dx = msg.pose.pose.position.x - self.previous_pose.pose.pose.position.x
            dy = msg.pose.pose.position.y - self.previous_pose.pose.pose.position.y
            distance = math.sqrt(dx**2 + dy**2)

            if distance > self.jump_distance_threshold:
                self.get_logger().warn(f"Robot teleportation detected! Jumped {distance:.2f}m")
                self.has_teleported = True

        self.previous_pose = msg

        # Update the last confident pose if our confidence is good and we haven't jumped
        if self.current_confidence >= self.confidence_threshold:
            if not self.has_teleported:
                self.last_confident_pose = msg
            else:
                # If we have high confidence again, we can reset the teleport flag 
                # because AMCL has successfully localized at the new location
                self.has_teleported = False
                self.last_confident_pose = msg

    def confidence_callback(self, msg):
        self.current_confidence = msg.data

    def lost_callback(self, msg):
        # We only care if the bot is actually lost and we aren't already recovering
        if msg.data is True and not self.recovery_in_progress:
            
            # Condition 1: Confidence is below 0.5
            # Condition 2: The bot didn't teleport/jump
            # Condition 3: We actually have a recorded confident pose to send
            if self.current_confidence < self.confidence_threshold:
                if not self.has_teleported:
                    if self.last_confident_pose is not None:
                        self.get_logger().info(
                            "Lost condition met (low confidence, no teleport). "
                            f"Initiating recovery in {self.recovery_delay_seconds} seconds..."
                        )
                        self.recovery_in_progress = True
                        
                        # Trigger the delayed publisher
                        self.timer = self.create_timer(
                            self.recovery_delay_seconds, 
                            self.publish_recovery_pose
                        )
                    else:
                        self.get_logger().warn("Bot is lost, but we have no prior confident pose to restore to.")
                else:
                    self.get_logger().warn("Bot is lost, but teleportation was detected earlier. Aborting recovery.")
            else:
                self.get_logger().info("Bot flagged as lost, but AMCL confidence is still high. Ignoring.")

    def publish_recovery_pose(self):
        # Cancel the timer so it only runs once (one-shot timer)
        self.timer.cancel()

        if self.last_confident_pose is not None:
            self.get_logger().info("Publishing last confident pose to /initialpose")
            
            # Update the timestamp before sending
            self.last_confident_pose.header.stamp = self.get_clock().now().to_msg()
            self.initial_pose_pub.publish(self.last_confident_pose)
        
        # Reset recovery states
        self.recovery_in_progress = False
        self.has_teleported = False

def main(args=None):
    rclpy.init(args=args)
    node = PoseRecoveryNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()