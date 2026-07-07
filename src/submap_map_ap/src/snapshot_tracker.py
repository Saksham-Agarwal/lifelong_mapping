#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Bool

class SnapshotTriggerNode(Node):
    def __init__(self):
        super().__init__('snapshot_trigger_node')
        
        # Subscriber to the bounding boxes topic
        self.subscription = self.create_subscription(
            Detection2DArray,
            '/cluster_positive',
            self.detection_callback,
            10
        )
        
        # Publisher for the snapshot trigger
        self.publisher_ = self.create_publisher(Bool, '/take_snapshot', 10)
        
        # State variable to ensure we only send ONE message per detection event
        # rather than spamming the topic every frame an object is present.
        self.object_currently_detected = False
        
        self.get_logger().info('Snapshot trigger node has been started.')

    def detection_callback(self, msg):
        # Check if there is at least one bounding box in the array
        if len(msg.detections) > 0:
            # Only trigger if we weren't already detecting an object
            if not self.object_currently_detected:
                self.get_logger().info('Positive change object detected! Sending snapshot trigger.')
                
                trigger_msg = Bool()
                trigger_msg.data = True
                self.publisher_.publish(trigger_msg)
                
                # Update state so we don't trigger again until the object leaves
                self.object_currently_detected = True
        else:
            # Reset the state when no objects are in the frame
            if self.object_currently_detected:
                self.get_logger().info('Object left the frame. Resetting trigger state.')
                self.object_currently_detected = False

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