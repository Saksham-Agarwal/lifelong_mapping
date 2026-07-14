#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

class BotPositionTracker(Node):
    def __init__(self):
        super().__init__('bot_position_tracker')
        
        # 1. Define Latching QoS
        # This guarantees that the last published message stays alive on the topic 
        # so new subscribers instantly get the current state.
        latching_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # 2. Publisher using the Latching QoS
        self.pub_is_lost = self.create_publisher(Bool, '/is_bot_lost', latching_qos)
        
        # 3. Standard Subscriber to AMCL confidence
        self.sub_confidence = self.create_subscription(
            Float32,
            '/amcl_confidence',
            self.confidence_callback,
            10
        )
        
        # Track state to ensure we only publish ONCE per change.
        # Initializing as None ensures we publish the baseline state on the very first callback.
        self.current_state = None
        
        self.get_logger().info('Tracker initialized. Monitoring /amcl_confidence...')

    def confidence_callback(self, msg):
        confidence = msg.data
        
        # Evaluate if confidence is below threshold
        is_lost = confidence < 0.4
        
        # Only trigger publish if the state has actually changed (or on first run)
        if self.current_state != is_lost:
            self.current_state = is_lost
            
            # Publish the new state (which gets latched)
            lost_msg = Bool()
            lost_msg.data = is_lost
            self.pub_is_lost.publish(lost_msg)
            
            # Log the change to the terminal
            if is_lost:
                self.get_logger().warn(f'STATE FLIP: Bot is LOST! Confidence dropped to {confidence:.2f}')
            else:
                self.get_logger().info(f'STATE FLIP: Bot recovered. Confidence restored to {confidence:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = BotPositionTracker()
    
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