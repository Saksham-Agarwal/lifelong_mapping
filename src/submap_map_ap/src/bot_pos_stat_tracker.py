#!/usr/bin/env python3

import rclpy
from rclpy.lifecycle import Node, State, TransitionCallbackReturn
from std_msgs.msg import Float32, Bool
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

class BotPositionTracker(Node):
    def __init__(self):
        super().__init__('bot_position_tracker')
        
        self.declare_parameter('confidence_threshold_lost', 0.25)
        
        # Initialize component variables to None until configured
        self.pub_is_lost = None
        self.sub_confidence = None
        
        # Track state to ensure we only publish ONCE per change.
        # Initializing as None ensures we publish the baseline state on the very first callback.
        self.current_state = None

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring Bot Position Tracker...')
        
        # 1. Define Latching QoS
        # This guarantees that the last published message stays alive on the topic 
        # so new subscribers instantly get the current state.
        latching_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # 2. Publisher using the Latching QoS
        self.pub_is_lost = self.create_lifecycle_publisher(Bool, '/is_bot_lost', latching_qos)
        
        # 3. Standard Subscriber to AMCL confidence
        self.sub_confidence = self.create_subscription(
            Float32,
            '/amcl_confidence',
            self.confidence_callback,
            10
        )
        
        self.get_logger().info('Tracker configured. Monitoring /amcl_confidence...')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Activating Bot Position Tracker.')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Deactivating Bot Position Tracker.')
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Cleaning up Bot Position Tracker.')
        if self.pub_is_lost: self.destroy_publisher(self.pub_is_lost)
        if self.sub_confidence: self.destroy_subscription(self.sub_confidence)
        
        # Reset state so the baseline is published again upon reactivation
        self.current_state = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info('Shutting down Bot Position Tracker.')
        return TransitionCallbackReturn.SUCCESS

    def confidence_callback(self, msg):
        confidence = msg.data
        # Fetch the latest parameter value dynamically
        threshold = self.get_parameter('confidence_threshold_lost').value
        
        # Evaluate if confidence is below the config threshold
        is_lost = confidence < threshold
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