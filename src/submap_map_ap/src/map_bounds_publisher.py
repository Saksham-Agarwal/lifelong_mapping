import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from nav_msgs.msg import OccupancyGrid

# Import your new custom message
from submap_map_ap.msg import MapBounds 

class MapBoundsPublisher(Node):
    def __init__(self):
        super().__init__('map_bounds_publisher')
        
        # QoS Profile: Reliable and Transient Local (Latching)
        latching_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # Subscriber for the global map
        self.sub_map = self.create_subscription(
            OccupancyGrid, 
            '/map', 
            self.map_callback, 
            latching_qos
        )
        
        # Publisher for the extracted boundaries (also latched!)
        self.pub_bounds = self.create_publisher(MapBounds, '/map_bounds', latching_qos)
        
        self.published = False
        self.get_logger().info('Waiting for /map to extract boundaries...')

    def map_callback(self, msg):
        # We only need to process and publish this once
        if not self.published:
            bounds_msg = MapBounds()
            
            bounds_msg.resolution = msg.info.resolution
            bounds_msg.width = msg.info.width
            bounds_msg.height = msg.info.height
            
            bounds_msg.min_x = msg.info.origin.position.x
            bounds_msg.min_y = msg.info.origin.position.y
            
            width_meters = bounds_msg.width * bounds_msg.resolution
            height_meters = bounds_msg.height * bounds_msg.resolution
            
            bounds_msg.max_x = bounds_msg.min_x + width_meters
            bounds_msg.max_y = bounds_msg.min_y + height_meters
            
            self.pub_bounds.publish(bounds_msg)
            self.published = True
            
            self.get_logger().info('Success! Map boundaries extracted and latched to /map_bounds.')
            self.get_logger().info(f'Bounds: X[{bounds_msg.min_x:.2f} to {bounds_msg.max_x:.2f}], Y[{bounds_msg.min_y:.2f} to {bounds_msg.max_y:.2f}]')

def main(args=None):
    rclpy.init(args=args)
    node = MapBoundsPublisher()
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