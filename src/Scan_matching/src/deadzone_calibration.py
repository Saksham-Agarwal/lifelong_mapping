import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class DeadzoneCalibrator(Node):
    def __init__(self):
        super().__init__('deadzone_calibrator')
        
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan_reliable',
            self.scan_callback,
            10
        )
        
        self.target_iterations = 10
        self.current_iteration = 0
        self.consistent_inf_indices = None
        
        self.get_logger().info('Deadzone calibrator started. Waiting for /scan_reliable...')

    def scan_callback(self, msg):
        if self.current_iteration >= self.target_iterations:
            return

        # Find infinite or blocked values
        current_infs = set(
            i for i, r in enumerate(msg.ranges) 
            if math.isinf(r) or math.isnan(r) or r > msg.range_max
        )

        if self.consistent_inf_indices is None:
            self.consistent_inf_indices = current_infs
        else:
            self.consistent_inf_indices.intersection_update(current_infs)

        self.current_iteration += 1
        self.get_logger().info(f'Processed scan {self.current_iteration}/{self.target_iterations}')

        if self.current_iteration == self.target_iterations:
            self.report_deadzones(msg)
            raise SystemExit

    def report_deadzones(self, msg):
        self.get_logger().info('--- CALIBRATION COMPLETE ---')
        
        if not self.consistent_inf_indices:
            self.get_logger().info('No consistent deadzones found. The Lidar sees everything!')
            return
            
        sorted_indices = sorted(list(self.consistent_inf_indices))
        
        # Group consecutive indices into distinct ranges
        index_ranges = []
        start_idx = sorted_indices[0]
        prev_idx = sorted_indices[0]
        
        for idx in sorted_indices[1:]:
            if idx == prev_idx + 1:
                prev_idx = idx
            else:
                index_ranges.append((start_idx, prev_idx))
                start_idx = idx
                prev_idx = idx
        # Append the final group
        index_ranges.append((start_idx, prev_idx))

        # Handle 360-degree wrap-around (if deadzone crosses the 0-index boundary)
        total_rays = len(msg.ranges)
        if len(index_ranges) > 1 and index_ranges[-1][1] == total_rays - 1 and index_ranges[0][0] == 0:
            self.get_logger().info('Note: A deadzone wraps around the 0-index boundary.')
            # Merge the last range into the first range
            index_ranges[0] = (index_ranges[-1][0], index_ranges[0][1])
            index_ranges.pop()

        self.get_logger().info(f'Found {len(index_ranges)} distinct deadzone region(s):')
        
        for i, (start, end) in enumerate(index_ranges):
            # Convert indices to radians, then to degrees
            start_angle_rad = msg.angle_min + (start * msg.angle_increment)
            end_angle_rad = msg.angle_min + (end * msg.angle_increment)
            
            start_angle_deg = math.degrees(start_angle_rad)
            end_angle_deg = math.degrees(end_angle_rad)
            
            # Output formatting
            if start == end:
                self.get_logger().info(
                    f'  Region {i+1}: Single Ray at Index {start} | Angle: {start_angle_deg:.2f}°'
                )
            else:
                self.get_logger().info(
                    f'  Region {i+1}: Indices {start} to {end} | '
                    f'Angles: {start_angle_deg:.2f}° to {end_angle_deg:.2f}°'
                )

def main(args=None):
    rclpy.init(args=args)
    node = DeadzoneCalibrator()
    
    try:
        rclpy.spin(node)
    except SystemExit:
        rclpy.logging.get_logger("DeadzoneCalibrator").info("Finished. Shutting down node.")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()