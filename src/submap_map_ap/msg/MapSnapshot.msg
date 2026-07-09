# AMCL Initial Guess
float64 amcl_x
float64 amcl_y
float64 amcl_yaw

# AMCL Confidence Score
float32 amcl_confidence

# Extracted Point Clouds
geometry_msgs/Point[] global_points
geometry_msgs/Point[] local_points
geometry_msgs/Point[] submap_points  