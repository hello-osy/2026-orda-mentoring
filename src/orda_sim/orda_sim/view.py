"""Live camera / lane mask / lidar viewer for the mentoring demonstration."""
import math
import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan

def run(config):
    rclpy.init()
    node = rclpy.create_node('orda_view')
    bridge = CvBridge()
    frames = {}
    size = config['viewer']['panel_size']
    def image(message, name):
        frames[name] = bridge.imgmsg_to_cv2(message, 'bgr8')
    def scan(message):
        canvas = np.full((size, size, 3), 30, np.uint8)
        scale = size / (2 * config['viewer']['lidar_range'])
        for index, distance in enumerate(message.ranges):
            if not math.isfinite(distance) or distance > config['viewer']['lidar_range']:
                continue
            angle = message.angle_min + index * message.angle_increment
            x = int(size/2 - math.sin(angle)*distance*scale)
            y = int(size/2 - math.cos(angle)*distance*scale)
            cv2.circle(canvas, (x, y), 2, (0, 220, 180), -1)
        cv2.arrowedLine(canvas, (size//2, size//2+12), (size//2, size//2-12), (255, 255, 255), 2)
        frames['LiDAR (+x forward)'] = canvas
    node.create_subscription(Image, config['topics']['image'], lambda msg: image(msg, 'Camera'), qos_profile_sensor_data)
    node.create_subscription(Image, config['topics']['debug_image'], lambda msg: image(msg, 'White mask / scan line'), qos_profile_sensor_data)
    node.create_subscription(LaserScan, config['topics']['scan'], scan, qos_profile_sensor_data)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=1/config['viewer']['rate_hz'])
            panels = []
            for name in ['Camera', 'White mask / scan line', 'LiDAR (+x forward)']:
                frame = frames.get(name, np.zeros((size, size, 3), np.uint8))
                frame = cv2.resize(frame, (size, size))
                cv2.rectangle(frame, (0, 0), (size, 30), (25, 25, 25), -1)
                cv2.putText(frame, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
                panels.append(frame)
            cv2.imshow('ORDA sensors - q to close', np.hstack(panels))
            if cv2.waitKey(1) & 255 == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
