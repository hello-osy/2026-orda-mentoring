"""Check dependencies, SDF, algorithm tests and optional live sensor stream."""
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .common import project_path

def run(config):
    for name in ['rclpy', 'cv_bridge', 'rosbag2_py', 'ackermann_msgs.msg', 'gz.transport13',
                 'cv2', 'torch', 'torchvision', 'ultralytics']:
        importlib.import_module(name)
        print('OK import', name)
    for name in ['gz', 'ros2']:
        if not shutil.which(name):
            raise SystemExit(f'{name} 미설치. Ubuntu24.04 + ROS2 Jazzy + ros-jazzy-ros-gz 필요')
    from .world import generate_world
    folder = project_path(config, config['paths']['generated']) / 'check'
    world = generate_world(config, folder)
    subprocess.run(['gz', 'sdf', '-k', str(world)], check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'src/tests'], check=True)
    if not config['check']['live']:
        print('정적 검사 완료. 실행 중인 센서 검사는 check.live: true 후 다시 실행.')
        return
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, LaserScan
    from nav_msgs.msg import Odometry
    from cv_bridge import CvBridge
    import cv2
    rclpy.init()
    node = rclpy.create_node('orda_sensor_check')
    messages = {}
    counts = {'image': 0, 'scan': 0, 'odometry': 0}
    def receive(name, message):
        messages[name] = message
        counts[name] += 1
    subscriptions = []
    for name, typ in [('image', Image), ('scan', LaserScan), ('odometry', Odometry)]:
        subscriptions.append(node.create_subscription(typ, config['topics'][name],
            lambda msg, key=name: receive(key, msg), qos_profile_sensor_data))
    start = time.monotonic()
    try:
        while time.monotonic() - start < config['check']['timeout'] and min(counts.values()) < 3:
            rclpy.spin_once(node, timeout_sec=.1)
        if min(counts.values()) < 3:
            raise RuntimeError(f'센서 수신 부족: {counts}')
        image = CvBridge().imgmsg_to_cv2(messages['image'], 'bgr8')
        output = project_path(config, config['paths']['logs'])
        output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output / 'camera_check.png'), image)
        report = {'messages_received': counts, 'image_shape': list(image.shape),
                  'scan_samples': len(messages['scan'].ranges), 'passed': True}
        (output / 'sensor_check.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()
