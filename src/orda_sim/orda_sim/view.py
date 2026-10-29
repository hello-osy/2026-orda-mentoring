"""Three read-only windows. All control values come from the answer node."""
import json
import math
import os
import time
import fcntl
from collections import deque

import cv2
import numpy as np

CYAN = (255, 220, 40)
GREEN = (90, 240, 90)
RED = (70, 90, 255)
WHITE = (235, 235, 235)


def label(image, text, y, color=WHITE, x=10, scale=.48):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def project(forward, left, shape, camera):
    """Same ground-to-camera projection as LaneVision; +left is positive."""
    h, w = shape[:2]
    dx = forward-camera['x']
    pitch = camera['pitch']
    depth = math.cos(pitch)*dx + math.sin(pitch)*camera['z']
    if depth <= .001:
        return None
    up = math.sin(pitch)*dx-math.cos(pitch)*camera['z']
    focal = w/(2*math.tan(math.radians(camera['hfov_deg'])/2))
    x = w/2-focal*(left-camera.get('y', 0.0))/depth
    y = h/2-focal*up/depth
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (int(round(x)), int(round(y)))


def preprocessing(frame, mask, packet, config):
    c = config['answer']['lane']
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array(c['white_min']), np.array(c['white_max']))
    left = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
    bird = np.zeros((c['height_px'], c['width_px']), np.uint8) if mask is None else mask
    right = cv2.cvtColor(bird, cv2.COLOR_GRAY2BGR)
    lane = packet.get('lane', {})
    for window in lane.get('windows', []):
        color = CYAN if window['side'] == 'left' else GREEN
        cv2.rectangle(right, (int(window['x0']), window['y0']),
                      (int(window['x1']), window['y1']), color, 1)
    for side, fit in lane.get('fits', {}).items():
        low, high = lane['fit_spans'][side]
        rows = np.linspace(low, high, 60)
        points = np.column_stack((np.polyval(fit, rows), rows)).astype(np.int32)
        cv2.polylines(right, [points], False, CYAN if side == 'left' else GREEN, 2)
    h, w = bird.shape
    if lane.get('valid'):
        forward = lane.get('forward', c['lookahead'])
        row = int((c['far']-forward)/(c['far']-c['near'])*(h-1))
        reference, detected = w//2, int(lane['center_px'])
        cv2.line(right, (reference, 0), (reference, h-1), CYAN, 1)
        cv2.line(right, (0, row), (w-1, row), (120, 120, 120), 1)
        cv2.arrowedLine(right, (reference, row), (detected, row), RED, 2, tipLength=.2)
        cv2.circle(right, (detected, row), 5, RED, -1)
    label(left, '1. HSV white mask (camera)', 22)
    label(right, '2. Clean BEV + actual search windows', 22)
    label(right, 'CYAN left / GREEN right / RED error', 44)
    if not lane.get('valid'):
        label(right, 'LANE NOT DETECTED', 68, RED)
    # Preserve each panel's aspect ratio with padding.
    size = config['viewer']['preprocess_panel_size']
    return np.hstack([fit_panel(left, size, size), fit_panel(right, size, size)])


def fit_panel(frame, width, height):
    scale = min(width/frame.shape[1], height/frame.shape[0])
    resized = cv2.resize(frame, (max(1, int(frame.shape[1]*scale)), max(1, int(frame.shape[0]*scale))))
    output = np.full((height, width, 3), 22, np.uint8)
    y, x = (height-resized.shape[0])//2, (width-resized.shape[1])//2
    output[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return output


def camera_overlay(frame, packet, config):
    image = frame.copy()
    h, w = image.shape[:2]
    lane, control = packet.get('lane', {}), packet.get('control', {})
    age = packet.get('age', float('inf'))
    if 0 <= age <= config['answer']['detector']['timeout']:
        for obj in packet.get('objects', []):
            box = obj.get('bbox')
            if not box or len(box) != 4:
                continue
            x0, y0, x1, y1 = [int(v*s) for v, s in zip(box, [w, h, w, h])]
            cv2.rectangle(image, (x0, y0), (x1, y1), GREEN, 2)
            label(image, f"{obj['name']} {obj['confidence']:.2f}", max(18, y0-5), GREEN, max(0, x0))
    if lane.get('valid'):
        forward = lane.get('forward', config['answer']['lane']['lookahead'])
        camera = config['robot']['camera']
        ref = project(forward, 0.0, image.shape, camera)
        target = project(forward, lane['error'], image.shape, camera)
        if ref is not None and target is not None:
            cv2.line(image, (ref[0], 0), (ref[0], h-1), CYAN, 1)
            cv2.line(image, (0, ref[1]), (w-1, ref[1]), (130, 130, 130), 1)
            cv2.arrowedLine(image, ref, target, RED, 3, tipLength=.2)
            cv2.circle(image, target, 6, RED, -1)
            label(image, f"ref x={ref[0]}  detected x={target[0]}  dx={target[0]-ref[0]:+d}px", h-88)
            label(image, f"error={lane['error']:+.3f}m (+left)  lookahead={forward:.2f}m", h-66)
    else:
        label(image, 'LANE NOT DETECTED - no measured x', h-88, RED)
    steering = control.get('steering', 0.0)
    base = (w-55, h-12)
    length = config['viewer']['steering_arrow_pixels']
    tip = (int(base[0]-length*math.sin(steering)), int(base[1]-length*math.cos(steering)))
    cv2.line(image, base, (base[0], base[1]-length), CYAN, 1)
    cv2.arrowedLine(image, base, tip, GREEN, 3, tipLength=.18)
    label(image, 'GREEN arrow = published steering (+left)', h-44, GREEN)
    label(image, f"v={control.get('speed', 0):+.3f}m/s  steer={steering:+.3f}rad ({math.degrees(steering):+.1f}deg)", 22)
    label(image, str(control.get('state', 'Waiting for answer telemetry')), 44, CYAN)
    pid = control.get('pid', {})
    label(image, f"Lane PID: P={pid.get('p', 0):+.3f} I={pid.get('i', 0):+.3f} D={pid.get('d', 0):+.3f}", 66)
    gain = control.get('pid_feedback_gain', config['answer']['pid'].get('feedback_gain', .35))
    label(image, f"sum={pid.get('total', 0):+.3f}  gain={gain:.2f}  correction={gain*pid.get('total', 0):+.3f}rad", 88)
    label(image, 'Lane PID is a proposal; mission/lidar may override it.', 110)
    label(image, f"YOLO age={age:.2f}s | image age={packet.get('image_age', 0):.2f}s", 132)
    if not control.get('sensor_ready', False):
        label(image, 'WAIT / SENSOR STOP - published command is zero', 156, RED)
    return image


def lidar_image(scan, drive, config):
    size, distance_limit = config['viewer']['panel_size'], config['viewer']['lidar_range']
    canvas = np.full((size, size, 3), 22, np.uint8)
    scale = size/(2*distance_limit)
    center = (size//2, size//2)
    for radius in np.arange(.5, distance_limit+.001, .5):
        cv2.circle(canvas, center, int(radius*scale), (65, 65, 65), 1)
        ring_y = center[1]-int(radius*scale)+15
        if ring_y > 70:
            label(canvas, f'{radius:.1f}m', ring_y, (130, 130, 130), center[0]+5, .35)
    cv2.line(canvas, (center[0], 0), (center[0], size), (65, 65, 65), 1)
    cv2.line(canvas, (0, center[1]), (size, center[1]), (65, 65, 65), 1)
    robot = config['robot']
    # Plot in rear-axle frame, including lidar mounting offset.
    def point(x, y):
        return (int(size/2-y*scale), int(size/2-x*scale))
    if scan is not None:
        for index, distance in enumerate(scan.ranges):
            if not math.isfinite(distance) or not scan.range_min <= distance <= min(scan.range_max, distance_limit):
                continue
            angle = scan.angle_min+index*scan.angle_increment
            x = robot['lidar']['x']+distance*math.cos(angle)
            y = robot['lidar'].get('y', 0.0)+distance*math.sin(angle)
            cv2.circle(canvas, point(x, y), 2, GREEN, -1)
    x0, x1 = (robot['wheelbase']-robot['length'])/2, (robot['wheelbase']+robot['length'])/2
    cv2.rectangle(canvas, point(x1, robot['width']/2), point(x0, -robot['width']/2), WHITE, 2)
    cv2.circle(canvas, point(robot['lidar']['x'], robot['lidar'].get('y', 0.0)), 4, CYAN, -1)
    steering, speed = drive
    arrow = point(.7*math.cos(steering), .7*math.sin(steering))
    cv2.arrowedLine(canvas, center, arrow, RED, 2, tipLength=.16)
    label(canvas, 'LiDAR: UP=forward LEFT=vehicle left', 22)
    label(canvas, f'v={speed:+.3f}m/s steer={steering:+.3f}rad', 44)
    label(canvas, 'RED: steering direction (not a planned path)', size-15, RED, scale=.4)
    return canvas


def run(config):
    from .common import project_path
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, LaserScan
    from std_msgs.msg import String
    from ackermann_msgs.msg import AckermannDriveStamped

    lock_path = project_path(config, 'src/runtime/viewer.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('시각화 창이 이미 실행 중입니다.', flush=True)
        return
    # OpenCV needs its bundled Qt plugins (Gazebo uses separate system Qt).
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node('orda_view')
    bridge = CvBridge()
    frames, masks = deque(maxlen=config['viewer']['frame_cache']), deque(maxlen=10)
    state = dict(packet={}, scan=None, drive=(0.0, 0.0), image_time=0.0, scan_time=0.0, telemetry_time=0.0)
    def stamp(msg):
        return msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
    def image(msg):
        frames.append((stamp(msg), bridge.imgmsg_to_cv2(msg, 'bgr8')))
        state['image_time'] = time.monotonic()
    def mask(msg):
        masks.append((stamp(msg), bridge.imgmsg_to_cv2(msg, 'mono8')))
    def scan(msg):
        state['scan'] = msg
        state['scan_time'] = time.monotonic()
    def telemetry(msg):
        state['packet'] = json.loads(msg.data)
        state['telemetry_time'] = time.monotonic()
    def drive(msg):
        state['drive'] = (msg.drive.steering_angle, msg.drive.speed)
    topics = config['topics']
    node.create_subscription(Image, topics['image'], image, qos_profile_sensor_data)
    node.create_subscription(Image, topics['debug_image'], mask, qos_profile_sensor_data)
    node.create_subscription(LaserScan, topics['scan'], scan, qos_profile_sensor_data)
    node.create_subscription(String, topics['detections'], telemetry, 10)
    node.create_subscription(AckermannDriveStamped, topics['drive'], drive, 10)
    names = ['ORDA 1 - Preprocess and Sliding Windows', 'ORDA 2 - Camera Control and YOLO', 'ORDA 3 - LiDAR']
    for name, position, dimensions in zip(names, config['viewer']['window_positions'], config['viewer']['window_sizes']):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, *dimensions)
        cv2.moveWindow(name, *position)
    next_draw = 0.0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.005)
            now = time.monotonic()
            if now < next_draw:
                continue
            next_draw = now+1/config['viewer']['rate_hz']
            packet = state['packet']
            render_config = dict(config, answer=packet.get('display', config['viewer']['display']))
            target = packet.get('lane_stamp')
            if target is not None and packet.get('image_stamp', target)-target > render_config['answer']['sensor_timeout']:
                target = None
            matched = next((frame for timestamp, frame in reversed(frames)
                            if target is not None and abs(timestamp-target) < 1e-6), None)
            frame = matched if matched is not None else (frames[-1][1] if frames else np.zeros((480, 640, 3), np.uint8))
            # Never place old lane geometry on an unrelated camera frame.
            shown = packet if matched is not None else dict(packet, lane={})
            bird = next((mask for timestamp, mask in reversed(masks)
                         if target is not None and abs(timestamp-target) < 1e-6), None)
            pre = preprocessing(frame, bird, shown if bird is not None else dict(shown, lane={}), render_config)
            overlay = camera_overlay(frame, shown, render_config)
            lidar = lidar_image(state['scan'], state['drive'], config)
            if now-state['image_time'] > config['viewer']['stale_wall_seconds']:
                label(pre, 'CAMERA STALE / WAITING', 70, RED)
                label(overlay, 'CAMERA STALE / WAITING', 180, RED)
            if now-state['telemetry_time'] > config['viewer']['stale_wall_seconds']:
                label(overlay, 'ANSWER TELEMETRY STALE / WAITING', 202, RED)
            if now-state['scan_time'] > config['viewer']['stale_wall_seconds']:
                label(lidar, 'LIDAR STALE / WAITING', 68, RED)
            for name, picture in zip(names, [pre, overlay, lidar]):
                cv2.imshow(name, picture)
            if cv2.waitKey(1) & 255 == ord('q'):
                break
            if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in names):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        lock.close()
