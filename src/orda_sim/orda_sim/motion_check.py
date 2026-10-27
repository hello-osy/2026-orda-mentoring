"""Short live test of combined drive command, steering direction and watchdog."""
import json
import math
import time
import rclpy
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from rclpy.parameter import Parameter
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from gz.transport13 import Node as GazeboNode
from gz.msgs10.pose_v_pb2 import Pose_V
from .common import project_path

def run(config):
    rclpy.init()
    node = rclpy.create_node('orda_motion_check', parameter_overrides=[Parameter('use_sim_time', value=True)])
    pub = node.create_publisher(AckermannDriveStamped, config['topics']['drive'], 10)
    state = {'odom': None, 'collection': False, 'world_pose': None}
    transport = GazeboNode()
    truth_topic = '/world/' + config['world']['name'] + '/dynamic_pose/info'
    def ground_truth(message):
        for p in message.pose:
            if p.name == config['robot']['name']:
                q = p.orientation
                state['world_pose'] = [p.position.x, p.position.y,
                    math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))]
    transport.subscribe(Pose_V, truth_topic, ground_truth)
    def odometry(message):
        state['odom'] = message
    def collection(message):
        state['collection'] = message.data
    node.create_subscription(Odometry, config['topics']['odometry'], odometry, qos_profile_sensor_data)
    node.create_subscription(Bool, config['topics']['collection_active'], collection,
                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    c = config['motion_check']
    def now():
        return node.get_clock().now().nanoseconds * 1e-9
    def pose():
        p, q = state['odom'].pose.pose.position, state['odom'].pose.pose.orientation
        return [p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y*q.y + q.z*q.z))]
    def phase(seconds, speed, steering, publish=True):
        start, wall = now(), time.monotonic()
        last_publish = 0.0
        while now() - start < seconds:
            rclpy.spin_once(node, timeout_sec=.01)
            if state['collection']:
                raise RuntimeError('데이터 수집 중입니다. record 완료 후 실행하세요.')
            if time.monotonic() - wall > c['wall_timeout']:
                raise RuntimeError('Gazebo clock 진행 없음')
            if publish and time.monotonic() - last_publish >= 1 / c['rate_hz']:
                message = AckermannDriveStamped()
                message.drive.speed, message.drive.steering_angle = float(speed), float(steering)
                pub.publish(message)
                last_publish = time.monotonic()
    try:
        wall = time.monotonic()
        while state['odom'] is None or state['world_pose'] is None:
            rclpy.spin_once(node, timeout_sec=.1)
            if time.monotonic() - wall > c['wall_timeout']:
                raise RuntimeError('Gazebo odom 또는 실제 모델 pose 수신 실패')
        phase(.3, 0, 0)
        before = pose()
        body_before = list(state['world_pose'])
        phase(c['straight_seconds'], c['speed'], 0)
        straight = pose()
        body_straight = list(state['world_pose'])
        phase(c['turn_seconds'], c['speed'], c['steering'])
        turn = pose()
        body_turn = list(state['world_pose'])
        # Stop publishing: it is the simulator watchdog that must stop the car.
        phase(c['watchdog_wait'], 0, 0, publish=False)
        velocity = abs(state['odom'].twist.twist.linear.x)
        distance = math.hypot(straight[0]-before[0], straight[1]-before[1])
        yaw = math.atan2(math.sin(turn[2]-straight[2]), math.cos(turn[2]-straight[2]))
        actual_distance = math.hypot(body_straight[0]-body_before[0], body_straight[1]-body_before[1])
        actual_yaw = math.atan2(math.sin(body_turn[2]-body_straight[2]), math.cos(body_turn[2]-body_straight[2]))
        report = {'straight_distance_m': distance, 'left_turn_yaw_rad': yaw,
                  'actual_model_distance_m': actual_distance, 'actual_model_yaw_rad': actual_yaw,
                  'watchdog_final_speed_mps': velocity,
                  'passed': distance > c['minimum_distance'] and yaw > c['minimum_yaw'] and
                            actual_distance > c['minimum_distance'] and actual_yaw > c['minimum_yaw'] and
                            velocity < c['stop_speed']}
        output = project_path(config, config['paths']['logs']) / 'motion_check.json'
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        if not report['passed']:
            raise RuntimeError('차량 이동 검사 실패: ' + str(output))
    finally:
        if rclpy.ok():
            pub.publish(AckermannDriveStamped())
        transport.unsubscribe(truth_topic)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
