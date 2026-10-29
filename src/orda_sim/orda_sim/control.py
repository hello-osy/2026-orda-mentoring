"""Drive adapter and changing scene; never supplies answers to the controller."""
import json
import math
import random
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock as ClockMessage
from std_msgs.msg import String
from gz.transport13 import Node as GazeboNode
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean

def command_to_twist(speed, steering, wheelbase):
    return speed, speed * math.tan(steering) / wheelbase

class Adapter(Node):
    def __init__(self, config, manifest):
        super().__init__('orda_simulator_adapter')
        self.c = config
        self.manifest = manifest
        self.speed = self.steering = 0.0
        self.last_command = -math.inf
        self.sim_time = 0.0
        self.last_clock = -math.inf
        self.stop_scene = threading.Event()
        self.gz = GazeboNode()
        self.publisher = self.create_publisher(Twist, config['topics']['cmd_vel'], 10)
        self.create_subscription(AckermannDriveStamped, config['topics']['drive'], self.on_drive, 10)
        self.create_subscription(ClockMessage, '/clock', self.on_clock, 10)
        transient = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                               reliability=ReliabilityPolicy.RELIABLE)
        self.status = self.create_publisher(String, '/simulation/scene', transient)
        self.create_timer(1 / config['simulator']['command_rate'], self.tick,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.worker = threading.Thread(target=self.scene_loop, daemon=True)
        self.worker.start()

    def on_drive(self, message):
        speed, steering = message.drive.speed, message.drive.steering_angle
        if not (math.isfinite(speed) and math.isfinite(steering)):
            self.speed = self.steering = 0.0
            return
        robot = self.c['robot']
        self.speed = max(-robot['max_speed'], min(robot['max_speed'], speed))
        self.steering = max(-robot['max_steering'], min(robot['max_steering'], steering))
        self.last_command = time.monotonic()

    def on_clock(self, message):
        self.sim_time = message.clock.sec + message.clock.nanosec * 1e-9
        self.last_clock = time.monotonic()

    def tick(self):
        now = time.monotonic()
        safe = (now - self.last_command < self.c['simulator']['command_timeout']
                and now - self.last_clock < self.c['simulator']['clock_timeout'])
        message = Twist()
        if safe:
            message.linear.x, message.angular.z = command_to_twist(
                self.speed, self.steering, self.c['robot']['wheelbase'])
        self.publisher.publish(message)

    def move(self, item, active):
        p = item['pose']
        request = Pose(name=item['name'])
        request.position.x, request.position.y = float(p[0]), float(p[1])
        request.position.z = float(p[2]) if active else -10.0
        cr, sr = math.cos(p[3]/2), math.sin(p[3]/2)
        cp, sp = math.cos(p[4]/2), math.sin(p[4]/2)
        cy, sy = math.cos(p[5]/2), math.sin(p[5]/2)
        request.orientation.w = cr*cp*cy + sr*sp*sy
        request.orientation.x = sr*cp*cy - cr*sp*sy
        request.orientation.y = cr*sp*cy + sr*cp*sy
        request.orientation.z = cr*cp*sy - sr*sp*cy
        ok, reply = self.gz.request('/world/' + self.c['world']['name'] + '/set_pose',
                                    request, Pose, Boolean, self.c['simulator']['service_timeout_ms'])
        return ok and reply.data

    def scene_loop(self):
        c = self.c['scene']
        rng = random.Random(c['random_seed'])
        selected = c['fork_initial']
        previous = {}
        while not self.stop_scene.wait(0.1):
            durations = c['traffic_seconds']
            phase = self.sim_time % sum(durations)
            traffic = 'traffic_red' if phase < durations[0] else (
                'traffic_green' if phase < durations[0] + durations[1] else 'traffic_yellow')
            # Replace the sign once while the program is already running.
            # Keep it stable while the car is committing to either branch.
            fork_epoch = int(self.sim_time >= c['fork_switch_after'])
            if previous.get('fork_epoch', -1) != fork_epoch:
                selected = rng.choice(['left', 'right']) if c['random_fork'] and fork_epoch else c['fork_initial']
            advertisement = 'advertisement_' + str(int(self.sim_time / c['advertisement_seconds']) % 3)
            target = {'traffic': traffic, 'fork': selected, 'advertisement': advertisement}
            all_ok = True
            for item in self.manifest['signs']:
                group = item['group']
                if group not in target or previous.get(group) == target[group]:
                    continue
                active = item.get('class_name') == target[group] or item['name'] == target[group]
                all_ok = self.move(item, active) and all_ok
            if all_ok:
                previous = dict(target, fork_epoch=fork_epoch)
                status = String(data=json.dumps(dict(target, stamp=self.sim_time)))
                self.status.publish(status)

    def close(self):
        self.stop_scene.set()
        if rclpy.ok():
            self.publisher.publish(Twist())
        self.worker.join(timeout=3)
        self.destroy_node()
