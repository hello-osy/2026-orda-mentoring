"""Observe a real Gazebo run. Ground truth is never sent to the driving node.

Course completion and measurable stop/speed/parking criteria are checked here.
The approximate reference corridor and footprint overlap are diagnostics, not an
implementation of the competition judges' full 100-point scoring rubric.
"""
from collections import deque
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import threading
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import yaml

from .common import project_path


def in_rectangle(point, rectangle, margin=0.0):
    x, y = point
    left, bottom, right, top = rectangle
    return left - margin <= x <= right + margin and bottom - margin <= y <= top + margin


def footprint(x, y, yaw, robot):
    """Four outer body corners; the model origin is the rear axle projection."""
    center_x = robot['wheelbase'] / 2
    c, s = math.cos(yaw), math.sin(yaw)
    result = []
    for local_x, local_y in [(center_x + robot['length']/2, robot['width']/2),
                             (center_x + robot['length']/2, -robot['width']/2),
                             (center_x - robot['length']/2, -robot['width']/2),
                             (center_x - robot['length']/2, robot['width']/2)]:
        result.append([x + c*local_x - s*local_y, y + s*local_x + c*local_y])
    return result


def point_to_segment(point, first, second):
    p, a, b = np.asarray(point), np.asarray(first), np.asarray(second)
    line = b - a
    fraction = np.clip(np.dot(p-a, line) / max(np.dot(line, line), 1e-12), 0, 1)
    return float(np.linalg.norm(p - a - fraction*line))


def rectangles_overlap(first, second):
    """Separating-axis footprint overlap proxy, not a physics contact sensor."""
    a, b = np.asarray(first), np.asarray(second)
    for polygon in [a, b]:
        for index in range(2):
            edge = polygon[index+1] - polygon[index]
            normal = np.array([-edge[1], edge[0]])
            pa, pb = a @ normal, b @ normal
            if pa.max() <= pb.min() or pb.max() <= pa.min():
                return False
    return True


def euler(quaternion):
    x, y, z, w = quaternion
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y-z*x))))
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return roll, pitch, yaw


class ConeRoute:
    """Northbound body-centre crossings, independent of cone contact checks.

    For a northbound car, right means greater world X and left means smaller X.
    Gate positions come from scene.cones; no coordinates are built into this class.
    """
    def __init__(self, positions, sides, x_limits, margin, robot):
        if len(positions) != len(sides) or any(side not in ['left', 'right'] for side in sides):
            raise ValueError('cone_inside_sides must contain one left/right entry per scene cone')
        if len(x_limits) != 2 or x_limits[0] >= x_limits[1] or margin < 0:
            raise ValueError('Invalid cone gate limits or margin')
        self.positions = positions
        self.sides = sides
        self.x_limits = x_limits
        self.margin = margin
        self.center_offset = robot['wheelbase']/2
        self.previous = None
        self.crossed = set()
        self.wrong = set()
        self.crossings = []

    def update(self, x, y, yaw, now):
        current = [x+self.center_offset*math.cos(yaw),
                   y+self.center_offset*math.sin(yaw), now]
        previous = self.previous
        self.previous = current
        records = []
        if previous is None or current[1] <= previous[1]:
            return records
        for index, (cone_x, cone_y) in enumerate(self.positions):
            if not previous[1] < cone_y <= current[1]:
                continue
            fraction = (cone_y-previous[1])/(current[1]-previous[1])
            crossing_x = previous[0]+fraction*(current[0]-previous[0])
            if not self.x_limits[0] <= crossing_x <= self.x_limits[1]:
                continue
            side = self.sides[index]
            signed_clearance = crossing_x-cone_x if side == 'right' else cone_x-crossing_x
            correct = signed_clearance > self.margin
            record = {'index': index, 'model': 'cone_'+str(index), 'cone': [cone_x, cone_y],
                      'required_side': side, 'body_center': [crossing_x, cone_y],
                      'sim_time': previous[2]+fraction*(now-previous[2]),
                      'signed_side_clearance_m': signed_clearance, 'correct': correct}
            self.crossed.add(index)
            if not correct:
                self.wrong.add(index)
            records.append(record)
        records.sort(key=lambda item:item['sim_time'])
        self.crossings.extend(records)
        return records

    def report(self):
        missing = sorted(set(range(len(self.positions)))-self.crossed)
        return {'passed': not missing and not self.wrong, 'missing_gates': missing,
                'wrong_side_gates': sorted(self.wrong), 'crossings': self.crossings,
                'reference': 'Body centre at rear axle plus half wheelbase; northbound crossings only.'}


class Evaluation:
    """Pure numerical state, separate from ROS/Gazebo so checks are testable."""
    def __init__(self, config):
        self.config = config
        self.c = config['evaluation']
        self.robot = config['robot']
        self.started = None
        self.last_time = None
        self.history = deque()
        self.index = 0
        self.checkpoints = []
        self.events = []
        self.rows = []
        self.anchor = None
        self.last_movement = None
        self.last_progress = None
        self.parked_since = None
        self.bounds_since = None
        self.tilt_since = None
        self.expected_fork = None
        self.fork_observation_stamp = None
        self.wrong_branch = False
        self.cones_disturbed = set()
        self.cone_route = ConeRoute(config['scene']['cones'], self.c['cone_inside_sides'],
                                    self.c['cone_gate_x_limits'], self.c['cone_gate_margin'], self.robot)
        self.overlap_models = set()
        self.lane_samples = 0
        self.outside_lane_samples = 0
        self.slow_speeds = []
        self.stop_checks = {name: {'best_hold': 0.0, 'closest_stopped_error': None,
                                  'hold_since': None, 'crossed': False, 'crossing': None,
                                  'previous_distance': None}
                            for name in ['pedestrian', 'traffic']}
        self.red_encountered = False
        self.green_since = None
        self.outcome = None
        self.last_row = None
        self.last_state = None

    def event(self, kind, **details):
        entry = {'event': kind, 'elapsed': 0.0 if self.started is None else self.last_time-self.started,
                 **details}
        self.events.append(entry)
        return entry

    def finish(self, reason):
        if self.outcome is None:
            self.outcome = reason
            self.event('finished', reason=reason)

    def corridor_inside(self, corner):
        if any(in_rectangle(corner, area) for area in self.c['allowed_areas']):
            return True
        for corridor in self.c['reference_corridors']:
            points = corridor['points']
            limit = corridor['half_width'] + self.c['lane_reference_margin']
            if any(point_to_segment(corner, a, b) <= limit for a, b in zip(points, points[1:])):
                return True
        return False

    def record_stop(self, name, front, now, speed, scene, scene_valid):
        c = self.c[name]
        check = self.stop_checks[name]
        in_region = in_rectangle(front, c['region'])
        distance = sum((c['line_point'][i]-front[i])*c['direction'][i] for i in range(2))
        if not in_region:
            check['hold_since'] = None
            return
        stopped = speed <= self.c['stopped_speed']
        if stopped:
            error = abs(distance)
            if check['closest_stopped_error'] is None or error < check['closest_stopped_error']:
                check['closest_stopped_error'] = error
        near = abs(distance) <= self.c['stop_line_tolerance']
        if stopped and near:
            if check['hold_since'] is None:
                check['hold_since'] = now
            check['best_hold'] = max(check['best_hold'], now-check['hold_since'])
        else:
            check['hold_since'] = None
        if name == 'traffic':
            light = scene.get('traffic') if scene_valid else None
            if 0 <= distance <= c['approach_distance'] and light in ['traffic_red', 'traffic_yellow']:
                self.red_encountered = True
            if near and light == 'traffic_green':
                if self.green_since is None:
                    self.green_since = now
            else:
                self.green_since = None
            if self.green_since is not None and now-self.green_since > c['maximum_green_departure_delay']:
                self.event('traffic_departure_timeout')
                self.finish('traffic_green_departure_timeout')
        previous = check['previous_distance']
        if previous is not None and previous > -self.c['stop_line_tolerance'] and distance <= -self.c['stop_line_tolerance'] and not check['crossed']:
            check['crossed'] = True
            check['crossing'] = {'elapsed': now-self.started, 'speed': speed,
                                 'signal': scene.get('traffic') if scene_valid else None,
                                 'hold_before_crossing': check['best_hold']}
            self.event(name+'_line_crossed', **check['crossing'])
        check['previous_distance'] = distance

    def update(self, pose, now, scene, command, state, cones):
        if self.outcome is not None:
            return
        if self.last_time is not None and now < self.last_time:
            self.finish('simulation_clock_reset')
            return
        if self.last_time is not None and now == self.last_time:
            return
        x, y, z, roll, pitch, yaw = pose
        if self.started is None:
            self.started = self.last_progress = self.last_movement = now
            self.anchor = [x, y]
        self.last_time = now
        if self.history:
            delta = math.dist([x, y], self.history[-1][1:3])
            if delta > self.c['maximum_pose_jump']:
                self.finish('teleport_detected_during_evaluation')
                return
        for crossing in self.cone_route.update(x, y, yaw, now):
            self.event('cone_gate_crossed', elapsed=crossing['sim_time']-self.started, **crossing)
        self.history.append((now, x, y))
        while len(self.history) > 2 and now-self.history[1][0] >= self.c['speed_window_seconds']:
            self.history.popleft()
        oldest = self.history[0]
        speed = math.dist([x, y], oldest[1:3]) / max(now-oldest[0], 1e-9)
        if math.dist([x, y], self.anchor) >= self.c['movement_radius']:
            self.last_movement = now
            self.anchor = [x, y]
        if state != self.last_state:
            self.event('controller_state', state=state)
            self.last_state = state
        if state.startswith('fault') and self.c['fail_on_controller_fault']:
            self.finish('controller_'+state)
        scene_valid = 'stamp' in scene and abs(now-scene['stamp']) <= self.c['scene_stale_seconds']
        if self.expected_fork is None and in_rectangle([x, y], self.c['fork_observation_region']):
            selected = scene.get('fork')
            if scene_valid and selected in ['left', 'right']:
                self.expected_fork = selected
                self.fork_observation_stamp = now
                self.event('fork_instruction_observed', direction=selected)
        if self.index < len(self.c['checkpoints']):
            checkpoint = self.c['checkpoints'][self.index]
            if 'choices' in checkpoint:
                selected = self.expected_fork
                for direction, point in checkpoint['choices'].items():
                    if math.dist([x,y], point) <= checkpoint['radius']:
                        if selected is None:
                            self.finish('fork_instruction_unavailable')
                        elif direction != selected:
                            self.wrong_branch = True
                            self.event('wrong_fork_branch', expected=selected, actual=direction)
                            if self.c['fail_on_wrong_branch']:
                                self.finish('wrong_fork_branch')
                        else:
                            self.checkpoints.append({'name': checkpoint['name'], 'elapsed': now-self.started,
                                                     'point': [x,y], 'branch': direction})
                            self.event('checkpoint', **self.checkpoints[-1])
                            self.index += 1
                            self.last_progress = now
                        break
            elif math.dist([x,y], checkpoint['point']) <= checkpoint['radius']:
                self.checkpoints.append({'name': checkpoint['name'], 'elapsed': now-self.started, 'point': [x,y]})
                self.event('checkpoint', **self.checkpoints[-1])
                self.index += 1
                self.last_progress = now
        front = [x + self.robot['front_axle_x']*math.cos(yaw),
                 y + self.robot['front_axle_x']*math.sin(yaw)]
        # Only score a mission after reaching its part of the ordered course.
        names = [item['name'] for item in self.checkpoints]
        if 'zigzag_exit' in names and 'three_lane_entry' not in names:
            self.record_stop('pedestrian', front, now, speed, scene, scene_valid)
        if 'traffic_approach' in names and 'traffic_exit' not in names:
            self.record_stop('traffic', front, now, speed, scene, scene_valid)
        if in_rectangle(front, self.c['slow_region']):
            self.slow_speeds.append(speed)
        corners = footprint(x, y, yaw, self.robot)
        all_inside = all(in_rectangle(corner, self.c['parking_bounds'], -self.c['footprint_margin']) for corner in corners)
        if self.index == len(self.c['checkpoints']) and all_inside and speed <= self.c['stopped_speed']:
            if self.parked_since is None:
                self.parked_since = now
            if now-self.parked_since >= self.c['parking_hold_seconds']:
                self.finish('course_complete')
        else:
            self.parked_since = None
        within_world = all(in_rectangle(corner, self.c['world_bounds'], self.c['out_of_bounds_margin']) for corner in corners)
        self.bounds_since = None if within_world else self.bounds_since if self.bounds_since is not None else now
        if self.bounds_since is not None and now-self.bounds_since >= self.c['out_of_bounds_seconds']:
            self.finish('outside_track_mat')
        upright = max(abs(roll), abs(pitch)) <= self.c['max_roll_pitch']
        self.tilt_since = None if upright else self.tilt_since if self.tilt_since is not None else now
        if self.tilt_since is not None and now-self.tilt_since >= self.c['tilt_seconds']:
            self.finish('vehicle_tilted')
        lane_exempt = any(in_rectangle([x,y], area) for area in self.c['lane_exempt_regions'])
        outside = not lane_exempt and not all(self.corridor_inside(corner) for corner in corners)
        if not lane_exempt:
            self.lane_samples += 1
            self.outside_lane_samples += int(outside)
        for index, original in enumerate(self.config['scene']['cones']):
            name = 'cone_'+str(index)
            if name in cones and name not in self.cones_disturbed:
                position, tilt = cones[name]
                if math.dist(position, original) > self.c['cone_shift_tolerance'] or tilt > self.c['cone_tilt_radians']:
                    self.cones_disturbed.add(name)
                    self.event('cone_disturbed', model=name, position=position, tilt=tilt)
        for group in ['obstacles', 'parking_obstacles']:
            for index, obstacle in enumerate(self.config['scene'][group]):
                name = group+'_'+str(index)
                # Parked model origins are at body centre, unlike the ego rear axle.
                ox = obstacle[0] - self.robot['wheelbase']/2*math.cos(obstacle[5])
                oy = obstacle[1] - self.robot['wheelbase']/2*math.sin(obstacle[5])
                other = footprint(ox, oy, obstacle[5], self.robot)
                if name not in self.overlap_models and rectangles_overlap(corners, other):
                    self.overlap_models.add(name)
                    self.event('obstacle_footprint_overlap_proxy', model=name)
        valid_hold = False
        for name in ['pedestrian', 'traffic']:
            check = self.stop_checks[name]
            if check['hold_since'] is not None:
                if name == 'pedestrian' and now-check['hold_since'] <= self.c[name]['maximum_departure_delay']:
                    valid_hold = True
                if name == 'traffic' and scene_valid and scene.get('traffic') != 'traffic_green':
                    valid_hold = True
        if now-self.last_movement >= self.c['stuck_seconds'] and not valid_hold:
            self.finish('stationary_timeout')
        if now-self.last_progress >= self.c['progress_timeout_seconds'] and not valid_hold:
            self.finish('checkpoint_progress_timeout')
        if now-self.started >= self.c['max_sim_seconds']:
            self.finish('simulation_time_limit')
        self.last_row = {'elapsed': now-self.started, 'sim_time': now, 'x': x, 'y': y, 'z': z,
                         'yaw': yaw, 'roll': roll, 'pitch': pitch, 'speed_measured': speed,
                         'speed_command': command[0], 'steering_command': command[1], 'state': state,
                         'checkpoint_index': self.index, 'traffic': scene.get('traffic', 'unknown'),
                         'fork': scene.get('fork', 'unknown'), 'outside_reference_corridor': outside,
                         'cone_gates_crossed': len(self.cone_route.crossed),
                         'cone_route_correct': self.cone_route.report()['passed'],
                         'four_corners_inside_bay': all_inside}
        self.rows.append(self.last_row)

    def report(self):
        pedestrian, traffic = self.stop_checks['pedestrian'], self.stop_checks['traffic']
        ped_ok = bool(pedestrian['crossed'] and pedestrian['crossing']['hold_before_crossing'] >= self.c['pedestrian']['minimum_hold'])
        require_traffic_stop = self.config['answer']['missions'].get('traffic_require_stop', True)
        traffic_hold = traffic['best_hold'] >= self.c['traffic']['minimum_hold']
        traffic_cross = bool(traffic['crossed'] and traffic['crossing']['signal'] == 'traffic_green')
        slow_ok = bool(self.slow_speeds and max(self.slow_speeds) <= self.c['slow_max_speed']+self.c['slow_speed_tolerance'])
        complete = self.outcome == 'course_complete'
        criteria = {'ordered_checkpoints_and_parked': complete, 'slow_speed': slow_ok,
                    'pedestrian_stop_3_seconds_at_line': ped_ok,
                    'traffic_crossed_on_green': traffic_cross,
                    'correct_fork': self.expected_fork is not None and not self.wrong_branch,
                    'cones_undisturbed': not self.cones_disturbed,
                    'cone_route_correct': self.cone_route.report()['passed'],
                    'no_obstacle_footprint_overlap_proxy': not self.overlap_models}
        if require_traffic_stop:
            criteria['traffic_stop_at_line'] = traffic_hold
        return {'status': self.outcome or 'running', 'passed': complete and all(criteria.values()),
                'course_complete': complete, 'criteria': criteria,
                'elapsed_sim_seconds': 0 if self.started is None else self.last_time-self.started,
                'checkpoints_passed': self.checkpoints,
                'next_checkpoint': self.c['checkpoints'][self.index]['name'] if self.index < len(self.c['checkpoints']) else 'park_inside_bay',
                'expected_fork': self.expected_fork, 'pedestrian': pedestrian, 'traffic': traffic,
                'traffic_stop_policy': {'required_at_line': require_traffic_stop,
                                        'minimum_hold_seconds': self.c['traffic']['minimum_hold'] if require_traffic_stop else None,
                                        'rule': 'Stop once at the line and cross on green.' if require_traffic_stop else 'Only the crossing signal is scored; crossing must be green.'},
                'red_or_yellow_encountered_on_approach': self.red_encountered,
                'slow_max_measured_speed': max(self.slow_speeds) if self.slow_speeds else None,
                'cones_disturbed': sorted(self.cones_disturbed),
                'cone_route': self.cone_route.report(),
                'obstacle_overlap_proxy': sorted(self.overlap_models),
                'outside_approximate_corridor_samples': self.outside_lane_samples,
                'non_exempt_lane_samples': self.lane_samples,
                'last_pose': self.last_row,
                'limitations': ['Reference corridor is approximate and is not official lane departure scoring.',
                                'Footprint overlap is a geometry proxy; no physics contact score is asserted.',
                                'Facilities and their evaluation positions are practice assumptions in config.yaml.',
                                'This observer never publishes commands or supplies ground truth to answer.py.']}


def plot_trajectory(config, evaluation, output):
    ppm = config['evaluation']['plot_pixels_per_meter']
    track = project_path(config, config['paths']['generated']) / 'track.png'
    image = cv2.imread(str(track))
    if image is None:
        image = np.full((6*ppm,10*ppm,3), 35, np.uint8)
    else:
        image = cv2.resize(image, (10*ppm,6*ppm))
    def pixel(point):
        return round(point[0]*ppm), round((6-point[1])*ppm)
    for index, checkpoint in enumerate(config['evaluation']['checkpoints']):
        points = checkpoint['choices'].values() if 'choices' in checkpoint else [checkpoint['point']]
        for point in points:
            color = (40,210,40) if index < evaluation.index else (100,100,100)
            cv2.circle(image,pixel(point),max(2,round(checkpoint['radius']*ppm)),color,1)
            cv2.putText(image, str(index+1), pixel(point), cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
    for row in evaluation.rows:
        color = (50,50,230) if row['outside_reference_corridor'] else (230,180,20)
        cv2.circle(image,pixel([row['x'],row['y']]),2,color,-1)
    if evaluation.last_row:
        row = evaluation.last_row
        corners = footprint(row['x'],row['y'],row['yaw'],config['robot'])
        cv2.polylines(image,[np.asarray([pixel(c) for c in corners],np.int32)],True,(0,210,255),2)
    cv2.imwrite(str(output/'trajectory.png'),image)


def run(config):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String, Bool
    from ackermann_msgs.msg import AckermannDriveStamped
    from cv_bridge import CvBridge
    from gz.transport13 import Node as TransportNode
    from gz.msgs10.pose_v_pb2 import Pose_V

    settings = config['evaluation']
    stamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')
    output = project_path(config,settings['output']) / ('run_'+stamp)
    output.mkdir(parents=True)
    (output/'config_used.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    evaluation = Evaluation(config)
    state_lock = threading.Lock()
    latest = {'pose': None, 'pose_time': None, 'pose_received': None, 'scene': {},
              'command': [0.0,0.0], 'state': 'waiting for answer', 'image': None,
              'cones': {}, 'collecting': False, 'detections': None}
    detections_file = (output/'detections.jsonl').open('w', buffering=1)
    rclpy.init()
    node = rclpy.create_node('orda_evaluator')
    transport = TransportNode()
    bridge = CvBridge()

    def pose_callback(message):
        now = message.header.stamp.sec + message.header.stamp.nsec*1e-9
        with state_lock:
            for item in message.pose:
                q = item.orientation
                angles = euler([q.x,q.y,q.z,q.w])
                if item.name == config['robot']['name']:
                    latest['pose'] = [item.position.x,item.position.y,item.position.z,*angles]
                    latest['pose_time'] = now
                    latest['pose_received'] = time.monotonic()
                elif item.name.startswith('cone_'):
                    latest['cones'][item.name] = ([item.position.x,item.position.y],max(abs(angles[0]),abs(angles[1])))

    transport.subscribe(Pose_V, '/world/'+config['world']['name']+'/dynamic_pose/info', pose_callback)
    def scene_callback(message):
        try:
            parsed = json.loads(message.data)
        except (ValueError,TypeError):
            return
        with state_lock:
            latest['scene'] = parsed
    def assign(key,value):
        with state_lock:
            latest[key] = value
    def detections_callback(message):
        # Debug evidence only: these objects never enter Evaluation.update().
        try:
            packet = json.loads(message.data)
            if not isinstance(packet, dict):
                raise ValueError('Expected a detection JSON object')
        except (ValueError, TypeError) as error:
            detections_file.write(json.dumps({'parse_error': str(error),
                                               'raw': message.data})+'\n')
            return
        with state_lock:
            received = latest['pose_time']
            stamp = packet.get('stamp')
            numeric_stamp = isinstance(stamp, (int, float)) and math.isfinite(stamp)
            record = dict(packet)
            record.update(received_sim_time=received, stamp=stamp,
                          age=received-stamp if received is not None and numeric_stamp else None,
                          publisher_age=packet.get('age'))
            record.setdefault('objects', [])
            latest['detections'] = record
        detections_file.write(json.dumps(record, ensure_ascii=False)+'\n')
    transient = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                           reliability=ReliabilityPolicy.RELIABLE)
    node.create_subscription(String, '/simulation/scene', scene_callback, transient)
    node.create_subscription(String, config['topics'].get('status','/answer/status'), lambda m:assign('state',m.data),10)
    node.create_subscription(String, config['topics'].get('detections','/answer/detections'), detections_callback,10)
    node.create_subscription(Image, config['topics']['image'],lambda m:assign('image',m),qos_profile_sensor_data)
    node.create_subscription(AckermannDriveStamped,config['topics']['drive'],lambda m:assign('command',[m.drive.speed,m.drive.steering_angle]),10)
    node.create_subscription(Bool,config['topics']['collection_active'],lambda m:assign('collecting',m.data),transient)
    started_wall = time.monotonic()
    last_sample = None
    last_snapshot = -1e9
    event_index = 0
    csv_file = (output/'trajectory.csv').open('w',newline='')
    writer = None
    events_file = (output/'events.jsonl').open('w')
    print('평가 시작:',output,flush=True)
    try:
        while rclpy.ok() and evaluation.outcome is None:
            rclpy.spin_once(node, timeout_sec=1/settings['sample_hz'])
            with state_lock:
                data = dict(latest)
                data['cones'] = dict(latest['cones'])
            wall = time.monotonic()
            if data['pose'] is None or data['collecting']:
                if evaluation.started is not None and data['collecting']:
                    evaluation.finish('collection_started_during_evaluation')
                elif wall-started_wall > settings['initial_wait_wall_seconds']:
                    evaluation.finish('initial_pose_unavailable_or_collection_active')
                continue
            if wall-data['pose_received'] > settings['sensor_timeout_wall_seconds']:
                evaluation.finish('ground_truth_pose_stream_stale')
                continue
            now = data['pose_time']
            if last_sample is not None and now < last_sample:
                evaluation.finish('simulation_clock_reset')
                continue
            if last_sample is not None and now-last_sample < 1/settings['sample_hz']:
                continue
            last_sample = now
            evaluation.update(data['pose'],now,data['scene'],data['command'],data['state'],data['cones'])
            if evaluation.last_row:
                if writer is None:
                    writer = csv.DictWriter(csv_file,fieldnames=list(evaluation.last_row))
                    writer.writeheader()
                writer.writerow(evaluation.last_row)
                csv_file.flush()
            events = evaluation.events[event_index:]
            if events or now-last_snapshot >= settings['snapshots_every_seconds']:
                if data['image'] is not None:
                    image = bridge.imgmsg_to_cv2(data['image'],'bgr8')
                    label = events[-1]['event'] if events else 'interval'
                    filename = f'{len(evaluation.rows):06d}_{label}.jpg'
                    cv2.imwrite(str(output/filename),image)
                    image_stamp = data['image'].header.stamp
                    metadata = {'image': filename, 'sim_time': now,
                                'image_stamp': image_stamp.sec+image_stamp.nanosec*1e-9,
                                'controller_state': data['state'],
                                'detections': data['detections'],
                                'note': 'Latest detector result; image and inference stamps may differ. Debug only.'}
                    if data['detections'] is not None:
                        metadata['detections'] = dict(data['detections'])
                        stamp = metadata['detections']['stamp']
                        if isinstance(stamp, (int,float)) and math.isfinite(stamp):
                            metadata['detections']['age'] = now-stamp
                    (output/Path(filename).with_suffix('.json')).write_text(
                        json.dumps(metadata, ensure_ascii=False, indent=2))
                last_snapshot = now
            for event in events:
                events_file.write(json.dumps(event,ensure_ascii=False)+'\n')
                print(json.dumps(event,ensure_ascii=False),flush=True)
            event_index = len(evaluation.events)
            events_file.flush()
            (output/'status.json').write_text(json.dumps(evaluation.report(),indent=2,ensure_ascii=False))
    except KeyboardInterrupt:
        evaluation.finish('interrupted')
    finally:
        if evaluation.outcome is None:
            evaluation.finish('ros_shutdown')
        for event in evaluation.events[event_index:]:
            events_file.write(json.dumps(event,ensure_ascii=False)+'\n')
        events_file.close()
        detections_file.close()
        csv_file.close()
        report = evaluation.report()
        (output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        plot_trajectory(config,evaluation,output)
        transport.unsubscribe('/world/'+config['world']['name']+'/dynamic_pose/info')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print(json.dumps({'report':str(output/'report.json'),'status':report['status'],'passed':report['passed']},ensure_ascii=False),flush=True)
    if not report['passed']:
        raise SystemExit(1)
