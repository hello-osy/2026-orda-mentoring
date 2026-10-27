"""Real-image parking alignment integration test; the only synthetic input is a P trigger.

Run with one YAML containing parking_alignment_test settings. Absolute Gazebo
pose is recorded for evaluation only; it is never passed to LaneVision/Driver.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import yaml
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from gz.transport13 import Node as GzNode
from gz.msgs10.pose_v_pb2 import Pose_V
from orda_sim.common import load_config, project_path
from orda_sim.dataset import _set_pose
from orda_sim.answer import LaneVision, Driver, scan_sectors
from orda_sim.evaluate import footprint, in_rectangle, rectangles_overlap, euler


def run(config):
    test = config['parking_alignment_test']
    output = project_path(config,test['output'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'config_used.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    source = project_path(config,'src/orda_sim/orda_sim/answer.py')
    code_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    rclpy.init()
    node = rclpy.create_node('parking_alignment_integration_test')
    gz = GzNode()
    bridge = CvBridge()
    lock = threading.Lock()
    data = {'pose':None,'clock':None,'image':None,'scan':None,'velocity':None,
            'image_stamp':None,'scan_stamp':None,'odom_stamp':None}
    rows, transitions = [], []
    overlaps = set()
    def pose_callback(message):
        with lock:
            for item in message.pose:
                if item.name == config['robot']['name']:
                    q = item.orientation
                    data['pose'] = [item.position.x,item.position.y,item.position.z,*euler([q.x,q.y,q.z,q.w])]
    def message_time(message):
        return message.header.stamp.sec + message.header.stamp.nanosec*1e-9
    def image_callback(message):
        data['image'] = bridge.imgmsg_to_cv2(message,'bgr8')
        data['image_stamp'] = message_time(message)
    def scan_callback(message):
        data['scan'] = scan_sectors(message.ranges,message.angle_min,message.angle_increment,
                                    message.range_min,message.range_max,config['answer']['lidar'])
        data['scan_stamp'] = message_time(message)
    def odometry_callback(message):
        data['velocity'] = message.twist.twist.linear.x
        data['odom_stamp'] = message_time(message)
    def clock_callback(message):
        data['clock'] = message.clock.sec + message.clock.nanosec*1e-9
    node.create_subscription(Image,config['topics']['image'],image_callback,qos_profile_sensor_data)
    node.create_subscription(LaserScan,config['topics']['scan'],scan_callback,qos_profile_sensor_data)
    node.create_subscription(Odometry,config['topics']['odometry'],odometry_callback,qos_profile_sensor_data)
    node.create_subscription(Clock,'/clock',clock_callback,10)
    gz.subscribe(Pose_V,'/world/'+config['world']['name']+'/dynamic_pose/info',pose_callback)
    publisher = node.create_publisher(AckermannDriveStamped,config['topics']['drive'],10)
    def command(speed,steering):
        message = AckermannDriveStamped()
        message.drive.speed, message.drive.steering_angle = float(speed),float(steering)
        publisher.publish(message)
    def spin_cycle():
        end = time.monotonic()+1/test['rate_hz']
        while time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=min(.01,max(0,end-time.monotonic())))
    def stopped_wait(seconds):
        start = data['clock']
        while data['clock']-start<seconds:
            command(0,0)
            spin_cycle()
    vision = LaneVision(config['answer']['lane'],config['robot']['camera'],config['robot']['front_axle_x'])
    driver = Driver(config)
    driver.stage_index = config['answer']['missions']['sequence'].index('parking')
    last_state = None
    last_picture = -1e9
    start_time = None
    parked_since = None
    report = None
    try:
        start_wall = time.monotonic()
        while any(data[key] is None for key in ['clock','image','scan','velocity','pose']):
            spin_cycle()
            if time.monotonic()-start_wall>30:
                raise RuntimeError('Required camera/lidar/odom/clock/pose unavailable')
        stopped_wait(.5)
        _set_pose(None,config['world']['name'],config['robot']['name'],test['initial_pose'],5000,3)
        stopped_wait(1.)
        start_time = data['clock']
        while data['clock']-start_time<test['max_sim_seconds']:
            now = data['clock']
            fresh = all(0 <= now-data[key] < config['answer']['sensor_timeout']
                        for key in ['image_stamp','scan_stamp','odom_stamp'])
            if not fresh:
                command(0,0)
                spin_cycle()
                continue
            lane = vision.analyze(data['image'],driver.fork_bias)
            # Deliberately synthetic confirmed P. No other mission or GT input.
            detection = [{'name':'parking','confidence':1.,'area':1.,'center':(.5,.5)}]
            speed,steering,state = driver.update(lane,data['scan'],detection,data['image_stamp'],data['velocity'],now)
            command(speed,steering)
            with lock:
                pose = data['pose'][:]
            corners = footprint(pose[0],pose[1],pose[5],config['robot'])
            for index,other in enumerate(config['scene']['parking_obstacles']):
                ox = other[0]-config['robot']['wheelbase']/2*math.cos(other[5])
                oy = other[1]-config['robot']['wheelbase']/2*math.sin(other[5])
                if rectangles_overlap(corners,footprint(ox,oy,other[5],config['robot'])):
                    overlaps.add(index)
            row = {'elapsed':now-start_time,'state':state,'x':pose[0],'y':pose[1],'yaw':pose[5],
                   'lane_valid':lane['valid'],'lane_error':lane['error'],'lane_heading':lane['heading'],
                   'camera_stop_distance':lane['stop_distance'],'tracked_line_distance':driver.line_distance,
                   'speed_command':speed,'steering_command':steering,'odom_speed':data['velocity'],
                   'lidar_front':data['scan']['front'],'lidar_rear':data['scan']['rear']}
            rows.append(row)
            if state != last_state:
                transition = dict(row)
                transitions.append(transition)
                print(json.dumps(transition),flush=True)
                last_state = state
                last_picture = -1e9
            if now-last_picture>=test['snapshot_interval']:
                image = data['image'].copy()
                cv2.putText(image,f'{state} err={lane["error"]:.3f} heading={lane["heading"]:.3f}',
                            (8,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                name = f'{len(rows):05d}_{state.replace(":","_").replace(" ","_")}'
                cv2.imwrite(str(output/(name+'_camera.jpg')),image)
                cv2.imwrite(str(output/(name+'_bev.png')),lane['mask'])
                last_picture = now
            if state.startswith('fault'):
                break
            if state == 'parked':
                if parked_since is None:parked_since=now
                if now-parked_since>=test['parked_hold_seconds']:break
            spin_cycle()
        command(0,0)
        stopped_wait(1.)
        with lock:pose=data['pose'][:]
        corners=footprint(pose[0],pose[1],pose[5],config['robot'])
        bounds=config['evaluation']['parking_bounds']
        margin=min(min(c[0]-bounds[0],bounds[2]-c[0],c[1]-bounds[1],bounds[3]-c[1]) for c in corners)
        inside=all(in_rectangle(c,bounds) for c in corners)
        report={'passed':driver.state=='parked' and inside and not overlaps,
                'controller_state':driver.state,'start_pose':test['initial_pose'],'end_pose':pose,
                'all_four_corners_inside':inside,'minimum_margin_m':margin,'corners':corners,
                'parking_neighbor_overlap_proxy':sorted(overlaps),'transitions':transitions,
                'answer_sha256':code_hash,'elapsed_sim_seconds':data['clock']-start_time,
                'control_inputs':['real RGB image','real lidar','real wheel odometry','simulation clock','synthetic confirmed parking trigger'],
                'ground_truth_usage':'Output evaluation and scenario initialization only, never controller feedback.',
                'scope':'Camera alignment and timed maneuver test; YOLO recognition is intentionally bypassed.'}
        (output/'report.json').write_text(json.dumps(report,indent=2))
        with (output/'trajectory.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        image=cv2.imread(str(project_path(config,config['paths']['generated'])/'track.png'));ppm=400
        def pixel(p):return round(p[0]*ppm),round((6-p[1])*ppm)
        for row in rows:cv2.circle(image,pixel([row['x'],row['y']]),3,(0,180,255),-1)
        cv2.polylines(image,[np.asarray([pixel(c)for c in corners],np.int32)],True,(0,255,0),3)
        cv2.imwrite(str(output/'trajectory.png'),image[int(.6*ppm):int(2.7*ppm),int(5.5*ppm):int(8.3*ppm)])
        print(json.dumps(report,indent=2),flush=True)
    finally:
        if data['clock'] is not None:stopped_wait(.5)
        _set_pose(None,config['world']['name'],config['robot']['name'],config['robot']['spawn'],5000,3)
        if data['clock'] is not None:stopped_wait(1.)
        command(0,0)
        gz.unsubscribe('/world/'+config['world']['name']+'/dynamic_pose/info')
        node.destroy_node()
        rclpy.shutdown()
    return report


if __name__ == '__main__':
    if len(sys.argv)!=2:raise SystemExit('Pass one YAML containing parking_alignment_test settings.')
    run(load_config(sys.argv[1]))
