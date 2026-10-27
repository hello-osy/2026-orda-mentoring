import csv,json,math,sys,time,threading
from pathlib import Path
import cv2,numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from gz.transport13 import Node as GzNode
from gz.msgs10.pose_v_pb2 import Pose_V
from orda_sim.common import load_config,project_path
from orda_sim.dataset import _set_pose
from orda_sim.evaluate import footprint,in_rectangle,rectangles_overlap,euler

config=load_config(sys.argv[1]);trial=sys.argv[2]
sequence=config['answer']['parking']['sequence']
if len(sys.argv)>3:
    sequence=json.loads(Path(sys.argv[3]).read_text())
folder=project_path(config,config['evaluation']['output'])/('parking_calibration_'+trial)
folder.mkdir(parents=True,exist_ok=True)
rclpy.init();node=rclpy.create_node('parking_calibration_observer')
gz=GzNode();lock=threading.Lock();state={'pose':None,'clock':None,'image':None}
rows=[];stage_results=[];overlaps=set();bridge=CvBridge()
def on_pose(message):
    with lock:
        for item in message.pose:
            if item.name==config['robot']['name']:
                q=item.orientation
                state['pose']=[item.position.x,item.position.y,item.position.z,*euler([q.x,q.y,q.z,q.w])]
                state['pose_time']=message.header.stamp.sec+message.header.stamp.nsec*1e-9

def receive_clock(msg):state['clock']=msg.clock.sec+msg.clock.nanosec*1e-9

def receive_image(msg):state['image']=msg
node.create_subscription(Clock,'/clock',receive_clock,10)
node.create_subscription(Image,config['topics']['image'],receive_image,qos_profile_sensor_data)
gz.subscribe(Pose_V,'/world/'+config['world']['name']+'/dynamic_pose/info',on_pose)
publisher=node.create_publisher(AckermannDriveStamped,config['topics']['drive'],10)
def command(speed,steering):
    msg=AckermannDriveStamped();msg.drive.speed=float(speed);msg.drive.steering_angle=float(steering);publisher.publish(msg)

def step(name,seconds,speed,steering):
    start=state['clock'];last=0
    while state['clock']-start < seconds:
        command(speed,steering)
        end=time.monotonic()+.05
        while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=min(.01,max(0,end-time.monotonic())))
        with lock:pose=state['pose'][:]
        if state['clock']!=last:
            row={'time':state['clock'],'stage':name,'speed_command':speed,'steering_command':steering,
                 'x':pose[0],'y':pose[1],'z':pose[2],'roll':pose[3],'pitch':pose[4],'yaw':pose[5]}
            rows.append(row);last=state['clock']
            corners=footprint(pose[0],pose[1],pose[5],config['robot'])
            for i,obstacle in enumerate(config['scene']['parking_obstacles']):
                ox=obstacle[0]-config['robot']['wheelbase']/2*math.cos(obstacle[5]);oy=obstacle[1]-config['robot']['wheelbase']/2*math.sin(obstacle[5])
                if rectangles_overlap(corners,footprint(ox,oy,obstacle[5],config['robot'])):overlaps.add(i)
    print(name,state['pose'],flush=True)
    stage_results.append({'stage':name,'pose':state['pose'][:],'sim_time':state['clock']})
try:
    startwall=time.monotonic()
    while state['pose'] is None or state['clock'] is None:
        rclpy.spin_once(node,timeout_sec=.1)
        if time.monotonic()-startwall>15:raise RuntimeError('missing sensors')
    step('stop_before_teleport',.5,0,0)
    _set_pose(None,config['world']['name'],config['robot']['name'],(json.loads(sys.argv[4]) if len(sys.argv)>4 else [7.05,4.84,config['robot']['spawn'][2],0,0,0]),5000,3)
    step('stabilize',1.,0,0)
    initial=state['pose'][:]
    for index,part in enumerate(sequence):step('segment_'+str(index),part['seconds'],part['speed'],part['steering'])
    step('settle',1.,0,0)
    end=state['pose'][:];corners=footprint(end[0],end[1],end[5],config['robot']);bounds=config['evaluation']['parking_bounds']
    margin=min(min(c[0]-bounds[0],bounds[2]-c[0],c[1]-bounds[1],bounds[3]-c[1]) for c in corners)
    report={'trial':trial,'initial_pose':initial,'sequence':sequence,'end_pose':end,'corners':corners,
            'all_four_corners_inside':all(in_rectangle(c,bounds) for c in corners),'minimum_margin_m':margin,
            'parking_neighbor_footprint_overlap_proxy':sorted(overlaps),'stages':stage_results,
            'scope':'Timed maneuver from prescribed normalized pose only. Does not validate YOLO trigger or visual alignment.'}
    (folder/'report.json').write_text(json.dumps(report,indent=2))
    with (folder/'trajectory.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    im=cv2.imread(str(project_path(config,config['paths']['generated'])/'track.png'));ppm=400
    def pixel(p):return (round(p[0]*ppm),round((6-p[1])*ppm))
    for row in rows:
        if row['stage'] not in ['stop_before_teleport','stabilize']:cv2.circle(im,pixel([row['x'],row['y']]),4,(0,180,255),-1)
    cv2.polylines(im,[np.asarray([pixel(c)for c in corners],np.int32)],True,(0,255,0),4)
    crop=im[int((6-5.4)*ppm):int((6-3.3)*ppm),int(5.8*ppm):int(8.1*ppm)]
    cv2.imwrite(str(folder/'trajectory.png'),crop)
    if state['image'] is not None:cv2.imwrite(str(folder/'camera.png'),bridge.imgmsg_to_cv2(state['image'],'bgr8'))
    print(json.dumps(report,indent=2),flush=True)
finally:
    if state['clock'] is not None:step('stop',.5,0,0)
    _set_pose(None,config['world']['name'],config['robot']['name'],config['robot']['spawn'],5000,3)
    if state['clock'] is not None:step('restore_start',1.,0,0)
    command(0,0);node.destroy_node();rclpy.shutdown()
