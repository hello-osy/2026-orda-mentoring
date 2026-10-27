import json
import os
import signal
import subprocess
import time
import fcntl
import re
import yaml
from .common import project_path
from .world import generate_world

def bridge_configuration(config):
    topics = config['topics']
    robot = config['robot']['name']
    pairs = [('/clock', '/clock', 'rosgraph_msgs/msg/Clock', 'gz.msgs.Clock'),
             (topics['image'], topics['image'], 'sensor_msgs/msg/Image', 'gz.msgs.Image'),
             (topics['camera_info'], topics['camera_info'], 'sensor_msgs/msg/CameraInfo', 'gz.msgs.CameraInfo'),
             (topics['scan'], topics['scan'], 'sensor_msgs/msg/LaserScan', 'gz.msgs.LaserScan'),
             (f'/model/{robot}/odometry', topics['odometry'], 'nav_msgs/msg/Odometry', 'gz.msgs.Odometry')]
    output = [dict(gz_topic_name=gz, ros_topic_name=ros, ros_type_name=rt,
                   gz_type_name=gt, direction='GZ_TO_ROS') for gz, ros, rt, gt in pairs]
    output.append(dict(gz_topic_name=topics['cmd_vel'], ros_topic_name=topics['cmd_vel'],
                       ros_type_name='geometry_msgs/msg/Twist', gz_type_name='gz.msgs.Twist',
                       direction='ROS_TO_GZ'))
    return output

def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

def run(config):
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from .control import Adapter
    partition = re.sub(r'[^A-Za-z0-9_-]', '_', config['environment']['gz_partition'])
    lock_path = project_path(config, 'src/runtime/simulator-' + partition + '.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('이 프로젝트의 시뮬레이터가 이미 실행 중입니다.')
    generated = project_path(config, config['paths']['generated'])
    generated.mkdir(parents=True, exist_ok=True)
    world = generate_world(config, generated)
    bridge_file = generated / 'bridge.yaml'
    bridge_file.write_text(yaml.safe_dump(bridge_configuration(config)))
    log_dir = project_path(config, config['paths']['logs'])
    log_dir.mkdir(parents=True, exist_ok=True)
    command = ['gz', 'sim', '-r', '-v', str(config['simulator']['verbosity'])]
    if config['simulator']['headless']:
        command += ['-s']
    if config['simulator']['egl_headless']:
        command += ['--headless-rendering']
    command.append(str(world))
    processes = []
    files = []
    node = None
    try:
        for name, args in [('gazebo', command), ('bridge', ['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                            '--ros-args', '-p', 'config_file:=' + str(bridge_file)])]:
            log = (log_dir / (name + '.log')).open('w')
            files.append(log)
            env = os.environ.copy()
            if name == 'gazebo':
                env.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
                env.pop('QT_QPA_FONTDIR', None)
                env['QT_QPA_PLATFORM'] = 'xcb'
                env.pop('WAYLAND_DISPLAY', None)
                if config['environment']['egl_software_only']:
                    shim = project_path(config, 'src/runtime/libegl_software.so')
                    if not shim.exists():
                        raise RuntimeError('EGL 보정 라이브러리 없음. setup.sh를 실행하세요.')
                    env['LD_PRELOAD'] = str(shim) + (':' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
            processes.append(subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = Adapter(config, json.loads((generated / 'manifest.json').read_text()))
        print(f'Gazebo 시작: {world}\n로그: {log_dir}\n종료: Ctrl+C', flush=True)
        while rclpy.ok():
            for name, process in zip(['Gazebo', 'ROS bridge'], processes):
                if process.poll() is not None:
                    raise RuntimeError(f'{name} 종료({process.returncode}). {log_dir} 로그 확인')
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node.close()
            if rclpy.ok():
                rclpy.shutdown()
        finally:
            for process in reversed(processes):
                stop_process(process)
            for log in files:
                log.close()
