"""Hold-to-drive terminal controls; release automatically stops."""
import select
import sys
import termios
import tty
import time
import rclpy
from rclpy.signals import SignalHandlerOptions
from ackermann_msgs.msg import AckermannDriveStamped

def run(config):
    if not sys.stdin.isatty():
        raise SystemExit('teleop은 일반 터미널에서 실행하세요.')
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node('orda_teleop')
    pub = node.create_publisher(AckermannDriveStamped, config['topics']['drive'], 10)
    old = termios.tcgetattr(sys.stdin)
    print('w/s 전진·후진, a/d 전진 좌·우, z/c 후진 좌·우, space 정지, q 종료. 키를 놓으면 정지.')
    cfg = config['teleop']
    keys = {'w': (1, 0), 's': (-1, 0), 'a': (1, 1), 'd': (1, -1),
            'z': (-1, 1), 'c': (-1, -1)}
    try:
        tty.setcbreak(sys.stdin)
        while rclpy.ok():
            key = sys.stdin.read(1) if select.select([sys.stdin], [], [], cfg['release_timeout'])[0] else ' '
            if key in ('q', '\x03'):
                break
            speed, steer = keys.get(key, (0, 0))
            message = AckermannDriveStamped()
            message.drive.speed = speed * cfg['speed']
            message.drive.steering_angle = steer * cfg['steering']
            pub.publish(message)
            rclpy.spin_once(node, timeout_sec=0)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                pub.publish(AckermannDriveStamped())
                time.sleep(0.1)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
