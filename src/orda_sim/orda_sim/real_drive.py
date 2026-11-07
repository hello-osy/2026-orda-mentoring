"""실차용: answer.py의 /drive(AckermannDriveStamped)를 LIMO 드라이버의 /cmd_vel(Twist)로 변환한다.

시뮬레이터의 control.py와 같은 변환(angular.z = v * tan(steer) / wheelbase)을 쓰지만
Gazebo에 의존하지 않는다. /drive가 command_timeout 동안 끊기면 정지 명령을 보낸다.
"""
import math
import time


def run(config):
    import rclpy
    from rclpy.node import Node
    from ackermann_msgs.msg import AckermannDriveStamped
    from geometry_msgs.msg import Twist

    robot, topics = config['robot'], config['topics']
    rate = float(config['simulator'].get('command_rate', 30.0))
    timeout = float(config['simulator'].get('command_timeout', 0.4))
    cmd_vel_topic = config.get('real_cmd_vel', '/cmd_vel')

    class DriveBridge(Node):
        def __init__(self):
            super().__init__('orda_real_drive')
            self.speed = self.steering = 0.0
            self.stamp = -1e9
            self.publisher = self.create_publisher(Twist, cmd_vel_topic, 10)
            self.create_subscription(AckermannDriveStamped, topics['drive'], self.on_drive, 10)
            self.create_timer(1.0 / rate, self.tick)
            self.get_logger().info(f"{topics['drive']} -> {cmd_vel_topic} 변환 시작 "
                                   f"(max_speed={robot['max_speed']}, max_steering={robot['max_steering']})")

        def on_drive(self, message):
            speed, steering = message.drive.speed, message.drive.steering_angle
            if not (math.isfinite(speed) and math.isfinite(steering)):
                speed = steering = 0.0
            self.speed = max(-robot['max_speed'], min(robot['max_speed'], speed))
            self.steering = max(-robot['max_steering'], min(robot['max_steering'], steering))
            self.stamp = time.monotonic()

        def tick(self):
            message = Twist()
            if time.monotonic() - self.stamp <= timeout:
                message.linear.x = self.speed
                message.angular.z = self.speed * math.tan(self.steering) / robot['wheelbase']
            self.publisher.publish(message)

        def stop(self):
            self.publisher.publish(Twist())

    rclpy.init()
    node = DriveBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
