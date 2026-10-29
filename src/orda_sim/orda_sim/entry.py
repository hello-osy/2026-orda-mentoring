"""Internal dispatch; users run scripts/*.sh with one YAML argument."""
import sys
import os
from .common import load_config

def main():
    action, filename = sys.argv[1:]
    config = load_config(filename)
    paths = config['environment']['gazebo_config_paths']
    os.environ['GZ_CONFIG_PATH'] = ':'.join(paths + [os.environ.get('GZ_CONFIG_PATH', '')])
    if action == 'simulator':
        from .simulator import run
        run(config)
    elif action == 'answer':
        from .answer import main as answer_main
        answer_main(filename)
    elif action == 'view':
        from .view import run
        run(config)
    elif action == 'ready':
        import rclpy
        import cv_bridge
        import ackermann_msgs.msg
        import gz.transport13
        import ultralytics
        print('설치 완료. simulator.sh와 answer.sh를 각각 실행하세요.')
    else:
        raise SystemExit(f'Unknown action: {action}')

if __name__ == '__main__':
    main()
