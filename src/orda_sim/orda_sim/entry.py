"""Internal dispatch; users run scripts/*.sh with one YAML argument."""
import sys
import os
from .common import load_config

def main():
    action, filename = sys.argv[1:]
    config = load_config(filename)
    # Sourcing ros_gz may hide the standalone Harmonic command manifests.
    paths = config['environment']['gazebo_config_paths']
    os.environ['GZ_CONFIG_PATH'] = ':'.join(paths + [os.environ.get('GZ_CONFIG_PATH', '')])
    if action == 'simulator':
        from .simulator import run
        run(config)
    elif action == 'answer':
        from .answer import main as answer_main
        answer_main(filename)
    elif action in ('record', 'extract'):
        from . import dataset
        getattr(dataset, action)(config)
    elif action == 'train':
        from .train import train
        train(config)
    elif action == 'teleop':
        from .teleop import run
        run(config)
    elif action == 'motion_check':
        from .motion_check import run
        run(config)
    elif action == 'evaluate':
        from .evaluate import run
        run(config)
    elif action == 'race_test':
        from .race_test import run
        run(config)
    elif action == 'check':
        from .verify import run
        run(config)
    elif action == 'slides':
        from .slides import run
        run(config)
    elif action == 'view':
        from .view import run
        run(config)
    else:
        raise SystemExit(f'Unknown action: {action}')

if __name__ == '__main__':
    main()
