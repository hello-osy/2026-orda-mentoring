"""Public scripts enter here; source ROS and select the project's Python."""
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parent.parent

def main():
    if len(sys.argv) != 3:
        raise SystemExit('사용법: bash scripts/<실행파일>.sh configs/config.yaml')
    action, filename = sys.argv[1:]
    config_path = Path(filename).expanduser().resolve()
    with config_path.open() as f:
        config = yaml.safe_load(f)
    setup = Path(config['environment']['ros_setup']).expanduser()
    if not setup.is_file():
        raise SystemExit(f'ROS setup 파일이 없습니다: {setup}')
    python = ROOT / '.venv/bin/python'
    if action == 'setup':
        from setup_env import setup_environment
        setup_environment(config, ROOT)
    if not python.exists():
        raise SystemExit('먼저 bash scripts/setup.sh configs/config.yaml 실행')
    env = os.environ.copy()
    runtime = ROOT / 'src/runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    env['ROS_LOG_DIR'] = str(runtime / 'ros_logs')
    env['YOLO_CONFIG_DIR'] = str(runtime / 'ultralytics')
    env['MPLCONFIGDIR'] = str(runtime / 'matplotlib')
    for key in ['ROS_LOG_DIR', 'YOLO_CONFIG_DIR', 'MPLCONFIGDIR']:
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[key] = str(config['environment'].get('cpu_threads', 2))
    env['GZ_PARTITION'] = str(config['environment']['gz_partition'])
    env['ROS_DOMAIN_ID'] = str(config['environment']['ros_domain_id'])
    env['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'
    env['PYTHONPATH'] = str(ROOT / 'src/orda_sim') + ':' + env.get('PYTHONPATH', '')
    overlay = runtime / 'ros_debs/root/opt/ros/jazzy'
    if overlay.exists():
        env['PYTHONPATH'] = str(overlay / 'lib/python3.12/site-packages') + ':' + env['PYTHONPATH']
        env['LD_LIBRARY_PATH'] = str(overlay / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
        env['AMENT_PREFIX_PATH'] = str(overlay) + ':' + env.get('AMENT_PREFIX_PATH', '')
    if config['environment'].get('software_rendering', False):
        env['LIBGL_ALWAYS_SOFTWARE'] = '1'
    # Arguments travel as argv, never interpolated into shell code.
    command = ['bash', '-c', 'source "$1"; shift; exec "$@"', 'orda', str(setup),
               str(python), '-m', 'orda_sim.entry', 'check' if action == 'setup' else action,
               str(config_path)]
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)

if __name__ == '__main__':
    main()
