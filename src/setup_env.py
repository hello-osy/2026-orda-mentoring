"""Install only project dependencies; ROS itself is already installed."""
from pathlib import Path
import importlib.util
import os
import subprocess
import tempfile
import urllib.request


def has_apt_candidate(package):
    result = subprocess.run(['apt-cache', 'policy', package], text=True, capture_output=True,
                            env={**os.environ, 'LC_ALL': 'C'}, check=True)
    for line in result.stdout.splitlines():
        if line.strip().startswith('Candidate:'):
            return line.split(':', 1)[1].strip() != '(none)'
    return False


def ensure_gazebo_repository(repository):
    """Use the official Harmonic repository only when apt cannot find Gazebo."""
    if has_apt_candidate(repository['package']):
        return
    architecture = subprocess.run(['dpkg', '--print-architecture'], text=True,
                                  capture_output=True, check=True).stdout.strip()
    source = (f"deb [arch={architecture} signed-by={repository['keyring']}] "
              f"{repository['url']} {repository['suite']} {repository['components']}\n")
    # Download without root, then install only the two documented apt files.
    # https://gazebosim.org/docs/harmonic/install_ubuntu/
    with tempfile.TemporaryDirectory(prefix='orda-gazebo-apt-') as folder:
        key = Path(folder) / 'gazebo.gpg'
        with urllib.request.urlopen(repository['key_url'], timeout=repository['download_timeout']) as response:
            key.write_bytes(response.read())
        if not key.stat().st_size:
            raise RuntimeError('The Gazebo repository signing key download was empty.')
        listing = Path(folder) / 'gazebo-stable.list'
        listing.write_text(source)
        for local, destination in [(key, repository['keyring']), (listing, repository['source_list'])]:
            subprocess.run(['sudo', 'install', '-d', '-m', '0755', str(Path(destination).parent)], check=True)
            subprocess.run(['sudo', 'install', '-m', '0644', str(local), destination], check=True)
    subprocess.run(['sudo', 'apt-get', 'update'], check=True)
    if not has_apt_candidate(repository['package']):
        raise RuntimeError('Gazebo has no apt candidate after repository setup; check environment.gazebo_repository.')

def setup_environment(config, root):
    missing = []
    for package in config['environment']['apt_packages']:
        result = subprocess.run(['dpkg-query', '-W', '-f=${Status}', package], text=True, capture_output=True)
        if result.returncode or result.stdout != 'install ok installed':
            missing.append(package)
    if missing:
        subprocess.run(['sudo', 'apt-get', 'update'], check=True)
        repository = config['environment']['gazebo_repository']
        if repository['package'] in missing:
            ensure_gazebo_repository(repository)
        subprocess.run(['sudo', 'apt-get', 'install', '-y', *missing], check=True)
    python = root / '.venv/bin/python'
    if not python.exists():
        subprocess.run(['/usr/bin/python3', '-m', 'venv', '--system-site-packages', str(root / '.venv')], check=True)
    (root / '.venv/COLCON_IGNORE').touch()
    if config['environment']['egl_software_only']:
        output = root / 'src/runtime/libegl_software.so'
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['cc', '-shared', '-fPIC', '-O2', str(root / 'src/runtime_support/egl_software.c'),
                        '-o', str(output), '-ldl', '-pthread'], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', *config['environment']['torch_packages'],
                    '--index-url', config['environment']['torch_index']], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', *config['environment']['pip_packages']], check=True)
    # Small ROS message package can be installed locally without sudo.
    ros_prefix = Path(config['environment']['ros_setup']).parent
    if not (ros_prefix / 'share/ackermann_msgs/package.xml').exists():
        folder = root / 'src/runtime/ros_debs'
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / 'root'
        if not (target / 'opt/ros/jazzy/share/ackermann_msgs/package.xml').exists():
            subprocess.run(['apt-get', 'download', 'ros-jazzy-ackermann-msgs'], cwd=folder, check=True)
            for package in folder.glob('ros-jazzy-ackermann-msgs_*.deb'):
                subprocess.run(['dpkg-deb', '-x', str(package), str(target)], check=True)
