#!/usr/bin/env bash
# 실차(Jetson, Ubuntu 22.04 + ROS 2 Humble) 공통 환경. answer_real.sh / drive_real.sh가 source 한다.
# ~/.bashrc는 메뉴 입력(read)이 있어서 직접 source 하지 않고 필요한 부분만 옮겼다.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
[ -f "$HOME/wego_ws/install/setup.bash" ] && source "$HOME/wego_ws/install/setup.bash"
export ROS_LOCALHOST_ONLY=1
export PYTHONWARNINGS=ignore
# ultralytics는 conda dl_env, torch(CUDA)는 ~/.local에 설치되어 있다.
export PYTHONPATH="$root/src/orda_sim:$HOME/miniconda3/envs/dl_env/lib/python3.10/site-packages:/usr/lib/python3.10/dist-packages:${PYTHONPATH:-}"
export YOLO_CONFIG_DIR="$root/src/runtime/ultralytics"
export ROS_LOG_DIR="$root/src/runtime/ros_logs"
mkdir -p "$YOLO_CONFIG_DIR" "$ROS_LOG_DIR"
cd "$root"
