#!/usr/bin/env bash
# 실차용: /drive -> /cmd_vel 변환. 이걸 켜는 순간 실차가 answer.py 명령대로 움직인다.
# 사용법: bash scripts/drive_real.sh configs/real_car.yaml
set -e
if [ "$#" -ne 1 ]; then
  echo "사용법: bash scripts/drive_real.sh configs/real_car.yaml" >&2
  exit 2
fi
config="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
source "$(dirname "$0")/real_env.sh"
exec /usr/bin/python3 -m orda_sim.entry real_drive "$config"
