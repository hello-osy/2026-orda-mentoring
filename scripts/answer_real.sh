#!/usr/bin/env bash
# 실차용: 주행 코드(answer.py) 실행. /drive만 발행하므로 차는 drive_real.sh를 켜야 움직인다.
# 사용법: bash scripts/answer_real.sh configs/real_car.yaml
set -e
if [ "$#" -ne 1 ]; then
  echo "사용법: bash scripts/answer_real.sh configs/real_car.yaml" >&2
  exit 2
fi
config="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
source "$(dirname "$0")/real_env.sh"
exec /usr/bin/python3 -m orda_sim.entry answer "$config"
