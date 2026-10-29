# 2026 ORDA 멘토링

Gazebo + ROS 2 시뮬레이터입니다. **사용자는 아래 두 파일만 수정·교체하면 됩니다.**

- (src/orda_sim/orda_sim/answer.py): 주행 코드. 
- (src/orda_sim/orda_sim/best.pt): YOLO 모델.

**설치** — Ubuntu 24.04 + ROS 2 Jazzy 기준입니다. 일반 Ubuntu와 **WSL2 + WSLg의 Ubuntu 24.04**에서 같은 명령을 사용합니다.
/2026-orda-mentoring 경로에서 명령어를 실행하세요.


```bash
bash scripts/setup.sh configs/config.yaml
```

**터미널 1 — 시뮬레이터**

```bash
bash scripts/simulator.sh configs/config.yaml
```

**터미널 2 — 주행 코드**

```bash
bash scripts/answer.sh configs/config.yaml
```
