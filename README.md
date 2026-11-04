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

**역할 분담 — 모두 `answer.py`에서 작업합니다.**

- 차선 2명: `LaneVision`·`LaneDriving`(인식/PID), `LaneChange`(전방 차량 감지/차선 변경)
- 주차 1명: `Parking`(진입 정렬/주차 시간표)
- 라바콘 1명: `ConeDriving`·`cone_steering`(라바콘 안쪽 통과)
- 공통: `Driver`(제어 선택/정지 조건), `AnswerNode`(센서 수신/YOLO/최종 `/drive` 발행)

각 담당의 파라미터는 파일 상단 `ANSWER_CONFIG`에 있습니다. 시뮬레이터는 차량 장애물을 차로마다 1대씩 총 3대 배치하며, 라바콘은 반지름 4.5cm·높이 24cm입니다.
