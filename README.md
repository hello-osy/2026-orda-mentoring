# 2026 ORDA 멘토링

첨부 CAD를 옮긴 **Gazebo + ROS 2** 연습장입니다. 카메라·라이다로 주행하고, YOLOv8 인식으로 미션 상태를 바꿉니다. 정답 예제는 **[answer.py](src/orda_sim/orda_sim/answer.py) 한 파일**입니다. 현재 연습 배치의 **최종 기본 코스에서 무개입 완주**했습니다. [검증 결과](src/materials/validation.md) · [도면 재현 범위](src/materials/track_notes.md)

**설치** — Ubuntu 24.04 / ROS 2 Jazzy 설치 상태에서 한 번 실행합니다.

```bash
cd /home/osy/OSY_STUDY/261027/2026-orda-mentoring
bash scripts/setup.sh configs/config.yaml
```

**실행** — 터미널 1은 시뮬레이터, 터미널 2는 정답 코드입니다. 종료는 각각 `Ctrl+C`입니다.

```bash
cd /home/osy/OSY_STUDY/261027/2026-orda-mentoring
bash scripts/simulator.sh configs/config.yaml
```

```bash
cd /home/osy/OSY_STUDY/261027/2026-orda-mentoring
bash scripts/answer.sh configs/config.yaml
```

**YOLO 다시 학습** — YAML의 `paths.dataset`을 새 폴더명으로 바꾼 뒤, 시뮬레이터만 켜고 다른 터미널에서 실행합니다. 촬영 중 차량은 여러 관측 위치로 이동합니다.

```bash
cd /home/osy/OSY_STUDY/261027/2026-orda-mentoring
bash scripts/record.sh configs/config.yaml
bash scripts/extract.sh configs/config.yaml
bash scripts/train.sh configs/config.yaml
```

설정은 **[configs/config.yaml](configs/config.yaml)만 수정**합니다. Gazebo 창은 `simulator.headless: false`와 `environment.software_rendering: true`, 차선만 시험은 `answer.detector.enabled: false`, 일반 주행 녹화는 `dataset.collection.enabled: false`입니다. 이 PC의 소프트웨어 렌더링에 맞춰 `world.real_time_factor: 0.6`으로 실행합니다. 산출물은 `src/artifacts/`에 저장됩니다.

속도·조향은 `/drive`의 `AckermannDriveStamped` 한 메시지로 보냅니다(`speed`: m/s, `steering_angle`: rad, 양수=좌회전). 센서는 `/camera/image_raw`, `/scan`, 속도 측정은 `/odom`입니다. 정답 코드는 전역 좌표·정답 라벨을 사용하지 않습니다. 시뮬레이터·정답·수동주행은 같은 YAML을 사용하세요.

수동 조작: `bash scripts/teleop.sh configs/config.yaml` (`w/s/a/d`, 정답 코드 먼저 종료). 검사: `bash scripts/check.sh configs/config.yaml` (`check.live: true`면 켜진 센서도 검사).

센서·차선 처리 화면: `bash scripts/view.sh configs/config.yaml` (`q` 종료).

자동 완주 시험: 기존 시뮬레이터·정답 코드를 종료하고 `bash scripts/race_test.sh configs/config.yaml` 실행. 새 시뮬레이터부터 종료까지 자동으로 처리하고 궤적·정지 오차·판정은 `src/artifacts/evaluations/`에 저장합니다.

**[멘토링 PPT](src/materials/mentoring_20261029.pptx)** · [PDF](src/materials/mentoring_20261029.pdf) · [구현 근거](src/materials/references.md) · 구조: `configs` 설정 / `scripts` 실행 / `src` 코드·자료·산출물
