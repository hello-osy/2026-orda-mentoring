# 실행 검증 기록

2026-10-28, **최종 기본 설정(왼쪽 A)으로 출발부터 주차까지 무개입 완주**했습니다. 13개 체크포인트와 9개 판정 기준이 모두 통과했고, 시뮬레이터·정답·평가 프로세스가 종료 코드0으로 끝났습니다.

| 시뮬레이션 주행 시간 | 보행자 정지 오차 / 대기 | 신호 정지 오차 / 대기 | 주차선 최소 여유 |
|---:|---:|---:|---:|
| 206.088초 | 3.50cm 앞 / 3.182초 | 3.52cm 앞 / 1.206초 | 10.61cm |

정지 오차는 정지 유지 중 **앞바퀴에서 선까지의 거리**입니다(허용 ±5cm). 신호는 녹색 접근도 한 번 완전히 정지한 뒤 녹색에서 통과합니다. 콘4개를 지정된 안쪽 방향으로 통과했으며, 실제 콘 변위는 수치오차 수준이었습니다. 차체 네 모서리가 주차칸 안에 들어간 뒤3초 이상 정차했습니다.

- [최종 판정·실측 감사](../artifacts/evaluations/run_20261028_034358_310086/audit.json) · [주행 궤적](../artifacts/evaluations/run_20261028_034358_310086/trajectory.png) · [원본 판정](../artifacts/evaluations/run_20261028_034358_310086/report.json)
- [코드·SDF·의존성 검사: 77개 테스트 통과](check_result.txt)
- [실패 → 수정 → 재시험 기록](iteration_log.json)

## 검증 범위와 추가 근거

최종 검증은 **기본 코스 전체1회 + 수정 구간 재현 + 카메라 중단 주차 시험**으로 구성했습니다. 우측 전체 완주 기록은 이전 버전 결과이며, 최종 소스의 우측 전체 재시험과 구분합니다.

| 시험 | 확인한 범위 |
|---|---|
| [카메라 중단 주차](../artifacts/evaluations/parking_camera_blackout_segment/run41/audit.json) | 실제 P 인식·정렬 후 영상2.038초/22프레임 차단, 영상 나이2.282초에서도 주차 완료. 네 모서리 최소여유10.214cm, 실제정지6.0초 |
| [차량 통로 실패 자세 재현](../artifacts/evaluations/vehicle_refinement_segment/run40/audit.json) | 기존 정지 자세에서 재출발→차량 구간→우측 B 분기 통과, 차량 겹침0 |
| [이전 버전 우측 전체 완주](../artifacts/evaluations/run_20261028_025008_297386/audit.json) | 206.232초, 13개 체크포인트·9개 기준 통과 |

카메라와 YOLO는 차선 주행·미션 판단·주차 정렬에 필요합니다. 정렬 완료 뒤의 고정 주차 시퀀스는 영상에 의존하지 않으며, 이 단계에서도 라이다·속도 정보·시계·명령 watchdog과 장애물 검사를 유지합니다. 라이다 또는 속도 정보가 끊기면 주차를 중단합니다. 이 동작은 `parking.require_camera_during_sequence`로 설정합니다.

## 재현

Ubuntu24.04 ARM64 / ROS2 Jazzy / Gazebo Harmonic, CPU 렌더링, `world.real_time_factor: 0.6`에서 검증했습니다. 기본은 왼쪽이며, 우측 표지는 `scene.random_fork: false`, `scene.fork_initial: right` 두 값으로 선택합니다.

```bash
cd /home/osy/OSY_STUDY/261027/2026-orda-mentoring
bash scripts/race_test.sh configs/config.yaml
```

평가 폴더의 `config_used.yaml`, 실행 로그 폴더의 `answer_snapshot.txt`·`provenance.json`·Gazebo/브리지 로그에 사용 설정·코드·모델 해시와 실행 기록을 보존했습니다.

## 학습과 센서

실제 Gazebo rosbag6회에서 RGB3,058프레임을 기록했습니다. 품질 검사 후469장(학습312/검증157)을 독립 수집 시드로 나눠 YOLOv8n을30epoch 학습했습니다. 최종 mAP50은0.9888, mAP50–95는0.9598입니다. [모델·클래스별 지표](../artifacts/models/training_report.json)

카메라640×480, 라이다720샘플의 토픽 발행과 주행 명령을 확인했습니다. 별도 동작 시험에서 전진0.229m·좌회전0.266rad, 명령 중단 후 속도0을 측정했습니다. 정답 코드는 카메라·라이다·오도메트리의 속도를 사용하며 실제 위치와 정답 라벨은 독립 평가·학습에서 사용합니다.

검증 범위는 첨부 CAD와 [명시된 연습 배치](track_notes.md), 현재 차량 모델입니다. 현재 기록은 반복 성공률 통계나 실물 LIMO 성능을 뜻하지 않습니다. 차선 범위는 근사 진단, 차량 겹침은 외곽선 기하 검사로서 공식 채점·물리 접촉 점수를 대신하지 않습니다.
