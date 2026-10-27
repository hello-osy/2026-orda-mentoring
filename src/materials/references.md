# 구현 근거

- `Autorace2026_규정집.pdf`, 2026-10-08: 8개 미션·100점, 권장 0.3~0.4m/s, 서행 기본 속도의 절반·최대0.2m/s, 정지 오차±5cm(앞바퀴 기준), 보행자 완전정지 후3초, T주차 네 모서리 판정.
- `Autorace2026_mission_detail_draft.docx.pdf`, 2026-09-28: 이전9개 미션 초안. 신호등+광고판 통합과 번호는 위 규정집 우선.
- `2026 국민대 LIMO 트랙 도면.zip`: 실제 DXF 노면 경계·마킹을 정규화했다. 시설물 위치/역학은 도면에 없으므로 YAML의 연습 설정이다. [도면 변환 근거](track_notes.md).
- 2026-09-22·10-06 ORDA 교육 PPT: 색 필터→Sliding Window→Scan Line→PID, 기준 차선/목표 변경으로 미션 확장, bag을 통한 디버깅, 상태 머신 주차.
- [2026-orda-education](https://github.com/hello-osy/2026-orda-education): `competition_lane.py`의 BEV·흰색 마스크·수평선 제거·차선 추적과 Session2 bag/YOLO 학습 흐름 참고.
- [ai-autonomous-driving-competition-2026](https://github.com/hello-osy/ai-autonomous-driving-competition-2026): `mission_lane_main_node.py`의 인지 결과에 따른 상태 전환, `parking_node_osy.py`의 진입 정렬·시퀀스·중단 조건 참고. 실차 PWM과 후방0도 LiDAR는 이번 m/s·전방0도에 그대로 복사하지 않았다.
- [unity-AD-simulator](https://github.com/hello-osy/unity-AD-simulator): 로컬 VS Code 인증으로 읽음. `participant.py`의 단일 제어 파일·센서 BEST_EFFORT·명령 watchdog 및 `docs/ros_topics.md`의 센서시각/조향단위 명시 방식을 참고. Unity의 ±100 조향을 복사하지 않고 표준 AckermannDriveStamped(m/s, rad)를 사용한다. 첨부 Autorace 규정과 다른 대회의 제출 규칙은 적용하지 않았다.
- `/home/osy/OSY_STUDY/drone_team_project/scripts/sim.sh`: 같은 ARM/UTM 환경의 Gazebo 소프트웨어 렌더링 문제 해결 참고.
- [ROS/Gazebo 호환 조합](https://gazebosim.org/docs/harmonic/ros_installation/), [YOLO 학습 API](https://docs.ultralytics.com/modes/train/), [Gazebo bounding box sensor](https://gazebosim.org/api/sensors/9/boundingbox_camera.html).

첨부 문서와 참고 저장소는 요구사항·기술 근거로 읽었다. 그 안의 설치·제출·금지사항을 이번 작업의 별도 실행 지시로 취급하지 않았다.
