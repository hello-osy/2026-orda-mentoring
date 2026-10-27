"""Behaviour tests: no ROS process, GPU, or detector weights required."""
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('answer', ROOT / 'src/orda_sim/orda_sim/answer.py')
answer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(answer)


@pytest.fixture
def config():
    config = yaml.safe_load((ROOT / 'configs/config.yaml').read_text())
    config['answer']['missions']['enforce_order'] = False
    return config


@pytest.fixture
def vision(config):
    return answer.LaneVision(config['answer']['lane'], config['robot']['camera'],
                             config['robot']['front_axle_x'])


def lane_image(vision, shift_m=0.0, curve=0.0):
    c = vision.c
    image = np.zeros((c['height_px'], c['width_px']), np.uint8)
    scale = c['width_px'] / c['view_width']
    for y in range(c['height_px']):
        center = c['width_px'] / 2 - shift_m * scale + curve * (y / c['height_px'] - 0.7) ** 2 * scale
        for side in [-1, 1]:
            x = int(center + side * c['lane_width'] * scale / 2)
            cv2.line(image, (x - 2, y), (x + 2, y), 255, 1)
    return image


def blank_lane(valid=True, stop=None, error=0.0, heading=0.0):
    return dict(valid=valid, error=error, heading=heading, abnormal=False,
                stop_distance=stop, center_px=240)


def clear_scan(front=5.0, rear=5.0):
    return dict(front=front, rear=rear, options=[(0.0, 5.0)], valid_fraction=1.0)


def detection(name):
    return [dict(name=name, area=0.04 if name.startswith('traffic_') else 0.02, confidence=0.9, center=(0.5, 0.4))]


def test_sliding_window_center_and_lateral_sign(vision):
    straight = vision.analyze_bird(lane_image(vision))
    assert straight['valid']
    assert abs(straight['error']) < 0.015
    vision.previous_center = None
    shifted = vision.analyze_bird(lane_image(vision, shift_m=0.12))
    assert shifted['error'] == pytest.approx(0.12, abs=0.015)


def test_sliding_window_curve_and_one_missing_boundary(vision):
    image = lane_image(vision, curve=0.22)
    image[:, :int(image.shape[1] / 2)] = 0
    result = vision.analyze_bird(image)
    assert result['valid']
    assert math.isfinite(result['heading'])
    assert abs(result['error']) < 0.15


def test_stop_line_is_measured_from_front_axle(vision):
    image = lane_image(vision)
    distance_from_base = 0.60
    row = int(vision.row_at(distance_from_base))
    scale = image.shape[1] / vision.c['view_width']
    half = int(vision.c['lane_width'] * scale / 2)
    cv2.rectangle(image, (image.shape[1] // 2 - half, row - 4),
                  (image.shape[1] // 2 + half, row), 255, -1)
    result = vision.analyze_bird(image)
    assert result['valid']
    assert result['stop_distance'] == pytest.approx(distance_from_base - vision.front_axle_x, abs=0.01)


def test_ground_map_centerline_projects_to_image_center(vision):
    x, y = vision.ground_maps((480, 640, 3))
    assert x[x.shape[0] // 2, x.shape[1] // 2] == pytest.approx(320, abs=3)
    assert np.isfinite(x).all() and np.isfinite(y).all()


def test_lidar_infinity_is_clear_but_nan_is_unsafe(config):
    c = config['answer']['lidar']
    ranges = np.full(360, np.inf)
    good = answer.scan_sectors(ranges, -math.pi, 2 * math.pi / 360, .02, 10, c)
    assert good['front'] == 10
    assert good['rear'] == 10
    ranges[180] = np.nan
    bad = answer.scan_sectors(ranges, -math.pi, 2 * math.pi / 360, .02, 10, c)
    assert bad['front'] == 0


def test_pid_direction_and_integral_bound(config):
    pid = answer.PID(config['answer']['pid'])
    for _ in range(300):
        output = pid.update(0.2, 0.05)
    assert output > 0
    assert pid.integral <= config['answer']['pid']['integral_limit']
    pid.reset()
    assert pid.update(-0.2, 0.05) < 0


def test_repeated_same_detection_frame_does_not_confirm(config):
    driver = answer.Driver(config)
    config['answer']['detector']['confirm_frames'] = 2
    for i in range(10):
        driver.update(blank_lane(stop=.5), clear_scan(), detection('pedestrian'), 1.0, 0.3, 1.0 + i * 0.05)
    assert driver.state == 'lane'
    driver.update(blank_lane(stop=.5), clear_scan(), detection('pedestrian'), 2.0, 0.3, 2.0)
    assert driver.state == 'pedestrian_approach'


def test_pedestrian_wait_starts_only_after_measured_stop(config):
    driver = answer.Driver(config)
    driver.transition('pedestrian_approach', 1.0)
    for now in np.arange(1.0, 2.0, .05):
        speed, _, _ = driver.update(blank_lane(stop=.02), clear_scan(), [], now, .08, now)
        assert speed == 0.0
        assert driver.state == 'pedestrian_approach'
    driver.update(blank_lane(stop=.02), clear_scan(), [], 2.0, 0.0, 2.0)
    assert driver.state == 'pedestrian_hold'
    driver.update(blank_lane(stop=.02), clear_scan(), [], 4.9, 0.0, 4.9)
    assert driver.state == 'pedestrian_hold'
    driver.update(blank_lane(stop=.02), clear_scan(), [], 5.3, 0.0, 5.3)
    assert driver.state == 'lane'


def test_traffic_hold_requires_confirmed_green(config):
    driver = answer.Driver(config)
    driver.transition('traffic_hold', 1)
    for t in [1., 2., 3.]:
        speed, _, _ = driver.update(blank_lane(), clear_scan(), [], t, 0, t)
        assert speed == 0.0
        assert driver.state == 'traffic_hold'
    driver.update(blank_lane(), clear_scan(), detection('traffic_green'), 4, 0, 4)
    driver.update(blank_lane(), clear_scan(), detection('traffic_green'), 5, 0, 5)
    assert driver.state == 'traffic_approach'


def test_lost_lane_eventually_stops(config):
    driver = answer.Driver(config)
    driver.update(blank_lane(), clear_scan(), [], 1, .2, 1)
    speed, _, _ = driver.update(blank_lane(valid=False), clear_scan(), [], 3, .2, 3)
    assert speed == 0


def test_reverse_parking_obstacle_aborts_sequence(config):
    driver = answer.Driver(config)
    config['answer']['parking']['sequence'] = [dict(seconds=10, speed=-.1, steering=.4)]
    driver.transition('parking_sequence', 1)
    driver.sequence_started = 1
    speed, _, state = driver.update(blank_lane(), clear_scan(rear=.05), [], 2, -.1, 2)
    assert speed == 0
    assert state == 'fault: obstacle during parking'


def test_slow_sign_respects_half_base_speed(config):
    driver = answer.Driver(config)
    driver.update(blank_lane(), clear_scan(), detection('slow'), 1, .3, 1)
    speed, _, _ = driver.update(blank_lane(), clear_scan(), detection('slow'), 2, .3, 2)
    assert speed <= min(.2, config['answer']['base_speed'] / 2)


def test_stop_line_tracker_does_not_jump_to_next_crosswalk_stripe(config):
    driver = answer.Driver(config)
    driver.transition('pedestrian_approach', 1)
    driver.update(blank_lane(stop=.30), clear_scan(), [], 1, .1, 1)
    driver.update(blank_lane(stop=.65), clear_scan(), [], 1.1, .1, 1.1)
    assert driver.line_distance == pytest.approx(.29, abs=.001)


def test_conflicting_red_and_green_does_not_release_traffic_hold(config):
    driver = answer.Driver(config)
    driver.transition('traffic_hold', 1)
    both = detection('traffic_green') + detection('traffic_red')
    driver.update(blank_lane(), clear_scan(), both, 1, 0, 1)
    speed, _, state = driver.update(blank_lane(), clear_scan(), both, 2, 0, 2)
    assert state == 'traffic_hold'
    assert speed == 0


def test_passed_unrelated_stripe_is_not_latched_for_later_mission(config):
    driver = answer.Driver(config)
    driver.line_distance = -.10
    driver.line_seen = 1
    driver.update(blank_lane(), clear_scan(), [], 1, .3, 1)
    assert driver.line_distance is None


def test_mission_order_ignores_signal_visible_from_initial_adjacent_road(config):
    config['answer']['missions']['enforce_order'] = True
    driver = answer.Driver(config)
    driver.update(blank_lane(), clear_scan(), detection('traffic_red'), 1, .3, 1)
    driver.update(blank_lane(), clear_scan(), detection('traffic_red'), 2, .3, 2)
    assert driver.state == 'lane'
    assert driver.stage() == 'slow'
    driver.update(blank_lane(), clear_scan(), detection('slow'), 3, .3, 3)
    driver.update(blank_lane(), clear_scan(), detection('slow'), 4, .3, 4)
    assert driver.stage() == 'pedestrian'


def test_saved_gazebo_startup_frame_has_straight_centered_lane(vision):
    path = ROOT / 'src/materials/simulator_camera.png'
    if not path.exists():
        pytest.skip('No recorded Gazebo startup image in this checkout')
    image = cv2.imread(str(path))
    assert image is not None
    result = vision.analyze(image)
    assert result['valid']
    assert abs(result['error']) < 0.06
    assert abs(result['heading']) < 0.10
    assert not result['abnormal']


def test_ground_projection_honors_configured_lateral_camera_offset(vision):
    original_x, _ = vision.ground_maps((480, 640, 3))
    vision.camera = dict(vision.camera, y=0.04)
    shifted_x, _ = vision.ground_maps((480, 640, 3))
    row, col = 100, 240
    # A camera mounted left of the body sees the body centerline to its right.
    assert shifted_x[row, col] > original_x[row, col]


def test_smooth_corner_is_not_misclassified_as_zigzag(vision):
    curved = vision.analyze_bird(lane_image(vision, curve=0.45))
    assert curved['valid']
    assert not curved['abnormal']


def test_zigzag_boundary_residual_requests_slow_driving(vision):
    c = vision.c
    image = np.zeros((c['height_px'], c['width_px']), np.uint8)
    scale = c['width_px'] / c['view_width']
    for y in range(c['height_px']):
        center = c['width_px'] / 2 + .065 * scale * math.sin(y / c['height_px'] * 8 * math.pi)
        for side in [-1, 1]:
            x = int(center + side * c['lane_width'] * scale / 2)
            cv2.line(image, (x - 2, y), (x + 2, y), 255, 1)
    result = vision.analyze_bird(image)
    assert result['valid']
    assert result['abnormal']


def test_one_curved_paint_component_cannot_become_two_boundaries(vision):
    vision.c['windows'] = 12
    image = cv2.imread(str(ROOT / 'src/tests/fixtures/corner_one_boundary.png'))
    result = vision.analyze(image)
    assert result['valid']
    assert len(vision.previous_boundaries) == 1


def test_per_class_detection_size_gate(config):
    config['answer']['detector']['minimum_area_by_class'] = {'parking': .03}
    driver = answer.Driver(config)
    assert not driver.confirmed(detection('parking'), 1)
    assert not driver.confirmed(detection('parking'), 2)
    assert not driver.confirmed(detection('slow'), 3)
    assert 'slow' in driver.confirmed(detection('slow'), 4)


def test_fork_choice_survives_waiting_for_obstacle(config):
    driver = answer.Driver(config)
    driver.update(blank_lane(), clear_scan(), detection('left'), 1, 0, 1)
    driver.update(blank_lane(), clear_scan(), detection('left'), 2, 0, 2)
    assert driver.fork_target > 0
    assert driver.fork_pending
    driver.update(blank_lane(), clear_scan(), [], 30, 0, 30)
    assert driver.fork_target > 0
    assert driver.fork_pending
    assert driver.fork_remaining > 0


def test_fork_waits_for_clear_travel_and_activates_only_once(config):
    driver = answer.Driver(config)
    occupied = clear_scan()
    occupied['wide_objects'] = 1
    driver.update(blank_lane(), occupied, detection('right'), 1, .1, 1.0)
    driver.update(blank_lane(), occupied, detection('right'), 2, .1, 1.1)
    assert driver.fork_pending and driver.fork_bias == 0
    driver.update(blank_lane(), clear_scan(), [], 3, .1, 1.2)
    assert driver.fork_pending  # One lost lidar cluster cannot release the arrow.
    driver.update(blank_lane(), occupied, [], 4, .1, 1.3)
    assert driver.fork_clear_distance == 0
    for t in np.arange(1.4, 3.6, .1):
        driver.update(blank_lane(), clear_scan(), [], t, .1, t)
    assert not driver.fork_pending and driver.fork_bias < 0
    remaining = driver.fork_remaining
    driver.update(blank_lane(), occupied, detection('right'), 30, 0, 30)
    driver.update(blank_lane(), occupied, detection('right'), 31, 0, 31)
    assert driver.fork_bias < 0  # New obstacle visibility does not disable it.
    assert driver.fork_remaining == pytest.approx(remaining)  # No re-arm extension.


def test_pending_fork_expires_without_leaking_into_cones(config):
    driver = answer.Driver(config)
    driver.fork_pending = True
    driver.fork_target = -.55
    driver.fork_remaining = .005
    occupied = clear_scan()
    occupied['wide_objects'] = 1
    driver.update(blank_lane(), occupied, [], 1, .2, 1)
    assert not driver.fork_pending
    assert driver.fork_bias == 0
    assert driver.fork_target == 0


def test_same_tick_arrow_and_first_car_keeps_road_memory(config, monkeypatch):
    config['answer']['missions']['enforce_order'] = True
    driver = answer.Driver(config)
    driver.stage_index = config['answer']['missions']['sequence'].index('fork')
    driver.update(blank_lane(), clear_scan(), detection('right'), 1, .1, 1)
    scan = clear_scan(front=.5)
    scan.update(wide_objects=1, points=np.array([[.5, 0.0]]))
    monkeypatch.setattr(answer, 'avoidance_steering', lambda *args: 0.0)
    driver.update(blank_lane(), scan, detection('right'), 2, .1, 1.1)
    assert driver.stage() == 'traffic'
    assert driver.fork_pending
    assert driver.road_memory


def test_pending_fork_keeps_road_memory_until_direction_can_activate(config):
    driver = answer.Driver(config)
    driver.road_memory = True
    driver.avoiding = True
    driver.avoidance_clear_distance = .4
    driver.fork_pending = True
    driver.fork_target = -.55
    driver.fork_remaining = 2.0
    scan = clear_scan()
    scan['wide_objects'] = 1
    driver.update(blank_lane(), scan, [], 1, 0, 1)
    assert driver.fork_pending
    assert driver.road_memory
    assert driver.avoiding


def test_cone_path_must_cross_on_side_toward_road_center(config):
    forward = np.array([.1, .4, .8, 1.1])
    cones = np.array([[.4, .15], [.8, -.15]])
    inside = np.array([0., -.05, .05, 0.])
    outside = np.array([0., -.05, -.35, -.35])
    assert answer.passes_inside_cones(forward, inside, cones, 0., config['answer']['lidar'])
    assert not answer.passes_inside_cones(forward, outside, cones, 0., config['answer']['lidar'])


def test_cone_crossing_uses_measured_horizon_without_extrapolation(config):
    forward = np.array([.1, .4, .8])
    lateral = np.array([0., -.1, -.2])
    cones = np.array([[.05, -.15], [1.2, -.15]])
    assert answer.passes_inside_cones(forward, lateral, cones, 0., config['answer']['lidar'])
    assert not answer.passes_inside_cones(forward[::-1], lateral, cones, 0., config['answer']['lidar'])


@pytest.mark.parametrize('filename', ['cone_stopped_scan.json', 'cone_stopped_scan_near.json', 'cone_stopped_scan_angled.json'])
def test_short_cone_arcs_fit_two_inside_gates_and_body_clearance(config, filename):
    evidence = json.loads((ROOT/'src/tests/fixtures'/filename).read_text())
    scan = answer.scan_sectors(**evidence['scan'], config=config['answer']['lidar'])
    scan['allow_edge_crossing'] = True
    heading, center = evidence['heading'], evidence['road_center']
    lane = blank_lane(heading=heading)
    lane.update(road_memory=True, road_center=center, fit_near=0., fit_far=2.,
                center_fit=[0., math.tan(heading), center/math.cos(heading)])
    steering = answer.avoidance_steering(scan, lane, config['robot'], config['answer']['lidar'], evidence['previous'])
    assert steering is not None


def test_pedestrian_false_positive_without_current_lane_stop_line_does_not_arm(config):
    driver = answer.Driver(config)
    driver.update(blank_lane(), clear_scan(), detection('pedestrian'), 1, .3, 1)
    driver.update(blank_lane(), clear_scan(), detection('pedestrian'), 2, .3, 2)
    assert driver.state == 'lane'
    driver.update(blank_lane(stop=.6), clear_scan(), detection('pedestrian'), 3, .3, 3)
    assert driver.state == 'pedestrian_approach'


def test_edge_truncated_sign_cannot_trigger_pedestrian(config):
    driver = answer.Driver(config)
    boxes = detection('pedestrian')
    boxes[0]['truncated'] = True
    assert not driver.confirmed(boxes, 1)
    assert not driver.confirmed(boxes, 2)


def test_side_post_does_not_activate_avoidance(config):
    scan = clear_scan()
    scan['points'] = np.array([[.3, -.42], [.31, -.42]])
    assert not answer.lane_obstructed(scan, blank_lane(), config['robot'], config['answer']['lidar'])
    scan['points'] = np.array([[.6, 0.0]])
    assert answer.lane_obstructed(scan, blank_lane(), config['robot'], config['answer']['lidar'])


def test_real_width_can_pass_thirty_centimeter_gap(config):
    scan = clear_scan()
    scan['points'] = np.array([[x, y] for x in np.linspace(.3, 1.3, 50) for y in [-.15, .15]])
    result = answer.avoidance_steering(scan, blank_lane(), config['robot'], config['answer']['lidar'], 0.0)
    assert result == pytest.approx(0.0, abs=.05)


def test_rectangle_footprint_rejects_gap_narrower_than_car(config):
    scan = clear_scan()
    scan['points'] = np.array([[x, y] for x in np.linspace(.0, 1.3, 60) for y in [-.10, .10]])
    assert answer.avoidance_steering(scan, blank_lane(), config['robot'], config['answer']['lidar'], 0.0) is None


def test_green_is_not_complete_until_line_crossing(config):
    driver = answer.Driver(config)
    driver.update(blank_lane(stop=.6), clear_scan(), detection('traffic_green'), 1, .2, 1)
    driver.update(blank_lane(stop=.58), clear_scan(), detection('traffic_green'), 2, .2, 1.1)
    assert driver.state == 'traffic_approach'
    driver.update(blank_lane(stop=.56), clear_scan(), detection('traffic_red'), 3, .2, 1.2)
    speed, _, state = driver.update(blank_lane(stop=.54), clear_scan(), detection('traffic_red'), 4, .2, 1.3)
    assert state == 'traffic_approach'
    assert speed <= config['answer']['missions']['stop_approach_speed']
    driver.traffic_stop_completed = True
    driver.line_distance = -.04
    driver.update(blank_lane(), clear_scan(), detection('traffic_green'), 5, .1, 1.4)
    driver.update(blank_lane(), clear_scan(), detection('traffic_green'), 6, .1, 1.5)
    assert driver.state == 'lane'


def test_green_approach_stops_once_using_measured_speed(config):
    driver = answer.Driver(config)
    driver.update(blank_lane(stop=.03), clear_scan(), detection('traffic_green'), 1, .1, 1.0)
    driver.update(blank_lane(stop=.03), clear_scan(), detection('traffic_green'), 2, 0, 1.1)
    assert driver.state == 'traffic_hold'
    assert not driver.traffic_stop_completed
    driver.update(blank_lane(stop=.03), clear_scan(), detection('traffic_green'), 3, .05, 1.3)
    assert driver.stopped_since is None
    driver.update(blank_lane(stop=.03), clear_scan(), detection('traffic_green'), 4, 0, 1.4)
    driver.update(blank_lane(stop=.03), clear_scan(), detection('traffic_green'), 5, 0, 1.6)
    assert driver.state == 'traffic_hold'
    driver.update(blank_lane(stop=.03), clear_scan(), detection('traffic_green'), 6, 0, 1.71)
    assert driver.state == 'traffic_approach'
    assert driver.traffic_stop_completed
    speed, _, state = driver.update(blank_lane(stop=.03), clear_scan(), detection('traffic_green'), 7, 0, 1.8)
    assert state == 'traffic_approach'
    assert speed > 0
    driver.line_distance = -.04
    driver.update(blank_lane(), clear_scan(), detection('traffic_green'), 8, .1, 1.9)
    assert driver.state == 'lane'
    driver.update(blank_lane(), clear_scan(), detection('traffic_green'), 9, .1, 2.0)
    assert driver.state == 'lane'


def test_same_camera_frame_does_not_rewind_stop_distance(config):
    driver = answer.Driver(config)
    driver.transition('pedestrian_approach', 1.0)
    lane = blank_lane(stop=.5)
    lane['stamp'] = 1.0
    driver.update(lane, clear_scan(), [], 1, .1, 1.0)
    driver.update(lane, clear_scan(), [], 1, .1, 1.2)
    driver.update(lane, clear_scan(), [], 1, .1, 1.4)
    assert driver.line_distance == pytest.approx(.46)
    assert driver.line_seen == 1.0


def test_delayed_camera_distance_is_compensated_to_now(config):
    driver = answer.Driver(config)
    lane = blank_lane(stop=.5)
    lane['stamp'] = 1.0
    driver.update(lane, clear_scan(), [], 1, .1, 1.5)
    assert driver.line_distance == pytest.approx(.45)


def test_lane_obstruction_does_not_extrapolate_beyond_measured_paint(config):
    scan = clear_scan()
    scan['points'] = np.array([[1.0, 0.0]])
    lane = blank_lane()
    lane.update(fit_near=.3, fit_far=.6, center_fit=[0, 0, 0])
    assert not answer.lane_obstructed(scan, lane, config['robot'], config['answer']['lidar'])


def test_lidar_distinguishes_narrow_cone_slice_from_post(config):
    angles = np.linspace(-math.pi, math.pi, 721)
    ranges = np.full(721, np.inf)
    ranges[np.abs(angles) < .023] = .8
    result = answer.scan_sectors(ranges, -math.pi, angles[1]-angles[0], .05, 12., config['answer']['lidar'])
    assert result['narrow_objects'] == 1
    ranges[:] = np.inf
    ranges[np.abs(angles) < .012] = .8
    result = answer.scan_sectors(ranges, -math.pi, angles[1]-angles[0], .05, 12., config['answer']['lidar'])
    assert result['narrow_objects'] == 0


def test_local_road_memory_uses_yaw_rate_and_waits_for_alignment(config):
    driver = answer.Driver(config)
    driver.road_memory = True
    driver.avoiding = True
    driver.road_heading = .5
    driver.avoidance_clear_distance = .4
    driver.last_time = 1.0
    driver.update(blank_lane(error=-.6), clear_scan(), [], 1, .1, 1.1, angular_velocity=.4)
    assert driver.road_heading == pytest.approx(.46)
    assert driver.avoiding
    assert driver.road_memory


def test_red_hold_preserves_selected_line_during_long_red(config):
    driver = answer.Driver(config)
    driver.transition('traffic_hold', 1.0)
    driver.line_distance = .03
    driver.line_seen = 1.0
    driver.update(blank_lane(stop=.58), clear_scan(), [], 1, 0, 15.0)
    assert driver.line_distance == pytest.approx(.03)
    assert driver.state == 'traffic_hold'


def test_reappearing_adjacent_lane_does_not_flip_same_boundary(vision):
    original = lane_image(vision)
    vision.analyze_bird(original)
    one_line = original.copy()
    one_line[:, :one_line.shape[1]//2] = 0
    before = vision.analyze_bird(one_line)
    extra = one_line.copy()
    x = int(extra.shape[1]/2 + 1.5*vision.c['lane_width']*extra.shape[1]/vision.c['view_width'])
    cv2.line(extra, (x, 0), (x, extra.shape[0]-1), 255, 5)
    after = vision.analyze_bird(extra)
    assert before['valid'] and after['valid']
    assert abs(after['error']-before['error']) < .04
    deliberate_change = vision.analyze_bird(extra, fork_bias=-.55)
    assert deliberate_change['error'] < -.2


def test_corner_histogram_starts_below_short_branch_paint(config):
    config['answer']['lane']['histogram_band_height'] = .12
    vision = answer.LaneVision(config['answer']['lane'], config['robot']['camera'], config['robot']['front_axle_x'])
    frame = cv2.imread(str(ROOT/'src/tests/fixtures/corner_branch_paint.png'))
    lane = vision.analyze(frame)
    assert lane['valid']
    assert lane['heading'] < -.5
    assert lane['error'] < -.2


def test_planner_respects_sensor_selected_fork_direction(config):
    scan = clear_scan()
    scan['points'] = np.array([[2.8, 0.0]])
    lane = blank_lane()
    lane.update(road_memory=True, fit_near=0., fit_far=2.)
    lane['center_fit'] = [0., 0., .55]
    left = answer.avoidance_steering(scan, lane, config['robot'], config['answer']['lidar'], 0.)
    lane['center_fit'] = [0., 0., -.55]
    right = answer.avoidance_steering(scan, lane, config['robot'], config['answer']['lidar'], 0.)
    assert left > .1
    assert right < -.1


def test_lidar_corridor_follows_safe_gap_toward_fork(config):
    scan = clear_scan()
    scan['points'] = np.array([[x, y] for x in [.4, .8, 1.2] for center in [-.52, 0, .52]
                               for y in np.linspace(center-.11, center+.11, 15)])
    left = answer.road_gap(scan, config['robot'], 0, 0, .55, config['answer']['lidar'])
    right = answer.road_gap(scan, config['robot'], 0, 0, -.55, config['answer']['lidar'])
    assert left == pytest.approx(.26, abs=.02)
    assert right == pytest.approx(-.26, abs=.02)


def test_crosswalk_tiles_are_not_longitudinal_lane_dashes(config):
    config['answer']['lane']['histogram_band_height'] = .12
    vision = answer.LaneVision(config['answer']['lane'], config['robot']['camera'], config['robot']['front_axle_x'])
    frame = cv2.imread(str(ROOT/'src/tests/fixtures/crosswalk_near.png'))
    lane = vision.analyze(frame)
    assert lane['valid']
    assert abs(lane['error']) < .03
    assert abs(lane['heading']) < .03


def test_closed_diamond_does_not_replace_open_zigzag_lanes(config):
    vision = answer.LaneVision(config['answer']['lane'], config['robot']['camera'], config['robot']['front_axle_x'])
    frame = cv2.imread(str(ROOT/'src/tests/fixtures/zigzag_diamond.png'))
    lane = vision.analyze(frame)
    assert lane['valid']
    assert abs(lane['error']) < .03
    assert abs(lane['heading']) < .15


def test_independent_angled_dashes_recover_road_direction(config):
    vision = answer.LaneVision(config['answer']['lane'], config['robot']['camera'], config['robot']['front_axle_x'])
    frame = cv2.imread(str(ROOT/'src/tests/fixtures/angled_dashed_road.png'))
    lane = vision.analyze(frame)
    # The nearby curved merge line points the other way. Separate straight
    # dashes still agree on the road direction, without a world pose input.
    assert lane['heading'] > .1
    assert lane['dash_heading'] == pytest.approx(-.36, abs=.03)


def test_heading_consensus_rejects_single_line_and_wide_crosswalk_tiles(vision):
    bird = np.zeros((vision.c['height_px'], vision.c['width_px']), np.uint8)
    scale = vision.c['width_px']/vision.c['view_width']
    rows = vision.c['height_px']/(vision.c['far']-vision.c['near'])
    cv2.rectangle(bird, (100, 100), (103, 100+int(.20*rows)), 255, -1)
    assert answer.estimate_dash_heading(bird, vision.c) is None
    bird[:] = 0
    for x in [100, 160]:
        cv2.rectangle(bird, (x, 100), (x+int(.07*scale), 100+int(.20*rows)), 255, -1)
    assert answer.estimate_dash_heading(bird, vision.c) is None


def test_obstacle_entry_uses_agreed_dash_heading(config, monkeypatch):
    driver = answer.Driver(config)
    driver.stage_index = config['answer']['missions']['sequence'].index('fork')
    lane = blank_lane()
    lane.update(heading=.15, dash_heading=-.35)
    scan = clear_scan(front=.5)
    scan.update(wide_objects=1, points=np.array([[.5, 0.0]]))
    monkeypatch.setattr(answer, 'avoidance_steering', lambda *args: 0.0)
    driver.update(lane, scan, [], 1, 0, 1)
    assert driver.road_memory
    assert driver.road_heading == pytest.approx(-.35)


def test_cones_do_not_inherit_static_obstacle_side_preference(config, monkeypatch):
    driver = answer.Driver(config)
    driver.stage_index = config['answer']['missions']['sequence'].index('traffic')
    driver.road_memory = True
    driver.avoiding = True
    captured = []
    monkeypatch.setattr(answer, 'avoidance_steering', lambda scan, lane, robot, cfg, previous: captured.append(lane) or 0.0)
    scan = clear_scan(front=.5)
    scan.update(narrow_objects=1, wide_objects=0, points=np.array([[.5, 0.0]]))
    driver.update(blank_lane(), scan, [], 1, 0, 1)
    assert driver.debug['cone_clearance']
    assert captured[0]['center_fit'][2] == pytest.approx(0.0)


def test_partial_car_cluster_during_fork_is_not_a_cone(config):
    driver = answer.Driver(config)
    driver.stage_index = config['answer']['missions']['sequence'].index('traffic')
    driver.fork_remaining = 2.0
    scan = clear_scan()
    scan.update(narrow_objects=1, wide_objects=0)
    driver.update(blank_lane(), scan, [], 1, 0, 1)
    assert not driver.debug['cone_clearance']


def test_fork_bias_does_not_relabel_single_current_lane_boundary(vision):
    one = lane_image(vision)
    one[:, one.shape[1]//2:] = 0
    lane = vision.analyze_bird(one, fork_bias=.55)
    assert lane['valid']
    assert abs(lane['error']) < .02


def test_fork_lane_arrival_releases_old_camera_identity(config):
    driver = answer.Driver(config)
    driver.road_memory = True
    driver.avoiding = True
    driver.avoidance_clear_distance = .4
    driver.road_heading = 0.0
    driver.road_center = -.55
    driver.fork_bias = .55
    driver.fork_remaining = 2.0
    driver.update(blank_lane(error=.6), clear_scan(), [], 1, 0, 1)
    assert not driver.avoiding
    assert not driver.road_memory


def test_failed_static_corridor_refines_grid_without_reducing_clearance(config):
    raw = json.loads((ROOT/'src/tests/fixtures/static_car_stopped_scan.json').read_text())
    heading = raw.pop('road_heading')
    center = raw.pop('road_center')
    lidar = config['answer']['lidar']
    scan = answer.scan_sectors(**raw, config=lidar)
    desired = answer.road_gap(scan, config['robot'], heading, center, lidar['road_default_bias'], lidar)
    lane = blank_lane(error=-.256)
    lane.update(heading=heading, center_fit=[0., math.tan(heading), (center+desired)/math.cos(heading)],
                fit_near=0., fit_far=2., road_memory=True, road_center=center)
    coarse = dict(lidar, planning_refine_on_failure=False)
    assert answer.avoidance_steering(scan, lane, config['robot'], coarse, .0325) is None
    refined = answer.avoidance_steering(scan, lane, config['robot'], lidar, .0325)
    assert 0 < refined < .0325


def test_static_grid_refinement_preserves_existing_success(config):
    scan = clear_scan()
    scan['points'] = np.array([[2.8, 0.0]])
    lane = blank_lane()
    lane.update(center_fit=[0., 0., .3], fit_near=0., fit_far=2., road_memory=True)
    lidar = config['answer']['lidar']
    ordinary = answer.avoidance_steering(scan, lane, config['robot'], dict(lidar, planning_refine_on_failure=False), 0.)
    assert ordinary is not None
    assert answer.avoidance_steering(scan, lane, config['robot'], lidar, 0.) == ordinary


def test_camera_blackout_is_optional_only_after_parking_alignment(config):
    ages = dict(image=2., detector=2., scan=.02, odometry=.02)
    available = dict(image=True, lane=True, scan=True, detector=True)
    assert answer.sensor_status(config['answer'], 'parking_align', ages, available) == (False, True)
    for state in ['parking_sequence', 'parked']:
        assert answer.sensor_status(config['answer'], state, ages, available) == (True, False)
    available['lane'] = False
    assert not answer.sensor_status(config['answer'], 'parking_sequence', ages, available)[0]
    available['lane'] = True
    config['answer']['parking']['require_camera_during_sequence'] = True
    assert answer.sensor_status(config['answer'], 'parking_sequence', ages, available) == (False, True)


@pytest.mark.parametrize('lost_sensor', ['scan', 'odometry'])
def test_required_sensor_loss_permanently_aborts_timed_parking(config, lost_sensor):
    driver = answer.Driver(config)
    driver.transition('parking_sequence', 1.)
    ages = dict(image=2., detector=2., scan=.02, odometry=.02)
    ages[lost_sensor] = 2.
    available = dict(image=True, lane=True, scan=True, detector=True)
    ready, required_camera = answer.sensor_status(config['answer'], driver.state, ages, available)
    assert not ready and not required_camera
    driver.sensor_failure(2.)
    assert driver.state == 'fault: sensor lost during parking'
    speed, steering, state = driver.update(blank_lane(), clear_scan(), [], 3., 0., 3.)
    assert speed == 0.
    assert state == 'fault: sensor lost during parking'
