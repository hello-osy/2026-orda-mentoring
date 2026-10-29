"""센서만 사용하는 교육용 정답 예제: 흰색 → sliding window → scan line → PID.

사용자는 이 파일과 같은 폴더의 best.pt만 수정·교체합니다.
주행 파라미터는 아래 ANSWER_CONFIG에 있습니다. config.yaml은 시뮬레이터 설정입니다.
월드 좌표, 정답 라벨, 경로 waypoint는 읽지 않습니다.
"""
import math
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import yaml


# 사용자 주행 설정: 속도, 차선 인식, PID, 회피, 주차 시간표
ANSWER_CONFIG = {'rate_hz': 20.0,
 'debug_rate_hz': 5.0,
 'sensor_timeout': 0.7,
 'odometry_timeout': 0.7,
 'base_speed': 0.4,
 'slow_speed': 0.2,
 'minimum_curve_speed': 0.15,
 'detector': {'enabled': True,
              'confidence': 0.45,
              'image_size': 320,
              'device': 'cpu',
              'rate_hz': 5.0,
              'timeout': 1.5,
              'minimum_area': 0.0006,
              'confirm_frames': 2,
              'minimum_area_by_class': {'parking': 0.02,
                                        'pedestrian': 0.012,
                                        'traffic_red': 0.025,
                                        'traffic_yellow': 0.025,
                                        'traffic_green': 0.025,
                                        'left': 0.006,
                                        'right': 0.006,
                                        'slow': 0.004},
              'cpu_threads': 2,
              'edge_margin': 0.01,
              'allow_truncated_classes': ['traffic_red', 'traffic_yellow', 'traffic_green'],
              'shutdown_timeout': 5.0},
 'lane': {'white_min': [0, 0, 165],
          'white_max': [179, 85, 255],
          'view_width': 2.4,
          'near': 0.2,
          'far': 1.7,
          'width_px': 480,
          'height_px': 420,
          'lane_width': 0.55,
          'windows': 12,
          'margin': 0.17,
          'minimum_pixels': 25,
          'lookahead': 0.65,
          'heading_gain': 0.0,
          'lost_grace': 1.2,
          'smoothing': 0.55,
          'stop_min_width': 0.32,
          'stop_max_thickness': 0.13,
          'component_min_height': 0.24,
          'fit_min_span': 0.16,
          'abnormal_fit_residual': 0.035,
          'histogram_band_height': 0.12,
          'max_missing_windows': 1,
          'dash_min_height': 0.12,
          'dash_max_aspect': 0.25,
          'dash_heading_max_angle': 1.2,
          'dash_heading_length_cap': 0.3,
          'dash_heading_angle_tolerance': 0.1,
          'dash_heading_min_components': 2,
          'dash_heading_support_ratio': 0.6,
          'identity_gate': 0.45,
          'histogram_smooth_pixels': 9,
          'minimum_peak': 4,
          'max_peaks': 8,
          'peak_suppression_ratio': 0.75,
          'target_history_weight': 0.35,
          'pair_width_min_ratio': 0.7,
          'pair_width_max_ratio': 1.75,
          'pair_width_weight': 0.3,
          'fit_minimum_windows': 3,
          'solid_row_fraction': 0.85,
          'fit_samples': 50,
          'stop_lane_width_ratio': 0.38,
          'stop_row_coverage': 0.7,
          'stop_edge_margin_pixels': 2,
          'heading_smoothing': 0.2,
          'abnormal_error_scale': 1.0,
          'closed_hole_min_area': 0.003},
 'pid': {'kp': 1.3,
         'ki': 0.03,
         'kd': 0.11,
         'integral_limit': 0.25,
         'derivative_alpha': 0.25,
         'feedback_gain': 0.35},
 'lidar': {'trigger_distance': 0.85,
           'emergency_distance': 0.18,
           'front_half_angle': 0.3,
           'avoidance_speed': 0.13,
           'planning_distance': 1.0,
           'planning_turn_distance': 0.4,
           'planning_heading_limit': 1.2,
           'lane_change_limit': 0.65,
           'activation_distance': 1.2,
           'planning_margin': 0.025,
           'release_distance': 0.25,
           'cluster_gap': 0.12,
           'small_object_min_width': 0.025,
           'small_object_max_width': 0.12,
           'small_object_range': 1.8,
           'small_object_clearance': 0.03,
           'cone_inside_required': True,
           'cone_side_deadband': 0.02,
           'cone_inside_margin': 0.01,
           'cone_turn_distances': [0.1, 0.25, 0.4],
           'cone_planning_distance': 0.8,
           'road_heading_limit': 0.75,
           'road_release_heading': 0.18,
           'road_release_error': 0.3,
           'road_half_width': 0.78,
           'small_object_memory_distance': 0.8,
           'planning_point_range': 3.0,
           'cluster_min_points': 3,
           'wide_object_half_width': 0.9,
           'small_object_half_width': 0.75,
           'planning_steps': 35,
           'planning_steering_samples': 33,
           'planning_return_samples': 33,
           'planning_refine_on_failure': True,
           'planning_heading_clip': 0.8,
           'road_heading_slack': 0.03,
           'planning_cost_heading': 1.0,
           'planning_cost_lateral': 2.0,
           'planning_cost_steering_change': 0.8,
           'planning_cost_clearance': 0.1,
           'planning_clearance_cap': 0.5,
           'vision_memory_distance': 2.0,
           'road_gap_range': 1.8,
           'road_gap_min_width': 0.005,
           'road_gap_center_width': 0.3,
           'planning_reference_distance': 0.65,
           'planning_cost_return_steering': 0.6,
           'planning_cost_reference': 1.5,
           'road_default_bias': 0.3},
 'missions': {'sign_cooldown': 12.0,
              'slow_seconds': 14.0,
              'fork_bias': 0.55,
              'pedestrian_wait': 3.2,
              'stop_offset': 0.015,
              'stop_tolerance': 0.025,
              'stop_approach_speed': 0.13,
              'traffic_require_stop': True,
              'traffic_min_stop': 0.3,
              'stop_gain': 0.9,
              'stopped_speed': 0.012,
              'approach_timeout': 15.0,
              'stop_line_memory': 8.0,
              'line_missing_timeout': 2.0,
              'line_match_tolerance': 0.1,
              'enforce_order': True,
              'sequence': ['slow', 'pedestrian', 'fork', 'traffic', 'parking'],
              'fork_distance': 5.0,
              'fork_wait_for_obstacles': True,
              'fork_clear_distance': 0.2,
              'pedestrian_require_line': True,
              'trigger_line_max_distance': 1.5,
              'trigger_line_freshness': 0.4,
              'fork_align_tolerance': 0.12},
 'parking': {'require_camera_during_sequence': False,
             'enabled': True,
             'align_timeout': 24.0,
             'align_speed': 0.08,
             'line_distance': 0.15,
             'distance_tolerance': 0.025,
             'lateral_tolerance': 0.035,
             'heading_tolerance': 0.09,
             'stable_seconds': 0.6,
             'sequence': [{'seconds': 2.1, 'speed': 0.12, 'steering': 0},
                          {'seconds': 6.65, 'speed': -0.1, 'steering': -0.5},
                          {'seconds': 3.0, 'speed': -0.1, 'steering': 0},
                          {'seconds': 3.2, 'speed': 0, 'steering': 0}]},
 'max_image_travel': 0.04,
 'curve_speed_gain': 0.6}

YOLO_CLASSES = ['traffic_red', 'traffic_yellow', 'traffic_green', 'left', 'right', 'pedestrian', 'slow', 'parking']


def clip(value, low, high):
    return float(max(low, min(high, value)))


class PID:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous = None
        self.derivative = 0.0
        self.terms = dict(p=0.0, i=0.0, d=0.0, total=0.0)

    def update(self, error, dt):
        c = self.config
        dt = clip(dt, 0.001, 0.2)
        self.integral = clip(self.integral + error * dt,
                             -c['integral_limit'], c['integral_limit'])
        derivative = 0.0 if self.previous is None else (error - self.previous) / dt
        alpha = c['derivative_alpha']
        self.derivative = alpha * derivative + (1 - alpha) * self.derivative
        self.previous = error
        p, i, d = c['kp'] * error, c['ki'] * self.integral, c['kd'] * self.derivative
        self.terms = dict(p=p, i=i, d=d, total=p+i+d)
        return p+i+d


def estimate_dash_heading(bird, config):
    """Local direction agreed by separate thin paint components; no world pose."""
    h, w = bird.shape
    lateral_scale = w / config['view_width']
    forward_scale = (h - 1) / (config['far'] - config['near'])
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bird)
    angles, weights = [], []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < config['minimum_pixels']:
            continue
        x, y, width, height = stats[label, :4]
        rows, cols = np.nonzero(labels[y:y+height, x:x+width] == label)
        points = np.column_stack((config['far'] - (rows+y)/forward_scale,
                                  (w/2-cols-x)/lateral_scale))
        short, long = sorted(cv2.minAreaRect(points.astype(np.float32))[1])
        if long < config['dash_min_height'] or short > long * config['dash_max_aspect']:
            continue
        _, axes = np.linalg.eigh(np.cov(points.T))
        direction = axes[:, -1]
        if direction[0] < 0:
            direction = -direction
        angle = math.atan2(direction[1], direction[0])
        if abs(angle) > config['dash_heading_max_angle']:
            continue
        angles.append(angle)
        weights.append(min(long, config['dash_heading_length_cap']))
    if not angles:
        return None
    angles, weights = np.asarray(angles), np.asarray(weights)
    tolerance = config['dash_heading_angle_tolerance']
    support = np.array([weights[np.abs(angles-angle) < tolerance].sum() for angle in angles])
    agreed = np.abs(angles - angles[support.argmax()]) < tolerance
    if (np.count_nonzero(agreed) < config['dash_heading_min_components'] or
            support.max() < config['dash_heading_support_ratio'] * weights.sum()):
        return None
    return float(np.average(angles[agreed], weights=weights[agreed]))


class LaneVision:
    """Flat-ground camera projection preserves metres for stop-line distance."""
    def __init__(self, lane, camera, front_axle_x):
        self.c = lane
        self.camera = camera
        self.front_axle_x = front_axle_x
        self.maps = None
        self.previous_center = None
        self.previous_boundaries = {}
        self.previous_fits = {}
        self.valid_bird = None

    def ground_maps(self, image_shape):
        h, w = image_shape[:2]
        c = self.c
        rows, cols = np.indices((c['height_px'], c['width_px']), dtype=np.float32)
        forward = c['far'] - rows / (c['height_px'] - 1) * (c['far'] - c['near'])
        left = (0.5 - cols / (c['width_px'] - 1)) * c['view_width']
        camera = self.camera
        dx = forward - camera['x']
        left = left - camera.get('y', 0.0)
        pitch = camera['pitch']
        depth = math.cos(pitch) * dx + math.sin(pitch) * camera['z']
        up = math.sin(pitch) * dx - math.cos(pitch) * camera['z']
        focal = w / (2 * math.tan(math.radians(camera['hfov_deg']) / 2))
        safe_depth = np.maximum(depth, 0.001)
        map_x = w / 2 - focal * left / safe_depth
        map_y = h / 2 - focal * up / safe_depth
        map_x[depth <= 0] = -1
        map_y[depth <= 0] = -1
        return map_x.astype(np.float32), map_y.astype(np.float32)

    def row_at(self, forward):
        c = self.c
        return (c['far'] - forward) / (c['far'] - c['near']) * (c['height_px'] - 1)

    def meters_at(self, row):
        c = self.c
        return c['far'] - row / (c['height_px'] - 1) * (c['far'] - c['near'])

    def analyze(self, bgr, fork_bias=0.0):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array(self.c['white_min']), np.array(self.c['white_max']))
        if self.maps is None or self.maps[0] != bgr.shape[:2]:
            self.maps = (bgr.shape[:2], self.ground_maps(bgr.shape))
        map_x, map_y = self.maps[1]
        self.valid_bird = (map_x >= 1) & (map_x < bgr.shape[1]-1) & (map_y >= 1) & (map_y < bgr.shape[0]-1)
        bird = cv2.remap(white, *self.maps[1], cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT)
        return self.analyze_bird(bird, fork_bias)

    def analyze_bird(self, bird, fork_bias=0.0):
        """Separated from ROS so lane recovery and stop distance can be tested."""
        c = self.c
        h, w = bird.shape
        scale = w / c['view_width']
        lane_width = c['lane_width'] * scale
        # Horizontal paint is a stop line/crosswalk, not a longitudinal lane.
        horizontal = cv2.morphologyEx(bird, cv2.MORPH_OPEN,
                                      np.ones((1, max(3, int(c['stop_min_width'] * scale))), np.uint8))
        lane_mask = cv2.subtract(bird, horizontal)
        # A diamond is a closed white ring, not a lane boundary. Remove its
        # whole component before the near-field histogram chooses lane starts.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(lane_mask)
        contours, hierarchy = cv2.findContours(lane_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        ring_labels = set()
        hole_area = c.get('closed_hole_min_area', .003) * scale * h / (c['far']-c['near'])
        if hierarchy is not None:
            for index, relation in enumerate(hierarchy[0]):
                parent = relation[3]
                if parent >= 0 and cv2.contourArea(contours[index]) >= hole_area:
                    x, y = contours[parent][0, 0]
                    ring_labels.add(int(labels[y, x]))
        clean = np.zeros_like(lane_mask)
        minimum_height = c.get('component_min_height', 0.24) / (c['far'] - c['near']) * h
        for label in range(1, count):
            if label in ring_labels:
                continue
            component_height = stats[label, cv2.CC_STAT_HEIGHT]
            component_width = stats[label, cv2.CC_STAT_WIDTH]
            x, y, width, height = stats[label, :4]
            rows, cols = np.nonzero(labels[y:y+height, x:x+width] == label)
            coordinates = np.column_stack((cols / scale, rows * (c['far']-c['near']) / (h-1))).astype(np.float32)
            short, long = sorted(cv2.minAreaRect(coordinates)[1])
            narrow_dash = long >= c['dash_min_height'] and short <= long * c['dash_max_aspect']
            if component_height >= minimum_height or narrow_dash:
                clean[labels == label] = 255
        lane_mask = clean
        occupied = np.flatnonzero(np.any(lane_mask > 0, axis=1))
        bottom = int(occupied.max()) + 1 if len(occupied) else h
        top = max(0, bottom - int(c.get('histogram_band_height', 0.20) / (c['far'] - c['near']) * h))
        histogram = np.sum(lane_mask[top:bottom] > 0, axis=0).astype(float)
        smooth = int(c.get('histogram_smooth_pixels', 9))
        histogram = np.convolve(histogram, np.ones(smooth) / smooth, mode='same')
        peaks = []
        peak_components = set()
        work = histogram.copy()
        while work.max() >= c.get('minimum_peak', 4) and len(peaks) < c.get('max_peaks', 8):
            peak = int(work.argmax())
            local = labels[top:bottom, max(0, peak-4):min(w, peak+5)]
            ids, counts = np.unique(local[local > 0], return_counts=True)
            component = int(ids[np.argmax(counts)]) if len(ids) else -1
            if component not in peak_components:
                peaks.append(peak)
                peak_components.add(component)
            radius = max(10, int(lane_width * c.get('peak_suppression_ratio', .75)))
            work[max(0, peak - radius):min(w, peak + radius + 1)] = 0
        peaks.sort()
        association_row = (top + bottom - 1) / 2
        previous_x = {side: float(np.polyval(fit, association_row)) for side, fit in self.previous_fits.items()}
        identity_gate = lane_width * c.get('identity_gate', .45)
        target = w / 2 - fork_bias * scale
        if self.previous_center is not None:
            weight = c.get('target_history_weight', .35)
            target = (1-weight)*target + weight*self.previous_center
        pairs = []
        for left in peaks:
            for right in peaks:
                gap = right - left
                if c.get('pair_width_min_ratio', .70)*lane_width < gap < c.get('pair_width_max_ratio', 1.75)*lane_width:
                    conflict = False
                    if abs(fork_bias) < .001:
                        for side, peak in [('left', left), ('right', right)]:
                            if previous_x:
                                previous_side = min(previous_x, key=lambda key: abs(previous_x[key]-peak))
                                if abs(previous_x[previous_side]-peak) < identity_gate and previous_side != side:
                                    conflict = True
                    if conflict:
                        continue
                    center = (left + right) / 2
                    cost = abs(center - target) + c.get('pair_width_weight', .3) * abs(gap - lane_width)
                    pairs.append((cost, left, right))
        starts = []
        if pairs:
            _, left, right = min(pairs)
            starts = [(left, 'left'), (right, 'right')]
        elif peaks:
            if previous_x:
                distance, nearest, side = min((abs(value-peak), peak, side)
                    for peak in peaks for side, value in previous_x.items())
                if distance >= identity_gate:
                    side = 'left' if nearest < w/2 else 'right'
            else:
                nearest = min(peaks, key=lambda x: abs(abs(x-target)-lane_width/2))
                side = 'left' if nearest < w/2 else 'right'
            starts = [(nearest, side)]
        nonzero_y, nonzero_x = lane_mask.nonzero()
        fits = {}
        fit_spans = {}
        solid_boundaries = {}
        fit_residuals = []
        windows = []
        for start, side in starts:
            current = float(start)
            selected = []
            last_mean = None
            missing_windows = 0
            for window in range(c['windows']):
                low = h - (window + 1) * h // c['windows']
                high = h - window * h // c['windows']
                inside = ((nonzero_y >= low) & (nonzero_y < high) &
                          (np.abs(nonzero_x - current) < c['margin'] * scale))
                windows.append(dict(side=side, x0=float(current-c['margin']*scale),
                                    x1=float(current+c['margin']*scale), y0=low, y1=high))
                indices = np.flatnonzero(inside)
                if len(indices) >= c['minimum_pixels']:
                    mean = float(np.mean(nonzero_x[indices]))
                    # Predict the next window along the observed boundary. This
                    # follows a tight bend without requiring a prewritten route.
                    change = 0.0 if last_mean is None else mean - last_mean
                    current = mean + clip(change, -c['margin'] * scale, c['margin'] * scale)
                    last_mean = mean
                    selected.extend(indices.tolist())
                    missing_windows = 0
                elif selected:
                    missing_windows += 1
                    if missing_windows > c.get('max_missing_windows', 1):
                        break
            if len(selected) >= c['minimum_pixels'] * c.get('fit_minimum_windows', 3):
                ys = nonzero_y[selected]
                xs = nonzero_x[selected]
                if np.ptp(ys) > h * c.get('fit_min_span', 0.16):
                    fits[side] = np.polyfit(ys, xs, 2)
                    fit_spans[side] = (float(ys.min()), float(ys.max()))
                    solid_boundaries[side] = len(np.unique(ys)) / (np.ptp(ys) + 1) > c.get('solid_row_fraction', .85)
                    fit_residuals.append(float(np.sqrt(np.mean((xs - np.polyval(fits[side], ys)) ** 2))) / scale)
        result = dict(valid=False, error=0.0, heading=0.0, abnormal=False,
                      stop_distance=None, stop_candidates=[], center_px=w / 2, mask=bird, lane_mask=lane_mask,
                      windows=windows, fits={side: fit.tolist() for side, fit in fits.items()},
                      fit_spans=fit_spans,
                      dash_heading=estimate_dash_heading(bird, c))
        if fits:
            self.previous_fits = {side: fit.copy() for side, fit in fits.items()}
            self.previous_boundaries = {side: float(np.polyval(fit, self.row_at(c['lookahead'])))
                                        for side, fit in fits.items()}
            # Shift each measured boundary by half a lane along its normal.
            # A horizontal pixel shift is wrong in tight bends and cuts corners.
            forwards, laterals = [], []
            boundaries = {}
            row_scale = (h - 1) / (c['far'] - c['near'])
            for side, fit in fits.items():
                rows = np.linspace(*fit_spans[side], int(c.get('fit_samples', 50)))
                forward = c['far'] - rows / row_scale
                lateral = (w / 2 - np.polyval(fit, rows)) / scale
                slope = np.polyval(np.polyder(fit), rows) * row_scale / scale
                normal_size = np.sqrt(1 + slope * slope)
                boundaries[side] = dict(fit=np.polyfit(forward, lateral, 2).tolist(),
                                        near=float(forward.min()), far=float(forward.max()))
                sign = 1 if side == 'left' else -1
                forwards.extend(forward + sign * c['lane_width'] / 2 * slope / normal_size)
                laterals.extend(lateral - sign * c['lane_width'] / 2 / normal_size)
            forwards, laterals = np.asarray(forwards), np.asarray(laterals)
            useful = (forwards > c['near']) & (forwards < c['far'])
            if np.count_nonzero(useful) < 5:
                return result
            forwards, laterals = forwards[useful], laterals[useful]
            center_fit = np.polyfit(forwards, laterals, 2)
            preview = clip(c['lookahead'], float(forwards.min()), float(forwards.max()))
            error = float(np.polyval(center_fit, preview))
            center = w / 2 - error * scale
            if self.previous_center is not None:
                center = c['smoothing'] * center + (1 - c['smoothing']) * self.previous_center
            self.previous_center = center
            error = (w / 2 - center) / scale
            heading = math.atan(float(np.polyval(np.polyder(center_fit), preview)))
            # Smooth corners also have large curvature. Zigzag paint instead
            # leaves a large residual around its smooth fitted boundary.
            residual = max(fit_residuals) if fit_residuals else 0.0
            result.update(valid=True, error=error, heading=heading,
                          abnormal=residual > c.get('abnormal_fit_residual', 0.035), center_px=center, forward=preview, solid=solid_boundaries, boundaries=boundaries,
                          center_fit=center_fit.tolist(), fit_near=float(forwards.min()), fit_far=float(forwards.max()))
        # A stripe must span the selected lane, not merely a parallel lane.
        center = result['center_px']
        x0 = int(clip(center - lane_width * c.get('stop_lane_width_ratio', .38), 0, w - 1))
        x1 = int(clip(center + lane_width * c.get('stop_lane_width_ratio', .38), x0 + 1, w))
        rows = np.flatnonzero(np.mean(horizontal[:, x0:x1] > 0, axis=1) > c.get('stop_row_coverage', .7))
        if len(rows):
            groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
            for group in reversed(groups):
                thickness = len(group) / (h - 1) * (c['far'] - c['near'])
                distance = self.meters_at(float(group.max())) - self.front_axle_x
                complete = True
                if self.valid_bird is not None and self.valid_bird.shape == bird.shape:
                    before = max(0, int(group.min())-int(c.get('stop_edge_margin_pixels', 2)))
                    after = min(h-1, int(group.max())+int(c.get('stop_edge_margin_pixels', 2)))
                    complete = bool(np.all(self.valid_bird[[before, after], x0:x1]))
                if distance > -0.04:
                    result['stop_candidates'].append(dict(distance=float(distance), thickness=float(thickness), complete=complete))
                    if complete and thickness <= c['stop_max_thickness'] and result['stop_distance'] is None:
                        result['stop_distance'] = float(distance)
        return result


def scan_sectors(ranges, angle_min, angle_increment, range_min, range_max, config):
    """Validate scan data and expose measured local obstacle clusters."""
    values = np.asarray(ranges, dtype=float)
    angles = angle_min + np.arange(len(values)) * angle_increment
    # Positive infinity is a valid no-return. NaN/negative/zero is unknown and unsafe.
    clean = np.where(np.isposinf(values), range_max, values)
    valid = np.isfinite(clean) & (clean >= range_min)
    clean = np.where(valid, np.minimum(clean, range_max), 0.0)
    def minimum(center, half):
        delta = np.arctan2(np.sin(angles - center), np.cos(angles - center))
        sector = clean[np.abs(delta) < half]
        return float(np.min(sector)) if len(sector) else 0.0
    front = minimum(0.0, config['front_half_angle'])
    rear = minimum(math.pi, config['front_half_angle'])
    hits = valid & (clean < min(range_max, config.get('planning_point_range', 3.0)))
    points = np.column_stack((clean[hits] * np.cos(angles[hits]), clean[hits] * np.sin(angles[hits])))
    narrow_objects = 0
    wide_objects = 0
    narrow_centers = []
    if len(points):
        breaks = np.flatnonzero(np.linalg.norm(np.diff(points, axis=0), axis=1) > config.get('cluster_gap', .12)) + 1
        for cluster in np.split(points, breaks):
            if len(cluster) < config.get('cluster_min_points', 3):
                continue
            width = float(np.linalg.norm(np.max(cluster, axis=0) - np.min(cluster, axis=0)))
            center = np.mean(cluster, axis=0)
            if width > config.get('small_object_max_width', .12) and 0 < center[0] < config.get('small_object_range', 1.8) and abs(center[1]) < config.get('wide_object_half_width', .9):
                wide_objects += 1
            if (config.get('small_object_min_width', .025) <= width <= config.get('small_object_max_width', .12) and
                    0 < center[0] < config.get('small_object_range', 1.8) and abs(center[1]) < config.get('small_object_half_width', .75)):
                narrow_objects += 1
                narrow_centers.append(center.tolist())
    return dict(front=front, rear=rear, points=points,
                narrow_objects=narrow_objects, narrow_centers=narrow_centers,
                wide_objects=wide_objects, valid_fraction=float(np.mean(valid)))


def lane_obstructed(scan, lane, robot, config):
    """Side posts are visible to lidar but do not obstruct the camera lane."""
    if scan['front'] < config['trigger_distance']:
        return True
    points = np.asarray(scan.get('points', []), dtype=float).reshape(-1, 2)
    if not len(points):
        return False
    forward = points[:, 0] + robot.get('lidar', {}).get('x', .10)
    lateral = points[:, 1] + robot.get('lidar', {}).get('y', 0.0)
    fit = lane.get('center_fit', [0.0, 0.0, lane['error']])
    center = np.polyval(fit, forward)
    half_width = robot['width']/2 + config.get('planning_margin', .025)
    if scan.get('allow_edge_crossing', False):
        half_width += config.get('small_object_clearance', .04)
    inside = (forward > 0) & (forward < config.get('activation_distance', 1.2))
    if 'fit_far' in lane:
        inside &= forward <= lane['fit_far']
    if 'fit_near' in lane:
        inside &= forward >= lane['fit_near']
    return bool(np.any(inside & (np.abs(lateral-center) < half_width)))


def road_gap(scan, robot, heading, road_center, desired, config):
    """Choose a free lateral corridor from lidar, in the remembered road frame."""
    points = np.asarray(scan.get('points', []), dtype=float).reshape(-1, 2).copy()
    if not len(points):
        return desired
    points[:, 0] += robot.get('lidar', {}).get('x', .10)
    points[:, 1] += robot.get('lidar', {}).get('y', 0.0)
    cosine, sine = math.cos(heading), math.sin(heading)
    forward = points[:, 0]*cosine + points[:, 1]*sine
    lateral = -points[:, 0]*sine + points[:, 1]*cosine - road_center
    lateral = lateral[(forward > 0) & (forward < config.get('road_gap_range', 1.8))]
    margin = robot['width']/2 + config.get('planning_margin', .025)
    bound = config.get('road_half_width', .78) - margin
    intervals = sorted((max(-bound, y-margin), min(bound, y+margin)) for y in lateral
                       if y+margin > -bound and y-margin < bound)
    gaps = []
    left = -bound
    for low, high in intervals:
        if low > left:
            gaps.append((left, low))
        left = max(left, high)
    if left < bound:
        gaps.append((left, bound))
    candidates = []
    for low, high in gaps:
        if high-low < config.get('road_gap_min_width', .005):
            continue
        goal = (low+high)/2 if high-low < config.get('road_gap_center_width', .30) else clip(desired, low, high)
        candidates.append((abs(goal-desired), -goal, goal))
    return desired if not candidates else float(min(candidates)[2])


def passes_inside_cones(path_forward, path_lateral, cones, road_center, config):
    """Pass each measured cone on the side toward the remembered road centre."""
    if not len(cones):
        return True
    if np.any(np.diff(path_forward) <= 0):
        return False
    side = road_center - cones[:, 1]
    future = ((cones[:, 0] > path_forward[0]) & (cones[:, 0] <= path_forward[-1]) &
              (np.abs(side) > config.get('cone_side_deadband', .02)))
    crossing = np.interp(cones[future, 0], path_forward, path_lateral)
    inside = (crossing - cones[future, 1]) * np.sign(side[future])
    return bool(np.all(inside >= config.get('cone_inside_margin', .01)))


def avoidance_steering(scan, lane, robot, config, previous, refine=False):
    """Check short Ackermann paths with the real rectangular vehicle footprint."""
    points = np.asarray(scan.get('points', []), dtype=float).reshape(-1, 2).copy()
    if not len(points):
        return None
    points[:, 0] += robot.get('lidar', {}).get('x', 0.10)
    points[:, 1] += robot.get('lidar', {}).get('y', 0.0)
    horizon = config.get('planning_distance', 1.0)
    if scan.get('allow_edge_crossing', False):
        horizon = config.get('cone_planning_distance', .8)
    margin = config.get('planning_margin', .025)
    if scan.get('allow_edge_crossing', False):
        margin += config.get('small_object_clearance', .04)
    half_length = robot['length'] / 2
    half_width = robot['width'] / 2
    steps = int(config.get('planning_steps', 35))
    samples = int(config.get('planning_steering_samples', 17))
    return_samples = int(config.get('planning_return_samples', 9))
    if steps < 1 or samples < 3 or return_samples < 3 or samples % 2 == 0 or return_samples % 2 == 0:
        raise ValueError('planning_steps >= 1; steering sample counts must be odd and >= 3')
    step = horizon / steps
    heading_clip = config.get('planning_heading_clip', .8)
    heading = clip(lane.get('heading', 0.0), -heading_clip, heading_clip)
    cones = np.empty((0, 2))
    if (scan.get('allow_edge_crossing', False) and lane.get('road_memory', False) and
            config.get('cone_inside_required', True)):
        centers = np.asarray(scan.get('narrow_centers', []), dtype=float).reshape(-1, 2).copy()
        centers[:, 0] += robot.get('lidar', {}).get('x', .10)
        centers[:, 1] += robot.get('lidar', {}).get('y', 0.0)
        cones = np.column_stack((centers[:, 0]*math.cos(heading) + centers[:, 1]*math.sin(heading),
                                 -centers[:, 0]*math.sin(heading) + centers[:, 1]*math.cos(heading)))
    center_at_origin = lane['error'] - math.tan(heading) * lane.get('forward', .65)
    center_fit = lane.get('center_fit', [0.0, math.tan(heading), center_at_origin])
    solid = lane.get('solid', {})
    candidates = []
    preview = min(horizon, config.get('planning_reference_distance', .65))
    target_y = center_fit[0]*preview*preview + center_fit[1]*preview + center_fit[2]
    reference_steering = math.atan2(2*robot['wheelbase']*target_y, preview*preview+target_y*target_y)
    steering_choices = np.linspace(-robot['max_steering'], robot['max_steering'], samples)
    return_choices = np.linspace(-robot['max_steering'], robot['max_steering'], return_samples)
    if refine:
        # A narrow safe corridor can fall between the regular steering samples.
        # Retry only a failed search; existing successful paths stay unchanged.
        fine = robot['max_steering'] / (samples-1)
        extra = np.clip([reference_steering, previous, -fine, fine],
                        -robot['max_steering'], robot['max_steering'])
        steering_choices = np.unique(np.r_[steering_choices, extra])
        return_choices = np.unique(np.r_[return_choices, extra])
    first_distances = [config.get('planning_turn_distance', .40)]
    if scan.get('allow_edge_crossing', False):
        first_distances = config.get('cone_turn_distances', [.10, .25, .40])
    if not first_distances or any(distance <= 0 or distance > horizon for distance in first_distances):
        raise ValueError('Arc switch distances must be positive and within the planning horizon')
    for first_distance in first_distances:
        for steering in steering_choices:
            curvature = math.tan(steering) / robot['wheelbase']
            for next_steering in return_choices:
                return_curvature = math.tan(next_steering) / robot['wheelbase']
                x, y, theta = 0.0, 0.0, 0.0
                path_x, path_y, path_theta = [], [], []
                for i in range(steps):
                    k = curvature if i * step < first_distance else return_curvature
                    theta += k * step
                    x += math.cos(theta) * step
                    y += math.sin(theta) * step
                    path_x.append(x + robot['wheelbase']/2 * math.cos(theta))
                    path_y.append(y + robot['wheelbase']/2 * math.sin(theta))
                    path_theta.append(theta)
                end = clip(x, lane.get('fit_near', .2), lane.get('fit_far', horizon))
                goal_heading = math.atan(2*center_fit[0]*end + center_fit[1])
                if abs(theta - goal_heading) > config.get('planning_heading_limit', 1.2):
                    continue
                if lane.get('road_memory', False) and max(abs(t-heading) for t in path_theta) > max(config.get('road_heading_limit', .75), abs(heading)+config.get('road_heading_slack', .03)):
                    continue
                x, y, theta_path = np.asarray(path_x), np.asarray(path_y), np.asarray(path_theta)
                cosine, sine = np.cos(theta_path), np.sin(theta_path)
                road_offset = y * math.cos(heading) - x * math.sin(heading)
                if np.max(np.abs(road_offset)) > config.get('lane_change_limit', .65):
                    continue
                if len(cones):
                    forward = np.r_[robot['wheelbase']/2*math.cos(heading),
                                    x*math.cos(heading) + y*math.sin(heading)]
                    lateral = np.r_[-robot['wheelbase']/2*math.sin(heading), road_offset]
                    if not passes_inside_cones(forward, lateral, cones, lane['road_center'], config):
                        continue
                # Use measured curved boundaries. A tangent at the preview point
                # extrapolated back to the car can incorrectly exclude every path.
                inside_lane = True
                if lane.get('road_memory', False) and 'road_center' in lane and not scan.get('allow_edge_crossing', False):
                    relative = theta_path-heading
                    extent = half_length*np.abs(np.sin(relative)) + (half_width+margin)*np.abs(np.cos(relative))
                    if np.any(np.abs(road_offset-lane['road_center'])+extent > config.get('road_half_width', .78)):
                        continue
                elif not scan.get('allow_edge_crossing', False):
                    for side, boundary in lane.get('boundaries', {}).items():
                        if not solid.get(side, False):
                            continue
                        for longitudinal in [-half_length, half_length]:
                            for lateral in [-half_width-margin, half_width+margin]:
                                corner_x = x + longitudinal*cosine - lateral*sine
                                corner_y = y + longitudinal*sine + lateral*cosine
                                measured = (corner_x >= boundary['near']) & (corner_x <= boundary['far'])
                                distance = corner_y - np.polyval(boundary['fit'], corner_x)
                                outside = distance > 0 if side == 'left' else distance < 0
                                if np.any(outside & measured):
                                    inside_lane = False
                    if not inside_lane:
                        continue
                # Transform each lidar hit into each future body frame. This keeps
                # the narrow .22 m body passable in a .30 m gap between parked cars.
                dx = points[None, :, 0] - x[:, None]
                dy = points[None, :, 1] - y[:, None]
                longitudinal = np.abs(dx*cosine[:, None] + dy*sine[:, None]) - half_length
                lateral = np.abs(-dx*sine[:, None] + dy*cosine[:, None]) - half_width
                separation = np.sqrt(np.maximum(longitudinal, 0)**2 + np.maximum(lateral, 0)**2)
                clearance = float(separation.min())
                if clearance < margin:
                    continue
                deviation = y[-1] - (center_fit[0]*end*end + center_fit[1]*end + center_fit[2])
                cost = (config.get('planning_cost_heading', 1.0)*(theta-goal_heading)**2 +
                        config.get('planning_cost_lateral', 2.0)*deviation**2 +
                        config.get('planning_cost_steering_change', .8)*(steering-previous)**2 +
                        config.get('planning_cost_return_steering', .6)*next_steering**2 +
                        config.get('planning_cost_reference', 1.5)*(steering-reference_steering)**2 -
                        config.get('planning_cost_clearance', .1)*min(clearance, config.get('planning_clearance_cap', .5)))
                candidates.append((cost, steering))
    if (not candidates and not refine and not scan.get('allow_edge_crossing', False) and
            config.get('planning_refine_on_failure', True)):
        return avoidance_steering(scan, lane, robot, config, previous, refine=True)
    return None if not candidates else float(min(candidates)[1])


def sensor_status(config, state, ages, available):
    """Require the inputs used by this state, keeping lidar/odometry mandatory."""
    timed_parking = state in ('parking_sequence', 'parked')
    required_camera = not timed_parking or config['parking'].get('require_camera_during_sequence', False)
    ready = (available['scan'] and 0 <= ages['scan'] < config['sensor_timeout'] and
             0 <= ages['odometry'] < config['odometry_timeout'])
    if required_camera:
        ready = ready and available['image'] and 0 <= ages['image'] < config['sensor_timeout']
        if config['detector']['enabled']:
            ready = ready and available['detector'] and 0 <= ages['detector'] < config['detector']['timeout']
    else:
        # The timed sequence must have been entered through camera alignment.
        ready = ready and available['lane']
    return bool(ready), required_camera


class Driver:
    """Perception/state machine. Absolute vehicle pose is intentionally unused."""
    def __init__(self, config):
        self.c = config['answer']
        self.robot = config['robot']
        self.pid = PID(self.c['pid'])
        self.state = 'lane'
        self.state_started = 0.0
        self.last_time = None
        self.last_lane_time = -1e9
        self.last_steering = 0.0
        self.last_heading = 0.0
        self.last_lane_error = 0.0
        self.last_lane_forward = .65
        self.lost_lane_distance = 0.0
        self.slow_until = 0.0
        self.fork_remaining = 0.0
        self.fork_bias = 0.0
        self.fork_target = 0.0
        self.fork_pending = False
        self.fork_clear_distance = 0.0
        self.cooldowns = {}
        self.stopped_since = None
        self.traffic_stop_completed = False
        self.line_distance = None
        self.line_seen = -1e9
        self.align_since = None
        self.sequence_index = 0
        self.sequence_started = 0.0
        self.sign_counts = {}
        self.last_detection_stamp = None
        self.last_lane_stamp = None
        self.stage_index = 0
        self.avoidance_previous = 0.0
        self.avoiding = False
        self.avoidance_clear_distance = 0.0
        self.road_memory = False
        self.road_heading = 0.0
        self.road_center = 0.0
        self.cone_remaining = 0.0
        self.debug = {}

    def transition(self, state, now):
        self.state = state
        self.state_started = now
        self.stopped_since = None
        self.align_since = None
        self.pid.reset()
        if state != 'lane':
            self.avoiding = False
            self.road_memory = False
            self.cone_remaining = 0.0

    def sensor_failure(self, now):
        self.stopped_since = None
        self.align_since = None
        if self.state == 'parking_sequence':
            self.transition('fault: sensor lost during parking', now)

    def stage(self):
        sequence = self.c['missions']['sequence']
        return sequence[self.stage_index] if self.stage_index < len(sequence) else 'complete'

    def allowed(self, mission):
        return not self.c['missions']['enforce_order'] or self.stage() == mission

    def advance(self, mission):
        if self.stage() == mission:
            self.stage_index += 1

    def confirmed(self, detections, stamp):
        truncated_allowed = self.c['detector'].get('allow_truncated_classes', ['traffic_red', 'traffic_yellow', 'traffic_green'])
        names = {d['name'] for d in detections
                 if (not d.get('truncated', False) or d['name'] in truncated_allowed) and d['area'] >= self.c['detector'].get('minimum_area_by_class', {}).get(d['name'], self.c['detector']['minimum_area'])}
        if stamp != self.last_detection_stamp:
            self.sign_counts = {name: self.sign_counts.get(name, 0) + 1 for name in names}
            self.last_detection_stamp = stamp
        return {name for name in names
                if self.sign_counts.get(name, 0) >= self.c['detector']['confirm_frames']}

    def update(self, lane, scan, detections, detection_stamp, velocity, now, angular_velocity=0.0):
        c = self.c
        m = c['missions']
        dt = 0.05 if self.last_time is None else clip(now - self.last_time, 0.001, 0.2)
        if self.last_time is not None and now < self.last_time:
            self.__init__({'answer': self.c, 'robot': self.robot})
        self.last_time = now
        if self.road_memory:
            self.road_heading -= angular_velocity * dt
            self.road_center += velocity * math.sin(self.road_heading) * dt
        self.cone_remaining = max(0.0, self.cone_remaining-max(0.0, velocity)*dt)
        self.fork_remaining = max(0.0, self.fork_remaining - max(0.0, velocity) * dt)
        if self.line_distance is not None:
            self.line_distance -= velocity * dt
        image_stamp = lane.get('stamp', now)
        new_image = image_stamp != self.last_lane_stamp
        self.last_lane_stamp = image_stamp
        if new_image and lane['stop_distance'] is not None:
            # A delayed image measures the stripe at exposure, not at this tick.
            candidate = lane['stop_distance'] - velocity * max(0.0, now-image_stamp)
            tracking_stop = self.state.endswith('_approach') or self.state.endswith('_hold') or self.state == 'parking_align'
            # Once the first stop stripe is selected, do not jump to a farther
            # crosswalk stripe when the first leaves the camera's field of view.
            same_line = self.line_distance is None or abs(candidate - self.line_distance) < m['line_match_tolerance']
            if not tracking_stop or same_line:
                self.line_distance = candidate
                self.line_seen = image_stamp
        if (now - self.line_seen > m['stop_line_memory'] and
                not self.state.endswith(('_hold', '_approach')) and self.state != 'parking_align'):
            self.line_distance = None
        if self.state == 'lane' and self.line_distance is not None and self.line_distance < -0.05:
            self.line_distance = None
        names = self.confirmed(detections, detection_stamp)
        ready = {name for name in names if now >= self.cooldowns.get(name, 0)}
        green_only = 'traffic_green' in names and not ({'traffic_red', 'traffic_yellow'} & names)
        if 'slow' in ready and self.allowed('slow'):
            self.advance('slow')
            self.slow_until = now + m['slow_seconds']
            self.cooldowns['slow'] = now + m['sign_cooldown']
        if ('left' in ready or 'right' in ready) and self.allowed('fork') and self.fork_remaining <= 0.0:
            self.advance('fork')
            self.fork_target = m['fork_bias'] if 'left' in ready else -m['fork_bias']
            self.fork_pending = m.get('fork_wait_for_obstacles', True)
            self.fork_clear_distance = 0.0
            self.fork_bias = 0.0 if self.fork_pending else self.fork_target
            self.fork_remaining = m.get('fork_distance', 1.6)
            self.cooldowns['left'] = self.cooldowns['right'] = now + m['sign_cooldown']
        # Read the arrow early, but finish the parked-car corridor before
        # changing sides. Require measured travel with no wide object ahead.
        if self.fork_pending:
            if scan.get('wide_objects', 0) > 0:
                self.fork_clear_distance = 0.0
            else:
                self.fork_clear_distance += max(0.0, velocity) * dt
            if self.fork_clear_distance >= m.get('fork_clear_distance', .20):
                self.fork_bias = self.fork_target
                self.fork_pending = False
        if self.fork_remaining <= 0.0:
            self.fork_bias = 0.0
            self.fork_target = 0.0
            self.fork_pending = False
        if self.state == 'lane':
            if ('traffic_red' in ready or 'traffic_yellow' in ready) and self.allowed('traffic'):
                self.traffic_stop_completed = False
                self.transition('traffic_approach', now)
            elif green_only and 'traffic_green' in ready and self.allowed('traffic'):
                self.traffic_stop_completed = False
                self.transition('traffic_approach', now)
            elif 'pedestrian' in ready and self.allowed('pedestrian'):
                fresh_line = (self.line_distance is not None and m['stop_offset'] < self.line_distance < m.get('trigger_line_max_distance', 1.5) and
                              now - self.line_seen < m.get('trigger_line_freshness', .4))
                if fresh_line or not m.get('pedestrian_require_line', True):
                    self.transition('pedestrian_approach', now)
            elif 'parking' in ready and c['parking']['enabled'] and self.allowed('parking'):
                self.transition('parking_align', now)
        if lane['valid']:
            self.last_lane_time = image_stamp
            self.last_lane_error = lane['error']
            self.last_lane_forward = lane.get('forward', c['lane']['lookahead'])
            self.lost_lane_distance = 0.0
            alpha = c['lane'].get('heading_smoothing', .2)
            self.last_heading = (1-alpha)*self.last_heading + alpha*lane['heading']
            error = lane['error']
            if lane['abnormal']:
                error *= c['lane'].get('abnormal_error_scale', 1.0)
            preview = lane.get('forward', c['lane']['lookahead'])
            geometric = math.atan2(2 * self.robot['wheelbase'] * error, preview * preview + error * error)
            steering = geometric + c['pid'].get('feedback_gain', 0.35) * self.pid.update(error, dt)
            steering += c['lane']['heading_gain'] * self.last_heading
            self.last_steering = clip(steering, -self.robot['max_steering'], self.robot['max_steering'])
        steering = self.last_steering
        speed = max(c['minimum_curve_speed'], c['base_speed'] * (1 - c.get('curve_speed_gain', .6) * abs(steering) / self.robot['max_steering']))
        if now < self.slow_until or lane['abnormal']:
            speed = min(speed, c['slow_speed'], c['base_speed'] / 2, 0.2)
        if not lane['valid']:
            self.lost_lane_distance += abs(velocity) * dt
            speed = min(speed, c['slow_speed'])
            if now - self.last_lane_time > c['lane']['lost_grace']:
                speed = 0.0
        if self.state.endswith('_approach'):
            crossing_on_green = (self.state == 'traffic_approach' and green_only and
                                 (not m.get('traffic_require_stop', True) or self.traffic_stop_completed))
            if now - self.state_started > m['approach_timeout']:
                self.transition('fault: stop line not reached', now)
            if self.line_distance is None:
                speed = min(speed, m['stop_approach_speed'])
                if now - self.state_started > m['line_missing_timeout']:
                    speed = 0.0
            elif crossing_on_green:
                # Seeing green far away is not mission completion. Keep watching
                # red/yellow until the measured front axle crosses this stripe.
                if self.line_distance < -m['stop_tolerance']:
                    for name in ['traffic_red', 'traffic_yellow', 'traffic_green']:
                        self.cooldowns[name] = now + m['sign_cooldown']
                    self.advance('traffic')
                    self.line_distance = None
                    self.transition('lane', now)
            else:
                remaining = self.line_distance - m['stop_offset']
                speed = min(speed, m['stop_approach_speed'], max(0.0, remaining * m['stop_gain']))
                if remaining <= m['stop_tolerance']:
                    speed = 0.0
                    if abs(velocity) <= m['stopped_speed']:
                        self.transition(self.state.replace('_approach', '_hold'), now)
        if self.state == 'pedestrian_hold':
            speed = 0.0
            if abs(velocity) > m['stopped_speed']:
                self.stopped_since = None
            elif self.stopped_since is None:
                self.stopped_since = now
            elif now - self.stopped_since >= max(3.0, m['pedestrian_wait']):
                self.cooldowns['pedestrian'] = now + m['sign_cooldown']
                self.advance('pedestrian')
                self.line_distance = None
                self.transition('lane', now)
        if self.state == 'traffic_hold':
            speed = 0.0
            if abs(velocity) > m['stopped_speed']:
                self.stopped_since = None
            elif self.stopped_since is None:
                self.stopped_since = now
            minimum_stop = m.get('traffic_min_stop', .3) if m.get('traffic_require_stop', True) else 0.0
            if self.stopped_since is not None and now - self.stopped_since >= minimum_stop:
                self.traffic_stop_completed = True
            if green_only and self.traffic_stop_completed:
                self.transition('traffic_approach', now)
        if self.state == 'parking_align':
            p = c['parking']
            speed = 0.0
            if now - self.state_started > p['align_timeout']:
                self.transition('fault: parking alignment timeout', now)
            elif lane['valid'] and self.line_distance is not None:
                distance_error = self.line_distance - p['line_distance']
                speed = clip(distance_error * 0.8, -p['align_speed'], p['align_speed'])
                aligned = (abs(distance_error) <= p['distance_tolerance'] and
                           abs(lane['error']) <= p['lateral_tolerance'] and
                           abs(lane['heading']) <= p['heading_tolerance'])
                if aligned:
                    speed = 0.0
                    if abs(velocity) <= m['stopped_speed']:
                        if self.align_since is None:
                            self.align_since = now
                        elif now - self.align_since >= p['stable_seconds']:
                            self.sequence_index = 0
                            self.sequence_started = now
                            self.transition('parking_sequence', now)
                else:
                    self.align_since = None
        if self.state == 'parking_sequence':
            sequence = c['parking']['sequence']
            while self.sequence_index < len(sequence) and now - self.sequence_started >= sequence[self.sequence_index]['seconds']:
                self.sequence_started += sequence[self.sequence_index]['seconds']
                self.sequence_index += 1
            if self.sequence_index >= len(sequence):
                self.advance('parking')
                self.transition('parked', now)
                speed = 0.0
            else:
                step = sequence[self.sequence_index]
                speed, steering = step['speed'], step['steering']
        if self.state.startswith('fault') or self.state == 'parked':
            speed = 0.0
        # Lidar runs for all missions, including reversing during parking.
        # Cones may legally be bypassed outside the painted lane. Trigger this
        # only after the fork and from measured narrow lidar clusters. Their
        # wider base needs more clearance than the thin slice at lidar height.
        scan = dict(scan)
        cone_hint = (self.state == 'lane' and self.stage() == 'traffic' and self.fork_remaining <= 0 and
                     scan.get('narrow_objects', 0) > 0 and scan.get('wide_objects', 0) == 0)
        if cone_hint:
            self.cone_remaining = c['lidar'].get('small_object_memory_distance', .8)
        scan['allow_edge_crossing'] = self.stage() == 'traffic' and self.cone_remaining > 0.0
        blocked = lane_obstructed(scan, lane, self.robot, c['lidar'])
        if blocked:
            remember_road = (not self.avoiding and scan.get('wide_objects', 0) > 0 and
                             (self.allowed('fork') or self.fork_remaining > 0))
            if (remember_road or scan['allow_edge_crossing']) and not self.road_memory:
                self.road_memory = True
                self.road_heading = lane['heading'] if lane['valid'] else self.last_heading
                if lane.get('dash_heading') is not None:
                    self.road_heading = lane['dash_heading']
                self.road_center = self.last_lane_error*math.cos(self.road_heading) - self.last_lane_forward*math.sin(self.road_heading)
            self.avoiding = True
            self.avoidance_clear_distance = 0.0
        elif self.avoiding:
            self.avoidance_clear_distance += max(0.0, velocity) * dt
            aligned = not self.road_memory or (abs(self.road_heading) < c['lidar'].get('road_release_heading', .18) and
                                                   abs(lane['error']) < c['lidar'].get('road_release_error', .30))
            if self.road_memory and self.fork_bias != 0:
                aligned = (abs(self.road_heading) < c['lidar'].get('road_release_heading', .18) and
                           abs(self.road_center+self.fork_bias) < m.get('fork_align_tolerance', .12))
            if self.road_memory and self.fork_pending:
                aligned = False
            if aligned and self.avoidance_clear_distance >= c['lidar'].get('release_distance', .25):
                self.avoiding = False
                self.road_memory = False
        if (self.road_memory and self.avoiding and not lane['valid'] and self.state == 'lane' and
                self.lost_lane_distance < c['lidar'].get('vision_memory_distance', 2.0)):
            speed = c['lidar']['avoidance_speed']
        planner_failed = False
        if speed > 0 and self.avoiding and self.state == 'lane':
            planning_lane = lane
            if self.road_memory:
                # Camera boundary identities can change while a parked car hides
                # paint. Remember only the local road direction and lateral line,
                # propagated with odometry velocities; never read absolute pose.
                heading = self.road_heading
                cosine = max(.2, math.cos(heading))
                fit = [0.0, math.tan(heading), self.road_center/cosine]
                half_road = c['lidar'].get('road_half_width', .78)
                boundaries = {side: dict(fit=[0.0, fit[1], fit[2]+sign*half_road/cosine], near=0.0, far=2.0)
                              for side, sign in [('left', 1), ('right', -1)]}
                desired = 0.0
                if not scan['allow_edge_crossing']:
                    desired = self.fork_bias if self.fork_bias != 0 else c['lidar'].get('road_default_bias', .30)
                    desired = road_gap(scan, self.robot, heading, self.road_center, desired, c['lidar'])
                target_fit = [fit[0], fit[1], fit[2]+desired/cosine]
                planning_lane = dict(lane, heading=heading, center_fit=target_fit, fit_near=0.0, fit_far=2.0,
                                     solid=dict(left=True, right=True), boundaries=boundaries, road_memory=True, road_center=self.road_center)
            planned = avoidance_steering(scan, planning_lane, self.robot, c['lidar'], self.avoidance_previous)
            if planned is None:
                if not blocked and lane['valid'] and not self.road_memory:
                    self.avoiding = False
                else:
                    speed = 0.0
                    planner_failed = True
            else:
                steering = planned
                self.avoidance_previous = planned
                speed = min(speed, c['lidar']['avoidance_speed'])
        clearance = scan['front'] if speed >= 0 else scan['rear']
        if scan['valid_fraction'] < 0.5 or clearance < c['lidar']['emergency_distance']:
            if self.state == 'parking_sequence':
                # A paused time schedule must never resume at a later step.
                self.transition('fault: obstacle during parking', now)
            speed = 0.0
        self.debug = dict(lane_valid=lane['valid'], lane_error=lane['error'], lane_heading=lane['heading'],
                          pid=dict(self.pid.terms), pid_error=self.pid.previous,
                          pid_feedback_gain=c['pid'].get('feedback_gain', .35),
                          dash_heading=lane.get('dash_heading'),
                          traffic_stop_completed=self.traffic_stop_completed,
                          lane_age=now-self.last_lane_time, front=scan['front'], rear=scan['rear'],
                          blocked=blocked, avoiding=self.avoiding, planner_failed=planner_failed,
                          cone_clearance=scan['allow_edge_crossing'], narrow_objects=scan.get('narrow_objects', 0),
                          wide_objects=scan.get('wide_objects', 0),
                          narrow_centers=scan.get('narrow_centers', []),
                          road_memory=self.road_memory, road_heading=self.road_heading, road_center=self.road_center,
                          lost_lane_distance=self.lost_lane_distance, fork_remaining=self.fork_remaining, fork_bias=self.fork_bias,
                          fork_pending=self.fork_pending, fork_target=self.fork_target,
                          line_distance=self.line_distance, stop_candidates=lane.get('stop_candidates', []), stage=self.stage(), speed=float(speed))
        return float(speed), clip(steering, -self.robot['max_steering'], self.robot['max_steering']), self.state


def main(config_path=None):
    # Keep ROS imports here so pure perception/control tests run without ROS installed.
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from ackermann_msgs.msg import AckermannDriveStamped
    from sensor_msgs.msg import Image, LaserScan
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from cv_bridge import CvBridge

    if config_path is None:
        if len(sys.argv) != 2:
            raise SystemExit('사용법: scripts/answer.sh configs/config.yaml')
        config_path = sys.argv[1]
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text())
    config['answer'] = ANSWER_CONFIG
    c = config['answer']
    detector = None
    if c['detector']['enabled']:
        weights = Path(__file__).with_name('best.pt')
        if not weights.is_file():
            raise SystemExit(f'YOLO 모델이 없습니다: {weights}\nanswer.py 옆에 best.pt 파일을 넣으세요.')
        from ultralytics import YOLO
        detector = YOLO(str(weights))
        import torch
        threads = int(c['detector'].get('cpu_threads', config['environment'].get('cpu_threads', 2)))
        torch.set_num_threads(threads)
        detector.add_callback('on_predict_start', lambda _: torch.set_num_threads(threads))
        expected = set(YOLO_CLASSES)
        if set(detector.names.values()) != expected:
            raise SystemExit('모델의 클래스 이름은 answer.py의 YOLO_CLASSES 8종과 같아야 합니다. best.pt를 확인하세요.')

    class AnswerNode(Node):
        def __init__(self):
            super().__init__('orda_answer', parameter_overrides=[Parameter('use_sim_time', value=True)])
            self.bridge = CvBridge()
            self.driver = Driver(config)
            self.vision = LaneVision(c['lane'], config['robot']['camera'], config['robot']['front_axle_x'])
            self.image = None
            self.image_stamp = -1e9
            self.lane_stamp = None
            self.lane_bias = None
            self.lane_cache = None
            self.scan = None
            self.scan_stamp = -1e9
            self.velocity = 0.0
            self.angular_velocity = 0.0
            self.odom_stamp = -1e9
            self.detections = []
            self.detection_stamp = -1e9
            self.lock = threading.Lock()
            self.stop_event = threading.Event()
            self.last_state = ''
            topics = config['topics']
            self.publisher = self.create_publisher(AckermannDriveStamped, topics['drive'], 10)
            self.status = self.create_publisher(String, topics.get('status', '/answer/status'), 10)
            self.debug = self.create_publisher(Image, topics.get('debug_image', '/answer/debug_image'), qos_profile_sensor_data)
            self.debug_stamp = -1e9
            self.detection_debug_stamp = -1e9
            self.detection_output = self.create_publisher(String, topics.get('detections', '/answer/detections'), 10)
            self.create_subscription(Image, topics['image'], self.on_image, qos_profile_sensor_data)
            self.create_subscription(LaserScan, topics['scan'], self.on_scan, qos_profile_sensor_data)
            self.create_subscription(Odometry, topics['odometry'], self.on_odometry, qos_profile_sensor_data)
            self.timer = self.create_timer(1.0 / c['rate_hz'], self.control)
            if detector is not None:
                self.worker = threading.Thread(target=self.detect, daemon=True)
                self.worker.start()
            self.get_logger().info('정답 코드 시작: YOLO + camera lane + lidar; absolute pose is unused.')

        def now(self):
            return self.get_clock().now().nanoseconds / 1e9

        def stamp(self, message):
            return message.header.stamp.sec + message.header.stamp.nanosec / 1e9

        def on_image(self, message):
            try:
                frame = self.bridge.imgmsg_to_cv2(message, 'bgr8')
                with self.lock:
                    self.image = frame
                    self.image_stamp = self.stamp(message)
            except Exception as error:
                self.get_logger().error(f'image conversion failed: {error}')

        def on_scan(self, message):
            self.scan = scan_sectors(message.ranges, message.angle_min, message.angle_increment,
                                     message.range_min, message.range_max, c['lidar'])
            self.scan_stamp = self.stamp(message)

        def on_odometry(self, message):
            self.velocity = message.twist.twist.linear.x
            self.angular_velocity = message.twist.twist.angular.z
            self.odom_stamp = self.stamp(message)

        def detect(self):
            last_stamp = None
            while not self.stop_event.is_set():
                with self.lock:
                    frame = None if self.image is None else self.image.copy()
                    stamp = self.image_stamp
                if frame is not None and stamp != last_stamp:
                    last_stamp = stamp
                    try:
                        results = detector.predict(frame, conf=c['detector']['confidence'],
                                                   imgsz=c['detector']['image_size'],
                                                   device=c['detector']['device'], verbose=False)[0]
                        torch.set_num_threads(threads)
                        objects = []
                        h, w = frame.shape[:2]
                        for box in results.boxes:
                            x0, y0, x1, y1 = box.xyxy[0].cpu().tolist()
                            margin = c['detector'].get('edge_margin', .01)
                            truncated = x0 <= margin*w or x1 >= (1-margin)*w or y0 <= margin*h or y1 >= (1-margin)*h
                            objects.append(dict(name=results.names[int(box.cls[0])],
                                                bbox=[x0/w, y0/h, x1/w, y1/h], truncated=truncated,
                                                confidence=float(box.conf[0]),
                                                area=(x1 - x0) * (y1 - y0) / (w * h),
                                                center=((x0 + x1) / (2 * w), (y0 + y1) / (2 * h))))
                        with self.lock:
                            self.detections = objects
                            self.detection_stamp = stamp
                    except Exception as error:
                        self.get_logger().error(f'YOLO failed: {error}')
                        self.stop_event.set()
                self.stop_event.wait(1.0 / c['detector']['rate_hz'])

        def publish(self, speed, steering):
            message = AckermannDriveStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = 'base_link'
            message.drive.speed = float(speed)
            message.drive.steering_angle = float(steering)
            self.publisher.publish(message)

        def control(self):
            now = self.now()
            with self.lock:
                frame = self.image
                image_stamp = self.image_stamp
                detections = self.detections
                detection_stamp = self.detection_stamp
            ages = dict(image=now-image_stamp, scan=now-self.scan_stamp,
                        odometry=now-self.odom_stamp, detector=now-detection_stamp)
            available = dict(image=frame is not None, scan=self.scan is not None,
                             lane=self.lane_cache is not None, detector=not self.stop_event.is_set())
            sensor_ready, required_camera = sensor_status(c, self.driver.state, ages, available)
            self.driver.debug['required_camera'] = required_camera
            if not sensor_ready or not math.isfinite(self.velocity) or not math.isfinite(self.angular_velocity):
                self.publish(0.0, 0.0)
                self.driver.debug['published_speed'] = 0.0
                self.driver.debug['speed'] = 0.0
                speed, steering = 0.0, 0.0
                self.driver.sensor_failure(now)
                state = 'waiting for fresh sensors / detector'
            else:
                if required_camera and (image_stamp != self.lane_stamp or self.driver.fork_bias != self.lane_bias):
                    self.lane_cache = self.vision.analyze(frame, self.driver.fork_bias)
                    self.lane_cache['stamp'] = image_stamp
                    self.lane_stamp = image_stamp
                    self.lane_bias = self.driver.fork_bias
                lane = self.lane_cache
                was_avoiding = self.driver.avoiding
                speed, steering, state = self.driver.update(lane, self.scan, detections,
                                                             detection_stamp, self.velocity, now, self.angular_velocity)
                if was_avoiding and not self.driver.avoiding:
                    self.vision.previous_fits = {}
                    self.vision.previous_boundaries = {}
                    self.vision.previous_center = None
                    self.lane_stamp = None
                if self.driver.state != 'parking_sequence':
                    age = max(now-image_stamp, .001)
                    limit = c.get('max_image_travel', .04) / age
                    speed = clip(speed, -limit, limit)
                self.publish(speed, steering)
                self.driver.debug['published_speed'] = speed
                self.driver.debug['speed'] = speed
                self.driver.debug['required_camera'] = required_camera
                if now - self.debug_stamp >= 1.0 / c['debug_rate_hz']:
                    message = self.bridge.cv2_to_imgmsg(lane['lane_mask'], 'mono8')
                    stamp = float(lane['stamp'])
                    message.header.stamp.sec = int(stamp)
                    message.header.stamp.nanosec = int((stamp-int(stamp))*1e9)
                    self.debug.publish(message)
                    self.debug_stamp = now
            # Telemetry follows the actual published command, including safety stops.
            if now - self.detection_debug_stamp >= 1.0 / c['debug_rate_hz']:
                lane = self.lane_cache or {}
                keys = ('valid', 'error', 'heading', 'center_px', 'forward', 'windows',
                        'fits', 'fit_spans', 'center_fit', 'fit_near', 'fit_far')
                control = dict(self.driver.debug, steering=steering, speed=speed,
                               state=state, sensor_ready=sensor_ready,
                               controller_state=self.driver.state)
                packet = dict(stamp=detection_stamp, age=now-detection_stamp, objects=detections,
                              control=control, image_age=now-image_stamp, scan_age=now-self.scan_stamp,
                              image_stamp=image_stamp, lane_stamp=lane.get('stamp'),
                              lane={key: lane[key] for key in keys if key in lane},
                              display=dict(lane=c['lane'], pid=c['pid'], detector=dict(timeout=c['detector']['timeout']),
                                           sensor_timeout=c['sensor_timeout']))
                self.detection_output.publish(String(data=json.dumps(packet)))
                self.detection_debug_stamp = now
            if state != self.last_state:
                self.get_logger().info(state)
                self.status.publish(String(data=state))
                self.last_state = state

    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = AnswerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_event.set()
        if rclpy.ok():
            node.publish(0.0, 0.0)
        if hasattr(node, 'worker'):
            node.worker.join(timeout=c['detector'].get('shutdown_timeout', 5.0))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
