"""Traffic scoring uses actual at-line stops, not an earlier red observation."""
from pathlib import Path

import yaml

from orda_sim.evaluate import Evaluation


def traffic_evaluation(required, samples):
    config = yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/config.yaml').read_text())
    config['answer']['missions']['traffic_require_stop'] = required
    evaluation = Evaluation(config)
    evaluation.started = 0.0
    traffic = config['evaluation']['traffic']
    for time, distance, speed, light in samples:
        front = [traffic['line_point'][i]-distance*traffic['direction'][i] for i in range(2)]
        evaluation.last_time = time
        evaluation.record_stop('traffic', front, time, speed, {'traffic': light}, True)
    return evaluation.report()


def test_mandatory_stop_rejects_green_arrival_without_stopping():
    report = traffic_evaluation(True, [(0,.1,.1,'traffic_green'),(1,-.1,.1,'traffic_green')])
    assert report['criteria']['traffic_crossed_on_green']
    assert not report['criteria']['traffic_stop_at_line']
    assert report['traffic_stop_policy']['required_at_line']


def test_mandatory_stop_accepts_measured_stop_even_if_arrival_is_green():
    report = traffic_evaluation(True, [(0,.02,0,'traffic_green'),(.4,.02,0,'traffic_green'),
                                       (1,-.1,.1,'traffic_green')])
    assert report['criteria']['traffic_stop_at_line']
    assert report['criteria']['traffic_crossed_on_green']
    assert report['traffic']['best_hold'] >= .4


def test_conditional_mode_scores_green_crossing_without_red_latch_penalty():
    report = traffic_evaluation(False, [(0,.4,.1,'traffic_red'),(1,.1,.1,'traffic_green'),
                                        (2,-.1,.1,'traffic_green')])
    assert report['red_or_yellow_encountered_on_approach']
    assert report['criteria']['traffic_crossed_on_green']
    assert 'traffic_stop_at_line' not in report['criteria']
    assert not report['traffic_stop_policy']['required_at_line']
