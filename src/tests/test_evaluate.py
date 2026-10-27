import math
from orda_sim.evaluate import ConeRoute, footprint, in_rectangle, rectangles_overlap, point_to_segment

ROBOT = {'wheelbase': .2, 'length': .322, 'width': .22}


def test_parking_uses_all_four_corners_and_rear_axle_offset():
    corners = footprint(6.88, 4.054, math.pi/2, ROBOT)
    assert all(in_rectangle(c,[6.63,3.97,7.13,4.57]) for c in corners)
    # The model origin is inside, but one side of the body protrudes.
    bad = footprint(7.08,4.054,math.pi/2,ROBOT)
    assert in_rectangle([7.08,4.054],[6.63,3.97,7.13,4.57])
    assert not all(in_rectangle(c,[6.63,3.97,7.13,4.57]) for c in bad)


def test_collision_proxy_handles_rotated_vehicles():
    first = footprint(0,0,0,ROBOT)
    assert rectangles_overlap(first, footprint(.1,.05,math.pi/4,ROBOT))
    assert not rectangles_overlap(first, footprint(.7,.05,math.pi/4,ROBOT))


def test_corridor_distance_is_to_segment_not_infinite_line():
    assert abs(point_to_segment([2,0],[0,0],[1,0])-1)<1e-9
    assert abs(point_to_segment([.5,.3],[0,0],[1,0])-.3)<1e-9


def test_cone_gates_interpolate_body_center_and_do_not_skip_multiple_gates():
    route = ConeRoute([[.63,2.1],[.94,2.6],[.63,3.1],[.94,3.6]],
                      ['right','left','right','left'],[0,1.5],0,ROBOT)
    route.update(.8,1.6,math.pi/2,0)
    first = route.update(.8,2.85,math.pi/2,2)
    assert [item['index'] for item in first] == [0,1]
    # The rear axle starts at1.6, but the body centre starts at1.7.
    assert abs(first[0]['sim_time']-.64) < 1e-9
    assert abs(first[0]['body_center'][0]-.8) < 1e-9
    route.update(.8,3.8,math.pi/2,4)
    assert route.report()['passed']
    assert len(route.crossings) == 4


def test_cone_outer_bypass_and_missing_gates_cannot_pass():
    positions = [[.63,2.1],[.94,2.6],[.63,3.1],[.94,3.6]]
    sides = ['right','left','right','left']
    outside = ConeRoute(positions,sides,[0,1.5],0,ROBOT)
    outside.update(.38,1.6,math.pi/2,0)
    outside.update(.38,3.8,math.pi/2,4)
    assert outside.report()['wrong_side_gates'] == [0,2]
    assert not outside.report()['passed']
    missing = ConeRoute(positions,sides,[0,1.5],0,ROBOT)
    missing.update(.8,1.6,math.pi/2,0)
    missing.update(.8,2.85,math.pi/2,2)
    assert missing.report()['missing_gates'] == [2,3]
    assert not missing.report()['passed']
