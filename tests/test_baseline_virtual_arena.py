from types import SimpleNamespace
import math

import pytest

from baseline_virtual_arena import (
    FixedCornerArena, InitialCarArena, fixed_arena, initial_position_arena, order_rectangle,
    cage_walls, virtual_wall_force, segment_distance,
)
from baseline_physical import BaselineController, PhysicalCommandFilter, physical_parameters, herder_cage_path_allowed
from relative_cage_core import (
    ArenaGeometryError, LocalObservation, SecondOrderDynamics, vec_norm,
)


def initial_poses(offset=(0, 0)):
    return {i: (offset[0] + ((i - 1) % 5) * 400,
                offset[1] + ((i - 1) // 5) * 400) for i in range(1, 26)}


def arena_fixture():
    from relative_cage_core import RelativeArena
    return RelativeArena((0, 0), (1, 0), (0, 1), 4000, 4000, 1600, 1600, 0, 90)


@pytest.mark.parametrize("offset", [(0, 0), (12000, -9000), (-5000, 7000)])
def test_auto_field_fits_initial_cars_at_arbitrary_world_location(offset):
    poses = initial_poses(offset)
    arena, corners = initial_position_arena(poses, range(18, 26), range(1, 18))
    assert arena.width_mm == arena.height_mm == 4000
    assert arena.cage_width_mm == arena.cage_height_mm == 1600
    assert all(arena.contains_field_world(p, 300 - 1e-6) for p in poses.values())
    assert all(not arena.contains_cage_world(poses[i], -75) for i in range(18, 26))
    assert all(not arena.contains_cage_world(poses[i]) for i in range(1, 18))
    assert arena.world_to_local(corners.left_down) == pytest.approx((0, 0))
    assert arena.world_to_local(corners.right_up) == pytest.approx((4000, 4000))


def test_translation_changes_only_world_origin():
    a, _ = initial_position_arena(initial_poses(), range(18, 26), range(1, 18))
    b, _ = initial_position_arena(initial_poses((9000, -6000)), range(18, 26), range(1, 18))
    assert b.origin_world == pytest.approx((a.origin_world[0] + 9000, a.origin_world[1] - 6000))
    assert b.x_axis_world == a.x_axis_world
    assert b.y_axis_world == a.y_axis_world


def test_spread_out_initial_cars_do_not_silently_enlarge_the_field():
    with pytest.raises(ValueError, match="do not fit"):
        initial_position_arena({1: (0, 0), 18: (5000, 0)}, (18,), (1,))


def test_opening_requires_space_for_a_car_and_margin():
    with pytest.raises(ValueError, match="opening must exceed"):
        initial_position_arena(initial_poses(), range(18, 26), range(1, 18), opening_mm=150)


def test_virtual_walls_repel_on_both_sides_but_gate_stays_enterable():
    arena = arena_fixture()
    outside = virtual_wall_force((200, 2300), arena, 700, 300, 900, 75)
    inside = virtual_wall_force((200, 2500), arena, 700, 300, 900, 75)
    gate = virtual_wall_force((800, 2400), arena, 700, 300, 900, 75)
    assert outside[1] < 0
    assert inside[1] > 0
    assert vec_norm(gate) < 1e-9


def test_predicted_crossing_of_closed_wall_is_rejected_but_gate_is_open():
    arena = arena_fixture()
    guard = PhysicalCommandFilter(cage_opening_mm=700)
    assert not guard.safe_velocity(1, (0, 200), {1: (200, 2300)}, {}, {}, arena, 75)
    assert guard.safe_velocity(1, (0, 200), {1: (800, 2300)}, {}, {}, arena, 75)
    assert segment_distance((200, 2300), (200, 2500), *cage_walls(arena, 700)[2]) == 0


def test_herder_inside_target_can_leave_and_cannot_go_deeper():
    arena = arena_fixture()
    assert herder_cage_path_allowed((1400, 3200), (1700, 3200), arena, 115)
    assert not herder_cage_path_allowed((1400, 3200), (1200, 3200), arena, 115)
    assert not herder_cage_path_allowed((1800, 3200), (1400, 3200), arena, 115)


def test_controller_repels_herder_from_virtual_field_boundary():
    arena = arena_fixture()
    params = physical_parameters(arena, 75)
    model = BaselineController(cage_opening_mm=700)
    model.elapsed = 2
    observations = {18: LocalObservation(18, (3900, 1000), 0, 0),
                    1: LocalObservation(1, (3400, 1000), 0, 0)}
    result = model(observations, (18,), (1,), (), arena, params, SecondOrderDynamics(), .05)
    assert result.accelerations[18][0] < 0
    assert vec_norm(result.velocities[18]) <= 160
    assert vec_norm(result.velocities[1]) <= 240


def test_filter_preserves_pwm_ramp_out_of_motor_dead_zone():
    import relative_cage_capture as runtime
    arena = arena_fixture()
    params = physical_parameters(arena, 75)
    guard = PhysicalCommandFilter(cage_opening_mm=700)
    controller = SimpleNamespace(last_motion="STOP", last_pwm=0, pwm_slew_credit=0.)
    cal = {"F": SimpleNamespace(direction_world=(1, 0), yaw=0.,
                                 measured_speed_mm_s=120., calibration_pwm=110)}
    obs = {18: LocalObservation(18, (2500, 1000), 0., 0.)}
    for step in range(10):
        decision = runtime.rate_limited_motion_decision(controller, "F", 1., 90, 240., .05)
        output = guard({18: decision}, obs, {18: SimpleNamespace(yaw=0)}, {18: cal},
                       arena, params, {}, (18,), step * .05)[18]
        assert output.command == "F"
        assert output.pwm > 0
        controller.last_motion, controller.last_pwm = output.command, output.pwm
    assert controller.last_pwm == 90


def test_runtime_virtual_field_needs_no_corner_subjects(monkeypatch):
    import baseline_vicon_herding as entry
    runtime = entry.runtime
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    def sleep(duration):
        clock.now += max(duration, 0.01)
    monkeypatch.setattr(runtime.time, "sleep", sleep)
    poses = initial_poses((5000, -3000))
    created = []
    class Tracker:
        closed = False
        def __init__(self, host, names, max_age):
            assert all(n.startswith("kedaya") for n in names)
            created.append(self)
        def connect(self, *_): pass
        def close(self): self.closed = True
        def get_snapshot(self):
            return SimpleNamespace(fresh_subjects={f"kedaya{i}" for i in poses},
                                   poses={f"kedaya{i}": SimpleNamespace(position=p) for i, p in poses.items()})
    monkeypatch.setattr(runtime, "ViconSubjectTracker", Tracker)
    monkeypatch.setattr(runtime, "sample_corner_positions", lambda *_: pytest.fail("Unexpected fixed corner read"))
    monkeypatch.setattr(runtime, "connect_requested_cars", lambda *_: pytest.fail("Unexpected motor connection in dry run"))
    assert entry.main(["--dry-run"]) == 0
    assert created[0].closed


def test_initial_sampling_skips_missing_car_and_freezes_geometry(monkeypatch):
    import baseline_virtual_arena as geometry
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr(geometry.time, "monotonic", lambda: clock.now)
    def sleep(duration): clock.now += duration
    monkeypatch.setattr(geometry.time, "sleep", sleep)
    poses = initial_poses()
    poses.pop(25)
    calls = SimpleNamespace(count=0)
    class Tracker:
        def get_snapshot(self):
            calls.count += 1
            # One extreme stale sample must not determine the median field.
            samples = dict(poses)
            if calls.count == 1:
                samples[1] = (1e6, 1e6)
            return SimpleNamespace(fresh_subjects={f"kedaya{i}" for i in samples},
                                   poses={f"kedaya{i}": SimpleNamespace(position=p) for i, p in samples.items()})
    args = SimpleNamespace(initial_visibility_sec=.1, initial_visibility_ratio=.7,
                           robot_diameter_mm=150, cage_width_ratio=.4, cage_height_ratio=.4)
    initializer = InitialCarArena(4000, 4000, 80, 700)
    arena, corners, reference = initializer(Tracker(), args, {i: f"kedaya{i}" for i in range(1, 26)},
                                             tuple(range(18, 26)), tuple(range(1, 18)))
    assert reference == {}
    assert 25 not in args.virtual_field_initial_positions
    assert args.virtual_field_initial_positions[1] == (0, 0)
    assert arena.contains_field_world(poses[1], 300)


def test_four_metre_150mm_cars_complete_capture_through_virtual_gate():
    import relative_cage_capture as runtime
    from relative_cage_core import CaptureMonitor
    arena = arena_fixture()
    params = physical_parameters(arena, 75)
    positions = {18: (2700., 3200.), 19: (2700., 2700.), 20: (2700., 3700.), 1: (2100., 3200.)}
    velocities = {i: (0, 0) for i in positions}
    controllers = {i: SimpleNamespace(last_motion="", last_pwm=None, pwm_slew_credit=0.) for i in positions}
    directions = {c: (math.cos(k * math.pi / 3), math.sin(k * math.pi / 3))
                  for k, c in enumerate(("F", "RF", "RB", "B", "LB", "LF"))}
    cal = {c: SimpleNamespace(direction_world=v, yaw=0., measured_speed_mm_s=120., calibration_pwm=110)
           for c, v in directions.items()}
    model = BaselineController(cage_opening_mm=700)
    guard = PhysicalCommandFilter(cage_opening_mm=700, isolate_failures=True)
    dynamics, monitor = SecondOrderDynamics(), CaptureMonitor(2.)
    minimum = math.inf
    captured = False
    dt = .05
    for step in range(1000):
        obs = {i: LocalObservation(i, p, 0, step * dt) for i, p in positions.items()}
        result = model(obs, (18, 19, 20), (1,), (), arena, params, dynamics, dt)
        if monitor.update(step * dt, (1,), (1,), result.inside_cage_evaders):
            captured = True
            break
        decisions = {}
        for i, v in result.velocities.items():
            speed = vec_norm(v)
            if speed < 25:
                decisions[i] = runtime.CommandDecision("STOP", 0., 0)
                continue
            controller = controllers[i]
            c, score = runtime.choose_calibrated_command(cal, v, 0, controller.last_motion, .5, .08)
            limits = params.herder_dynamics if i >= 18 else params.evader_dynamics
            minimum_pwm, maximum_pwm = (90, 170) if i >= 18 else (95, 230)
            pwm = runtime.speed_to_pwm(speed, limits.max_speed, minimum_pwm, maximum_pwm, 25., 120., 110)
            if controller.last_pwm in (None, 0):
                pwm = min(pwm, minimum_pwm)
            decisions[i] = runtime.rate_limited_motion_decision(controller, c, score, pwm, 240., dt)
        output = guard(decisions, obs, {i: SimpleNamespace(yaw=0) for i in positions},
                       {i: cal for i in positions}, arena, params, velocities, (18, 19, 20), step * dt)
        for i, d in output.items():
            velocities[i] = (0, 0) if d.command == "STOP" else tuple(v * 120 * d.pwm / 110 for v in directions[d.command])
            previous = positions[i]
            positions[i] = tuple(previous[k] + velocities[i][k] * dt for k in (0, 1))
            assert all(segment_distance(previous, positions[i], *wall) >= 115 - 1e-6 for wall in cage_walls(arena, 700))
            controllers[i].last_motion, controllers[i].last_pwm = d.command, d.pwm
            if d.command == "STOP" and decisions[i].command != "STOP":
                model.reset(i)
        minimum = min(minimum, *(math.dist(positions[i], positions[j])
                                 for i in positions for j in positions if i < j))
        assert all(arena.contains_field_local(p, 75) for p in positions.values())
    assert captured, "Ideal car must enter through a gate and stay in the cage for two seconds"
    assert minimum >= 230


FIELD = ((4000.0, 1000.0), (8000.0, 1000.0), (8000.0, 5000.0), (4000.0, 5000.0))
CAGE = ((4000.0, 3400.0), (5600.0, 3400.0), (5600.0, 5000.0), (4000.0, 5000.0))
CORNERS = {
    "top-left": ((0.0, 2400.0), (1600.0, 2400.0), (1600.0, 4000.0), (0.0, 4000.0)),
    "top-right": ((2400.0, 2400.0), (4000.0, 2400.0), (4000.0, 4000.0), (2400.0, 4000.0)),
    "bottom-left": ((0.0, 0.0), (1600.0, 0.0), (1600.0, 1600.0), (0.0, 1600.0)),
    "bottom-right": ((2400.0, 0.0), (4000.0, 0.0), (4000.0, 1600.0), (2400.0, 1600.0)),
}


def test_hard_coded_coordinates_reproduce_the_same_arena():
    arena, corners = fixed_arena(FIELD, CAGE, 700.0, 150.0, 80.0)
    assert (arena.width_mm, arena.height_mm) == pytest.approx((4000.0, 4000.0))
    assert (arena.cage_width_mm, arena.cage_height_mm) == pytest.approx((1600.0, 1600.0))
    assert arena.origin_world == pytest.approx((4000.0, 1000.0))
    assert arena.x_axis_world == pytest.approx((1.0, 0.0))
    assert arena.y_axis_world == pytest.approx((0.0, 1.0))
    assert arena.world_to_local(corners.left_down) == pytest.approx((0.0, 0.0))
    assert arena.world_to_local(corners.right_up) == pytest.approx((4000.0, 4000.0))
    # The cage is the local top-left corner; both openings face the field interior.
    assert arena.world_to_local((4000.0, 5000.0)) == pytest.approx((0.0, 4000.0))
    assert arena.world_to_local((5600.0, 3400.0)) == pytest.approx((1600.0, 2400.0))


@pytest.mark.parametrize("corner", sorted(CORNERS))
def test_cage_at_any_field_corner_keeps_local_top_left_placement(corner):
    base_field = ((0.0, 0.0), (4000.0, 0.0), (4000.0, 4000.0), (0.0, 4000.0))
    field = tuple((x + 1000.0, y + 2000.0) for x, y in base_field)
    cage = tuple((x + 1000.0, y + 2000.0) for x, y in CORNERS[corner])
    shuffled_field = (field[2], field[0], field[3], field[1])
    shuffled_cage = (cage[1], cage[3], cage[0], cage[2])
    arena, corners = fixed_arena(shuffled_field, shuffled_cage, 700.0, 150.0, 80.0)
    assert (arena.cage_width_mm, arena.cage_height_mm) == pytest.approx((1600.0, 1600.0))
    assert arena.cage_left == 0.0
    assert arena.cage_top == pytest.approx(arena.height_mm)
    cage_local = sorted(arena.world_to_local(point) for point in cage)
    assert min(point[0] for point in cage_local) == pytest.approx(0.0, abs=1e-6)
    assert max(point[1] for point in cage_local) == pytest.approx(arena.height_mm, abs=1e-6)


def test_cage_away_from_a_field_corner_is_rejected():
    with pytest.raises(ArenaGeometryError, match="cage must touch one corner"):
        fixed_arena(((0.0, 0.0), (4000.0, 0.0), (4000.0, 4000.0), (0.0, 4000.0)),
                    ((1000.0, 1000.0), (2000.0, 1000.0), (2000.0, 2000.0), (1000.0, 2000.0)),
                    700.0, 150.0, 80.0)


def test_field_that_is_not_a_rectangle_is_rejected():
    with pytest.raises(ArenaGeometryError, match="do not form a rectangle"):
        fixed_arena(((0.0, 0.0), (4000.0, 0.0), (4000.0, 3800.0), (0.0, 4000.0)),
                    ((0.0, 2400.0), (1600.0, 2400.0), (1600.0, 4000.0), (0.0, 4000.0)),
                    700.0, 150.0, 80.0)


def test_rectangle_ordering_is_independent_of_input_order():
    points = [(4000.0, 4000.0), (0.0, 0.0), (0.0, 4000.0), (4000.0, 0.0)]
    ordered = order_rectangle(points)
    assert ordered == [(0.0, 0.0), (4000.0, 0.0), (4000.0, 4000.0), (0.0, 4000.0)]


def fixed_initializer(monkeypatch, poses, arm=True, seconds=0.05):
    import baseline_virtual_arena as geometry
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(geometry.time, "monotonic", lambda: clock.now)
    def sleep(duration):
        clock.now += max(duration, 0.01)
    monkeypatch.setattr(geometry.time, "sleep", sleep)
    initializer = FixedCornerArena(FIELD, CAGE, 700.0, 150.0, 80.0)
    class Tracker:
        def get_snapshot(self):
            return SimpleNamespace(
                fresh_subjects={f"kedaya{i}" for i in poses},
                poses={f"kedaya{i}": SimpleNamespace(position=position)
                       for i, position in poses.items()},
            )
    args = SimpleNamespace(initial_visibility_sec=seconds, initial_visibility_ratio=0.5, arm=arm)
    subjects = {i: f"kedaya{i}" for i in poses}
    return initializer, Tracker(), args, subjects


def test_fixed_initializer_reports_inside_cars(monkeypatch, capsys):
    poses = {18: (5000.0, 3000.0), 1: (6000.0, 2000.0)}
    initializer, tracker, args, subjects = fixed_initializer(monkeypatch, poses)
    arena, corners, reference = initializer(
        tracker, args, subjects, (18,), (1,))
    assert reference == {}
    assert arena.contains_field_world(poses[18], 75)
    assert arena.contains_field_world(poses[1], 75)
    assert args.virtual_field_initial_positions[18] == pytest.approx(poses[18])
    printed = capsys.readouterr().out
    assert "Fixed field" in printed and "cage" in printed


def test_fixed_initializer_refuses_cars_outside_an_armed_field(monkeypatch):
    poses = {18: (5000.0, 3000.0), 1: (12000.0, 2000.0)}
    initializer, tracker, args, subjects = fixed_initializer(monkeypatch, poses)
    with pytest.raises(ArenaGeometryError, match="outside the configured field"):
        initializer(tracker, args, subjects, (18,), (1,))


def test_fixed_initializer_only_warns_outside_cars_in_dry_run(monkeypatch, capsys):
    poses = {18: (5000.0, 3000.0), 1: (12000.0, 2000.0)}
    initializer, tracker, args, subjects = fixed_initializer(monkeypatch, poses, arm=False)
    arena, _, _ = initializer(tracker, args, subjects, (18,), (1,))
    assert arena.width_mm == 4000.0
    assert "OUTSIDE the field body margin" in capsys.readouterr().out
