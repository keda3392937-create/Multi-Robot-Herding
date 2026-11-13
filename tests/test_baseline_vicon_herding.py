import ast
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from baseline_forces import BaselineForces
from baseline_physical import (
    BaselineController, PhysicalCommandFilter, physical_parameters,
    segment_hits_cage, swept_distance,
)
from relative_cage_core import RelativeArena, LocalObservation, SecondOrderDynamics, vec_norm


@dataclass(frozen=True)
class Decision:
    command: str
    score: float
    pwm: int


@pytest.fixture
def arena():
    return RelativeArena((0, 0), (1, 0), (0, 1), 5000, 5000, 2500, 2500, 0, 90)


def observation(i, x, y):
    return LocalObservation(i, (x, y), 0.0, 0.0)


def reference_class():
    # Load the archived uploaded source without GUI imports or running main.
    tree = ast.parse((Path(__file__).parents[1] / "simulation/baseline.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    namespace = {"np": np, "ConvexHull": ConvexHull}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "simulation/baseline.py", "exec"), namespace)
    return namespace[cls.name]


def force_fixture():
    m = BaselineForces()
    m.numHerders, m.numEvaders = 4, 1
    m.positionsHerder = np.array([[29., 30.], [60., 35.], [57., 60.], [30., 59.]])
    m.positionsEvader = np.array([[33., 30.]])
    m.herder_directions = np.array([0., math.pi, math.pi, -math.pi / 2])
    m.cageLeft, m.cageRight, m.cageBottom, m.cageTop = 0., 16., 64., 80.
    m.cageCenter = np.array([8., 72.])
    m.virtual_target = False
    return m


def test_herder_forces_match_uploaded_algorithm():
    m = force_fixture()
    ref = object.__new__(reference_class())
    ref.__dict__.update(copy.deepcopy(m.__dict__))
    ref.convex_hull_line = SimpleNamespace(set_data=lambda *_: None)
    expected = ref.update_herders_with_convex_hull(2.0)
    actual = m.update_herders_with_convex_hull(2.0)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(m.herder_directions, ref.herder_directions)


@pytest.mark.parametrize("point", [(33., 30.), (8., 72.), (18., 68.), (1., 25.)])
def test_evader_forces_match_uploaded_algorithm(point):
    m = force_fixture()
    m.positionsEvader = np.array([point])
    ref = object.__new__(reference_class())
    ref.__dict__.update(copy.deepcopy(m.__dict__))
    ref.velocitiesEvader = np.zeros((1, 2))
    ref.speedLimitEvader = 1e30  # Observe the original force before speed saturation.
    ref.update_evaders()
    np.testing.assert_allclose(m.evader_accelerations()[0], ref.velocitiesEvader[0] / ref.dt)


def test_virtual_cage_is_enterable_not_a_repelling_solid_wall():
    m = force_fixture()
    m.positionsEvader = np.array([[18., 68.]])
    assert m.calculate_cage_force(0)[0] > 0
    m.virtual_target = True
    np.testing.assert_array_equal(m.calculate_cage_force(0), [0, 0])


def test_variable_velocity_integration_and_speed_ratio(arena):
    params = physical_parameters(arena, 150)
    assert params.evader_dynamics.max_speed / params.herder_dynamics.max_speed == pytest.approx(1.5)
    model, dynamics = BaselineController(), SecondOrderDynamics(0)
    observations = {1: observation(1, 2700, 2000), 9: observation(9, 3200, 2000)}
    slow = model(observations, (1,), (9,), (), arena, params, dynamics, 0.05)
    fast = model(observations, (1,), (9,), (), arena, params, dynamics, 0.05)
    assert 0 < vec_norm(slow.velocities[9]) < vec_norm(fast.velocities[9])
    assert vec_norm(fast.velocities[9]) <= params.evader_dynamics.max_speed
    assert vec_norm(fast.accelerations[9]) <= params.evader_dynamics.max_acceleration
    # The model must not integrate or clip the measured position.
    assert observations[9].position == (3200, 2000)


def test_rectangular_arena_boundary_is_not_square():
    m = force_fixture()
    m.arenaWidth = 100
    m.positionsEvader = np.array([[78., 40.]])
    assert m.calculate_boundary_force(0)[0] == 0
    m.positionsEvader[0, 0] = 99
    assert m.calculate_boundary_force(0)[0] < 0


def test_swept_check_detects_crossing_even_with_separated_endpoints():
    assert swept_distance((1000, 0), (-2000, 0), 1.0) == pytest.approx(0)
    assert swept_distance((1000, 700), (-2000, 0), 1.0) == pytest.approx(700)


def calibration(speed=150):
    directions = {"F": (1, 0), "RF": (0.5, math.sqrt(3) / 2),
                  "RB": (-0.5, math.sqrt(3) / 2), "B": (-1, 0),
                  "LB": (-0.5, -math.sqrt(3) / 2), "LF": (0.5, -math.sqrt(3) / 2)}
    return {c: SimpleNamespace(direction_world=d, yaw=0, measured_speed_mm_s=speed,
                               calibration_pwm=110) for c, d in directions.items()}


def run_guard(arena, observations, decisions, herders=(), measured=None, guard=None, now=1):
    guard = guard or PhysicalCommandFilter()
    poses = {i: SimpleNamespace(yaw=0) for i in observations}
    calibrations = {i: calibration() for i in decisions}
    return guard(decisions, observations, poses, calibrations, arena,
                 physical_parameters(arena, 150), measured or {}, herders, now)


@pytest.mark.parametrize("herders", [(), (1,), (1, 2)])
def test_opposing_motion_never_authorizes_crossing_for_any_roles(arena, herders):
    obs = {1: observation(1, 3000, 1000), 2: observation(2, 3500, 1000)}
    decisions = {1: Decision("F", 1, 110), 2: Decision("B", 1, 110)}
    output = run_guard(arena, obs, decisions, herders)
    chosen = {}
    for i, d in output.items():
        ref = calibration()[d.command] if d.command != "STOP" else None
        chosen[i] = ((0, 0) if ref is None else tuple(v * 150 * d.pwm / 110 * 1.4 for v in ref.direction_world))
    assert swept_distance((-500, 0), tuple(chosen[1][k] - chosen[2][k] for k in (0, 1)), 0.95) >= 380


def test_failed_car_is_still_an_obstacle(arena):
    obs = {9: observation(9, 3000, 1000), 50: observation(50, 3400, 1000)}
    output = run_guard(arena, obs, {9: Decision("F", 1, 110)})
    assert output[9].command != "F"


def test_close_body_clearance_rejects_batch_even_inside_cage(arena):
    obs = {9: observation(9, 1000, 4000), 10: observation(10, 1250, 4000)}
    with pytest.raises(RuntimeError, match="below 380"):
        run_guard(arena, obs, {9: Decision("F", 1, 110), 10: Decision("STOP", 0, 0)})


def test_calibration_clearance_parameter_does_not_apply_inter_car_spacing(arena):
    pose = SimpleNamespace(position=(1000, 1000))
    nearby = {2: SimpleNamespace(position=(1100, 1000))}
    import relative_cage_capture as runtime
    safe, reason = runtime.calibration_position_is_safe(
        1, pose, {1: pose, **nearby}, arena, 100, 0,
    )
    assert safe, reason


def test_measured_coasting_is_checked_before_new_direction(arena):
    guard = PhysicalCommandFilter()
    positions = {1: (3000, 1000), 2: (3400, 1000)}
    assert not guard.safe_velocity(1, (-100, 0), positions, {1: (200, 0)}, {}, arena, 150)


def test_boundary_outward_motion_is_changed_or_stopped(arena):
    obs = {9: observation(9, 4750, 1000)}
    output = run_guard(arena, obs, {9: Decision("F", 1, 110)})
    assert output[9].command != "F"


def test_herder_yields_to_boundary_trapped_evader(arena):
    obs = {1: observation(1, 3950, 1000), 9: observation(9, 4700, 1000)}
    output = run_guard(arena, obs, {1: Decision("F", 1, 110)}, herders=(1,))
    assert output[1].command == "B"


def test_stop_request_never_restarts_car(arena):
    output = run_guard(arena, {9: observation(9, 1000, 4000)}, {9: Decision("STOP", 0, 0)})
    assert output[9].command == "STOP"


def test_lost_obstacle_pose_stops_instead_of_being_forgotten(arena):
    guard = PhysicalCommandFilter()
    obs = {9: observation(9, 3000, 1000), 50: observation(50, 4200, 1000)}
    run_guard(arena, obs, {9: Decision("F", 1, 110)}, guard=guard)
    with pytest.raises(RuntimeError, match="Lost previously visible obstacle"):
        run_guard(arena, {9: obs[9]}, {9: Decision("F", 1, 110)}, guard=guard, now=2)


def test_cage_crossing_test_includes_segment_not_only_endpoint(arena):
    assert segment_hits_cage((2000, 2000), (3000, 4000), arena, 0)
    assert not segment_hits_cage((3000, 1000), (4000, 1000), arena, 0)


def test_parked_cars_far_outside_field_do_not_block_experiment(arena):
    obs = {9: observation(9, 3000, 1000), 40: observation(40, -2000, -2000),
           41: observation(41, -2010, -2010)}
    output = run_guard(arena, obs, {9: Decision("F", 1, 110)})
    assert output[9].command == "F"


def test_all_fifty_ids_have_independent_finite_velocity_states(arena):
    params = physical_parameters(arena, 100)
    model, dynamics = BaselineController(), SecondOrderDynamics()
    obs = {i: observation(i, 500 + ((i - 1) % 10) * 450,
                         500 + ((i - 1) // 10) * 600) for i in range(1, 51)}
    result = model(obs, tuple(range(1, 9)), tuple(range(9, 51)), (), arena, params, dynamics, 0.05)
    assert set(result.velocities) == set(range(1, 51))
    assert all(math.isfinite(v) for velocity in result.velocities.values() for v in velocity)
    assert len(result.visible_evaders) == 42


def test_entry_defaults_and_runtime_hooks(monkeypatch):
    import baseline_vicon_herding as entry
    args = entry.parse_args([])
    assert args.herders == ",".join(map(str, range(18, 26)))
    assert args.evaders == ",".join(map(str, range(1, 18)))
    assert args.robot_diameter_mm == 150
    assert args.field_width_mm == args.field_height_mm == 4000
    assert args.isolate_car_failures
    assert args.parallel_calibration
    assert args.recalibrate
    assert args.calibration_speed == 255
    assert args.calibration_move_sec == pytest.approx(0.45)
    assert args.min_calibration_displacement_ratio * 4000 == pytest.approx(30.0)
    assert args.calibration_tag.startswith("baseline-real-v3")
    assert args.arm and args.skip_wifi
    assert args.vicon_host == entry.VICON_HOST
    called = {}
    def fake_main(args, **kwargs):
        called.update(kwargs)
        return 0
    monkeypatch.setattr(entry.runtime, "main", fake_main)
    # The test must not depend on the coordinates currently configured in the entry.
    monkeypatch.setattr(entry, "FIELD_CORNERS_WORLD", (
        (4000.0, 1000.0), (8000.0, 1000.0), (8000.0, 5000.0), (4000.0, 5000.0)))
    monkeypatch.setattr(entry, "CAGE_CORNERS_WORLD", (
        (4000.0, 3400.0), (5600.0, 3400.0), (5600.0, 5000.0), (4000.0, 5000.0)))
    assert entry.main([]) == 0
    assert isinstance(called["control_step"], BaselineController)
    assert isinstance(called["decision_filter"], PhysicalCommandFilter)
    assert called["decision_filter"].isolate_failures
    assert set(called["extra_obstacle_ids"]) == set(range(1, 51))
    from baseline_virtual_arena import FixedCornerArena
    assert isinstance(called["arena_initializer"], FixedCornerArena)
    assert called["calibrator"] is entry.calibrate_like_twelve_car
    arena = called["arena_initializer"].arena
    assert (arena.width_mm, arena.height_mm) == (4000, 4000)
    assert (arena.cage_width_mm, arena.cage_height_mm) == (1600, 1600)


def test_invalid_field_coordinates_stop_before_any_car_moves(monkeypatch):
    import baseline_vicon_herding as entry
    monkeypatch.setattr(entry.runtime, "main", lambda *a, **k: pytest.fail("must not run"))
    monkeypatch.setattr(entry, "FIELD_CORNERS_WORLD", (
        (4000.0, 1000.0), (8000.0, 1000.0), (8000.0, 5000.0), (4000.0, 5000.0)))
    monkeypatch.setattr(entry, "CAGE_CORNERS_WORLD", (
        (4000.0, 2000.0), (5600.0, 2000.0), (5600.0, 3600.0), (4000.0, 3600.0)))
    assert entry.main([]) == 2



def test_lost_car_is_retained_with_clearance_but_distant_car_continues(arena):
    guard = PhysicalCommandFilter(isolate_failures=True)
    obs = {9: observation(9, 3000, 1000), 10: observation(10, 4200, 1000)}
    run_guard(arena, obs, {9: Decision("B", 1, 110)}, guard=guard)
    output = run_guard(arena, {9: obs[9]}, {9: Decision("B", 1, 110)}, guard=guard, now=2)
    assert output[9].command == "B"
    assert guard.known_positions[10] == obs[10].position
    assert guard.lost_padding[10] > 0
    # Never forget an excluded car just because it has not reappeared.
    run_guard(arena, {9: obs[9]}, {9: Decision("B", 1, 110)}, guard=guard, now=3)
    assert 10 in guard.known_positions


def test_only_nearby_cars_stop_for_a_close_failed_obstacle(arena):
    guard = PhysicalCommandFilter(isolate_failures=True)
    obs = {9: observation(9, 3000, 1000), 10: observation(10, 3300, 1000),
           11: observation(11, 1000, 1000)}
    output = run_guard(arena, obs, {9: Decision("B", 1, 110), 11: Decision("F", 1, 110)}, guard=guard)
    assert output[9].command == "STOP"
    assert output[11].command == "F"


@pytest.mark.parametrize("failure", ["tcp", "no_motion", "pose"])
@pytest.mark.parametrize("failed_id", [4, 10])
def test_runtime_excludes_failed_role_and_keeps_healthy_cars_moving(arena, monkeypatch, failure, failed_id):
    import relative_cage_capture as runtime
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    def sleep(seconds):
        clock.now += max(seconds, 0.01)
    monkeypatch.setattr(runtime.time, "sleep", sleep)
    args = runtime.parse_args([], defaults={"isolate_car_failures": True,
                                            "max_runtime_sec": 0.30,
                                            "motion_response_timeout": 0.08})
    ids = (1, 2, 3, 4, 9, 10)
    obs = {i: observation(i, 700 + k * 650, 1000) for k, i in enumerate(ids)}
    model_calls, resets, events, stops, motions = [], [], [], [], []
    frame = SimpleNamespace(number=0)

    class Controller:
        def __init__(self, i):
            self.i, self.active, self.last_pwm = i, True, None
            self.last_motion, self.last_ping_at = "", 100.0
            self.pwm_slew_credit, self.failure_reason = 0.0, ""
        def set_motion(self, command, pwm, now, refresh):
            if failure == "tcp" and self.i == failed_id:
                self.active = False
                self.failure_reason = "test TCP write failed"
                return False
            self.last_motion, self.last_pwm = command, pwm
            motions.append((clock.now, self.i, command))
            return True
        def send_stop(self, force=False):
            stops.append((clock.now, self.i))
            return True
        def close(self, send_stop=True):
            if send_stop:
                self.send_stop()
            self.active = False

    class Tracker:
        def get_snapshot(self):
            frame.number += 1
            return SimpleNamespace(received_at=clock.now, frame_number=frame.number)

    class Estimator:
        raw_this_update = {}
        def __init__(self, *_): pass
        def update(self, observations, active_ids):
            return {i: ((0, 0) if i == failed_id and failure == "no_motion" else (120, 0)) for i in active_ids}
        def reset(self, *_): pass
        def reset_many(self, *_): pass

    def current(*_):
        visible = {i: o for i, o in obs.items()
                   if not (failure == "pose" and frame.number >= 2 and i == failed_id)}
        return visible, {i: SimpleNamespace(position=o.position, yaw=0.0) for i, o in visible.items()}

    def model(observations, herders, evaders, obstacles, *_args, **_kwargs):
        model_calls.append((tuple(herders), tuple(evaders), tuple(obstacles)))
        return SimpleNamespace(
            velocities={i: (120, 0) for i in herders + evaders},
            visible_herders=tuple(herders), visible_evaders=tuple(evaders),
            inside_cage_evaders=(), herder_hull_ids=tuple(herders),
            augmented_hull_herder_ids=tuple(herders), contained_evaders=(),
        )
    model.reset = resets.append
    monkeypatch.setattr(runtime, "check_corner_reference", lambda *_: (True, ""))
    monkeypatch.setattr(runtime, "current_local_observations", current)
    monkeypatch.setattr(runtime, "ViconVelocityEstimator", Estimator)
    monkeypatch.setattr(runtime, "log_control_states", lambda *_: None)
    # Buffered-frame draining needs a real Vicon stream; this test targets the loop.
    monkeypatch.setattr(runtime, "drain_vicon_frames", lambda *_: None)
    controllers = {i: Controller(i) for i in ids}
    calibrations = {i: calibration() for i in ids}
    ignored = set()
    logger = SimpleNamespace(event=lambda *args: events.append(args))
    runtime.run_control_loop(
        args, Tracker(), {i: f"kedaya{i}" for i in ids}, {}, {},
        arena, physical_parameters(arena, 150), (1, 2, 3, 4), (9, 10),
        (1, 2, 3, 4), (9, 10), controllers, calibrations, ignored, logger,
        control_step=model, decision_filter=lambda decisions, *_: decisions,
    )
    assert failed_id in ignored and failed_id not in calibrations
    assert not controllers[failed_id].active
    assert any(e[0] == "car_removed" and e[2] == failed_id for e in events)
    assert failed_id not in model_calls[-1][0] + model_calls[-1][1]
    assert all(i == failed_id for i in resets)  # Healthy integrators keep running.
    assert any(i != failed_id and t >= 0.15 and c != "STOP" for t, i, c in motions)
    # The only whole-fleet stop is the configured end of the test run.
    assert all(i == failed_id or t >= args.max_runtime_sec for t, i in stops)


def _run_with_speed_spikes(monkeypatch, arena, spike_cycles):
    """Drive run_control_loop with a fake estimator that reports huge raw speeds."""
    import relative_cage_capture as runtime
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    def sleep(seconds):
        clock.now += max(seconds, 0.01)
    monkeypatch.setattr(runtime.time, "sleep", sleep)
    monkeypatch.setattr(runtime, "drain_vicon_frames", lambda *_: None)
    ids = (1, 2, 3, 9)
    obs = {i: observation(i, 700 + 650 * k, 1000) for k, i in enumerate(ids)}
    updates = []

    class Tracker:
        def get_snapshot(self):
            clock.now += 0.005
            return SimpleNamespace(received_at=clock.now, frame_number=int(clock.now / 0.01))

    class Estimator:
        def __init__(self, *_):
            self.index = len(updates)
            updates.append(0)
        def update(self, observations, active_ids):
            self.index = len(updates)
            updates.append(self.index)
            spike = self.index in spike_cycles
            self.raw_this_update = {
                marker_id: ((1800.0, 0.0) if spike else (10.0, 0.0))
                for marker_id in active_ids
            }
            return {marker_id: (10.0, 0.0) for marker_id in active_ids}
        def reset(self, *_):
            pass
        def reset_many(self, *_):
            pass

    class Controller:
        def __init__(self):
            self.active, self.last_pwm, self.last_motion = True, 0, ""
            self.pwm_slew_credit, self.failure_reason = 0.0, ""
            self.last_ping_at = 0.0
        def set_motion(self, *_args, **_kwargs):
            return True
        def send_stop(self, force=False):
            return True
        def close(self, send_stop=True):
            self.active = False
        def ping(self):
            return True

    def current(*_args):
        return obs, {i: SimpleNamespace(position=o.position, yaw=0.0) for i, o in obs.items()}

    def model(observations, herders, evaders, *_args, **_kwargs):
        return SimpleNamespace(
            velocities={i: (0.0, 0.0) for i in herders + evaders},
            visible_herders=tuple(herders), visible_evaders=tuple(evaders),
            inside_cage_evaders=(), herder_hull_ids=tuple(herders),
            augmented_hull_herder_ids=tuple(herders), contained_evaders=(),
        )
    model.reset = lambda *_: None

    monkeypatch.setattr(runtime, "check_corner_reference", lambda *_: (True, ""))
    monkeypatch.setattr(runtime, "current_local_observations", current)
    monkeypatch.setattr(runtime, "ViconVelocityEstimator", Estimator)
    monkeypatch.setattr(runtime, "log_control_states", lambda *_: None)
    args = runtime.parse_args([], defaults={"isolate_car_failures": True, "max_runtime_sec": 0.30})
    controllers = {i: Controller() for i in ids}
    runtime.run_control_loop(
        args, Tracker(), {i: f"kedaya{i}" for i in ids}, {}, {},
        arena, physical_parameters(arena, 150), (1, 2, 3), (9,),
        (1, 2, 3), (9,), controllers, {i: calibration() for i in ids}, set(),
        SimpleNamespace(event=lambda *_: None),
        control_step=model, decision_filter=lambda decisions, *_: decisions,
    )
    return len(updates)


def test_single_frame_vicon_speed_spike_does_not_stop_the_fleet(arena, monkeypatch, capsys):
    """A stale/current frame pair looks like a jump once; that must not abort the run."""
    cycles = _run_with_speed_spikes(monkeypatch, arena, spike_cycles={1})
    assert cycles >= 3
    assert "one-frame Vicon speed spike" in capsys.readouterr().out


def test_sustained_speed_violation_still_stops_the_fleet(arena, monkeypatch):
    import relative_cage_capture as runtime
    with pytest.raises(runtime.SafetyStop, match="speed/jump exceeded"):
        _run_with_speed_spikes(monkeypatch, arena, spike_cycles={1, 2, 3, 4, 5})


def test_calibration_drops_an_implausible_vicon_jump_direction(monkeypatch, capsys):
    """A 4 m teleport during one pulse must not become a calibration direction."""
    import baseline_vicon_herding as entry
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(entry.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(entry.time, "sleep",
                        lambda seconds: setattr(clock, "now", clock.now + max(seconds, 0.01)))
    ids = (1, 18)
    positions = {1: (1000.0, 1000.0), 18: (2000.0, 1000.0)}
    steps = {"F": (100.0, 0.0), "RF": (50.0, 86.6), "RB": (-50.0, 86.6),
             "B": (-100.0, 0.0), "LB": (-50.0, -86.6), "LF": (50.0, -86.6)}
    teleported = []
    commanded = []

    class Controller:
        def __init__(self, marker_id):
            self.marker_id, self.active, self.failure_reason = marker_id, True, ""
            self.last_motion, self.last_pwm, self.pwm_slew_credit = "", None, 0.0
        def set_motion(self, command, pwm, now, refresh):
            commanded.append((self.marker_id, command))
            dx, dy = steps[command]
            x, y = positions[self.marker_id]
            positions[self.marker_id] = (x + dx * 0.05, y + dy * 0.05)
            if self.marker_id == 18 and not teleported:
                # Vicon relabelled the body for one frame in the middle of F.
                positions[18] = (positions[18][0] + 4000.0, positions[18][1])
                teleported.append(command)
            return True
        def send_stop(self, force=False):
            return self.active
        def close(self, send_stop=True):
            self.active = False

    class Tracker:
        def get_snapshot(self):
            return SimpleNamespace(
                fresh_subjects={f"kedaya{i}" for i in positions},
                poses={f"kedaya{i}": SimpleNamespace(
                    position=position, yaw=0.0, seen_at=clock.now)
                    for i, position in positions.items()},
            )

    class Logger:
        def event(self, *args):
            pass

    args = SimpleNamespace(
        min_calibration_displacement_ratio=30.0 / 4000.0,
        calibration_speed=255, calibration_move_sec=0.45, calibration_settle_sec=0.35,
        max_calibration_displacement_ratio=0.12,
    )
    controllers = {i: Controller(i) for i in ids}
    results, excluded = entry.calibrate_like_twelve_car(
        list(ids), controllers, Tracker(), {i: f"kedaya{i}" for i in ids},
        SimpleNamespace(scale_mm=4000.0), args, Logger(),
    )
    assert teleported == ["F"]
    assert "F" not in results[18]
    assert set(results[18]) == set(entry.runtime.MOTION_COMMANDS) - {"F"}
    assert set(results[1]) == set(entry.runtime.MOTION_COMMANDS)
    assert all(value.displacement_mm <= 480.0 for value in results[18].values())
    assert "Vicon jump?" in capsys.readouterr().out


def test_cage_packing_warning_flags_a_cage_too_small_for_the_evaders(capsys):
    import baseline_vicon_herding as entry
    from baseline_virtual_arena import fixed_arena
    field = ((0.0, 0.0), (4000.0, 0.0), (4000.0, 4000.0), (0.0, 4000.0))
    small, _ = fixed_arena(field, ((0.0, 3000.0), (1000.0, 3000.0),
                                   (1000.0, 4000.0), (0.0, 4000.0)), 700.0, 150.0, 80.0)
    assert entry.cage_packing_warning(small, 17)
    printed = capsys.readouterr().out
    assert "tight for 17 evaders" in printed and "1340x1340mm" in printed
    large, _ = fixed_arena(field, ((0.0, 2400.0), (1600.0, 2400.0),
                                   (1600.0, 4000.0), (0.0, 4000.0)), 700.0, 150.0, 80.0)
    assert not entry.cage_packing_warning(large, 17)
    assert capsys.readouterr().out == ""


def test_ping_reassembles_a_pong_split_across_tcp_segments():
    """The firmware writes "PONG id=" and the number separately."""
    import relative_cage_capture as runtime
    class FragmentedSocket:
        def __init__(self):
            self.chunks = [b"PONG id=", TimeoutError("timed out"), b"16\r\n"]
            self.sent = []
        def sendall(self, data):
            self.sent.append(data)
        def recv(self, _count):
            chunk = self.chunks.pop(0)
            if isinstance(chunk, Exception):
                raise chunk
            return chunk
        def close(self):
            pass
    controller = runtime.ExperimentCarController(16, "192.168.33.120", 0.6, 0.08)
    controller.sock = FragmentedSocket()
    controller.active = True
    assert controller.ping()
    assert controller.failure_reason == ""
    assert controller.sock.sent.count(b"PING\n") == 1


def test_ping_keeps_a_partial_line_for_the_next_call():
    import relative_cage_capture as runtime
    class SlowSocket:
        def __init__(self):
            self.chunks = [b"PONG id="]
            self.sent = []
        def sendall(self, data):
            self.sent.append(data)
        def recv(self, _count):
            if self.chunks:
                return self.chunks.pop(0)
            raise TimeoutError("timed out")
        def close(self):
            pass
    controller = runtime.ExperimentCarController(16, "192.168.33.120", 0.6, 0.08)
    sock = SlowSocket()
    controller.sock = sock
    controller.active = True
    assert not controller.ping(timeout_sec=0.03)
    assert "timed out" in controller.failure_reason
    assert bytes(controller.ping_buffer) == b"PONG id="
    sock.chunks.append(b"16\r\n")
    assert controller.ping()
    assert bytes(controller.ping_buffer) == b""


def test_ping_does_not_retry_a_wrong_identity():
    import relative_cage_capture as runtime
    class WrongCarSocket:
        def __init__(self):
            self.reads = 0
            self.sent = []
        def sendall(self, data):
            self.sent.append(data)
        def recv(self, _count):
            self.reads += 1
            return b"PONG id=8\n"
        def close(self):
            pass
    controller = runtime.ExperimentCarController(7, "192.168.33.111", 0.6, 0.08)
    sock = WrongCarSocket()
    controller.sock = sock
    controller.active = True
    assert not controller.ping()
    assert "identity mismatch" in controller.failure_reason
    assert sock.reads == 1
    assert sock.sent == [b"PING\n"]


def test_slow_vicon_frame_pauses_instead_of_aborting(arena, monkeypatch, capsys):
    """Vicon shares the access point with the cars; a 0.3 s frame must not end the run."""
    import relative_cage_capture as runtime
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(runtime.time, "sleep",
                        lambda seconds: setattr(clock, "now", clock.now + max(seconds, 0.01)))
    monkeypatch.setattr(runtime, "drain_vicon_frames", lambda *_: None)
    ids = (1, 2, 3, 9)
    obs = {i: observation(i, 700 + 650 * k, 1000) for k, i in enumerate(ids)}
    snapshots = SimpleNamespace(count=0)

    class Tracker:
        def get_snapshot(self):
            snapshots.count += 1
            clock.now += 0.005
            # Frame 2 is 0.3 s late (hold), frame 3 is 0.7 s late (pause).
            age = {2: 0.3, 3: 0.7}.get(snapshots.count, 0.005)
            return SimpleNamespace(received_at=clock.now - age, frame_number=snapshots.count)

    class Estimator:
        def __init__(self, *_):
            self.raw_this_update = {}
        def update(self, *_args):
            self.raw_this_update = {}
            return {}
        def reset(self, *_):
            pass
        def reset_many(self, *_):
            pass

    class Controller:
        def __init__(self):
            self.active, self.last_pwm, self.last_motion = True, 0, ""
            self.pwm_slew_credit, self.failure_reason = 0.0, ""
            self.last_ping_at = 0.0
        def set_motion(self, *_args, **_kwargs):
            return True
        def send_stop(self, force=False):
            return True
        def close(self, send_stop=True):
            self.active = False
        def ping(self):
            return True

    def current(*_args):
        return obs, {i: SimpleNamespace(position=o.position, yaw=0.0) for i, o in obs.items()}

    def model(observations, herders, evaders, *_args, **_kwargs):
        return SimpleNamespace(
            velocities={i: (0.0, 0.0) for i in herders + evaders},
            visible_herders=tuple(herders), visible_evaders=tuple(evaders),
            inside_cage_evaders=(), herder_hull_ids=tuple(herders),
            augmented_hull_herder_ids=tuple(herders), contained_evaders=(),
        )
    model.reset = lambda *_: None

    monkeypatch.setattr(runtime, "check_corner_reference", lambda *_: (True, ""))
    monkeypatch.setattr(runtime, "current_local_observations", current)
    monkeypatch.setattr(runtime, "ViconVelocityEstimator", Estimator)
    monkeypatch.setattr(runtime, "log_control_states", lambda *_: None)
    args = runtime.parse_args([], defaults={"isolate_car_failures": True, "max_runtime_sec": 0.30})
    runtime.run_control_loop(
        args, Tracker(), {i: f"kedaya{i}" for i in ids}, {}, {},
        arena, physical_parameters(arena, 150), (1, 2, 3), (9,),
        (1, 2, 3), (9,), {i: Controller() for i in ids},
        {i: calibration() for i in ids}, set(), SimpleNamespace(event=lambda *_: None),
        control_step=model, decision_filter=lambda decisions, *_: decisions,
    )
    assert snapshots.count >= 4
    printed = capsys.readouterr().out
    assert "Holding last commands: Vicon frame 300ms late" in printed
    assert "Paused: Vicon frame slower than the collision timing budget" in printed
    assert "Control resumed" in printed


def test_vicon_outage_still_stops_the_fleet(arena, monkeypatch):
    import relative_cage_capture as runtime
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(runtime.time, "sleep",
                        lambda seconds: setattr(clock, "now", clock.now + max(seconds, 0.01)))
    monkeypatch.setattr(runtime, "drain_vicon_frames", lambda *_: None)
    ids = (1, 2, 3, 9)
    obs = {i: observation(i, 700 + 650 * k, 1000) for k, i in enumerate(ids)}

    class Tracker:
        def get_snapshot(self):
            clock.now += 0.005
            return SimpleNamespace(received_at=clock.now - 3.0, frame_number=1)

    class Estimator:
        def __init__(self, *_):
            self.raw_this_update = {}
        def update(self, *_args):
            self.raw_this_update = {}
            return {}
        def reset(self, *_):
            pass
        def reset_many(self, *_):
            pass

    class Controller:
        def __init__(self):
            self.active, self.last_pwm, self.last_motion = True, 0, ""
            self.pwm_slew_credit, self.failure_reason = 0.0, ""
            self.last_ping_at = 0.0
        def set_motion(self, *_args, **_kwargs):
            return True
        def send_stop(self, force=False):
            return True
        def close(self, send_stop=True):
            self.active = False
        def ping(self):
            return True

    def current(*_args):
        return obs, {i: SimpleNamespace(position=o.position, yaw=0.0) for i, o in obs.items()}

    monkeypatch.setattr(runtime, "check_corner_reference", lambda *_: (True, ""))
    monkeypatch.setattr(runtime, "current_local_observations", current)
    monkeypatch.setattr(runtime, "ViconVelocityEstimator", Estimator)
    monkeypatch.setattr(runtime, "log_control_states", lambda *_: None)
    args = runtime.parse_args([], defaults={"isolate_car_failures": True, "max_runtime_sec": 1.0})
    with pytest.raises(runtime.SafetyStop, match="stopped delivering frames"):
        runtime.run_control_loop(
            args, Tracker(), {i: f"kedaya{i}" for i in ids}, {}, {},
            arena, physical_parameters(arena, 150), (1, 2, 3), (9,),
            (1, 2, 3), (9,), {i: Controller() for i in ids},
            {i: calibration() for i in ids}, set(), SimpleNamespace(event=lambda *_: None),
            control_step=lambda *a, **k: pytest.fail("must not compute on stale data"),
            decision_filter=lambda decisions, *_: decisions,
        )


def test_stopped_evader_restarts_instead_of_deadlocking_below_the_deadband(arena, monkeypatch):
    """Regression: STOP because the desired speed is tiny must not zero the integrator.

    Previously the dynamics were reset on every STOP decision, so a car whose
    acceleration per cycle is below ``motion_deadband`` could never restart and
    sat in a field corner for tens of seconds (evader 1, run 20261113_150730).
    """
    import relative_cage_capture as runtime
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(runtime.time, "sleep",
                        lambda seconds: setattr(clock, "now", clock.now + max(seconds, 0.01)))
    monkeypatch.setattr(runtime, "drain_vicon_frames", lambda *_: None)
    herders = (10, 11, 12)
    evader = 1
    poses = {10: (3000.0, 3000.0), 11: (3600.0, 1800.0), 12: (2400.0, 3600.0),
             evader: (150.0, 1000.0)}  # near the left field wall, far from the cage
    motions = []

    class Tracker:
        def get_snapshot(self):
            clock.now += 0.005
            return SimpleNamespace(
                received_at=clock.now, frame_number=int(clock.now / 0.01),
                poses={f"kedaya{i}": SimpleNamespace(
                    subject_name=f"kedaya{i}", position=p, yaw=0.0, seen_at=clock.now)
                    for i, p in poses.items()},
                fresh_subjects={f"kedaya{i}" for i in poses},
            )

    class Controller:
        def __init__(self, marker_id):
            self.marker_id, self.active, self.last_pwm = marker_id, True, 0
            self.last_motion, self.last_ping_at = "", 0.0
            self.pwm_slew_credit, self.failure_reason = 0.0, ""
        def set_motion(self, command, pwm, now, refresh):
            motions.append((clock.now, self.marker_id, command))
            self.last_motion, self.last_pwm = command, pwm
            return True
        def send_stop(self, force=False):
            self.last_motion, self.last_pwm = "STOP", 0
            return True
        def close(self, send_stop=True):
            self.active = False
        def ping(self):
            return True

    monkeypatch.setattr(runtime, "log_control_states", lambda *_: None)
    args = runtime.parse_args([], defaults={"isolate_car_failures": True, "max_runtime_sec": 1.0})
    ids = herders + (evader,)
    runtime.run_control_loop(
        args, Tracker(), {i: f"kedaya{i}" for i in ids}, {}, {},
        arena, physical_parameters(arena, 150), herders, (evader,),
        herders, (evader,), {i: Controller(i) for i in ids},
        {i: calibration() for i in ids}, set(), SimpleNamespace(event=lambda *_: None),
        control_step=BaselineController(cage_opening_mm=700),
        decision_filter=lambda decisions, *_: decisions,
    )
    evader_moves = [t for t, i, command in motions if i == evader and command != "STOP"]
    assert evader_moves, "evader stayed parked: STOP reset the desired velocity every cycle"
    assert min(evader_moves) < 0.4


def test_absurd_vicon_jump_excludes_that_car_instead_of_the_fleet(arena, monkeypatch):
    import relative_cage_capture as runtime
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(runtime.time, "sleep",
                        lambda seconds: setattr(clock, "now", clock.now + max(seconds, 0.01)))
    monkeypatch.setattr(runtime, "drain_vicon_frames", lambda *_: None)
    ids = (1, 2, 3, 9, 10)
    obs = {i: observation(i, 700 + 650 * k, 1000) for k, i in enumerate(ids)}
    events = []

    class Tracker:
        def get_snapshot(self):
            clock.now += 0.005
            return SimpleNamespace(received_at=clock.now, frame_number=int(clock.now / 0.01))

    class Estimator:
        def __init__(self, *_):
            self.raw_this_update = {}
        def update(self, observations, active_ids):
            # Car 9 teleports: 100 m/s is impossible for a car with a 393 mm/s limit.
            self.raw_this_update = {
                i: ((100_000.0, 0.0) if i == 9 else (10.0, 0.0)) for i in active_ids
            }
            return {i: (10.0, 0.0) for i in active_ids}
        def reset(self, *_):
            pass
        def reset_many(self, *_):
            pass

    class Controller:
        def __init__(self):
            self.active, self.last_pwm, self.last_motion = True, 0, ""
            self.pwm_slew_credit, self.failure_reason = 0.0, ""
            self.last_ping_at = 0.0
        def set_motion(self, *_args, **_kwargs):
            return True
        def send_stop(self, force=False):
            return True
        def close(self, send_stop=True):
            self.active = False
        def ping(self):
            return True

    def current(*_args):
        return obs, {i: SimpleNamespace(position=o.position, yaw=0.0) for i, o in obs.items()}

    def model(observations, herders, evaders, *_args, **_kwargs):
        return SimpleNamespace(
            velocities={i: (0.0, 0.0) for i in herders + evaders},
            visible_herders=tuple(herders), visible_evaders=tuple(evaders),
            inside_cage_evaders=(), herder_hull_ids=tuple(herders),
            augmented_hull_herder_ids=tuple(herders), contained_evaders=(),
        )
    model.reset = lambda *_: None

    monkeypatch.setattr(runtime, "check_corner_reference", lambda *_: (True, ""))
    monkeypatch.setattr(runtime, "current_local_observations", current)
    monkeypatch.setattr(runtime, "ViconVelocityEstimator", Estimator)
    monkeypatch.setattr(runtime, "log_control_states", lambda *_: None)
    args = runtime.parse_args([], defaults={"isolate_car_failures": True, "max_runtime_sec": 0.5})
    ignored = set()
    runtime.run_control_loop(
        args, Tracker(), {i: f"kedaya{i}" for i in ids}, {}, {},
        arena, physical_parameters(arena, 150), (1, 2, 3), (9, 10),
        (1, 2, 3), (9, 10), {i: Controller() for i in ids},
        {i: calibration() for i in ids}, ignored,
        SimpleNamespace(event=lambda *a: events.append(a)),
        control_step=model, decision_filter=lambda decisions, *_: decisions,
    )
    assert 9 in ignored
    assert any(event[0] == "car_removed" and event[2] == 9 for event in events)
    assert 10 not in ignored


def test_measured_zero_does_not_erase_startup_acceleration(arena):
    params = physical_parameters(arena, 150)
    model, measured_state = BaselineController(), SecondOrderDynamics()
    obs = {1: observation(1, 2900, 1200), 2: observation(2, 3900, 1900),
           3: observation(3, 4100, 2900), 9: observation(9, 2900, 2300)}
    peak = 0.0
    for _ in range(35):
        for i in obs:
            measured_state.observe_velocity(i, (0, 0), params.herder_dynamics if i < 9 else params.evader_dynamics)
        result = model(obs, (1, 2, 3), (9,), (), arena, params, measured_state, 0.05)
        peak = max(peak, *(vec_norm(result.velocities[i]) for i in (1, 2, 3)))
    assert peak > 25  # Cross the runtime's motor deadband despite measured zero.


def test_ideal_closed_loop_reaches_cage_without_contact(arena):
    """Exercise model -> discrete commands/PWM -> guard -> ideal car -> Vicon."""
    params = physical_parameters(arena, 150)
    positions = {1: (2900., 1200.), 2: (3900., 1900.),
                 3: (4100., 2900.), 9: (2900., 2300.)}
    velocities = {i: (0, 0) for i in positions}
    model, dynamics, guard = BaselineController(), SecondOrderDynamics(), PhysicalCommandFilter()
    cal = calibration()
    dt = 0.05
    minimum_distance = math.inf
    for step in range(600):
        obs = {i: observation(i, *p) for i, p in positions.items()}
        for i, v in velocities.items():
            dynamics.observe_velocity(i, v, params.herder_dynamics if i < 9 else params.evader_dynamics)
        result = model(obs, (1, 2, 3), (9,), (), arena, params, dynamics, dt)
        decisions = {}
        for i, v in result.velocities.items():
            speed = vec_norm(v)
            if speed < 25:
                decisions[i] = Decision("STOP", 0, 0)
            else:
                c = max(cal, key=lambda c: sum(cal[c].direction_world[k] * v[k] for k in (0, 1)))
                pwm = max(90 if i < 9 else 95, min(170 if i < 9 else 230, round(speed / 150 * 110)))
                decisions[i] = Decision(c, 1, pwm)
        out = run_guard(arena, obs, decisions, (1, 2, 3), velocities, guard, step * dt)
        for i, d in out.items():
            velocities[i] = ((0, 0) if d.command == "STOP" else
                             tuple(v * 150 * d.pwm / 110 for v in cal[d.command].direction_world))
            positions[i] = tuple(positions[i][k] + velocities[i][k] * dt for k in (0, 1))
            if d.command == "STOP" and decisions[i].command != "STOP":
                model.reset(i)
        minimum_distance = min(minimum_distance, *(math.dist(positions[i], positions[j])
                                                  for i in positions for j in positions if i < j))
        assert all(arena.contains_field_local(p, 150) for p in positions.values())
        if arena.contains_cage_local(positions[9], params.cage_hold_margin_mm):
            break
    assert arena.contains_cage_local(positions[9], params.cage_hold_margin_mm)
    assert minimum_distance >= 380


def test_six_direction_calibration_matches_twelve_car_behaviour(monkeypatch):
    """Strong full-PWM pulses keep healthy cars; stalled or unseen cars drop out."""
    import baseline_vicon_herding as entry
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(entry.time, "monotonic", lambda: clock.now)
    def sleep(seconds):
        clock.now += max(seconds, 0.01)
    monkeypatch.setattr(entry.time, "sleep", sleep)

    ids = (1, 2, 3, 18)
    stalled, unseen = 2, 3
    positions = {i: (900.0 * i, 500.0) for i in ids if i != unseen}
    steps = {"F": (100.0, 0.0), "RF": (50.0, 86.6), "RB": (-50.0, 86.6),
             "B": (-100.0, 0.0), "LB": (-50.0, -86.6), "LF": (50.0, -86.6)}
    sent = []

    class Controller:
        def __init__(self, marker_id):
            self.marker_id, self.active, self.failure_reason = marker_id, True, ""
            self.last_motion, self.last_pwm, self.pwm_slew_credit = "", None, 0.0
        def set_motion(self, command, pwm, now, refresh):
            sent.append((self.marker_id, command, pwm))
            if self.marker_id != stalled:
                dx, dy = steps[command]
                x, y = positions[self.marker_id]
                positions[self.marker_id] = (x + dx * 0.05, y + dy * 0.05)
            return True
        def send_stop(self, force=False):
            return self.active
        def close(self, send_stop=True):
            self.active = False

    class Tracker:
        def get_snapshot(self):
            return SimpleNamespace(
                fresh_subjects={f"kedaya{i}" for i in positions},
                poses={f"kedaya{i}": SimpleNamespace(
                    position=position, yaw=0.1, seen_at=clock.now)
                    for i, position in positions.items()},
            )

    class Logger:
        def __init__(self):
            self.events = []
        def event(self, *args):
            self.events.append(args)

    args = SimpleNamespace(
        min_calibration_displacement_ratio=30.0 / 4000.0,
        calibration_speed=255, calibration_move_sec=0.45, calibration_settle_sec=0.35,
        max_calibration_displacement_ratio=0.12,
    )
    controllers = {i: Controller(i) for i in ids}
    logger = Logger()
    results, excluded = entry.calibrate_like_twelve_car(
        list(ids), controllers, Tracker(), {i: f"kedaya{i}" for i in ids},
        SimpleNamespace(scale_mm=4000.0), args, logger,
    )
    assert results[stalled] == {}
    assert results[unseen] == {}
    assert excluded[unseen] == "calibration: no fresh Vicon start pose"
    assert not controllers[unseen].active
    for marker_id in (1, 18):
        assert set(results[marker_id]) == set(entry.runtime.MOTION_COMMANDS)
        assert all(value.calibration_pwm == 255 for value in results[marker_id].values())
        assert all(value.displacement_mm >= 30 for value in results[marker_id].values())
    assert any(pwm == 255 for _, _, pwm in sent)
    # Every direction is driven in the fixed order of the shared runtime module.
    driven = [command for marker_id, command, _ in sent if marker_id == 1]
    assert set(driven) == set(entry.runtime.MOTION_COMMANDS)


def test_calibration_reports_tcp_failure_instead_of_aborting(monkeypatch):
    import baseline_vicon_herding as entry
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(entry.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(entry.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + max(seconds, 0.01)))
    ids = (1, 2)
    positions = {i: (1000.0 * i, 500.0) for i in ids}

    class Controller:
        def __init__(self, marker_id):
            self.marker_id, self.active = marker_id, True
            self.last_motion, self.last_pwm, self.pwm_slew_credit = "", None, 0.0
            self.failure_reason = ""
        def set_motion(self, command, pwm, now, refresh):
            if self.marker_id == 2:
                self.active = False
                self.failure_reason = "TCP send failed"
                return False
            x, y = positions[self.marker_id]
            positions[self.marker_id] = (x + 3.0, y)
            return True
        def send_stop(self, force=False):
            return self.active
        def close(self, send_stop=True):
            self.active = False

    class Tracker:
        def get_snapshot(self):
            return SimpleNamespace(
                fresh_subjects={f"kedaya{i}" for i in ids},
                poses={f"kedaya{i}": SimpleNamespace(position=position, yaw=0.0, seen_at=clock.now)
                       for i, position in positions.items()},
            )

    class Logger:
        def event(self, *args):
            pass

    args = SimpleNamespace(
        min_calibration_displacement_ratio=30.0 / 4000.0,
        calibration_speed=255, calibration_move_sec=0.45, calibration_settle_sec=0.35,
        max_calibration_displacement_ratio=0.12,
    )
    controllers = {i: Controller(i) for i in ids}
    results, excluded = entry.calibrate_like_twelve_car(
        list(ids), controllers, Tracker(), {i: f"kedaya{i}" for i in ids},
        SimpleNamespace(scale_mm=4000.0), args, Logger(),
    )
    assert excluded[2] == "calibration pulse failed: TCP send failed"
    assert results[1]  # the healthy car keeps calibrating
    assert not controllers[2].active
