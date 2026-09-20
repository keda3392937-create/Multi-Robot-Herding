import json
import math
import sys
import threading
import time
from itertools import combinations
from types import SimpleNamespace

import pytest
import relative_cage_capture as capture_module

from relative_cage_capture import (
    CommandDecision,
    ExperimentCarController,
    MotionCalibration,
    ViconSubjectTracker,
    ViconVelocityEstimator,
    calibration_max_gap_deg,
    cage_has_capacity,
    choose_calibrated_command,
    connect_requested_cars,
    filtered_calibrations,
    load_calibration_file,
    motion_response_failed,
    parse_id_spec,
    parse_args,
    rate_limited_motion_decision,
    save_calibration_file,
    slew_pwm,
    slew_pwm_per_second,
    speed_to_pwm,
    valid_calibration_set,
)
from relative_cage_core import (
    ArenaGeometryError,
    CaptureMonitor,
    ControlParameters,
    DynamicsLimits,
    FieldCorners,
    LocalObservation,
    SecondOrderDynamics,
    active_role_ids,
    compute_control_step,
    convex_hull_indices,
    field_boundary_force,
    fit_relative_arena,
    formation_targets,
    hull_gap_surround_force,
    pair_repulsion,
    point_in_convex_polygon,
    select_shared_target_evader,
    smooth_repulsion,
    vec_norm,
)


def transform(point, angle_deg=0.0, offset=(0.0, 0.0)):
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        offset[0] + cosine * point[0] - sine * point[1],
        offset[1] + sine * point[0] + cosine * point[1],
    )


def make_corners(width=6000.0, height=4000.0, angle_deg=0.0, offset=(0.0, 0.0)):
    return FieldCorners(
        left_up=transform((0.0, height), angle_deg, offset),
        left_down=transform((0.0, 0.0), angle_deg, offset),
        right_up=transform((width, height), angle_deg, offset),
        right_down=transform((width, 0.0), angle_deg, offset),
    )


def make_arena():
    return fit_relative_arena(make_corners())


def observation(marker_id, position):
    return LocalObservation(marker_id, position, 0.0, 1.0)


def test_axis_aligned_field_and_internal_top_left_cage():
    arena = make_arena()
    assert arena.world_to_local((0.0, 0.0)) == pytest.approx((0.0, 0.0))
    assert arena.world_to_local((0.0, 4000.0)) == pytest.approx((0.0, 4000.0))
    assert arena.world_to_local((6000.0, 4000.0)) == pytest.approx((6000.0, 4000.0))
    assert arena.cage_top_left_local == pytest.approx((0.0, 4000.0))
    assert arena.cage_right_down_local == pytest.approx((2400.0, 2400.0))
    assert arena.contains_cage_local(arena.cage_center_local)
    assert not arena.contains_cage_local((3000.0, 3000.0))


def test_rotated_translated_field_round_trip():
    arena = fit_relative_arena(make_corners(angle_deg=37.0, offset=(1234.0, -876.0)))
    for local in ((0.0, 0.0), (6000.0, 0.0), (0.0, 4000.0), (6000.0, 4000.0), (1234.0, 2345.0)):
        world = transform(local, 37.0, (1234.0, -876.0))
        assert arena.world_to_local(world) == pytest.approx(local, abs=1e-6)
        assert arena.local_to_world(local) == pytest.approx(world, abs=1e-6)


def test_invalid_corner_labels_are_rejected():
    corners = make_corners()
    with pytest.raises(ArenaGeometryError):
        fit_relative_arena(
            FieldCorners(
                left_up=corners.left_up,
                left_down=corners.left_down,
                right_up=corners.right_down,
                right_down=corners.right_up,
            )
        )


def test_excessive_trapezoid_is_rejected():
    with pytest.raises(ArenaGeometryError):
        fit_relative_arena(
            FieldCorners(
                left_up=(0.0, 4000.0),
                left_down=(0.0, 0.0),
                right_up=(6000.0, 4000.0),
                right_down=(4500.0, 0.0),
            )
        )


def test_field_boundary_force_is_bounded_and_points_inward():
    arena = make_arena()
    margin = 300.0
    left = field_boundary_force((20.0, 2000.0), arena, margin, 2.0)
    right = field_boundary_force((5980.0, 2000.0), arena, margin, 2.0)
    top = field_boundary_force((3000.0, 3980.0), arena, margin, 2.0)
    bottom = field_boundary_force((3000.0, 20.0), arena, margin, 2.0)
    center = field_boundary_force((3000.0, 2000.0), arena, margin, 2.0)
    outside = field_boundary_force((6200.0, 2000.0), arena, margin, 2.0)
    assert left[0] > 0.0 and abs(left[1]) < 1e-9
    assert right[0] < 0.0 and abs(right[1]) < 1e-9
    assert top[1] < 0.0 and abs(top[0]) < 1e-9
    assert bottom[1] > 0.0 and abs(bottom[0]) < 1e-9
    assert center == (0.0, 0.0)
    assert outside[0] < 0.0
    assert math.isfinite(vec_norm(outside)) and vec_norm(outside) <= 6.0


def test_smooth_repulsion_is_continuous_and_zero_at_radius():
    radius = 500.0
    weight = 900.0
    just_inside = smooth_repulsion((radius - 1e-3, 0.0), radius, weight)
    at_radius = smooth_repulsion((radius, 0.0), radius, weight)
    just_outside = smooth_repulsion((radius + 1e-3, 0.0), radius, weight)
    assert at_radius == (0.0, 0.0)
    assert just_outside == (0.0, 0.0)
    assert vec_norm(just_inside) < 1e-6
    assert vec_norm(smooth_repulsion((0.5 * radius, 0.0), radius, weight)) > vec_norm(just_inside)


def test_coincident_herders_receive_opposite_separation_forces():
    positions = {1: (100.0, 100.0), 2: (100.0, 100.0)}
    first = pair_repulsion(1, (1, 2), positions, 300.0, 10.0)
    second = pair_repulsion(2, (1, 2), positions, 300.0, 10.0)
    assert first == pytest.approx((-second[0], -second[1]))
    assert vec_norm(first) > 0.0


def test_convex_hull_and_point_classification():
    points = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (1.0, 1.0)]
    hull = [points[index] for index in convex_hull_indices(points)]
    assert len(hull) == 4
    assert point_in_convex_polygon((1.0, 1.0), hull)
    assert point_in_convex_polygon((0.0, 1.0), hull)
    assert not point_in_convex_polygon((3.0, 1.0), hull)


def test_second_order_dynamics_limits_acceleration_and_speed():
    model = SecondOrderDynamics(seed=1)
    limits = DynamicsLimits(max_speed=2.0, max_acceleration=2.0, drag_per_second=0.0)
    first = model.step(1, (100.0, 0.0), 0.1, limits)
    assert first == pytest.approx((0.2, 0.0))
    velocity = first
    for _ in range(50):
        velocity = model.step(1, (100.0, 100.0), 0.1, limits)
    assert vec_norm(velocity) <= 2.0 + 1e-9
    assert all(math.isfinite(component) for component in velocity)
    model.reset(1)
    assert 1 not in model.velocity_by_id


def test_second_order_dynamics_rejects_negative_dt():
    model = SecondOrderDynamics()
    with pytest.raises(ValueError):
        model.step(1, (1.0, 0.0), -0.1, DynamicsLimits(1.0, 1.0, 0.0))


def test_second_order_dynamics_accepts_measured_mm_velocity_and_limits_acceleration():
    model = SecondOrderDynamics()
    limits = DynamicsLimits(
        max_speed=500.0,
        max_acceleration=100.0,
        drag_per_second=0.0,
    )
    assert model.observe_velocity(7, (700.0, 0.0), limits) == pytest.approx((500.0, 0.0))
    model.set_velocity(7, (200.0, 0.0), limits)
    velocity = model.step(7, (0.0, 1000.0), 0.1, limits)
    assert velocity == pytest.approx((200.0, 10.0))
    assert vec_norm((velocity[0] - 200.0, velocity[1])) <= limits.max_acceleration * 0.1 + 1e-9
    with pytest.raises(TypeError):
        model.set_velocity(7, (1000.0, 0.0))


def test_drag_and_control_share_the_same_total_acceleration_limit():
    model = SecondOrderDynamics(seed=1)
    limits = DynamicsLimits(max_speed=320.0, max_acceleration=960.0, drag_per_second=1.0)
    previous = model.observe_velocity(1, (320.0, 0.0), limits)
    velocity = model.step(1, (-960.0, 0.0), 0.05, limits)
    delta = (velocity[0] - previous[0], velocity[1] - previous[1])
    assert vec_norm(delta) <= limits.max_acceleration * 0.05 + 1e-9


def test_control_parameters_use_physical_units_and_robot_capture_margin():
    arena = make_arena()
    params = ControlParameters.from_arena(arena, robot_radius_mm=180.0)
    assert params.herder_dynamics.max_speed == pytest.approx(320.0)
    assert params.herder_dynamics.max_acceleration == pytest.approx(960.0)
    assert params.evader_dynamics.max_speed == pytest.approx(440.0)
    assert params.cage_hold_margin_mm >= 200.0
    assert params.boundary_margin_mm >= 220.0
    assert params.herder_repel_radius_mm >= 400.0
    assert params.evader_keepout_radius_mm >= 400.0
    assert params.obstacle_radius_mm >= 400.0


def test_control_step_stops_captured_evader_and_herders():
    arena = make_arena()
    params = ControlParameters.from_arena(arena)
    herders = (1, 2, 3, 4)
    evaders = (5,)
    observations = {
        1: observation(1, (500.0, 500.0)),
        2: observation(2, (5500.0, 500.0)),
        3: observation(3, (5500.0, 3500.0)),
        4: observation(4, (500.0, 2500.0)),
        5: observation(5, arena.cage_center_local),
    }
    result = compute_control_step(
        observations,
        herders,
        evaders,
        (),
        arena,
        params,
        SecondOrderDynamics(seed=2),
        0.05,
    )
    assert result.inside_cage_evaders == (5,)
    assert result.outside_cage_evaders == ()
    assert result.velocities[5] == (0.0, 0.0)
    assert all(result.velocities[marker_id] == (0.0, 0.0) for marker_id in herders)


def test_control_step_forms_targets_and_respects_model_limits():
    arena = make_arena()
    params = ControlParameters.from_arena(arena)
    herders = (1, 2, 3, 4)
    evaders = (5, 6)
    observations = {
        1: observation(1, (1000.0, 500.0)),
        2: observation(2, (5000.0, 500.0)),
        3: observation(3, (5200.0, 3000.0)),
        4: observation(4, (1800.0, 2200.0)),
        5: observation(5, (3500.0, 1700.0)),
        6: observation(6, (3900.0, 2100.0)),
    }
    result = compute_control_step(
        observations,
        herders,
        evaders,
        (),
        arena,
        params,
        SecondOrderDynamics(seed=3),
        0.05,
    )
    assert set(result.targets) == set(herders)
    assert len(result.herder_hull_ids) >= 3
    assert len(result.augmented_hull_herder_ids) >= 1
    selected_targets = {
        target_id for target_id in result.target_evader_by_herder.values() if target_id is not None
    }
    assert len(selected_targets) == 1
    assert all(vec_norm(result.velocities[marker_id]) <= params.herder_dynamics.max_speed for marker_id in herders)
    assert all(vec_norm(result.velocities[marker_id]) <= params.evader_dynamics.max_speed for marker_id in evaders)


def test_shared_target_prioritizes_uncontained_then_farthest_evader():
    arena = make_arena()
    positions = {
        5: (5800.0, 200.0),
        6: (3000.0, 2000.0),
        7: (2200.0, 3000.0),
    }
    target = select_shared_target_evader(
        (5, 6, 7),
        contained_evaders=(5,),
        positions=positions,
        cage_center=arena.cage_center_local,
    )
    assert target == 6
    assert select_shared_target_evader(
        (5, 6),
        contained_evaders=(5, 6),
        positions=positions,
        cage_center=arena.cage_center_local,
    ) == 5


def test_control_keeps_shared_target_until_it_enters_cage():
    arena = make_arena()
    params = ControlParameters.from_arena(arena)
    dynamics = SecondOrderDynamics(seed=1)
    positions = {
        1: (3300.0, 1000.0),
        2: (4300.0, 1200.0),
        3: (4100.0, 2400.0),
        4: (3200.0, 2200.0),
        10: (5000.0, 1000.0),
        11: (3000.0, 1800.0),
    }

    def step():
        observations = {
            marker_id: LocalObservation(marker_id, position, 0.0, 1.0)
            for marker_id, position in positions.items()
        }
        return compute_control_step(
            observations,
            (1, 2, 3, 4),
            (10, 11),
            (),
            arena,
            params,
            dynamics,
            0.05,
        )

    first = step()
    first_target = first.target_evader_by_herder[1]
    assert first_target in (10, 11)
    other_target = 11 if first_target == 10 else 10

    positions[first_target] = (3500.0, 1700.0)
    positions[other_target] = (5800.0, 300.0)
    second = step()
    assert set(second.target_evader_by_herder.values()) == {first_target}

    positions[first_target] = arena.cage_center_local
    third = step()
    assert set(third.target_evader_by_herder.values()) == {other_target}


def test_formation_slots_stay_inside_robot_safe_field_margin():
    arena = make_arena()
    params = ControlParameters.from_arena(arena, robot_radius_mm=150.0)
    positions = {
        1: (3000.0, 2000.0),
        2: (4000.0, 2000.0),
        3: (5000.0, 2000.0),
        4: (4000.0, 3000.0),
        5: (5900.0, 100.0),
    }
    targets = formation_targets((1, 2, 3, 4), positions, (5,), arena, params)
    margin = max(params.boundary_margin_mm, params.cage_hold_margin_mm)
    assert set(targets) == {1, 2, 3, 4}
    assert all(arena.contains_field_local(target, margin) for target in targets.values())
    minimum_spacing = 2.0 * params.robot_radius_mm + 40.0
    assert all(
        vec_norm((a[0] - b[0], a[1] - b[1])) >= minimum_spacing - 1e-9
        for a, b in combinations(targets.values(), 2)
    )
    assert all(
        not arena.contains_cage_local(target, -params.robot_radius_mm)
        for target in targets.values()
    )


def test_formation_slot_hysteresis_ignores_one_millimeter_grid_crossing():
    arena = make_arena()
    params = ControlParameters.from_arena(arena, robot_radius_mm=150.0)
    positions = {
        1: (2000.0, 500.0),
        2: (2600.0, 500.0),
        3: (3000.0, 1200.0),
        4: (2200.0, 1500.0),
        7: (2610.0, 1000.0),
    }
    first = formation_targets((1, 2, 3, 4), positions, (7,), arena, params)
    positions[7] = (2611.0, 1000.0)
    second = formation_targets(
        (1, 2, 3, 4),
        positions,
        (7,),
        arena,
        params,
        previous_targets=first,
    )
    assert second == first


def test_hull_gap_force_moves_toward_larger_angular_gap_and_is_bounded():
    center = (0.0, 0.0)
    marker_position = (0.5, math.sqrt(3.0) * 0.5)
    positions = {2: marker_position}
    hull_points = (
        (1.0, 0.0),
        marker_position,
        (-1.0, 0.0),
        (0.0, -1.0),
    )
    force = hull_gap_surround_force(
        2,
        (2,),
        hull_points,
        positions,
        center,
        maximum_force=100.0,
    )
    tangent_ccw = (-math.sin(math.pi / 3.0), math.cos(math.pi / 3.0))
    assert force[0] * tangent_ccw[0] + force[1] * tangent_ccw[1] > 0.0
    assert vec_norm(force) <= 100.0 + 1e-9


def test_capture_monitor_requires_every_active_evader_and_hold_time():
    monitor = CaptureMonitor(hold_sec=2.0)
    assert not monitor.update(10.0, (5, 6), (5, 6), (5, 6))
    assert not monitor.update(11.9, (5, 6), (5, 6), (5, 6))
    assert monitor.update(12.0, (5, 6), (5, 6), (5, 6))
    assert not monitor.update(12.1, (5, 6), (5,), (5,))
    assert monitor.started_at is None
    assert not monitor.update(20.0, (), (), ())


def test_active_roles_ignore_unusable_cars():
    herders, evaders = active_role_ids((1, 2, 3, 4), (5, 6), {1, 3, 4, 6})
    assert herders == (1, 3, 4)
    assert evaders == (6,)


def make_six_direction_calibration():
    result = {}
    for command, angle_deg in zip(("F", "RF", "RB", "B", "LB", "LF"), (0, -60, -120, 180, 120, 60)):
        angle = math.radians(angle_deg)
        result[command] = MotionCalibration(
            direction_world=(math.cos(angle), math.sin(angle)),
            yaw=0.0,
            displacement_mm=100.0,
            measured_speed_mm_s=300.0,
            calibration_pwm=110,
            calibrated_at_unix=time.time(),
        )
    return result


def test_calibration_coverage_and_yaw_compensated_command_choice():
    calibrations = make_six_direction_calibration()
    valid, _summary = valid_calibration_set(calibrations, 4, 150.0)
    assert valid
    assert calibration_max_gap_deg(calibrations) == pytest.approx(60.0)
    command, score = choose_calibrated_command(
        calibrations,
        desired_world=(0.0, 1.0),
        current_yaw=math.pi / 2.0,
        previous_command="",
        min_score=0.5,
        switch_margin=0.08,
    )
    assert command == "F"
    assert score == pytest.approx(1.0)


def test_bad_direction_coverage_is_rejected():
    calibrations = make_six_direction_calibration()
    clustered = {command: calibrations[command] for command in ("F", "RF", "LF", "LB")}
    valid, reason = valid_calibration_set(clustered, 4, 150.0)
    assert not valid
    assert "gap" in reason


def test_pwm_mapping_and_slew():
    assert speed_to_pwm(0.01, 1.0, 90, 170, 0.08) == 0
    assert speed_to_pwm(1.0, 1.0, 90, 170, 0.08) == 170
    assert 90 <= speed_to_pwm(0.5, 1.0, 90, 170, 0.08) <= 170
    assert slew_pwm(100, 160, 12) == 112
    assert slew_pwm(160, 100, 12) == 148
    assert slew_pwm(150, 0, 12) == 0
    pwm = 0
    credit = 0.0
    for _ in range(19):
        pwm, credit = slew_pwm_per_second(pwm, 100, 1.0, 0.05, credit)
        assert pwm == 0
    pwm, credit = slew_pwm_per_second(pwm, 100, 1.0, 0.05, credit)
    assert pwm == 1
    assert credit == pytest.approx(0.0, abs=1e-9)


def test_pwm_mapping_uses_measured_calibration_speed():
    assert speed_to_pwm(200.0, 320.0, 90, 170, 25.0, 200.0, 110) == 110
    assert speed_to_pwm(320.0, 320.0, 90, 170, 25.0, 200.0, 110) == 170


def test_motion_response_watchdog_requires_sustained_nonresponse():
    waiting = {}
    moving = CommandDecision("F", 1.0, 100)
    assert not motion_response_failed(1, moving, 0.0, 90, 10.0, 2.0, 15.0, waiting)
    assert not motion_response_failed(1, moving, 0.0, 90, 11.9, 2.0, 15.0, waiting)
    assert motion_response_failed(1, moving, 0.0, 90, 12.0, 2.0, 15.0, waiting)
    assert not motion_response_failed(1, moving, 20.0, 90, 12.1, 2.0, 15.0, waiting)
    assert waiting == {}


def test_invalid_cached_commands_are_removed_before_selection():
    calibrations = make_six_direction_calibration()
    calibrations["F"] = MotionCalibration(
        direction_world=(20.0, 0.0),
        yaw=0.0,
        displacement_mm=100.0,
        measured_speed_mm_s=300.0,
        calibration_pwm=110,
        calibrated_at_unix=time.time(),
    )
    filtered = filtered_calibrations(calibrations)
    assert "F" not in filtered
    assert len(filtered) == 5


def test_vicon_velocity_estimator_filters_physical_velocity():
    estimator = ViconVelocityEstimator(alpha=0.5)
    first = {1: LocalObservation(1, (100.0, 200.0), 0.0, 1.0)}
    second = {1: LocalObservation(1, (120.0, 200.0), 0.0, 1.1)}
    third = {1: LocalObservation(1, (130.0, 200.0), 0.0, 1.2)}
    assert estimator.update(first, (1,)) == {}
    assert estimator.update(second, (1,))[1] == pytest.approx((200.0, 0.0))
    duplicate = {1: LocalObservation(1, (999.0, 200.0), 0.0, 1.1)}
    assert estimator.update(duplicate, (1,)) == {}
    assert estimator.filtered[1] == pytest.approx((200.0, 0.0))
    assert estimator.update(third, (1,))[1] == pytest.approx((150.0, 0.0))
    estimator.reset(1)
    assert estimator.update(third, (1,)) == {}


def test_vicon_tracker_does_not_recount_the_same_frame():
    class FakeViconClient:
        frame = 42

        def GetFrame(self):
            return True

        def GetFrameNumber(self):
            return self.frame

        def GetSubjectRootSegmentName(self, subject_name):
            return "root"

        def GetSegmentGlobalTranslation(self, subject_name, root_segment):
            return (100.0, 200.0, 0.0), False

        def GetSegmentGlobalRotationEulerXYZ(self, subject_name, root_segment):
            return (0.0, 0.0, 0.25), False

    tracker = ViconSubjectTracker("unused", ("car",), 1.0)
    tracker.client = FakeViconClient()
    first = tracker.get_snapshot()
    second = tracker.get_snapshot()
    assert first.fresh_subjects == {"car"}
    assert second.fresh_subjects == set()
    assert second.poses["car"].seen_at == first.poses["car"].seen_at


def test_stop_command_clears_firmware_and_local_pwm():
    class FakeSocket:
        def __init__(self):
            self.sent = []

        def sendall(self, payload):
            self.sent.append(payload)

    controller = ExperimentCarController(1, "192.0.2.1", 0.1, 0.1)
    controller.sock = FakeSocket()
    controller.active = True
    controller.last_pwm = 150
    controller.last_motion = "F"
    assert controller.send_stop(force=True)
    assert controller.last_pwm == 0
    assert controller.sock.sent == [b"STOP\n", b"SPD 0\n"]

    controller.sock.sent.clear()
    controller.last_pwm = 140
    controller.last_motion = "F"
    assert controller.set_motion("STOP", 0, 1.0, 0.15)
    assert controller.sock.sent == [b"STOP\n", b"SPD 0\n"]

    controller.sock.sent.clear()
    controller.last_pwm = 100
    controller.last_motion = "F"
    controller.last_motion_at = 2.0
    assert controller.set_motion("F", 80, 2.01, 0.15)
    assert controller.sock.sent == [b"SPD 80\n", b"F\n"]


def test_normal_direction_change_ramps_old_direction_to_zero_first():
    controller = ExperimentCarController(1, "192.0.2.1", 0.1, 0.1)
    controller.last_motion = "F"
    controller.last_pwm = 100
    decision = rate_limited_motion_decision(controller, "B", -1.0, 150, 200.0, 0.1)
    assert decision == CommandDecision("F", -1.0, 80)

    controller.last_pwm = 0
    controller.pwm_slew_credit = 0.0
    decision = rate_limited_motion_decision(controller, "B", 1.0, 150, 200.0, 0.1)
    assert decision == CommandDecision("B", 1.0, 20)


def test_ping_verifies_car_identity_or_requires_discovery_proof():
    class FakeSocket:
        def __init__(self, response):
            self.response = response
            self.sent = []
            self.closed = False

        def sendall(self, payload):
            self.sent.append(payload)

        def recv(self, size):
            response, self.response = self.response[:size], self.response[size:]
            return response

        def close(self):
            self.closed = True

    controller = ExperimentCarController(3, "192.0.2.3", 0.1, 0.1)
    controller.sock = FakeSocket(b"PONG id=3\n")
    controller.active = True
    assert controller.ping()

    mismatch = ExperimentCarController(3, "192.0.2.50", 0.1, 0.1)
    mismatch.sock = FakeSocket(b"PONG id=50\n")
    mismatch.active = True
    assert not mismatch.ping()
    assert not mismatch.active
    assert "mismatch" in mismatch.failure_reason

    legacy = ExperimentCarController(3, "192.0.2.3", 0.1, 0.1, allow_plain_pong=True)
    legacy.sock = FakeSocket(b"PONG\n")
    legacy.active = True
    assert legacy.ping()


def test_connection_preflight_rejects_duplicate_ips_and_runs_in_parallel(monkeypatch):
    barrier = threading.Barrier(3, timeout=2.0)

    class FakeController:
        def __init__(
            self,
            marker_id,
            ip,
            connect_timeout_sec,
            send_timeout_sec,
            allow_plain_pong=False,
        ):
            self.marker_id = marker_id
            self.ip = ip
            self.allow_plain_pong = allow_plain_pong
            self.active = False
            self.failure_reason = ""

        def connect(self):
            barrier.wait()
            self.active = True
            return True

        def close(self, send_stop=False):
            self.active = False

    args = SimpleNamespace(
        skip_discovery=False,
        discovery_timeout=0.1,
        bind_ip=[],
        broadcast_ip=[],
        tcp_connect_timeout=0.1,
        tcp_send_timeout=0.1,
    )
    monkeypatch.setattr(capture_module, "ExperimentCarController", FakeController)
    monkeypatch.setattr(
        capture_module,
        "discover_car_ips",
        lambda *args, **kwargs: {
            1: "192.0.2.1",
            2: "192.0.2.2",
            3: "192.0.2.3",
        },
    )
    controllers, excluded = connect_requested_cars((1, 2, 3), {}, args)
    assert set(controllers) == {1, 2, 3}
    assert excluded == {}
    assert all(controller.allow_plain_pong for controller in controllers.values())

    monkeypatch.setattr(
        capture_module,
        "discover_car_ips",
        lambda *args, **kwargs: {
            1: "192.0.2.10",
            2: "192.0.2.10",
            3: "192.0.2.30",
        },
    )
    barrier = threading.Barrier(1, timeout=2.0)
    controllers, excluded = connect_requested_cars((1, 2, 3), {}, args)
    assert set(controllers) == {3}
    assert set(excluded) == {1, 2}


def test_calibration_cache_keeps_per_command_age_pwm_and_expected_identity(tmp_path):
    calibrations = make_six_direction_calibration()
    calibrations["F"] = MotionCalibration(
        direction_world=(1.0, 0.0),
        yaw=0.0,
        displacement_mm=100.0,
        measured_speed_mm_s=300.0,
        calibration_pwm=145,
        calibrated_at_unix=time.time() - 48.0 * 3600.0,
    )
    path = tmp_path / "calibration.json"
    save_calibration_file(
        path,
        {1: calibrations, 50: make_six_direction_calibration()},
        {1: "kedaya1", 50: "kedaya50"},
        "hardware-a",
    )
    loaded = load_calibration_file(path, {1: "kedaya1"}, "hardware-a", 24.0)
    assert set(loaded) == {1}
    assert "F" not in loaded[1]
    assert loaded[1]["RF"].calibration_pwm == 110
    assert load_calibration_file(path, {1: "kedaya1"}, "other-tag", 24.0) == {}


@pytest.mark.parametrize(
    "payload",
    (
        [],
        {"version": 3, "calibration_tag": "default", "cars": []},
        {
            "version": 3,
            "calibration_tag": "default",
            "cars": {"1": {"subject": "kedaya1", "commands": {"F": []}}},
        },
    ),
)
def test_malformed_calibration_cache_is_ignored(tmp_path, payload):
    path = tmp_path / "bad-calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_calibration_file(path, {1: "kedaya1"}) == {}


def test_closed_loop_single_evader_reaches_internal_cage_within_limits():
    arena = make_arena()
    params = ControlParameters.from_arena(arena, robot_radius_mm=150.0)
    dynamics = SecondOrderDynamics(seed=3)
    positions = {
        1: (4700.0, 600.0),
        2: (5000.0, 1100.0),
        3: (5000.0, 1900.0),
        4: (4600.0, 2400.0),
        7: (4000.0, 1500.0),
    }
    dt = 0.05
    captured = False
    for step in range(600):
        observations = {
            marker_id: LocalObservation(marker_id, position, 0.0, step * dt)
            for marker_id, position in positions.items()
        }
        result = compute_control_step(
            observations,
            active_herders=(1, 2, 3, 4),
            active_evaders=(7,),
            obstacle_ids=(),
            arena=arena,
            params=params,
            dynamics=dynamics,
            dt=dt,
        )
        for marker_id, velocity in result.velocities.items():
            limit = params.herder_dynamics if marker_id != 7 else params.evader_dynamics
            assert vec_norm(velocity) <= limit.max_speed + 1e-9
            assert vec_norm(result.accelerations[marker_id]) <= limit.max_acceleration + 1e-9
            positions[marker_id] = (
                positions[marker_id][0] + velocity[0] * dt,
                positions[marker_id][1] + velocity[1] * dt,
            )
        if arena.contains_cage_local(positions[7], params.cage_hold_margin_mm):
            captured = True
            break
    assert captured


def test_closed_loop_two_evaders_are_herded_without_target_oscillation():
    arena = make_arena()
    params = ControlParameters.from_arena(arena, robot_radius_mm=150.0)
    dynamics = SecondOrderDynamics(seed=6)
    positions = {
        1: (4800.0, 200.0),
        2: (5200.0, 700.0),
        3: (5300.0, 1400.0),
        4: (5100.0, 2100.0),
        5: (4600.0, 2600.0),
        6: (4000.0, 2800.0),
        7: (4300.0, 900.0),
        8: (3500.0, 3700.0),
    }
    targets = []
    captured = set()
    dt = 0.05
    for step in range(500):
        observations = {
            marker_id: LocalObservation(marker_id, position, 0.0, step * dt)
            for marker_id, position in positions.items()
        }
        result = compute_control_step(
            observations,
            (1, 2, 3, 4, 5, 6),
            (7, 8),
            (),
            arena,
            params,
            dynamics,
            dt,
        )
        target = result.target_evader_by_herder[1]
        if not targets or target != targets[-1]:
            targets.append(target)
        for marker_id, velocity in result.velocities.items():
            positions[marker_id] = (
                positions[marker_id][0] + velocity[0] * dt,
                positions[marker_id][1] + velocity[1] * dt,
            )
        captured.update(
            marker_id
            for marker_id in (7, 8)
            if arena.contains_cage_local(positions[marker_id], params.cage_hold_margin_mm)
        )
        if len(captured) == 2:
            break
    assert captured == {7, 8}
    assert targets == [8, 7]


def test_id_parser_and_cage_capacity():
    assert parse_id_spec("1-3,5;7") == (1, 2, 3, 5, 7)
    with pytest.raises(ValueError):
        parse_id_spec("0")
    arena = make_arena()
    params = ControlParameters.from_arena(arena)
    enough, usable, required = cage_has_capacity(arena, 6, 300.0, params.cage_hold_margin_mm)
    assert enough
    assert usable > required


@pytest.mark.parametrize(
    ("flag", "value"),
    (
        ("--actual-speed-limit-factor", "nan"),
        ("--corner-shift-ratio", "nan"),
        ("--max-runtime-sec", "inf"),
        ("--command-refresh", "0.31"),
    ),
)
def test_cli_rejects_nonfinite_and_watchdog_unsafe_values(monkeypatch, flag, value):
    monkeypatch.setattr(sys, "argv", ["relative_cage_capture.py", flag, value])
    with pytest.raises(SystemExit):
        parse_args()
