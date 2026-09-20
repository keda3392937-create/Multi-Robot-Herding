import math
import unittest

from relative_cage_core import FieldCorners, MotionLimits, build_relative_arena
from relative_vicon_cage_capture import (
    CarSpec,
    CommandCalibration,
    SafeTcpCarController,
    TrackedPose,
    ViconSnapshot,
    cage_capacity_ok,
    calibration_angle_gap_deg,
    calibration_position_safe,
    calibration_valid,
    choose_calibrated_command,
    parse_id_spec,
    slew_pwm,
    velocity_to_pwm,
)


def axis_aligned_arena():
    return build_relative_arena(
        FieldCorners(
            left_up=(0.0, 4000.0),
            left_down=(0.0, 0.0),
            right_up=(6000.0, 4000.0),
            right_down=(6000.0, 0.0),
        ),
        cage_width_ratio=0.30,
        cage_depth_ratio=0.30,
    )


def calibration(direction, yaw=0.0):
    return CommandCalibration(
        direction_world=direction,
        yaw_world=yaw,
        displacement_mm=100.0,
        speed_mm_s=300.0,
        yaw_change_deg=0.0,
    )


class ParsingAndCalibrationTests(unittest.TestCase):
    def test_parse_id_ranges_and_deduplicates(self):
        self.assertEqual(parse_id_spec("1-3,5;3,7"), (1, 2, 3, 5, 7))

    def test_calibration_coverage(self):
        commands = {
            f"C{index}": calibration((math.cos(angle), math.sin(angle)))
            for index, angle in enumerate(
                [0.0, math.pi / 3, 2 * math.pi / 3, math.pi, 4 * math.pi / 3, 5 * math.pi / 3]
            )
        }
        self.assertAlmostEqual(calibration_angle_gap_deg(commands), 60.0, places=6)
        valid, reason = calibration_valid(commands, min_commands=4, max_gap_deg=150.0)
        self.assertTrue(valid, reason)

        clustered = {
            "A": calibration((1.0, 0.0)),
            "B": calibration((0.9, 0.1)),
            "C": calibration((0.8, 0.2)),
            "D": calibration((0.7, 0.3)),
        }
        valid, _reason = calibration_valid(clustered, min_commands=4, max_gap_deg=150.0)
        self.assertFalse(valid)

    def test_command_selection_applies_yaw_and_minimum_score(self):
        commands = {
            "F": calibration((1.0, 0.0)),
            "B": calibration((-1.0, 0.0)),
        }
        command, score = choose_calibrated_command(
            commands,
            desired_world_velocity=(0.0, 1.0),
            current_yaw=math.pi / 2,
            current_command="STOP",
            min_score=0.5,
            hysteresis=0.05,
        )
        self.assertEqual(command, "F")
        self.assertGreater(score, 0.999)

        command, score = choose_calibrated_command(
            {"F": commands["F"]},
            desired_world_velocity=(-1.0, 0.0),
            current_yaw=0.0,
            current_command="STOP",
            min_score=0.5,
            hysteresis=0.05,
        )
        self.assertEqual(command, "STOP")
        self.assertLess(score, 0.0)

    def test_command_hysteresis_keeps_nearly_equal_direction(self):
        angle_60 = math.radians(60.0)
        commands = {
            "F": calibration((1.0, 0.0)),
            "RF": calibration((math.cos(angle_60), math.sin(angle_60))),
        }
        desired_angle = math.radians(35.0)
        command, _score = choose_calibrated_command(
            commands,
            (math.cos(desired_angle), math.sin(desired_angle)),
            current_yaw=0.0,
            current_command="F",
            min_score=0.5,
            hysteresis=0.10,
        )
        self.assertEqual(command, "F")


class ActuationTests(unittest.TestCase):
    def test_velocity_maps_to_pwm_and_stop_deadband(self):
        limits = MotionLimits(400.0, 1000.0, 1.0)
        self.assertEqual(velocity_to_pwm((0.0, 0.0), limits, 80, 200, 0.1), 0)
        self.assertEqual(velocity_to_pwm((200.0, 0.0), limits, 80, 200, 0.1), 140)
        self.assertEqual(velocity_to_pwm((800.0, 0.0), limits, 80, 200, 0.1), 200)

    def test_pwm_slew_and_immediate_stop(self):
        self.assertEqual(slew_pwm(120, 200, 0.1, 100.0), 130)
        self.assertEqual(slew_pwm(120, 0, 0.1, 100.0), 0)

    def test_tcp_send_failure_is_latched_without_reconnect(self):
        class FailingSocket:
            def sendall(self, _data):
                raise OSError("simulated link loss")

            def close(self):
                pass

        controller = SafeTcpCarController(CarSpec(1, "kedaya1", "192.0.2.1"), 0.1, 0.1)
        controller.sock = FailingSocket()
        controller.active = True
        self.assertFalse(controller.send_line("F"))
        self.assertFalse(controller.active)
        self.assertIsNone(controller.sock)
        self.assertIn("simulated link loss", controller.failure_reason)
        self.assertFalse(controller.send_line("F"))


class SafetyPreflightTests(unittest.TestCase):
    def test_calibration_clearance_respects_walls_but_not_entry(self):
        arena = axis_aligned_arena()

        def pose_at(local):
            return TrackedPose("kedaya1", arena.local_to_world(local), 0.0, 1.0)

        near_top = pose_at((3000.0, 3950.0))
        snapshot = ViconSnapshot(1, 1.0, {"car:1": near_top}, ("car:1",))
        safe, _reason = calibration_position_safe(
            1,
            near_top,
            snapshot,
            active_ids=(1,),
            arena=arena,
            clearance_mm=200.0,
            separation_mm=400.0,
        )
        self.assertFalse(safe)

        near_solid_bottom = pose_at((3000.0, 50.0))
        snapshot = ViconSnapshot(2, 1.0, {"car:1": near_solid_bottom}, ("car:1",))
        safe, _reason = calibration_position_safe(
            1,
            near_solid_bottom,
            snapshot,
            active_ids=(1,),
            arena=arena,
            clearance_mm=200.0,
            separation_mm=400.0,
        )
        self.assertFalse(safe)

        at_open_entry = pose_at((900.0, 50.0))
        snapshot = ViconSnapshot(3, 1.0, {"car:1": at_open_entry}, ("car:1",))
        safe, reason = calibration_position_safe(
            1,
            at_open_entry,
            snapshot,
            active_ids=(1,),
            arena=arena,
            clearance_mm=200.0,
            separation_mm=400.0,
        )
        self.assertTrue(safe, reason)

    def test_cage_capacity_check(self):
        arena = axis_aligned_arena()
        okay, usable, required = cage_capacity_ok(arena, evader_count=6, robot_radius_mm=120.0)
        self.assertTrue(okay)
        self.assertGreater(usable, required)


if __name__ == "__main__":
    unittest.main()
