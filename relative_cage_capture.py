from __future__ import annotations

import argparse
import concurrent.futures
import csv
import ipaddress
import json
import math
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


VICON_SDK_PATH = Path(r"D:\ViconDataStream\Win64\Python\vicon_dssdk")
if VICON_SDK_PATH.exists():
    sys.path.insert(0, str(VICON_SDK_PATH))

from vicon_dssdk import ViconDataStream

from relative_cage_core import (
    ArenaGeometryError,
    CaptureMonitor,
    ControlParameters,
    FieldCorners,
    LocalObservation,
    RelativeArena,
    SecondOrderDynamics,
    Vec2,
    active_role_ids,
    angle_wrap,
    compute_control_step,
    fit_relative_arena,
    vec_dot,
    vec_norm,
    vec_normalize,
    vec_scale,
    vec_sub,
)
from six_car_repel import connect_wifi, discover_car_ips


DEFAULT_VICON_HOST = "192.168.30.100"
DEFAULT_WIFI_SSID = "YAYA"
CONTROL_PORT = 23
MOTION_COMMANDS = ("F", "RF", "RB", "B", "LB", "LF")
CORNER_KEYS = ("left-up", "left-down", "right-up", "right-down")
CALIBRATION_VERSION = 3
FIRMWARE_COMMAND_TIMEOUT_SEC = 0.70
MAX_SAFE_CONTROL_PERIOD_SEC = 0.20
MAX_SAFE_COMMAND_REFRESH_SEC = 0.30
MAX_SAFE_POSE_AGE_SEC = 0.30
MAX_SAFE_TCP_IO_TIMEOUT_SEC = 0.20


@dataclass(frozen=True)
class WorldPose:
    subject_name: str
    position: Vec2
    yaw: float
    seen_at: float
    frame_number: int


@dataclass(frozen=True)
class ViconSnapshot:
    frame_number: int
    received_at: float
    poses: Dict[str, WorldPose]
    fresh_subjects: Set[str]


@dataclass(frozen=True)
class MotionCalibration:
    direction_world: Vec2
    yaw: float
    displacement_mm: float
    measured_speed_mm_s: float
    calibration_pwm: int
    calibrated_at_unix: float


@dataclass(frozen=True)
class CommandDecision:
    command: str
    score: float
    pwm: int


class ViconSubjectTracker:
    def __init__(self, host: str, subject_names: Iterable[str], max_pose_age_sec: float):
        self.host = host
        self.subject_names = tuple(dict.fromkeys(subject_names))
        self.max_pose_age_sec = max_pose_age_sec
        self.client = ViconDataStream.Client()
        self.root_segments: Dict[str, str] = {}
        self.cached_poses: Dict[str, WorldPose] = {}
        self.last_frame_number: Optional[int] = None

    def connect(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        self.client.SetConnectionTimeout(1000)
        print(f"Connecting to Vicon server {self.host} ...")
        while not self.client.IsConnected() and time.monotonic() < deadline:
            try:
                self.client.Connect(self.host)
            except ViconDataStream.DataStreamException as exc:
                print(f"Vicon connect failed: {exc}")
                time.sleep(0.5)
        if not self.client.IsConnected():
            raise RuntimeError(f"Could not connect to Vicon server {self.host} within {timeout_sec:.1f}s")

        self.client.EnableSegmentData()
        self.client.SetStreamMode(ViconDataStream.Client.StreamMode.EClientPull)
        self.client.SetAxisMapping(
            ViconDataStream.Client.AxisMapping.EForward,
            ViconDataStream.Client.AxisMapping.ELeft,
            ViconDataStream.Client.AxisMapping.EUp,
        )
        print("Vicon connected.")

    def get_snapshot(self) -> ViconSnapshot:
        received_at = time.monotonic()
        fresh_subjects: Set[str] = set()
        frame_number = -1
        try:
            has_frame = self.client.GetFrame()
        except ViconDataStream.DataStreamException as exc:
            print(f"Vicon GetFrame failed: {exc}")
            has_frame = False

        if has_frame:
            try:
                frame_number = int(self.client.GetFrameNumber())
            except (TypeError, ValueError, ViconDataStream.DataStreamException):
                frame_number = -1

            if frame_number >= 0 and frame_number == self.last_frame_number:
                has_frame = False
            elif frame_number >= 0:
                self.last_frame_number = frame_number

        if has_frame:
            for subject_name in self.subject_names:
                root_segment = self.root_segments.get(subject_name)
                if root_segment is None:
                    try:
                        root_segment = self.client.GetSubjectRootSegmentName(subject_name)
                    except ViconDataStream.DataStreamException:
                        continue
                    self.root_segments[subject_name] = root_segment

                try:
                    translation, translation_occluded = self.client.GetSegmentGlobalTranslation(
                        subject_name,
                        root_segment,
                    )
                    rotation, rotation_occluded = self.client.GetSegmentGlobalRotationEulerXYZ(
                        subject_name,
                        root_segment,
                    )
                except ViconDataStream.DataStreamException:
                    continue
                if translation_occluded or rotation_occluded:
                    continue

                x_mm, y_mm, _z_mm = translation
                _rx, _ry, yaw = rotation
                if not all(math.isfinite(float(value)) for value in (x_mm, y_mm, yaw)):
                    continue
                self.cached_poses[subject_name] = WorldPose(
                    subject_name=subject_name,
                    position=(float(x_mm), float(y_mm)),
                    yaw=float(yaw),
                    seen_at=received_at,
                    frame_number=frame_number,
                )
                fresh_subjects.add(subject_name)

        active = {
            subject_name: pose
            for subject_name, pose in self.cached_poses.items()
            if received_at - pose.seen_at <= self.max_pose_age_sec
        }
        return ViconSnapshot(
            frame_number=frame_number,
            received_at=received_at,
            poses=active,
            fresh_subjects=fresh_subjects,
        )

    def close(self) -> None:
        try:
            if self.client.IsConnected():
                self.client.Disconnect()
        except (AttributeError, ViconDataStream.DataStreamException):
            pass


class ExperimentCarController:
    """One-shot TCP controller: a failed car stays excluded for this run."""

    def __init__(
        self,
        marker_id: int,
        ip: str,
        connect_timeout_sec: float,
        send_timeout_sec: float,
        allow_plain_pong: bool = False,
    ):
        self.marker_id = marker_id
        self.ip = ip
        self.connect_timeout_sec = connect_timeout_sec
        self.send_timeout_sec = send_timeout_sec
        self.allow_plain_pong = allow_plain_pong
        self.sock: Optional[socket.socket] = None
        self.active = False
        self.failure_reason = ""
        self.last_motion = ""
        self.last_motion_at = 0.0
        self.last_pwm: Optional[int] = None
        self.pwm_slew_credit = 0.0
        self.last_ping_at = 0.0

    def connect(self) -> bool:
        self.close(send_stop=False)
        try:
            sock = socket.create_connection((self.ip, CONTROL_PORT), timeout=self.connect_timeout_sec)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(self.send_timeout_sec)
            self.sock = sock
            self.active = True
            self.failure_reason = ""
            self.last_motion = ""
            self.last_motion_at = 0.0
            self.last_pwm = None
            self.pwm_slew_credit = 0.0
            self.last_ping_at = 0.0
            if not self.send_stop(force=True) or not self.ping():
                return False
            print(f"Connected car{self.marker_id} at {self.ip}:{CONTROL_PORT}")
            return self.active
        except OSError as exc:
            self.failure_reason = f"initial TCP connection failed: {exc}"
            self.close(send_stop=False)
            return False

    def _send_line(self, line: str) -> bool:
        if not self.active or self.sock is None:
            return False
        try:
            self.sock.sendall((line + "\n").encode("ascii"))
            return True
        except OSError as exc:
            self.failure_reason = f"TCP send failed: {exc}"
            self.close(send_stop=False)
            return False

    def set_motion(self, command: str, pwm: int, now: float, refresh_sec: float) -> bool:
        pwm = max(0, min(255, int(pwm)))
        if command not in MOTION_COMMANDS and command != "STOP":
            command = "STOP"
        if command == "STOP":
            return self.send_stop(force=self.last_motion != "STOP" or self.last_pwm != 0)
        pwm_changed = self.last_pwm != pwm
        if pwm_changed:
            if not self._send_line(f"SPD {pwm}"):
                return False
            self.last_pwm = pwm
        if pwm_changed or command != self.last_motion or now - self.last_motion_at >= refresh_sec:
            if not self._send_line(command):
                return False
            self.last_motion = command
            self.last_motion_at = now
        return True

    def send_stop(self, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and self.last_motion == "STOP" and self.last_pwm == 0 and now - self.last_motion_at < 0.20:
            return self.active
        if not self._send_line("STOP"):
            return False
        self.last_motion = "STOP"
        self.last_motion_at = now
        if (force or self.last_pwm != 0) and not self._send_line("SPD 0"):
            return False
        self.last_pwm = 0
        self.pwm_slew_credit = 0.0
        return True

    def ping(self) -> bool:
        if not self._send_line("PING") or self.sock is None:
            return False
        try:
            response = bytearray()
            while len(response) < 64:
                chunk = self.sock.recv(64 - len(response))
                if not chunk:
                    raise ConnectionError("TCP peer closed during PING")
                response.extend(chunk)
                if b"\n" in response:
                    break
            line = bytes(response).splitlines()[0].strip().upper()
            if line == b"PONG" and self.allow_plain_pong:
                pass
            elif line.startswith(b"PONG ID="):
                try:
                    response_id = int(line.removeprefix(b"PONG ID="))
                except ValueError as exc:
                    raise ConnectionError(f"invalid PING identity response: {line!r}") from exc
                if response_id != self.marker_id:
                    raise ConnectionError(
                        f"PING identity mismatch: expected car{self.marker_id}, received car{response_id}"
                    )
            else:
                raise ConnectionError(f"unexpected PING response: {line!r}")
            self.last_ping_at = time.monotonic()
            return True
        except OSError as exc:
            self.failure_reason = f"TCP PING failed: {exc}"
        except ConnectionError as exc:
            self.failure_reason = str(exc)
        self.close(send_stop=False)
        return False

    def close(self, send_stop: bool = True) -> None:
        if self.sock is not None:
            if send_stop and self.active:
                try:
                    self.sock.sendall(b"STOP\nSPD 0\n")
                except OSError:
                    pass
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.active = False


def parse_id_spec(text: str) -> Tuple[int, ...]:
    result: List[int] = []
    seen = set()
    normalized = text.replace(";", ",").replace(" ", ",")
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start_id = int(start_text)
            end_id = int(end_text)
            step = 1 if start_id <= end_id else -1
            values = range(start_id, end_id + step, step)
        else:
            values = (int(token),)
        for marker_id in values:
            if not 1 <= marker_id <= 255:
                raise ValueError("Car IDs must be in the range 1..255")
            if marker_id not in seen:
                result.append(marker_id)
                seen.add(marker_id)
    if not result:
        raise ValueError("ID list cannot be empty")
    return tuple(result)


def parse_overrides(values: Sequence[str], label: str, valid_ids: Sequence[int]) -> Dict[int, str]:
    valid = set(valid_ids)
    result: Dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --{label} value {value!r}; expected ID=VALUE")
        marker_text, override = value.split("=", 1)
        marker_id = int(marker_text)
        if marker_id not in valid:
            raise ValueError(f"Invalid car ID {marker_id} for --{label}")
        if not override.strip():
            raise ValueError(f"Empty value for --{label} car{marker_id}")
        result[marker_id] = override.strip()
    return result


def format_ids(ids: Iterable[int]) -> str:
    values = tuple(ids)
    return ",".join(str(marker_id) for marker_id in values) if values else "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline-inspired real-car herding with a Vicon-defined relative field and an "
            "internal top-left virtual cage. Without --arm, only the field geometry is read."
        )
    )
    parser.add_argument("--herders", default="1-6")
    parser.add_argument("--evaders", default="7-16")
    parser.add_argument("--subject", action="append", default=[], metavar="ID=NAME")
    parser.add_argument("--car-ip", action="append", default=[], metavar="ID=IP")

    parser.add_argument("--vicon-host", default=DEFAULT_VICON_HOST)
    parser.add_argument("--left-up-subject", default="left-up")
    parser.add_argument("--left-down-subject", default="left-down")
    parser.add_argument("--right-up-subject", default="right-up")
    parser.add_argument("--right-down-subject", default="right-down")
    parser.add_argument("--vicon-connect-timeout", type=float, default=15.0)
    parser.add_argument("--pose-max-age", type=float, default=0.25)
    parser.add_argument("--corner-samples", type=int, default=25)
    parser.add_argument("--corner-sample-timeout", type=float, default=5.0)
    parser.add_argument("--corner-shift-ratio", type=float, default=0.02)
    parser.add_argument("--initial-visibility-sec", type=float, default=1.5)
    parser.add_argument("--initial-visibility-ratio", type=float, default=0.70)

    parser.add_argument("--cage-width-ratio", type=float, default=0.40)
    parser.add_argument("--cage-height-ratio", type=float, default=0.40)
    parser.add_argument("--robot-diameter-mm", type=float, default=300.0)
    parser.add_argument("--allow-small-cage", action="store_true")

    parser.add_argument("--wifi-ssid", default=DEFAULT_WIFI_SSID)
    parser.add_argument("--wifi-password", default=os.environ.get("VICON_CAR_WIFI_PASSWORD", ""))
    parser.add_argument("--wifi-wait", type=float, default=4.0)
    parser.add_argument("--skip-wifi", action="store_true")
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--discovery-timeout", type=float, default=5.0)
    parser.add_argument("--bind-ip", action="append", default=[])
    parser.add_argument("--broadcast-ip", action="append", default=[])
    parser.add_argument("--tcp-connect-timeout", type=float, default=0.60)
    parser.add_argument("--tcp-send-timeout", type=float, default=0.08)
    parser.add_argument("--ping-interval", type=float, default=1.0)

    parser.add_argument("--arm", action="store_true", help="Allow calibration and motor commands.")
    parser.add_argument("--passive-evaders", action="store_true")
    parser.add_argument("--allow-partial-vicon", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--control-period", type=float, default=0.05)
    parser.add_argument("--command-refresh", type=float, default=0.15)
    parser.add_argument("--status-interval", type=float, default=0.50)
    parser.add_argument("--capture-hold-sec", type=float, default=2.0)
    parser.add_argument("--max-runtime-sec", type=float, default=300.0)

    parser.add_argument("--herder-min-pwm", type=int, default=90)
    parser.add_argument("--herder-max-pwm", type=int, default=170)
    parser.add_argument("--evader-min-pwm", type=int, default=95)
    parser.add_argument("--evader-max-pwm", type=int, default=230)
    parser.add_argument("--pwm-slew-per-sec", type=float, default=240.0)
    parser.add_argument("--pwm-slew-per-cycle", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--motion-deadband", type=float, default=25.0, help="Command speed deadband in mm/s.")
    parser.add_argument("--measured-velocity-alpha", type=float, default=0.35)
    parser.add_argument("--actual-speed-limit-factor", type=float, default=1.75)
    parser.add_argument("--motion-response-timeout", type=float, default=2.0)
    parser.add_argument("--motion-response-min-speed", type=float, default=15.0)
    parser.add_argument("--min-direction-score", type=float, default=0.50)
    parser.add_argument("--command-switch-margin", type=float, default=0.08)

    parser.add_argument("--calibration-file", type=Path, default=Path("relative_cage_calibration.json"))
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--calibration-tag", default="default")
    parser.add_argument("--calibration-max-age-hours", type=float, default=24.0)
    parser.add_argument("--calibration-speed", type=int, default=110)
    parser.add_argument("--calibration-move-sec", type=float, default=0.30)
    parser.add_argument("--calibration-settle-sec", type=float, default=0.30)
    parser.add_argument("--calibration-samples", type=int, default=4)
    parser.add_argument("--min-calibration-displacement-ratio", type=float, default=0.006)
    parser.add_argument("--max-calibration-displacement-ratio", type=float, default=0.12)
    parser.add_argument("--calibration-clearance-ratio", type=float, default=0.06)
    parser.add_argument("--calibration-separation-ratio", type=float, default=0.10)
    parser.add_argument("--max-calibration-yaw-change-deg", type=float, default=15.0)
    parser.add_argument("--min-calibrated-commands", type=int, default=4)
    parser.add_argument("--max-calibration-angle-gap-deg", type=float, default=150.0)

    parser.add_argument("--log-root", type=Path, default=Path("relative_cage_runs"))
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    if args.pwm_slew_per_cycle is not None:
        args.pwm_slew_per_sec = args.pwm_slew_per_cycle / max(args.control_period, 1e-9)

    finite_float_fields = (
        "vicon_connect_timeout",
        "pose_max_age",
        "corner_sample_timeout",
        "corner_shift_ratio",
        "initial_visibility_sec",
        "initial_visibility_ratio",
        "cage_width_ratio",
        "cage_height_ratio",
        "robot_diameter_mm",
        "wifi_wait",
        "discovery_timeout",
        "tcp_connect_timeout",
        "tcp_send_timeout",
        "ping_interval",
        "control_period",
        "command_refresh",
        "status_interval",
        "capture_hold_sec",
        "max_runtime_sec",
        "pwm_slew_per_sec",
        "motion_deadband",
        "measured_velocity_alpha",
        "actual_speed_limit_factor",
        "motion_response_timeout",
        "motion_response_min_speed",
        "min_direction_score",
        "command_switch_margin",
        "calibration_max_age_hours",
        "calibration_move_sec",
        "calibration_settle_sec",
        "min_calibration_displacement_ratio",
        "max_calibration_displacement_ratio",
        "calibration_clearance_ratio",
        "calibration_separation_ratio",
        "max_calibration_yaw_change_deg",
        "max_calibration_angle_gap_deg",
    )
    invalid_finite = [
        field_name
        for field_name in finite_float_fields
        if not math.isfinite(float(getattr(args, field_name)))
    ]
    if invalid_finite:
        parser.error("numeric arguments must be finite: " + ", ".join(invalid_finite))

    if args.control_period <= 0.0 or args.command_refresh <= 0.0 or args.status_interval <= 0.0:
        parser.error("control timing values must be positive")
    if args.control_period > MAX_SAFE_CONTROL_PERIOD_SEC:
        parser.error(
            f"control-period must be <= {MAX_SAFE_CONTROL_PERIOD_SEC:.2f}s "
            f"for the {FIRMWARE_COMMAND_TIMEOUT_SEC:.2f}s firmware watchdog"
        )
    if args.command_refresh > MAX_SAFE_COMMAND_REFRESH_SEC:
        parser.error(
            f"command-refresh must be <= {MAX_SAFE_COMMAND_REFRESH_SEC:.2f}s "
            f"for the {FIRMWARE_COMMAND_TIMEOUT_SEC:.2f}s firmware watchdog"
        )
    if (
        args.pose_max_age <= 0.0
        or args.pose_max_age > MAX_SAFE_POSE_AGE_SEC
        or args.corner_samples < 3
        or args.corner_sample_timeout <= 0.0
        or args.vicon_connect_timeout <= 0.0
    ):
        parser.error(
            f"pose-max-age must be in (0, {MAX_SAFE_POSE_AGE_SEC:.2f}], "
            "Vicon timeout positive, and corner-samples at least 3"
        )
    if args.initial_visibility_sec <= 0.0 or args.discovery_timeout < 0.0 or args.wifi_wait < 0.0:
        parser.error("visibility time must be positive; discovery and Wi-Fi waits cannot be negative")
    if not 0.0 < args.initial_visibility_ratio <= 1.0:
        parser.error("initial-visibility-ratio must be in (0, 1]")
    if not 0.0 <= args.min_direction_score <= 1.0:
        parser.error("min-direction-score must be in [0, 1]")
    if args.tcp_connect_timeout <= 0.0 or args.tcp_send_timeout <= 0.0 or args.ping_interval <= 0.0:
        parser.error("TCP timeouts and ping interval must be positive")
    if args.tcp_send_timeout > MAX_SAFE_TCP_IO_TIMEOUT_SEC:
        parser.error(f"tcp-send-timeout must be <= {MAX_SAFE_TCP_IO_TIMEOUT_SEC:.2f}s")
    if not (0.05 <= args.cage_width_ratio <= 0.80 and 0.05 <= args.cage_height_ratio <= 0.80):
        parser.error("cage width and height ratios must each be in [0.05, 0.80]")
    if not 50.0 <= args.robot_diameter_mm <= 1000.0:
        parser.error("robot-diameter-mm must be in [50, 1000]")
    if not (1 <= args.herder_min_pwm <= args.herder_max_pwm <= 255):
        parser.error("herder PWM values must satisfy 1 <= min <= max <= 255")
    if not (1 <= args.evader_min_pwm <= args.evader_max_pwm <= 255):
        parser.error("evader PWM values must satisfy 1 <= min <= max <= 255")
    if args.pwm_slew_per_sec <= 0.0 or args.motion_deadband < 0.0:
        parser.error("PWM slew must be positive and motion deadband cannot be negative")
    if not 0.0 < args.measured_velocity_alpha <= 1.0:
        parser.error("measured-velocity-alpha must be in (0, 1]")
    if not 1.0 < args.actual_speed_limit_factor <= 3.0:
        parser.error("actual-speed-limit-factor must be in (1, 3]")
    if not 0.0 < args.motion_response_timeout <= 10.0 or args.motion_response_min_speed <= 0.0:
        parser.error("motion response timeout and minimum speed must be positive")
    if not 0.0 <= args.command_switch_margin < 1.0:
        parser.error("command-switch-margin must be in [0, 1)")
    if args.calibration_move_sec <= 0.0 or args.calibration_settle_sec < 0.0:
        parser.error("calibration move time must be positive and settle time cannot be negative")
    if args.calibration_samples < 2:
        parser.error("calibration-samples must be at least 2")
    if not 1 <= args.calibration_speed <= 255:
        parser.error("calibration-speed must be in 1..255")
    if args.calibration_max_age_hours <= 0.0 or not args.calibration_tag.strip():
        parser.error("calibration max age must be positive and calibration-tag cannot be empty")
    if not (
        0.0 < args.min_calibration_displacement_ratio
        < args.max_calibration_displacement_ratio
        <= 0.25
    ):
        parser.error("calibration displacement ratios must satisfy 0 < min < max <= 0.25")
    if args.calibration_clearance_ratio < 0.0 or args.calibration_separation_ratio < 0.0:
        parser.error("calibration clearance and separation ratios cannot be negative")
    if not 1 <= args.min_calibrated_commands <= len(MOTION_COMMANDS):
        parser.error(f"min-calibrated-commands must be in 1..{len(MOTION_COMMANDS)}")
    if not 0.0 < args.max_calibration_angle_gap_deg <= 360.0:
        parser.error("max-calibration-angle-gap-deg must be in (0, 360]")
    if not 0.0 < args.corner_shift_ratio <= 0.10:
        parser.error("corner shift ratio must be in (0, 0.10]")
    if not 0.0 < args.max_calibration_yaw_change_deg <= 90.0:
        parser.error("max-calibration-yaw-change-deg must be in (0, 90]")
    if args.capture_hold_sec <= 0.0 or args.max_runtime_sec <= 0.0:
        parser.error("capture-hold-sec and max-runtime-sec must be positive")
    return args


def corner_subjects(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "left-up": args.left_up_subject,
        "left-down": args.left_down_subject,
        "right-up": args.right_up_subject,
        "right-down": args.right_down_subject,
    }


def car_subjects(
    car_ids: Sequence[int],
    subject_overrides: Mapping[int, str],
) -> Dict[int, str]:
    return {
        marker_id: subject_overrides.get(marker_id, f"kedaya{marker_id}")
        for marker_id in car_ids
    }


def median_angle(angles: Sequence[float]) -> float:
    if not angles:
        raise ValueError("angles cannot be empty")
    return math.atan2(
        sum(math.sin(angle) for angle in angles),
        sum(math.cos(angle) for angle in angles),
    )


def sample_corner_positions(
    tracker: ViconSubjectTracker,
    subjects: Mapping[str, str],
    sample_count: int,
    timeout_sec: float,
) -> Tuple[FieldCorners, Dict[str, Vec2]]:
    samples: Dict[str, List[Vec2]] = {key: [] for key in CORNER_KEYS}
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline and any(len(values) < sample_count for values in samples.values()):
        snapshot = tracker.get_snapshot()
        for key, subject_name in subjects.items():
            if subject_name not in snapshot.fresh_subjects or len(samples[key]) >= sample_count:
                continue
            samples[key].append(snapshot.poses[subject_name].position)
        time.sleep(0.01)

    missing = [key for key, values in samples.items() if len(values) < sample_count]
    if missing:
        details = ", ".join(f"{key}={len(samples[key])}/{sample_count}" for key in CORNER_KEYS)
        raise RuntimeError(f"Could not collect all Vicon field corners ({details})")

    positions = {
        key: (
            median(point[0] for point in samples[key]),
            median(point[1] for point in samples[key]),
        )
        for key in CORNER_KEYS
    }
    return (
        FieldCorners(
            left_up=positions["left-up"],
            left_down=positions["left-down"],
            right_up=positions["right-up"],
            right_down=positions["right-down"],
        ),
        positions,
    )


def sample_pose(
    tracker: ViconSubjectTracker,
    subject_name: str,
    sample_count: int,
    timeout_sec: float,
) -> Optional[WorldPose]:
    samples: List[WorldPose] = []
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline and len(samples) < sample_count:
        snapshot = tracker.get_snapshot()
        if subject_name in snapshot.fresh_subjects:
            samples.append(snapshot.poses[subject_name])
        time.sleep(0.01)
    if len(samples) < sample_count:
        return None
    return WorldPose(
        subject_name=subject_name,
        position=(
            median(pose.position[0] for pose in samples),
            median(pose.position[1] for pose in samples),
        ),
        yaw=median_angle([pose.yaw for pose in samples]),
        seen_at=max(pose.seen_at for pose in samples),
        frame_number=max(pose.frame_number for pose in samples),
    )


def collect_initial_visibility(
    tracker: ViconSubjectTracker,
    subjects_by_id: Mapping[int, str],
    duration_sec: float,
    required_ratio: float,
) -> Tuple[Set[int], Dict[int, float]]:
    counts = {marker_id: 0 for marker_id in subjects_by_id}
    frame_count = 0
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        snapshot = tracker.get_snapshot()
        if snapshot.fresh_subjects:
            frame_count += 1
            for marker_id, subject_name in subjects_by_id.items():
                if subject_name in snapshot.fresh_subjects:
                    counts[marker_id] += 1
        time.sleep(0.01)
    denominator = max(1, frame_count)
    ratios = {marker_id: count / denominator for marker_id, count in counts.items()}
    visible = {marker_id for marker_id, ratio in ratios.items() if ratio >= required_ratio}
    return visible, ratios


def print_arena(arena: RelativeArena, corners: FieldCorners) -> None:
    print("Relative field initialized from Vicon subjects:")
    for name, position in corners.as_dict().items():
        local = arena.world_to_local(position)
        print(
            f"  {name:>10}: world=({position[0]:.1f},{position[1]:.1f}) "
            f"local=({local[0]:.1f},{local[1]:.1f})"
        )
    cage_top_left_world = arena.local_to_world(arena.cage_top_left_local)
    cage_right_down_world = arena.local_to_world(arena.cage_right_down_local)
    print(
        f"Field size: W={arena.width_mm:.1f}mm H={arena.height_mm:.1f}mm, "
        f"corner angle={arena.corner_angle_deg:.2f}deg, fit error={arena.corner_fit_error_mm:.1f}mm"
    )
    print(
        "Cage local range: "
        f"x=[0,{arena.cage_width_mm:.1f}] "
        f"y=[{arena.cage_bottom:.1f},{arena.cage_top:.1f}]"
    )
    print(
        "Cage world corners: "
        f"left-up=({cage_top_left_world[0]:.1f},{cage_top_left_world[1]:.1f}) "
        f"right-down=({cage_right_down_world[0]:.1f},{cage_right_down_world[1]:.1f})"
    )


def cage_has_capacity(
    arena: RelativeArena,
    evader_count: int,
    robot_diameter_mm: float,
    hold_margin_mm: float,
) -> Tuple[bool, float, float]:
    usable_width = max(0.0, arena.cage_width_mm - 2.0 * hold_margin_mm)
    usable_height = max(0.0, arena.cage_height_mm - 2.0 * hold_margin_mm)
    usable_area = usable_width * usable_height
    required_area = evader_count * (1.30 * robot_diameter_mm) ** 2
    return usable_area >= required_area, usable_area, required_area


def stop_all(controllers: Mapping[int, ExperimentCarController]) -> None:
    for controller in controllers.values():
        controller.send_stop(force=True)


def rotate(vector: Vec2, angle: float) -> Vec2:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        vector[0] * cosine - vector[1] * sine,
        vector[0] * sine + vector[1] * cosine,
    )


def calibration_max_gap_deg(calibrations: Mapping[str, MotionCalibration]) -> float:
    if len(calibrations) < 2:
        return 360.0
    angles = sorted(
        angle_wrap(math.atan2(value.direction_world[1], value.direction_world[0]) - value.yaw)
        for value in calibrations.values()
    )
    gaps = [angles[index + 1] - angles[index] for index in range(len(angles) - 1)]
    gaps.append(angles[0] + 2.0 * math.pi - angles[-1])
    return math.degrees(max(gaps))


def filtered_calibrations(
    calibrations: Mapping[str, MotionCalibration],
) -> Dict[str, MotionCalibration]:
    result: Dict[str, MotionCalibration] = {}
    for command, value in calibrations.items():
        components = (
            value.direction_world[0],
            value.direction_world[1],
            value.yaw,
            value.displacement_mm,
            value.measured_speed_mm_s,
            value.calibrated_at_unix,
        )
        direction_norm = vec_norm(value.direction_world)
        if (
            command not in MOTION_COMMANDS
            or not all(math.isfinite(component) for component in components)
            or not 0.80 <= direction_norm <= 1.20
            or value.displacement_mm <= 0.0
            or value.measured_speed_mm_s <= 0.0
            or not 1 <= value.calibration_pwm <= 255
            or value.calibrated_at_unix <= 0.0
        ):
            continue
        result[command] = MotionCalibration(
            direction_world=vec_normalize(value.direction_world),
            yaw=value.yaw,
            displacement_mm=value.displacement_mm,
            measured_speed_mm_s=value.measured_speed_mm_s,
            calibration_pwm=value.calibration_pwm,
            calibrated_at_unix=value.calibrated_at_unix,
        )
    return result


def valid_calibration_set(
    calibrations: Mapping[str, MotionCalibration],
    min_commands: int,
    max_gap_deg: float,
) -> Tuple[bool, str]:
    valid_commands = filtered_calibrations(calibrations)
    if len(valid_commands) < min_commands:
        return False, f"only {len(valid_commands)}/{len(MOTION_COMMANDS)} valid directions"
    gap = calibration_max_gap_deg(valid_commands)
    if gap > max_gap_deg:
        return False, f"direction coverage gap {gap:.1f}deg exceeds {max_gap_deg:.1f}deg"
    return True, f"{len(valid_commands)} directions, max gap {gap:.1f}deg"


def load_calibration_file(
    path: Path,
    expected_subjects: Mapping[int, str],
    calibration_tag: str = "default",
    max_age_hours: float = 24.0,
) -> Dict[int, Dict[str, MotionCalibration]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Ignoring unreadable calibration file {path}: {exc}")
        return {}
    if not isinstance(payload, dict):
        print(f"Ignoring calibration file {path}: top-level JSON must be an object")
        return {}
    if payload.get("version") != CALIBRATION_VERSION:
        print(f"Ignoring calibration file {path}: unsupported version")
        return {}
    if payload.get("calibration_tag") != calibration_tag:
        print(f"Ignoring calibration file {path}: calibration tag does not match")
        return {}
    result: Dict[int, Dict[str, MotionCalibration]] = {}
    now_unix = time.time()
    oldest_allowed = now_unix - max_age_hours * 3600.0
    newest_allowed = now_unix + 300.0
    cars_payload = payload.get("cars", {})
    if not isinstance(cars_payload, dict):
        print(f"Ignoring calibration file {path}: cars must be an object")
        return {}
    for marker_text, car_payload in cars_payload.items():
        try:
            marker_id = int(marker_text)
            if marker_id not in expected_subjects:
                continue
            if not isinstance(car_payload, dict):
                continue
            if car_payload.get("subject") != expected_subjects[marker_id]:
                continue
            command_payload = car_payload.get("commands", {})
            if not isinstance(command_payload, dict):
                continue
            commands: Dict[str, MotionCalibration] = {}
            for command, value in command_payload.items():
                if command not in MOTION_COMMANDS or not isinstance(value, dict):
                    continue
                try:
                    calibrated_at_unix = float(value["calibrated_at_unix"])
                    if not oldest_allowed <= calibrated_at_unix <= newest_allowed:
                        continue
                    commands[command] = MotionCalibration(
                        direction_world=(
                            float(value["direction_world"][0]),
                            float(value["direction_world"][1]),
                        ),
                        yaw=float(value["yaw"]),
                        displacement_mm=float(value["displacement_mm"]),
                        measured_speed_mm_s=float(value.get("measured_speed_mm_s", 0.0)),
                        calibration_pwm=int(value["calibration_pwm"]),
                        calibrated_at_unix=calibrated_at_unix,
                    )
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
            if commands:
                result[marker_id] = commands
        except (TypeError, ValueError):
            continue
    return result


def save_calibration_file(
    path: Path,
    calibrations: Mapping[int, Mapping[str, MotionCalibration]],
    subjects_by_id: Mapping[int, str],
    calibration_tag: str = "default",
) -> None:
    payload = {
        "version": CALIBRATION_VERSION,
        "saved_at": datetime.now().astimezone().isoformat(),
        "calibration_tag": calibration_tag,
        "cars": {
            str(marker_id): {
                "subject": subjects_by_id.get(marker_id, f"kedaya{marker_id}"),
                "commands": {
                    command: asdict(value)
                    for command, value in command_map.items()
                },
            }
            for marker_id, command_map in calibrations.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def calibration_position_is_safe(
    marker_id: int,
    pose: WorldPose,
    latest_poses: Mapping[int, WorldPose],
    arena: RelativeArena,
    clearance_mm: float,
    separation_mm: float,
) -> Tuple[bool, str]:
    local = arena.world_to_local(pose.position)
    edge_clearance = min(
        local[0],
        arena.width_mm - local[0],
        local[1],
        arena.height_mm - local[1],
    )
    if edge_clearance < clearance_mm:
        return False, f"only {edge_clearance:.1f}mm from virtual field boundary"
    nearest = min(
        (
            vec_norm(vec_sub(pose.position, other_pose.position))
            for other_id, other_pose in latest_poses.items()
            if other_id != marker_id
        ),
        default=float("inf"),
    )
    if nearest < separation_mm:
        return False, f"nearest car is only {nearest:.1f}mm away"
    return True, "safe"


def calibrate_one_car(
    marker_id: int,
    subject_name: str,
    controller: ExperimentCarController,
    all_controllers: Mapping[int, ExperimentCarController],
    tracker: ViconSubjectTracker,
    subjects_by_id: Mapping[int, str],
    corner_subject_map: Mapping[str, str],
    corner_reference: Mapping[str, Vec2],
    arena: RelativeArena,
    args: argparse.Namespace,
    logger: "ExperimentLogger",
) -> Dict[str, MotionCalibration]:
    calibration: Dict[str, MotionCalibration] = {}
    min_displacement = args.min_calibration_displacement_ratio * arena.scale_mm
    max_displacement = args.max_calibration_displacement_ratio * arena.scale_mm
    planned_move = min(
        max_displacement,
        max(0.5 * args.robot_diameter_mm, 0.03 * arena.scale_mm),
    )
    online_clearance = 0.5 * args.robot_diameter_mm + args.calibration_clearance_ratio * arena.scale_mm
    online_separation = args.robot_diameter_mm + args.calibration_separation_ratio * arena.scale_mm
    start_clearance = online_clearance + planned_move
    start_separation = online_separation + planned_move
    maximum_corner_shift = args.corner_shift_ratio * arena.scale_mm
    print(f"Sequentially calibrating car{marker_id} ({subject_name}) ...")

    for command in MOTION_COMMANDS:
        stop_all(all_controllers)
        time.sleep(args.calibration_settle_sec)
        start = sample_pose(tracker, subject_name, args.calibration_samples, 2.0)
        if start is None:
            print(f"  {command}: no fresh start pose")
            logger.event("calibration_aborted", f"{command}: no fresh start pose", marker_id)
            calibration.clear()
            break

        snapshot = tracker.get_snapshot()
        latest_poses = {
            other_id: snapshot.poses[other_subject]
            for other_id, other_subject in subjects_by_id.items()
            if other_subject in snapshot.poses
        }
        safe, reason = calibration_position_is_safe(
            marker_id,
            start,
            latest_poses,
            arena,
            start_clearance,
            start_separation,
        )
        if not safe:
            print(f"  {command}: skipped because calibration position is unsafe ({reason})")
            logger.event("calibration_aborted", f"{command}: {reason}", marker_id)
            calibration.clear()
            break

        missing_nearby = [
            other_id
            for other_id, other_controller in all_controllers.items()
            if other_controller.active and subjects_by_id[other_id] not in snapshot.poses
        ]
        corners_ok, corner_reason = check_corner_reference(
            snapshot,
            corner_subject_map,
            corner_reference,
            maximum_corner_shift,
        )
        if missing_nearby or not corners_ok:
            reason = (
                "missing Vicon car(s): " + format_ids(missing_nearby)
                if missing_nearby
                else corner_reason
            )
            print(f"  {command}: calibration aborted ({reason})")
            logger.event("calibration_aborted", f"{command}: {reason}", marker_id)
            stop_all(all_controllers)
            raise SafetyStop(reason)

        logger.event("calibration_pulse_started", command, marker_id)
        move_started = time.monotonic()
        deadline = move_started + args.calibration_move_sec
        abort_reason = ""
        fatal_reason = ""
        pulse_end = start
        while time.monotonic() < deadline and controller.active:
            now = time.monotonic()
            if not controller.set_motion(command, args.calibration_speed, now, 0.10):
                abort_reason = controller.failure_reason or "TCP command failed"
                break
            pulse_snapshot = tracker.get_snapshot()
            corners_ok, corner_reason = check_corner_reference(
                pulse_snapshot,
                corner_subject_map,
                corner_reference,
                maximum_corner_shift,
            )
            if not corners_ok:
                fatal_reason = corner_reason
                break
            current = pulse_snapshot.poses.get(subject_name)
            if current is None:
                abort_reason = "target Vicon pose became stale"
                break
            if current.seen_at > pulse_end.seen_at:
                pulse_end = current
            moved = vec_norm(vec_sub(current.position, start.position))
            if moved > max_displacement:
                abort_reason = f"live displacement {moved:.1f}mm exceeded {max_displacement:.1f}mm"
                break
            latest_poses = {
                other_id: pulse_snapshot.poses[other_subject]
                for other_id, other_subject in subjects_by_id.items()
                if other_subject in pulse_snapshot.poses
            }
            missing_nearby = [
                other_id
                for other_id, other_controller in all_controllers.items()
                if other_controller.active and subjects_by_id[other_id] not in pulse_snapshot.poses
            ]
            if missing_nearby:
                fatal_reason = "nearby Vicon car(s) became stale: " + format_ids(missing_nearby)
                break
            safe, reason = calibration_position_is_safe(
                marker_id,
                current,
                latest_poses,
                arena,
                online_clearance,
                online_separation,
            )
            if not safe:
                abort_reason = reason
                break
            time.sleep(0.02)
        controller.send_stop(force=True)
        if fatal_reason:
            logger.event("calibration_aborted", f"{command}: {fatal_reason}", marker_id)
            stop_all(all_controllers)
            raise SafetyStop(fatal_reason)
        if abort_reason or not controller.active:
            reason = abort_reason or controller.failure_reason or "controller failed during calibration"
            print(f"  {command}: emergency stop during calibration ({reason})")
            logger.event("calibration_aborted", f"{command}: {reason}", marker_id)
            calibration.clear()
            break

        settle_deadline = time.monotonic() + args.calibration_settle_sec
        settle_abort_reason = ""
        settle_fatal_reason = ""
        while time.monotonic() < settle_deadline:
            settle_snapshot = tracker.get_snapshot()
            corners_ok, corner_reason = check_corner_reference(
                settle_snapshot,
                corner_subject_map,
                corner_reference,
                maximum_corner_shift,
            )
            if not corners_ok:
                settle_fatal_reason = corner_reason
                break
            settled_pose = settle_snapshot.poses.get(subject_name)
            if settled_pose is None:
                settle_abort_reason = "target Vicon pose became stale while settling"
                break
            settled_distance = vec_norm(vec_sub(settled_pose.position, start.position))
            if settled_distance > max_displacement:
                settle_abort_reason = (
                    f"settled displacement {settled_distance:.1f}mm exceeded "
                    f"{max_displacement:.1f}mm"
                )
                break
            latest_poses = {
                other_id: settle_snapshot.poses[other_subject]
                for other_id, other_subject in subjects_by_id.items()
                if other_subject in settle_snapshot.poses
            }
            missing_nearby = [
                other_id
                for other_id, other_controller in all_controllers.items()
                if other_controller.active and subjects_by_id[other_id] not in settle_snapshot.poses
            ]
            if missing_nearby:
                settle_fatal_reason = (
                    "nearby Vicon car(s) became stale while settling: "
                    + format_ids(missing_nearby)
                )
                break
            safe, reason = calibration_position_is_safe(
                marker_id,
                settled_pose,
                latest_poses,
                arena,
                online_clearance,
                online_separation,
            )
            if not safe:
                settle_abort_reason = reason
                break
            time.sleep(0.02)
        if settle_fatal_reason:
            logger.event("calibration_aborted", f"{command}: {settle_fatal_reason}", marker_id)
            stop_all(all_controllers)
            raise SafetyStop(settle_fatal_reason)
        if settle_abort_reason:
            logger.event("calibration_aborted", f"{command}: {settle_abort_reason}", marker_id)
            print(f"  {command}: calibration aborted while settling ({settle_abort_reason})")
            calibration.clear()
            break
        end = sample_pose(tracker, subject_name, args.calibration_samples, 2.0)
        if end is None:
            print(f"  {command}: no fresh end pose")
            logger.event("calibration_direction_rejected", f"{command}: no end pose", marker_id)
            continue

        displacement = vec_sub(end.position, start.position)
        distance = vec_norm(displacement)
        powered_displacement = vec_sub(pulse_end.position, start.position)
        powered_distance = vec_norm(powered_displacement)
        powered_duration = max(pulse_end.seen_at - start.seen_at, 1e-6)
        yaw_change_deg = abs(math.degrees(angle_wrap(end.yaw - start.yaw)))
        warning = ""
        if powered_distance < min_displacement:
            warning = f"too little displacement (<{min_displacement:.1f}mm)"
        elif distance > max_displacement:
            warning = f"unrealistic displacement (>{max_displacement:.1f}mm)"
        elif yaw_change_deg > args.max_calibration_yaw_change_deg:
            warning = f"yaw changed {yaw_change_deg:.1f}deg"
        if warning:
            print(f"  {command}: rejected, dist={distance:.1f}mm, {warning}")
            logger.event("calibration_direction_rejected", f"{command}: {warning}", marker_id)
            continue

        calibration[command] = MotionCalibration(
            direction_world=vec_normalize(powered_displacement),
            yaw=start.yaw,
            displacement_mm=powered_distance,
            measured_speed_mm_s=powered_distance / powered_duration,
            calibration_pwm=args.calibration_speed,
            calibrated_at_unix=time.time(),
        )
        direction = calibration[command].direction_world
        print(
            f"  {command}: dist={distance:.1f}mm yaw_change={yaw_change_deg:.1f}deg "
            f"speed={calibration[command].measured_speed_mm_s:.1f}mm/s "
            f"direction=({direction[0]:+.3f},{direction[1]:+.3f})"
        )
        logger.event(
            "calibration_direction_accepted",
            f"{command}: displacement={distance:.1f}mm yaw_change={yaw_change_deg:.1f}deg",
            marker_id,
        )

    stop_all(all_controllers)
    return calibration


def choose_calibrated_command(
    calibrations: Mapping[str, MotionCalibration],
    desired_world: Vec2,
    current_yaw: float,
    previous_command: str,
    min_score: float,
    switch_margin: float,
) -> Tuple[str, float]:
    desired_direction = vec_normalize(desired_world)
    if vec_norm(desired_direction) <= 1e-9 or not calibrations:
        return "STOP", 0.0

    scores: Dict[str, float] = {}
    for command, calibration in calibrations.items():
        current_direction = rotate(
            calibration.direction_world,
            current_yaw - calibration.yaw,
        )
        scores[command] = vec_dot(current_direction, desired_direction)
    best_command = max(scores, key=scores.get)
    best_score = scores[best_command]
    if previous_command in scores and scores[previous_command] >= best_score - switch_margin:
        best_command = previous_command
        best_score = scores[previous_command]
    if best_score < min_score:
        return "STOP", best_score
    return best_command, best_score


def speed_to_pwm(
    speed: float,
    max_model_speed: float,
    minimum_pwm: int,
    maximum_pwm: int,
    deadband: float,
    reference_speed: Optional[float] = None,
    reference_pwm: Optional[int] = None,
) -> int:
    if speed < deadband or max_model_speed <= deadband:
        return 0
    if (
        reference_speed is not None
        and reference_pwm is not None
        and deadband < reference_speed < max_model_speed
        and minimum_pwm <= reference_pwm <= maximum_pwm
    ):
        if speed <= reference_speed:
            fraction = clamp_value(
                (speed - deadband) / (reference_speed - deadband),
                0.0,
                1.0,
            )
            return round(minimum_pwm + fraction * (reference_pwm - minimum_pwm))
        fraction = clamp_value(
            (speed - reference_speed) / (max_model_speed - reference_speed),
            0.0,
            1.0,
        )
        return round(reference_pwm + fraction * (maximum_pwm - reference_pwm))
    fraction = clamp_value((speed - deadband) / (max_model_speed - deadband), 0.0, 1.0)
    return round(minimum_pwm + fraction * (maximum_pwm - minimum_pwm))


def clamp_value(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def slew_pwm(current: Optional[int], requested: int, maximum_step: int) -> int:
    if current is None or requested == 0:
        return requested
    return int(clamp_value(requested, current - maximum_step, current + maximum_step))


def slew_pwm_per_second(
    current: Optional[int],
    requested: int,
    rate_per_sec: float,
    dt: float,
    credit: float,
    immediate_zero: bool = True,
) -> Tuple[int, float]:
    if requested == 0 and immediate_zero:
        return 0, 0.0
    if current is None:
        return requested, 0.0
    if requested == current:
        return requested, 0.0
    available = max(0.0, credit) + max(0.0, rate_per_sec) * max(0.0, dt)
    maximum_step = int(math.floor(available + 1e-9))
    if maximum_step <= 0:
        return current, available
    if requested == 0 and not immediate_zero:
        updated = int(clamp_value(requested, current - maximum_step, current + maximum_step))
    else:
        updated = slew_pwm(current, requested, maximum_step)
    used = abs(updated - current)
    return updated, max(0.0, available - used)


def rate_limited_motion_decision(
    controller: ExperimentCarController,
    desired_command: str,
    direction_score: float,
    requested_pwm: int,
    pwm_rate_per_sec: float,
    dt: float,
) -> CommandDecision:
    if desired_command not in MOTION_COMMANDS:
        controller.pwm_slew_credit = 0.0
        return CommandDecision("STOP", direction_score, 0)

    previous_motion = controller.last_motion
    previous_pwm = controller.last_pwm or 0
    if previous_motion in MOTION_COMMANDS and previous_motion != desired_command and previous_pwm > 0:
        pwm, controller.pwm_slew_credit = slew_pwm_per_second(
            previous_pwm,
            0,
            pwm_rate_per_sec,
            dt,
            controller.pwm_slew_credit,
            immediate_zero=False,
        )
        return CommandDecision(previous_motion, direction_score, pwm)

    pwm, controller.pwm_slew_credit = slew_pwm_per_second(
        controller.last_pwm,
        requested_pwm,
        pwm_rate_per_sec,
        dt,
        controller.pwm_slew_credit,
    )
    return CommandDecision(desired_command, direction_score, pwm)


def motion_response_failed(
    marker_id: int,
    decision: CommandDecision,
    measured_speed_mm_s: float,
    minimum_drive_pwm: int,
    now: float,
    timeout_sec: float,
    minimum_speed_mm_s: float,
    waiting_since: Dict[int, float],
) -> bool:
    expecting_motion = decision.command in MOTION_COMMANDS and decision.pwm >= minimum_drive_pwm
    if not expecting_motion or measured_speed_mm_s >= minimum_speed_mm_s:
        waiting_since.pop(marker_id, None)
        return False
    started_at = waiting_since.setdefault(marker_id, now)
    return now - started_at >= timeout_sec


class ExperimentLogger:
    STATE_FIELDS = (
        "wall_time",
        "monotonic_time",
        "elapsed_sec",
        "frame",
        "car_id",
        "role",
        "active",
        "tcp_active",
        "visible",
        "pose_age_sec",
        "world_x_mm",
        "world_y_mm",
        "local_x_mm",
        "local_y_mm",
        "yaw_rad",
        "accel_x",
        "accel_y",
        "model_vx",
        "model_vy",
        "measured_vx_mm_s",
        "measured_vy_mm_s",
        "target_x_mm",
        "target_y_mm",
        "command",
        "direction_score",
        "pwm",
        "inside_cage",
        "herder_hull",
        "augmented_hull",
    )

    def __init__(self, root: Path, enabled: bool):
        self.enabled = enabled
        self.run_dir: Optional[Path] = None
        self.state_file = None
        self.event_file = None
        self.state_writer = None
        self.event_writer = None
        self.last_flush = 0.0
        if not enabled:
            return

        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        run_dir = root / stamp
        suffix = 1
        while run_dir.exists():
            run_dir = root / f"{stamp}_{suffix:02d}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        self.run_dir = run_dir
        self.state_file = (run_dir / "states.csv").open("w", newline="", encoding="utf-8")
        self.event_file = (run_dir / "events.csv").open("w", newline="", encoding="utf-8")
        self.state_writer = csv.DictWriter(self.state_file, fieldnames=self.STATE_FIELDS)
        self.event_writer = csv.DictWriter(
            self.event_file,
            fieldnames=("wall_time", "monotonic_time", "event", "car_id", "details"),
        )
        self.state_writer.writeheader()
        self.event_writer.writeheader()

    def write_metadata(self, metadata: Mapping[str, object]) -> None:
        if not self.enabled or self.run_dir is None:
            return
        (self.run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def event(self, event: str, details: str = "", marker_id: Optional[int] = None) -> None:
        if not self.enabled or self.event_writer is None:
            return
        self.event_writer.writerow(
            {
                "wall_time": datetime.now().astimezone().isoformat(),
                "monotonic_time": f"{time.monotonic():.6f}",
                "event": event,
                "car_id": "" if marker_id is None else marker_id,
                "details": details,
            }
        )
        self.flush_if_due(force=True)

    def state(self, row: Mapping[str, object]) -> None:
        if not self.enabled or self.state_writer is None:
            return
        self.state_writer.writerow({field: row.get(field, "") for field in self.STATE_FIELDS})

    def flush_if_due(self, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_flush < 1.0:
            return
        if self.state_file is not None:
            self.state_file.flush()
        if self.event_file is not None:
            self.event_file.flush()
        self.last_flush = now

    def close(self) -> None:
        self.flush_if_due(force=True)
        if self.state_file is not None:
            self.state_file.close()
        if self.event_file is not None:
            self.event_file.close()


def serializable_args(args: argparse.Namespace) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in vars(args).items():
        if key == "wifi_password":
            result[key] = "<redacted>" if value else ""
        elif isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def connect_requested_cars(
    requested_ids: Sequence[int],
    ip_overrides: Mapping[int, str],
    args: argparse.Namespace,
) -> Tuple[Dict[int, ExperimentCarController], Dict[int, str]]:
    discovered: Dict[int, str] = {}
    if requested_ids and not args.skip_discovery:
        discovered = discover_car_ips(
            requested_ids,
            args.discovery_timeout,
            bind_ips=args.bind_ip,
            broadcast_ips=args.broadcast_ip,
        )

    excluded: Dict[int, str] = {}
    candidates: Dict[int, Tuple[str, bool]] = {}
    for marker_id in requested_ids:
        ip = ip_overrides.get(marker_id) or discovered.get(marker_id, "")
        if not ip:
            excluded[marker_id] = "not discovered and no --car-ip override"
            continue
        try:
            parsed_ip = ipaddress.ip_address(ip)
        except ValueError:
            excluded[marker_id] = f"invalid IPv4 address {ip!r}"
            continue
        if parsed_ip.version != 4:
            excluded[marker_id] = f"non-IPv4 address {ip!r} is not supported"
            continue
        ip = str(parsed_ip)
        if marker_id in ip_overrides and marker_id in discovered and discovered[marker_id] != ip:
            excluded[marker_id] = (
                f"--car-ip {ip} conflicts with UDP identity at {discovered[marker_id]}"
            )
            continue
        discovered_owner = next(
            (
                other_id
                for other_id, discovered_ip in discovered.items()
                if discovered_ip == ip and other_id != marker_id
            ),
            None,
        )
        if discovered_owner is not None:
            excluded[marker_id] = f"IP {ip} announced itself as car{discovered_owner}"
            continue
        candidates[marker_id] = (ip, discovered.get(marker_id) == ip)

    ids_by_ip: Dict[str, List[int]] = {}
    for marker_id, (ip, _identity_verified) in candidates.items():
        ids_by_ip.setdefault(ip, []).append(marker_id)
    for ip, marker_ids in ids_by_ip.items():
        if len(marker_ids) < 2:
            continue
        reason = f"duplicate IP {ip} assigned to IDs {format_ids(marker_ids)}"
        for marker_id in marker_ids:
            excluded[marker_id] = reason
            candidates.pop(marker_id, None)

    pending = {
        marker_id: ExperimentCarController(
            marker_id,
            candidate[0],
            args.tcp_connect_timeout,
            args.tcp_send_timeout,
            allow_plain_pong=candidate[1],
        )
        for marker_id, candidate in candidates.items()
    }
    controllers: Dict[int, ExperimentCarController] = {}
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(pending))) as executor:
            future_by_id = {
                executor.submit(controller.connect): marker_id
                for marker_id, controller in pending.items()
            }
            for future in concurrent.futures.as_completed(future_by_id):
                marker_id = future_by_id[future]
                controller = pending[marker_id]
                try:
                    connected = future.result()
                except Exception as exc:
                    connected = False
                    controller.failure_reason = f"initial TCP probe failed: {exc}"
                    controller.close(send_stop=False)
                if connected:
                    controllers[marker_id] = controller
                else:
                    excluded[marker_id] = controller.failure_reason or "initial TCP/PONG probe failed"
    return controllers, excluded


def prepare_calibrations(
    controlled_ids: Sequence[int],
    controllers: Mapping[int, ExperimentCarController],
    tracker: ViconSubjectTracker,
    subjects_by_id: Mapping[int, str],
    corner_subject_map: Mapping[str, str],
    corner_reference: Mapping[str, Vec2],
    arena: RelativeArena,
    args: argparse.Namespace,
    logger: "ExperimentLogger",
) -> Tuple[Dict[int, Dict[str, MotionCalibration]], Dict[int, str]]:
    loaded = (
        {}
        if args.recalibrate
        else load_calibration_file(
            args.calibration_file,
            subjects_by_id,
            args.calibration_tag,
            args.calibration_max_age_hours,
        )
    )
    result: Dict[int, Dict[str, MotionCalibration]] = {}
    excluded: Dict[int, str] = {}

    for marker_id in controlled_ids:
        controller = controllers.get(marker_id)
        if controller is None or not controller.active:
            excluded[marker_id] = "no active TCP controller"
            continue

        existing = filtered_calibrations(loaded.get(marker_id, {}))
        valid, summary = valid_calibration_set(
            existing,
            args.min_calibrated_commands,
            args.max_calibration_angle_gap_deg,
        )
        if valid:
            result[marker_id] = existing
            print(f"Loaded car{marker_id} calibration: {summary}")
            logger.event("calibration_loaded", summary, marker_id)
            continue

        logger.event("calibration_started", f"subject={subjects_by_id[marker_id]}", marker_id)
        measured = calibrate_one_car(
            marker_id,
            subjects_by_id[marker_id],
            controller,
            controllers,
            tracker,
            subjects_by_id,
            corner_subject_map,
            corner_reference,
            arena,
            args,
            logger,
        )
        measured = filtered_calibrations(measured)
        valid, summary = valid_calibration_set(
            measured,
            args.min_calibrated_commands,
            args.max_calibration_angle_gap_deg,
        )
        if not valid:
            excluded[marker_id] = f"calibration rejected: {summary}"
            print(f"Ignoring car{marker_id}: {excluded[marker_id]}")
            logger.event("calibration_rejected", excluded[marker_id], marker_id)
            controller.send_stop(force=True)
            continue
        result[marker_id] = measured
        print(f"Accepted car{marker_id} calibration: {summary}")
        logger.event("calibration_accepted", summary, marker_id)

    if result:
        saved = dict(loaded)
        saved.update(result)
        save_calibration_file(args.calibration_file, saved, subjects_by_id, args.calibration_tag)
        print(f"Calibration saved to {args.calibration_file.resolve()}")
    return result, excluded


def check_corner_reference(
    snapshot: ViconSnapshot,
    subjects: Mapping[str, str],
    reference_positions: Mapping[str, Vec2],
    maximum_shift_mm: float,
) -> Tuple[bool, str]:
    missing = [key for key, subject in subjects.items() if subject not in snapshot.poses]
    if missing:
        return False, "missing corner Vicon subjects: " + ",".join(missing)
    moved = []
    for key, subject in subjects.items():
        distance = vec_norm(vec_sub(snapshot.poses[subject].position, reference_positions[key]))
        if distance > maximum_shift_mm:
            moved.append(f"{key}:{distance:.1f}mm")
    if moved:
        return False, "field corner moved: " + ", ".join(moved)
    return True, "ok"


def current_local_observations(
    snapshot: ViconSnapshot,
    subjects_by_id: Mapping[int, str],
    arena: RelativeArena,
) -> Tuple[Dict[int, LocalObservation], Dict[int, WorldPose]]:
    local: Dict[int, LocalObservation] = {}
    world: Dict[int, WorldPose] = {}
    for marker_id, subject_name in subjects_by_id.items():
        pose = snapshot.poses.get(subject_name)
        if pose is None:
            continue
        world[marker_id] = pose
        local[marker_id] = LocalObservation(
            marker_id=marker_id,
            position=arena.world_to_local(pose.position),
            yaw_world=pose.yaw,
            seen_at=pose.seen_at,
        )
    return local, world


class ViconVelocityEstimator:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.previous: Dict[int, Tuple[Vec2, float]] = {}
        self.filtered: Dict[int, Vec2] = {}
        self.raw_this_update: Dict[int, Vec2] = {}

    def reset(self, marker_id: int) -> None:
        self.previous.pop(marker_id, None)
        self.filtered.pop(marker_id, None)

    def reset_many(self, marker_ids: Iterable[int]) -> None:
        for marker_id in marker_ids:
            self.reset(marker_id)

    def update(
        self,
        observations: Mapping[int, LocalObservation],
        marker_ids: Iterable[int],
    ) -> Dict[int, Vec2]:
        measured: Dict[int, Vec2] = {}
        self.raw_this_update = {}
        for marker_id in marker_ids:
            observation = observations.get(marker_id)
            if observation is None:
                self.reset(marker_id)
                continue
            previous = self.previous.get(marker_id)
            if previous is None:
                self.previous[marker_id] = (observation.position, observation.seen_at)
                continue
            previous_position, previous_seen_at = previous
            elapsed = observation.seen_at - previous_seen_at
            if elapsed < 0.005:
                continue
            self.previous[marker_id] = (observation.position, observation.seen_at)
            if elapsed > 0.50:
                self.filtered.pop(marker_id, None)
                continue
            raw = vec_scale(vec_sub(observation.position, previous_position), 1.0 / elapsed)
            if not all(math.isfinite(component) for component in raw):
                self.reset(marker_id)
                continue
            self.raw_this_update[marker_id] = raw
            old = self.filtered.get(marker_id, raw)
            filtered = (
                self.alpha * raw[0] + (1.0 - self.alpha) * old[0],
                self.alpha * raw[1] + (1.0 - self.alpha) * old[1],
            )
            self.filtered[marker_id] = filtered
            measured[marker_id] = filtered
        return measured


class SafetyStop(RuntimeError):
    pass


def log_control_states(
    logger: ExperimentLogger,
    started_at: float,
    snapshot: ViconSnapshot,
    requested_herders: Sequence[int],
    requested_evaders: Sequence[int],
    active_herders: Sequence[int],
    active_evaders: Sequence[int],
    controllers: Mapping[int, ExperimentCarController],
    local_observations: Mapping[int, LocalObservation],
    world_poses: Mapping[int, WorldPose],
    result,
    decisions: Mapping[int, CommandDecision],
    measured_velocities: Mapping[int, Vec2],
) -> None:
    now = time.monotonic()
    active_set = set(active_herders) | set(active_evaders)
    herder_set = set(requested_herders)
    inside_set = set(result.inside_cage_evaders)
    for marker_id in tuple(dict.fromkeys(tuple(requested_herders) + tuple(requested_evaders))):
        world_pose = world_poses.get(marker_id)
        local_observation = local_observations.get(marker_id)
        acceleration = result.accelerations.get(marker_id)
        velocity = result.velocities.get(marker_id)
        measured_velocity = measured_velocities.get(marker_id)
        target = result.targets.get(marker_id)
        decision = decisions.get(marker_id)
        controller = controllers.get(marker_id)
        logger.state(
            {
                "wall_time": datetime.now().astimezone().isoformat(),
                "monotonic_time": f"{now:.6f}",
                "elapsed_sec": f"{now - started_at:.6f}",
                "frame": snapshot.frame_number,
                "car_id": marker_id,
                "role": "herder" if marker_id in herder_set else "evader",
                "active": marker_id in active_set,
                "tcp_active": bool(controller and controller.active),
                "visible": world_pose is not None,
                "pose_age_sec": "" if world_pose is None else f"{now - world_pose.seen_at:.6f}",
                "world_x_mm": "" if world_pose is None else f"{world_pose.position[0]:.3f}",
                "world_y_mm": "" if world_pose is None else f"{world_pose.position[1]:.3f}",
                "local_x_mm": "" if local_observation is None else f"{local_observation.position[0]:.3f}",
                "local_y_mm": "" if local_observation is None else f"{local_observation.position[1]:.3f}",
                "yaw_rad": "" if world_pose is None else f"{world_pose.yaw:.6f}",
                "accel_x": "" if acceleration is None else f"{acceleration[0]:.6f}",
                "accel_y": "" if acceleration is None else f"{acceleration[1]:.6f}",
                "model_vx": "" if velocity is None else f"{velocity[0]:.6f}",
                "model_vy": "" if velocity is None else f"{velocity[1]:.6f}",
                "measured_vx_mm_s": (
                    "" if measured_velocity is None else f"{measured_velocity[0]:.6f}"
                ),
                "measured_vy_mm_s": (
                    "" if measured_velocity is None else f"{measured_velocity[1]:.6f}"
                ),
                "target_x_mm": "" if target is None else f"{target[0]:.3f}",
                "target_y_mm": "" if target is None else f"{target[1]:.3f}",
                "command": "" if decision is None else decision.command,
                "direction_score": "" if decision is None else f"{decision.score:.6f}",
                "pwm": "" if decision is None else decision.pwm,
                "inside_cage": marker_id in inside_set,
                "herder_hull": marker_id in result.herder_hull_ids,
                "augmented_hull": marker_id in result.augmented_hull_herder_ids,
            }
        )
    logger.flush_if_due()


def run_control_loop(
    args: argparse.Namespace,
    tracker: ViconSubjectTracker,
    subjects_by_id: Mapping[int, str],
    corner_subject_map: Mapping[str, str],
    corner_reference: Mapping[str, Vec2],
    arena: RelativeArena,
    params: ControlParameters,
    requested_herders: Sequence[int],
    requested_evaders: Sequence[int],
    initial_herders: Sequence[int],
    initial_evaders: Sequence[int],
    controllers: Dict[int, ExperimentCarController],
    calibrations: Dict[int, Dict[str, MotionCalibration]],
    ignored_ids: Set[int],
    logger: ExperimentLogger,
) -> None:
    active_herders = list(initial_herders)
    active_evaders = list(initial_evaders)
    dynamics = SecondOrderDynamics(args.seed)
    velocity_estimator = ViconVelocityEstimator(args.measured_velocity_alpha)
    capture_monitor = CaptureMonitor(args.capture_hold_sec)
    maximum_corner_shift = args.corner_shift_ratio * arena.scale_mm
    controlled_ids = set(active_herders)
    if not args.passive_evaders:
        controlled_ids.update(active_evaders)

    started_at = time.monotonic()
    previous_loop_at = started_at
    last_status_at = 0.0
    last_pause_reason = ""
    motion_waiting_since: Dict[int, float] = {}
    print(
        "Control armed. "
        f"Herders={format_ids(active_herders)} Evaders={format_ids(active_evaders)} "
        f"Controlled={format_ids(sorted(controlled_ids))}. Press Ctrl+C to stop."
    )
    logger.event("control_started", f"herders={active_herders}; evaders={active_evaders}")

    while True:
        loop_started = time.monotonic()
        if loop_started - started_at >= args.max_runtime_sec:
            stop_all(controllers)
            logger.event("runtime_timeout", f"limit={args.max_runtime_sec:.1f}s")
            print(f"Maximum runtime {args.max_runtime_sec:.1f}s reached; all cars stopped.")
            return
        dt = clamp_value(loop_started - previous_loop_at, 0.01, 0.25)
        previous_loop_at = loop_started
        snapshot = tracker.get_snapshot()

        corners_ok, corner_reason = check_corner_reference(
            snapshot,
            corner_subject_map,
            corner_reference,
            maximum_corner_shift,
        )
        if not corners_ok:
            stop_all(controllers)
            for marker_id in controlled_ids:
                dynamics.reset(marker_id)
            velocity_estimator.reset_many(controlled_ids)
            capture_monitor.reset()
            motion_waiting_since.clear()
            if corner_reason.startswith("field corner moved"):
                raise SafetyStop(corner_reason)
            if corner_reason != last_pause_reason:
                print(f"Paused: {corner_reason}")
                logger.event("control_paused", corner_reason)
                last_pause_reason = corner_reason
            time.sleep(args.control_period)
            continue

        local_observations, world_poses = current_local_observations(
            snapshot,
            subjects_by_id,
            arena,
        )
        active_ids = tuple(active_herders + active_evaders)
        missing_active = [marker_id for marker_id in active_ids if marker_id not in local_observations]
        for marker_id in missing_active:
            dynamics.reset(marker_id)
            velocity_estimator.reset(marker_id)
            motion_waiting_since.pop(marker_id, None)
            controller = controllers.get(marker_id)
            if controller is not None and marker_id in controlled_ids:
                controller.send_stop(force=True)
        if missing_active and not args.allow_partial_vicon:
            stop_all(controllers)
            for marker_id in controlled_ids:
                dynamics.reset(marker_id)
            velocity_estimator.reset_many(controlled_ids)
            capture_monitor.reset()
            motion_waiting_since.clear()
            pause_reason = "missing active Vicon car(s): " + format_ids(missing_active)
            if pause_reason != last_pause_reason:
                print(f"Paused: {pause_reason}")
                logger.event("control_paused", pause_reason)
                last_pause_reason = pause_reason
            time.sleep(args.control_period)
            continue

        visible_herders = [marker_id for marker_id in active_herders if marker_id in local_observations]
        if len(visible_herders) < 3:
            stop_all(controllers)
            for marker_id in controlled_ids:
                dynamics.reset(marker_id)
            velocity_estimator.reset_many(controlled_ids)
            capture_monitor.reset()
            motion_waiting_since.clear()
            pause_reason = f"only {len(visible_herders)} visible herders; at least 3 are required"
            if pause_reason != last_pause_reason:
                print(f"Paused: {pause_reason}")
                logger.event("control_paused", pause_reason)
                last_pause_reason = pause_reason
            time.sleep(args.control_period)
            continue

        if last_pause_reason:
            print("Control resumed: all required Vicon data is fresh.")
            logger.event("control_resumed", last_pause_reason)
            last_pause_reason = ""

        body_clearance_violations = [
            marker_id
            for marker_id in active_ids
            if marker_id in local_observations
            and not arena.contains_field_local(
                local_observations[marker_id].position,
                params.robot_radius_mm,
            )
        ]
        if body_clearance_violations:
            stop_all(controllers)
            raise SafetyStop(
                "active car center entered the chassis clearance at the virtual field boundary: "
                + format_ids(body_clearance_violations)
            )

        measured_velocities = velocity_estimator.update(local_observations, active_ids)
        raw_velocities = velocity_estimator.raw_this_update
        overspeed = []
        for marker_id, raw_velocity in raw_velocities.items():
            limits = (
                params.herder_dynamics
                if marker_id in active_herders
                else params.evader_dynamics
            )
            measured_speed = vec_norm(raw_velocity)
            if measured_speed > args.actual_speed_limit_factor * limits.max_speed:
                overspeed.append(
                    f"ID{marker_id}:{measured_speed:.1f}>{args.actual_speed_limit_factor:.2f}x"
                    f"{limits.max_speed:.1f}mm/s"
                )
                continue
        if overspeed:
            stop_all(controllers)
            raise SafetyStop("Vicon measured speed/jump exceeded safety limit: " + ", ".join(overspeed))
        for marker_id, measured_velocity in measured_velocities.items():
            if marker_id not in controlled_ids:
                continue
            limits = (
                params.herder_dynamics
                if marker_id in active_herders
                else params.evader_dynamics
            )
            dynamics.observe_velocity(marker_id, measured_velocity, limits)

        visible_obstacles = [marker_id for marker_id in ignored_ids if marker_id in local_observations]
        result = compute_control_step(
            local_observations,
            active_herders,
            active_evaders,
            visible_obstacles,
            arena,
            params,
            dynamics,
            dt,
            drive_evaders=not args.passive_evaders,
        )

        capture_confirmed = capture_monitor.update(
            loop_started,
            active_evaders,
            result.visible_evaders,
            result.inside_cage_evaders,
        )
        decisions: Dict[int, CommandDecision] = {}
        failed_ids: List[int] = []

        for marker_id in tuple(sorted(controlled_ids)):
            controller = controllers.get(marker_id)
            pose = world_poses.get(marker_id)
            calibration = calibrations.get(marker_id)
            local_velocity = result.velocities.get(marker_id, (0.0, 0.0))
            speed = vec_norm(local_velocity)
            if controller is None or not controller.active:
                failed_ids.append(marker_id)
                continue
            if pose is None or calibration is None or speed < args.motion_deadband or capture_confirmed:
                decision = CommandDecision("STOP", 0.0, 0)
            else:
                desired_world = arena.local_vector_to_world(local_velocity)
                previous_command = controller.last_motion if controller.last_motion in MOTION_COMMANDS else ""
                command, score = choose_calibrated_command(
                    calibration,
                    desired_world,
                    pose.yaw,
                    previous_command,
                    args.min_direction_score,
                    args.command_switch_margin,
                )
                if marker_id in active_herders:
                    reference = calibration.get(command)
                    requested_pwm = speed_to_pwm(
                        speed,
                        params.herder_dynamics.max_speed,
                        args.herder_min_pwm,
                        args.herder_max_pwm,
                        args.motion_deadband,
                        None if reference is None else reference.measured_speed_mm_s,
                        None if reference is None else reference.calibration_pwm,
                    )
                    minimum_pwm = args.herder_min_pwm
                else:
                    reference = calibration.get(command)
                    requested_pwm = speed_to_pwm(
                        speed,
                        params.evader_dynamics.max_speed,
                        args.evader_min_pwm,
                        args.evader_max_pwm,
                        args.motion_deadband,
                        None if reference is None else reference.measured_speed_mm_s,
                        None if reference is None else reference.calibration_pwm,
                    )
                    minimum_pwm = args.evader_min_pwm
                if command != "STOP" and controller.last_pwm in (None, 0):
                    requested_pwm = min(requested_pwm, minimum_pwm)
                decision = rate_limited_motion_decision(
                    controller,
                    command,
                    score,
                    requested_pwm,
                    args.pwm_slew_per_sec,
                    dt,
                )
            decisions[marker_id] = decision
            if not controller.set_motion(
                decision.command,
                decision.pwm,
                loop_started,
                args.command_refresh,
            ):
                failed_ids.append(marker_id)
                continue
            minimum_drive_pwm = (
                args.herder_min_pwm if marker_id in active_herders else args.evader_min_pwm
            )
            measured_speed = vec_norm(measured_velocities.get(marker_id, (0.0, 0.0)))
            if motion_response_failed(
                marker_id,
                decision,
                measured_speed,
                minimum_drive_pwm,
                loop_started,
                args.motion_response_timeout,
                args.motion_response_min_speed,
                motion_waiting_since,
            ):
                controller.failure_reason = (
                    f"no Vicon motion response >= {args.motion_response_min_speed:.1f}mm/s "
                    f"for {args.motion_response_timeout:.1f}s"
                )
                failed_ids.append(marker_id)

        heartbeat_at = time.monotonic()
        due_ping_ids = [
            marker_id
            for marker_id in controlled_ids
            if marker_id in controllers
            and controllers[marker_id].active
            and heartbeat_at - controllers[marker_id].last_ping_at >= args.ping_interval
        ]
        ping_id = (
            min(due_ping_ids, key=lambda marker_id: controllers[marker_id].last_ping_at)
            if due_ping_ids
            else None
        )
        if ping_id is not None and not controllers[ping_id].ping():
            failed_ids.append(ping_id)

        unique_failed_ids = sorted(set(failed_ids))
        if unique_failed_ids:
            stop_all(controllers)
        for marker_id in unique_failed_ids:
            if marker_id not in active_herders and marker_id not in active_evaders:
                continue
            failed_controller = controllers.get(marker_id)
            reason = (
                failed_controller.failure_reason
                if failed_controller is not None and failed_controller.failure_reason
                else "TCP controller became inactive"
            )
            print(f"Ignoring car{marker_id} for the rest of this run: {reason}")
            logger.event("car_removed", reason, marker_id)
            if failed_controller is not None:
                failed_controller.close(send_stop=False)
            dynamics.reset(marker_id)
            velocity_estimator.reset(marker_id)
            motion_waiting_since.pop(marker_id, None)
            calibrations.pop(marker_id, None)
            controlled_ids.discard(marker_id)
            ignored_ids.add(marker_id)
            if marker_id in active_herders:
                active_herders.remove(marker_id)
            if marker_id in active_evaders:
                active_evaders.remove(marker_id)
            capture_monitor.reset()

        if len(active_herders) < 3:
            raise SafetyStop("fewer than 3 active herders remain after a controller failure")
        if not active_evaders:
            raise SafetyStop("no active evaders remain after a controller failure")

        if unique_failed_ids:
            for marker_id in controlled_ids:
                dynamics.reset(marker_id)
            velocity_estimator.reset_many(controlled_ids)
            motion_waiting_since.clear()
            logger.event(
                "topology_changed",
                f"active herders={active_herders}; active evaders={active_evaders}",
            )
            elapsed = time.monotonic() - loop_started
            time.sleep(max(0.0, args.control_period - elapsed))
            continue

        log_control_states(
            logger,
            started_at,
            snapshot,
            requested_herders,
            requested_evaders,
            active_herders,
            active_evaders,
            controllers,
            local_observations,
            world_poses,
            result,
            decisions,
            measured_velocities,
        )

        if loop_started - last_status_at >= args.status_interval:
            command_text = " ".join(
                f"ID{marker_id}:{decision.command}@{decision.pwm}"
                for marker_id, decision in sorted(decisions.items())
            )
            print(
                f"H={len(result.visible_herders)}/{len(active_herders)} "
                f"E={len(result.visible_evaders)}/{len(active_evaders)} "
                f"hull={format_ids(result.herder_hull_ids)} "
                f"aug_hull={format_ids(result.augmented_hull_herder_ids)} "
                f"contained={format_ids(result.contained_evaders)} "
                f"in_cage={len(result.inside_cage_evaders)}/{len(active_evaders)}:"
                f"{format_ids(result.inside_cage_evaders)} | {command_text}"
            )
            last_status_at = loop_started

        if capture_confirmed:
            stop_all(controllers)
            logger.event(
                "capture_confirmed",
                f"all active evaders held inside cage for {args.capture_hold_sec:.2f}s",
            )
            print(
                f"Capture complete: all {len(active_evaders)} active evaders remained inside "
                f"the cage for {args.capture_hold_sec:.1f}s."
            )
            return

        elapsed = time.monotonic() - loop_started
        time.sleep(max(0.0, args.control_period - elapsed))


def main() -> int:
    args = parse_args()
    requested_herders = parse_id_spec(args.herders)
    requested_evaders = parse_id_spec(args.evaders)
    overlap = sorted(set(requested_herders) & set(requested_evaders))
    if overlap:
        raise ValueError("The same car cannot be both herder and evader: " + format_ids(overlap))
    if len(requested_herders) < 3:
        raise ValueError("At least three herder IDs must be configured")
    if not requested_evaders:
        raise ValueError("At least one evader ID must be configured")

    requested_ids = tuple(dict.fromkeys(requested_herders + requested_evaders))
    subject_overrides = parse_overrides(args.subject, "subject", requested_ids)
    ip_overrides = parse_overrides(args.car_ip, "car-ip", requested_ids)
    subjects_by_id = car_subjects(requested_ids, subject_overrides)
    corner_subject_map = corner_subjects(args)
    all_subject_names = tuple(corner_subject_map.values()) + tuple(subjects_by_id.values())
    if len(set(corner_subject_map.values())) != len(CORNER_KEYS):
        raise ValueError("The four corner Vicon subject names must be distinct")
    if len(set(subjects_by_id.values())) != len(subjects_by_id):
        raise ValueError("Every configured car must use a distinct Vicon subject name")
    subject_collisions = set(corner_subject_map.values()) & set(subjects_by_id.values())
    if subject_collisions:
        raise ValueError(
            "Corner subjects cannot also be car subjects: " + ",".join(sorted(subject_collisions))
        )

    controllers: Dict[int, ExperimentCarController] = {}
    tracker = ViconSubjectTracker(args.vicon_host, all_subject_names, args.pose_max_age)
    logger = ExperimentLogger(args.log_root, enabled=False)
    exit_code = 0

    try:
        if not args.skip_wifi:
            connect_wifi(args.wifi_ssid, args.wifi_wait, args.wifi_password)
        tracker.connect(args.vicon_connect_timeout)
        corners, corner_reference = sample_corner_positions(
            tracker,
            corner_subject_map,
            args.corner_samples,
            args.corner_sample_timeout,
        )
        arena = fit_relative_arena(
            corners,
            cage_width_ratio=args.cage_width_ratio,
            cage_height_ratio=args.cage_height_ratio,
        )
        params = ControlParameters.from_arena(arena, 0.5 * args.robot_diameter_mm)
        controlled_speed_limits = [params.herder_dynamics.max_speed]
        if not args.passive_evaders:
            controlled_speed_limits.append(params.evader_dynamics.max_speed)
        minimum_speed_limit = min(controlled_speed_limits)
        if args.motion_deadband >= minimum_speed_limit:
            raise ValueError(
                f"motion-deadband {args.motion_deadband:.1f}mm/s must be below the model "
                f"speed limit {minimum_speed_limit:.1f}mm/s"
            )
        if args.motion_response_min_speed >= minimum_speed_limit:
            raise ValueError(
                f"motion-response-min-speed {args.motion_response_min_speed:.1f}mm/s must be "
                f"below the model speed limit {minimum_speed_limit:.1f}mm/s"
            )
        print_arena(arena, corners)

        capacity_ok, usable_area, required_area = cage_has_capacity(
            arena,
            len(requested_evaders),
            args.robot_diameter_mm,
            params.cage_hold_margin_mm,
        )
        print(
            f"Cage capacity check: usable={usable_area / 1e6:.2f}m^2, "
            f"estimated required={required_area / 1e6:.2f}m^2 for {len(requested_evaders)} evader(s)."
        )
        if not capacity_ok:
            print("Warning: the cage may be too small for every requested evader.")

        if not args.arm:
            print(
                "Geometry check complete. No TCP connection, calibration, or motor command was sent. "
                "Run again with --arm only after verifying the printed field and cage corners."
            )
            return 0

        logger = ExperimentLogger(args.log_root, enabled=not args.no_log)
        logger.event("armed", "Motor commands and sequential calibration enabled")
        logger.write_metadata(
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "phase": "preflight",
                "arguments": serializable_args(args),
                "corner_subjects": corner_subject_map,
                "corner_positions_world": corner_reference,
                "arena": arena.as_dict(),
                "control_parameters": asdict(params),
                "requested_herders": requested_herders,
                "requested_evaders": requested_evaders,
            }
        )

        network_required_ids = tuple(requested_herders)
        if not args.passive_evaders:
            network_required_ids += tuple(requested_evaders)
        controllers, excluded = connect_requested_cars(network_required_ids, ip_overrides, args)
        for marker_id, reason in sorted(excluded.items()):
            print(f"Ignoring car{marker_id}: {reason}")

        visible_ids, visibility_ratios = collect_initial_visibility(
            tracker,
            subjects_by_id,
            args.initial_visibility_sec,
            args.initial_visibility_ratio,
        )
        for marker_id in requested_ids:
            if marker_id not in visible_ids:
                excluded.setdefault(
                    marker_id,
                    (
                        f"Vicon visibility ratio {visibility_ratios[marker_id]:.2f} is below "
                        f"{args.initial_visibility_ratio:.2f}"
                    ),
                )
                controller = controllers.get(marker_id)
                if controller is not None:
                    controller.send_stop(force=True)
                    controller.close(send_stop=False)

        initially_usable = set(controllers) & visible_ids - set(excluded)
        if args.passive_evaders:
            initially_usable.update(set(requested_evaders) & visible_ids)
        active_herders, active_evaders = active_role_ids(
            requested_herders,
            requested_evaders,
            initially_usable,
        )
        if len(active_herders) < 3:
            raise SafetyStop(
                f"only {len(active_herders)} usable herders remain after discovery/TCP/Vicon checks"
            )
        if not active_evaders:
            raise SafetyStop("no usable evader remains after discovery/TCP/Vicon checks")

        controlled_for_calibration = tuple(active_herders)
        if not args.passive_evaders:
            controlled_for_calibration += tuple(active_evaders)
        calibrations, calibration_excluded = prepare_calibrations(
            controlled_for_calibration,
            controllers,
            tracker,
            subjects_by_id,
            corner_subject_map,
            corner_reference,
            arena,
            args,
            logger,
        )
        excluded.update(calibration_excluded)

        final_usable = set(initially_usable)
        final_usable.difference_update(calibration_excluded)
        if not args.passive_evaders:
            final_usable.intersection_update(calibrations)
        else:
            final_usable = {
                marker_id
                for marker_id in final_usable
                if marker_id in active_evaders or marker_id in calibrations
            }
        active_herders, active_evaders = active_role_ids(
            requested_herders,
            requested_evaders,
            final_usable,
        )
        if len(active_herders) < 3:
            raise SafetyStop(f"only {len(active_herders)} calibrated herders remain")
        if not active_evaders:
            raise SafetyStop("no active evaders remain after calibration")

        final_capacity_ok, final_usable_area, final_required_area = cage_has_capacity(
            arena,
            len(active_evaders),
            args.robot_diameter_mm,
            params.cage_hold_margin_mm,
        )
        if not final_capacity_ok and not args.allow_small_cage:
            raise SafetyStop(
                "virtual cage is too small for the final active evaders "
                f"({final_usable_area / 1e6:.2f}m^2 usable, "
                f"{final_required_area / 1e6:.2f}m^2 estimated required); adjust cage ratios or "
                "use --allow-small-cage after a physical safety review"
            )

        active_control_ids = set(active_herders)
        if not args.passive_evaders:
            active_control_ids.update(active_evaders)
        for marker_id, controller in tuple(controllers.items()):
            if marker_id not in active_control_ids:
                controller.send_stop(force=True)
                controller.close(send_stop=False)

        ignored_ids = set(requested_ids) - set(active_herders) - set(active_evaders)
        for marker_id in ignored_ids:
            excluded.setdefault(marker_id, "not in final active role set")

        metadata = {
            "created_at": datetime.now().astimezone().isoformat(),
            "phase": "control",
            "arguments": serializable_args(args),
            "corner_subjects": corner_subject_map,
            "corner_positions_world": corner_reference,
            "arena": arena.as_dict(),
            "control_parameters": asdict(params),
            "requested_herders": requested_herders,
            "requested_evaders": requested_evaders,
            "active_herders": active_herders,
            "active_evaders": active_evaders,
            "excluded_cars": excluded,
            "visibility_ratios": visibility_ratios,
            "calibrations": {
                str(marker_id): {
                    command: asdict(value)
                    for command, value in command_map.items()
                }
                for marker_id, command_map in calibrations.items()
            },
        }
        logger.write_metadata(metadata)
        for marker_id, reason in sorted(excluded.items()):
            logger.event("car_excluded", reason, marker_id)
        if logger.run_dir is not None:
            print(f"Experiment log: {logger.run_dir.resolve()}")

        run_control_loop(
            args,
            tracker,
            subjects_by_id,
            corner_subject_map,
            corner_reference,
            arena,
            params,
            requested_herders,
            requested_evaders,
            active_herders,
            active_evaders,
            controllers,
            calibrations,
            ignored_ids,
            logger,
        )
    except KeyboardInterrupt:
        print("Interrupted. Stopping all connected cars.")
        logger.event("interrupted", "KeyboardInterrupt")
        exit_code = 130
    except (ArenaGeometryError, SafetyStop, RuntimeError, ValueError) as exc:
        print(f"Stopped safely: {exc}")
        logger.event("safety_stop", str(exc))
        exit_code = 2
    finally:
        stop_all(controllers)
        for controller in controllers.values():
            controller.close(send_stop=False)
        tracker.close()
        logger.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
