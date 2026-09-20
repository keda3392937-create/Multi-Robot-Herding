import argparse
import csv
import datetime as dt
import json
import math
import os
import socket
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


VICON_SDK_PATH = Path(r"D:\ViconDataStream\Win64\Python\vicon_dssdk")
if VICON_SDK_PATH.exists():
    sys.path.insert(0, str(VICON_SDK_PATH))

from vicon_dssdk import ViconDataStream

from relative_cage_core import (
    ArenaGeometryError,
    CageHerdingController,
    CaptureHoldMonitor,
    ControlParameters,
    FieldCorners,
    MotionLimits,
    RelativeArena,
    Vec2,
    build_relative_arena,
    select_participating_roles,
    vec_add,
    vec_dot,
    vec_limit,
    vec_norm,
    vec_normalize,
    vec_scale,
    vec_sub,
)
from six_car_repel import connect_wifi, discover_car_ips


DEFAULT_VICON_HOST = "192.168.30.101"
DEFAULT_WIFI_SSID = "YAYA"
CONTROL_PORT = 23
MOTION_COMMANDS = ("F", "RF", "RB", "B", "LB", "LF")
CORNER_KEYS = ("left-up", "left-down", "right-up", "right-down")
COMMAND_REFRESH_SEC = 0.18


def car_key(marker_id: int) -> str:
    return f"car:{marker_id}"


@dataclass(frozen=True)
class CarSpec:
    marker_id: int
    subject_name: str
    ip: str = ""


@dataclass(frozen=True)
class TrackedPose:
    subject_name: str
    position: Vec2
    yaw: float
    seen_at: float


@dataclass(frozen=True)
class ViconSnapshot:
    frame_number: int
    received_at: float
    poses: Dict[str, TrackedPose]
    fresh_keys: Tuple[str, ...]


@dataclass(frozen=True)
class CommandCalibration:
    direction_world: Vec2
    yaw_world: float
    displacement_mm: float
    speed_mm_s: float
    yaw_change_deg: float


CalibrationMap = Dict[int, Dict[str, CommandCalibration]]


class MultiSubjectViconTracker:
    def __init__(
        self,
        host: str,
        subject_by_key: Mapping[str, str],
        max_pose_age_sec: float,
    ):
        self.host = host
        self.subject_by_key = dict(subject_by_key)
        self.max_pose_age_sec = max_pose_age_sec
        self.client = ViconDataStream.Client()
        self.root_by_key: Dict[str, str] = {}
        self.cached_by_key: Dict[str, TrackedPose] = {}
        self.last_frame_number = -1

    def connect(self, timeout_sec: float) -> None:
        print(f"Connecting to Vicon server {self.host} ...")
        self.client.SetConnectionTimeout(1000)
        deadline = time.monotonic() + timeout_sec
        last_error = "connection timed out"
        while not self.client.IsConnected() and time.monotonic() < deadline:
            try:
                self.client.Connect(self.host)
            except ViconDataStream.DataStreamException as exc:
                last_error = str(exc)
                time.sleep(0.5)
        if not self.client.IsConnected():
            raise RuntimeError(f"Could not connect to Vicon server {self.host}: {last_error}")

        self.client.EnableSegmentData()
        self.client.SetStreamMode(ViconDataStream.Client.StreamMode.EClientPull)
        self.client.SetAxisMapping(
            ViconDataStream.Client.AxisMapping.EForward,
            ViconDataStream.Client.AxisMapping.ELeft,
            ViconDataStream.Client.AxisMapping.EUp,
        )
        print("Vicon connected.")

    def close(self) -> None:
        try:
            if self.client.IsConnected():
                self.client.Disconnect()
        except (AttributeError, ViconDataStream.DataStreamException):
            pass

    def get_snapshot(self) -> ViconSnapshot:
        now = time.monotonic()
        try:
            has_frame = self.client.GetFrame()
        except ViconDataStream.DataStreamException as exc:
            print(f"Vicon GetFrame failed: {exc}")
            return self._cached_snapshot(now, ())

        if not has_frame:
            return self._cached_snapshot(now, ())

        try:
            self.last_frame_number = int(self.client.GetFrameNumber())
        except ViconDataStream.DataStreamException:
            pass

        fresh_keys: List[str] = []
        for key, subject_name in self.subject_by_key.items():
            root_segment = self.root_by_key.get(key)
            if root_segment is None:
                try:
                    root_segment = self.client.GetSubjectRootSegmentName(subject_name)
                except ViconDataStream.DataStreamException:
                    continue
                self.root_by_key[key] = root_segment

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
            self.cached_by_key[key] = TrackedPose(
                subject_name=subject_name,
                position=(float(x_mm), float(y_mm)),
                yaw=float(yaw),
                seen_at=now,
            )
            fresh_keys.append(key)

        return self._cached_snapshot(now, tuple(fresh_keys))

    def _cached_snapshot(self, now: float, fresh_keys: Tuple[str, ...]) -> ViconSnapshot:
        active = {
            key: pose
            for key, pose in self.cached_by_key.items()
            if now - pose.seen_at <= self.max_pose_age_sec
        }
        return ViconSnapshot(
            frame_number=self.last_frame_number,
            received_at=now,
            poses=active,
            fresh_keys=fresh_keys,
        )


class SafeTcpCarController:
    def __init__(
        self,
        spec: CarSpec,
        connect_timeout_sec: float,
        io_timeout_sec: float,
    ):
        self.spec = spec
        self.connect_timeout_sec = connect_timeout_sec
        self.io_timeout_sec = io_timeout_sec
        self.sock: Optional[socket.socket] = None
        self.active = False
        self.failure_reason = ""
        self.last_motion_command = "STOP"
        self.last_motion_sent_at = 0.0
        self.last_pwm: Optional[int] = None

    def connect(self) -> bool:
        self.close(send_stop=False)
        try:
            sock = socket.create_connection(
                (self.spec.ip, CONTROL_PORT),
                timeout=self.connect_timeout_sec,
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(self.io_timeout_sec)
            self.sock = sock
            self.active = True
            self.failure_reason = ""
            self.last_motion_command = "STOP"
            self.last_motion_sent_at = 0.0
            self.last_pwm = None
            return True
        except OSError as exc:
            self._fail(f"initial TCP connect failed: {exc}")
            return False

    def _fail(self, reason: str) -> None:
        self.failure_reason = reason
        self.active = False
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def send_line(self, message: str) -> bool:
        if not self.active or self.sock is None:
            return False
        try:
            self.sock.sendall((message + "\n").encode("ascii"))
            return True
        except OSError as exc:
            self._fail(f"TCP send failed: {exc}")
            return False

    def set_motion(self, command: str, pwm: int, force: bool = False) -> bool:
        if command not in MOTION_COMMANDS and command != "STOP":
            raise ValueError(f"Unsupported motion command: {command}")
        if not self.active:
            return False

        pwm = max(0, min(255, int(pwm)))
        if command != "STOP" and (self.last_pwm is None or abs(pwm - self.last_pwm) >= 2):
            if not self.send_line(f"SPD {pwm}"):
                return False
            self.last_pwm = pwm

        now = time.monotonic()
        should_send = (
            force
            or command != self.last_motion_command
            or now - self.last_motion_sent_at >= COMMAND_REFRESH_SEC
        )
        if should_send:
            if not self.send_line(command):
                return False
            self.last_motion_command = command
            self.last_motion_sent_at = now
        return True

    def stop(self, force: bool = True) -> bool:
        return self.set_motion("STOP", 0, force=force)

    def close(self, send_stop: bool = True) -> None:
        if send_stop and self.active and self.sock is not None:
            try:
                self.sock.sendall(b"STOP\n")
            except OSError:
                pass
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.active = False


class ExperimentLogger:
    STATE_HEADER = (
        "wall_time",
        "elapsed_sec",
        "frame",
        "car_id",
        "role",
        "active",
        "visible",
        "seen_age_sec",
        "world_x_mm",
        "world_y_mm",
        "local_x_mm",
        "local_y_mm",
        "yaw_deg",
        "velocity_x_mm_s",
        "velocity_y_mm_s",
        "acceleration_x_mm_s2",
        "acceleration_y_mm_s2",
        "target_x_mm",
        "target_y_mm",
        "command",
        "pwm",
        "direction_score",
        "inside_cage",
        "held",
        "hull_herder",
    )

    def __init__(self, root: Path, enabled: bool):
        self.enabled = enabled
        self.started_monotonic = time.monotonic()
        self.run_dir: Optional[Path] = None
        self.state_file = None
        self.event_file = None
        self.state_writer = None
        self.event_writer = None
        self.last_flush = 0.0
        if not enabled:
            return

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = root / timestamp
        suffix = 1
        while self.run_dir.exists():
            self.run_dir = root / f"{timestamp}_{suffix:02d}"
            suffix += 1
        self.run_dir.mkdir(parents=True)
        self.state_file = (self.run_dir / "states.csv").open("w", newline="", encoding="utf-8")
        self.event_file = (self.run_dir / "events.csv").open("w", newline="", encoding="utf-8")
        self.state_writer = csv.writer(self.state_file)
        self.event_writer = csv.writer(self.event_file)
        self.state_writer.writerow(self.STATE_HEADER)
        self.event_writer.writerow(("wall_time", "elapsed_sec", "event", "details"))

    def write_metadata(self, metadata: Mapping[str, object]) -> None:
        if not self.enabled or self.run_dir is None:
            return
        (self.run_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def event(self, event: str, details: str) -> None:
        if not self.enabled or self.event_writer is None:
            return
        now = dt.datetime.now().isoformat(timespec="milliseconds")
        elapsed = time.monotonic() - self.started_monotonic
        self.event_writer.writerow((now, f"{elapsed:.3f}", event, details))
        self._flush_if_needed(force=True)

    def state(self, row: Sequence[object]) -> None:
        if not self.enabled or self.state_writer is None:
            return
        self.state_writer.writerow(row)
        self._flush_if_needed()

    def _flush_if_needed(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_flush < 1.0:
            return
        if self.state_file is not None:
            self.state_file.flush()
        if self.event_file is not None:
            self.event_file.flush()
        self.last_flush = now

    def close(self) -> None:
        self._flush_if_needed(force=True)
        if self.state_file is not None:
            self.state_file.close()
        if self.event_file is not None:
            self.event_file.close()


def parse_id_spec(text: str) -> Tuple[int, ...]:
    result: List[int] = []
    seen: Set[int] = set()
    normalized = text.replace(";", ",").replace(" ", ",")
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if start <= end else -1
            values: Iterable[int] = range(start, end + step, step)
        else:
            values = (int(token),)
        for marker_id in values:
            if marker_id <= 0:
                raise ValueError("Car IDs must be positive")
            if marker_id not in seen:
                result.append(marker_id)
                seen.add(marker_id)
    if not result:
        raise ValueError("Car ID list cannot be empty")
    return tuple(result)


def parse_id_overrides(
    values: Sequence[str],
    label: str,
    valid_ids: Sequence[int],
) -> Dict[int, str]:
    valid = set(valid_ids)
    result: Dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --{label} value {value!r}; expected ID=VALUE")
        marker_text, override = value.split("=", 1)
        marker_id = int(marker_text)
        if marker_id not in valid:
            raise ValueError(f"Car ID {marker_id} is not in the configured role lists")
        if not override.strip():
            raise ValueError(f"Invalid empty --{label} value for car{marker_id}")
        result[marker_id] = override.strip()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline-inspired real-car cage capture using four named Vicon field corners. "
            "Without --arm the script only validates and prints the relative geometry."
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
    parser.add_argument("--corner-samples", type=int, default=25)
    parser.add_argument("--corner-sample-timeout", type=float, default=6.0)
    parser.add_argument("--corner-max-shift-ratio", type=float, default=0.02)
    parser.add_argument("--vicon-connect-timeout", type=float, default=15.0)
    parser.add_argument("--max-pose-age", type=float, default=0.20)
    parser.add_argument("--initial-visibility-sec", type=float, default=1.0)
    parser.add_argument("--initial-visibility-ratio", type=float, default=0.60)

    parser.add_argument("--wifi-ssid", default=DEFAULT_WIFI_SSID)
    parser.add_argument("--wifi-password", default=os.environ.get("VICON_CAR_WIFI_PASSWORD", ""))
    parser.add_argument("--wifi-wait", type=float, default=4.0)
    parser.add_argument("--skip-wifi", action="store_true")
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--discovery-timeout", type=float, default=5.0)
    parser.add_argument("--bind-ip", action="append", default=[])
    parser.add_argument("--broadcast-ip", action="append", default=[])
    parser.add_argument("--tcp-connect-timeout", type=float, default=0.35)
    parser.add_argument("--tcp-io-timeout", type=float, default=0.15)

    parser.add_argument("--cage-width-ratio", type=float, default=0.32)
    parser.add_argument("--cage-depth-ratio", type=float, default=0.32)
    parser.add_argument("--robot-radius-mm", type=float, default=150.0)
    parser.add_argument("--allow-tight-cage", action="store_true")
    parser.add_argument("--max-outside-distance-ratio", type=float, default=0.03)

    parser.add_argument("--calibration-file", default="relative_cage_calibration.json")
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--calibration-pwm", type=int, default=110)
    parser.add_argument("--calibration-move-sec", type=float, default=0.30)
    parser.add_argument("--calibration-settle-sec", type=float, default=0.30)
    parser.add_argument("--calibration-samples", type=int, default=5)
    parser.add_argument("--min-calibration-displacement-ratio", type=float, default=0.006)
    parser.add_argument("--max-calibration-displacement-ratio", type=float, default=0.12)
    parser.add_argument("--max-calibration-yaw-change-deg", type=float, default=20.0)
    parser.add_argument("--min-calibrated-commands", type=int, default=4)
    parser.add_argument("--max-calibration-angle-gap-deg", type=float, default=150.0)
    parser.add_argument("--calibration-clearance-ratio", type=float, default=0.05)
    parser.add_argument("--calibration-separation-ratio", type=float, default=0.09)
    parser.add_argument("--min-direction-score", type=float, default=0.50)
    parser.add_argument("--command-hysteresis", type=float, default=0.08)

    parser.add_argument("--herder-min-pwm", type=int, default=85)
    parser.add_argument("--herder-max-pwm", type=int, default=170)
    parser.add_argument("--evader-min-pwm", type=int, default=90)
    parser.add_argument("--evader-max-pwm", type=int, default=230)
    parser.add_argument("--pwm-slew-per-sec", type=float, default=180.0)
    parser.add_argument("--stop-speed-ratio", type=float, default=0.08)
    parser.add_argument("--capture-hold-sec", type=float, default=2.0)
    parser.add_argument("--allow-partial-vicon", action="store_true")
    parser.add_argument("--max-runtime-sec", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--log-root", default="relative_cage_runs")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--confirm-external-cage", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    herder_ids = parse_id_spec(args.herders)
    evader_ids = parse_id_spec(args.evaders)
    overlap = sorted(set(herder_ids) & set(evader_ids))
    if overlap:
        raise ValueError(f"Cars cannot be both herder and evader: {overlap}")
    if len(herder_ids) < 3:
        raise ValueError("At least three herders must be configured")
    if args.corner_samples < 3 or args.calibration_samples < 3:
        raise ValueError("Corner and calibration sample counts must be at least three")
    if not 0.0 < args.initial_visibility_ratio <= 1.0:
        raise ValueError("--initial-visibility-ratio must be in (0, 1]")
    if args.robot_radius_mm <= 0.0:
        raise ValueError("--robot-radius-mm must be positive")
    if args.arm and not args.confirm_external_cage:
        raise ValueError(
            "--arm requires --confirm-external-cage because the requested cage extends below the marked field"
        )
    return herder_ids, evader_ids


def make_subject_map(
    args: argparse.Namespace,
    car_specs: Mapping[int, CarSpec],
) -> Dict[str, str]:
    subjects = {
        "left-up": args.left_up_subject,
        "left-down": args.left_down_subject,
        "right-up": args.right_up_subject,
        "right-down": args.right_down_subject,
    }
    subjects.update({car_key(marker_id): spec.subject_name for marker_id, spec in car_specs.items()})
    return subjects


def sample_field_corners(
    tracker: MultiSubjectViconTracker,
    sample_count: int,
    timeout_sec: float,
) -> FieldCorners:
    samples: Dict[str, List[Vec2]] = {key: [] for key in CORNER_KEYS}
    seen_frames: Dict[str, Set[int]] = {key: set() for key in CORNER_KEYS}
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline and any(len(values) < sample_count for values in samples.values()):
        snapshot = tracker.get_snapshot()
        for key in CORNER_KEYS:
            if key not in snapshot.fresh_keys or key not in snapshot.poses:
                continue
            if snapshot.frame_number in seen_frames[key]:
                continue
            samples[key].append(snapshot.poses[key].position)
            seen_frames[key].add(snapshot.frame_number)
        time.sleep(0.005)

    missing = [key for key, values in samples.items() if len(values) < sample_count]
    if missing:
        counts = ", ".join(f"{key}={len(samples[key])}/{sample_count}" for key in CORNER_KEYS)
        raise RuntimeError(f"Could not collect all four field corners ({counts}); missing: {missing}")

    medians = {
        key: (
            float(statistics.median(point[0] for point in values)),
            float(statistics.median(point[1] for point in values)),
        )
        for key, values in samples.items()
    }
    return FieldCorners(
        left_up=medians["left-up"],
        left_down=medians["left-down"],
        right_up=medians["right-up"],
        right_down=medians["right-down"],
    )


def print_arena(arena: RelativeArena) -> None:
    cage_right_bottom_world = arena.local_to_world(arena.cage_right_bottom_local)
    print(
        f"Relative field: width={arena.width_mm:.1f}mm height={arena.height_mm:.1f}mm "
        f"axis_angle={arena.raw_axis_angle_deg:.2f}deg fit_error={arena.max_corner_error_mm:.1f}mm"
    )
    print(
        f"Local frame: origin(left-down)=({arena.origin_world[0]:.1f},{arena.origin_world[1]:.1f}) "
        f"x_axis=({arena.x_axis_world[0]:+.5f},{arena.x_axis_world[1]:+.5f}) "
        f"y_axis=({arena.y_axis_world[0]:+.5f},{arena.y_axis_world[1]:+.5f})"
    )
    print(
        f"External cage: local TL=(0,0) BR=({arena.cage_width_mm:.1f},{-arena.cage_depth_mm:.1f}) "
        f"world BR=({cage_right_bottom_world[0]:.1f},{cage_right_bottom_world[1]:.1f})"
    )
    print("The shared segment y=0, 0<=x<=cage_width is an open entrance, not a virtual wall.")


def cage_capacity_ok(
    arena: RelativeArena,
    evader_count: int,
    robot_radius_mm: float,
) -> Tuple[bool, float, float]:
    usable_width = max(0.0, arena.cage_width_mm - 2.0 * robot_radius_mm)
    usable_depth = max(0.0, arena.cage_depth_mm - 2.0 * robot_radius_mm)
    usable_area = usable_width * usable_depth
    required_area = evader_count * (2.0 * robot_radius_mm) ** 2 * 1.5
    return usable_area >= required_area, usable_area, required_area


def sample_initial_visibility(
    tracker: MultiSubjectViconTracker,
    marker_ids: Sequence[int],
    duration_sec: float,
    required_ratio: float,
) -> Tuple[int, ...]:
    counts = {marker_id: 0 for marker_id in marker_ids}
    frame_count = 0
    seen_frames: Set[int] = set()
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        snapshot = tracker.get_snapshot()
        if snapshot.frame_number in seen_frames:
            time.sleep(0.005)
            continue
        seen_frames.add(snapshot.frame_number)
        frame_count += 1
        fresh = set(snapshot.fresh_keys)
        for marker_id in marker_ids:
            if car_key(marker_id) in fresh:
                counts[marker_id] += 1
        time.sleep(0.005)
    if frame_count == 0:
        return ()
    return tuple(
        marker_id
        for marker_id in marker_ids
        if counts[marker_id] / frame_count >= required_ratio
    )


def stop_all(controllers: Mapping[int, SafeTcpCarController]) -> None:
    for controller in controllers.values():
        controller.stop(force=True)


def circular_mean(angles: Sequence[float]) -> float:
    return math.atan2(
        sum(math.sin(angle) for angle in angles),
        sum(math.cos(angle) for angle in angles),
    )


def angle_difference(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def collect_pose_median(
    tracker: MultiSubjectViconTracker,
    key: str,
    sample_count: int,
    timeout_sec: float,
) -> Optional[TrackedPose]:
    samples: List[TrackedPose] = []
    frames: Set[int] = set()
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline and len(samples) < sample_count:
        snapshot = tracker.get_snapshot()
        if (
            key in snapshot.fresh_keys
            and key in snapshot.poses
            and snapshot.frame_number not in frames
        ):
            samples.append(snapshot.poses[key])
            frames.add(snapshot.frame_number)
        time.sleep(0.005)
    if len(samples) < sample_count:
        return None
    return TrackedPose(
        subject_name=samples[-1].subject_name,
        position=(
            float(statistics.median(pose.position[0] for pose in samples)),
            float(statistics.median(pose.position[1] for pose in samples)),
        ),
        yaw=circular_mean([pose.yaw for pose in samples]),
        seen_at=max(pose.seen_at for pose in samples),
    )


def corners_health(
    snapshot: ViconSnapshot,
    frozen_corners: FieldCorners,
    max_shift_mm: float,
) -> Tuple[bool, str]:
    references = frozen_corners.as_dict()
    missing = [key for key in CORNER_KEYS if key not in snapshot.poses]
    if missing:
        return False, "missing corner pose: " + ",".join(missing)
    moved = []
    for key, reference in references.items():
        shift = vec_norm(vec_sub(snapshot.poses[key].position, reference))
        if shift > max_shift_mm:
            moved.append(f"{key}:{shift:.1f}mm")
    if moved:
        return False, "corner moved: " + ",".join(moved)
    return True, ""


def rotate(vector: Vec2, angle: float) -> Vec2:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        vector[0] * cos_a - vector[1] * sin_a,
        vector[0] * sin_a + vector[1] * cos_a,
    )


def calibration_angle_gap_deg(calibrations: Mapping[str, CommandCalibration]) -> float:
    if len(calibrations) < 2:
        return 360.0
    angles = sorted(
        (math.atan2(value.direction_world[1], value.direction_world[0]) - value.yaw_world)
        % (2.0 * math.pi)
        for value in calibrations.values()
    )
    gaps = [angles[index + 1] - angles[index] for index in range(len(angles) - 1)]
    gaps.append(angles[0] + 2.0 * math.pi - angles[-1])
    return math.degrees(max(gaps))


def calibration_valid(
    calibrations: Mapping[str, CommandCalibration],
    min_commands: int,
    max_gap_deg: float,
) -> Tuple[bool, str]:
    if len(calibrations) < min_commands:
        return False, f"only {len(calibrations)} valid commands; need {min_commands}"
    gap = calibration_angle_gap_deg(calibrations)
    if gap > max_gap_deg:
        return False, f"direction coverage gap {gap:.1f}deg exceeds {max_gap_deg:.1f}deg"
    return True, ""


def load_calibrations(
    path: Path,
    specs: Mapping[int, CarSpec],
    args: argparse.Namespace,
) -> CalibrationMap:
    if args.recalibrate or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Ignoring unreadable calibration file {path}: {exc}")
        return {}
    if payload.get("version") != 1:
        print(f"Ignoring calibration file {path}: unsupported version")
        return {}

    result: CalibrationMap = {}
    cars = payload.get("cars", {})
    for marker_id, spec in specs.items():
        car_data = cars.get(str(marker_id), {})
        if car_data.get("subject_name") != spec.subject_name:
            continue
        commands: Dict[str, CommandCalibration] = {}
        for command, value in car_data.get("commands", {}).items():
            if command not in MOTION_COMMANDS:
                continue
            try:
                commands[command] = CommandCalibration(
                    direction_world=(
                        float(value["direction_world"][0]),
                        float(value["direction_world"][1]),
                    ),
                    yaw_world=float(value["yaw_world"]),
                    displacement_mm=float(value["displacement_mm"]),
                    speed_mm_s=float(value["speed_mm_s"]),
                    yaw_change_deg=float(value.get("yaw_change_deg", 0.0)),
                )
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        valid, reason = calibration_valid(
            commands,
            args.min_calibrated_commands,
            args.max_calibration_angle_gap_deg,
        )
        if valid:
            result[marker_id] = commands
        elif commands:
            print(f"Ignoring saved calibration for car{marker_id}: {reason}")
    return result


def save_calibrations(
    path: Path,
    calibrations: CalibrationMap,
    specs: Mapping[int, CarSpec],
    vicon_host: str,
) -> None:
    payload = {
        "version": 1,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "vicon_host": vicon_host,
        "cars": {
            str(marker_id): {
                "subject_name": specs[marker_id].subject_name,
                "commands": {
                    command: asdict(value)
                    for command, value in commands.items()
                },
            }
            for marker_id, commands in calibrations.items()
            if marker_id in specs
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def calibration_position_safe(
    marker_id: int,
    pose: TrackedPose,
    snapshot: ViconSnapshot,
    active_ids: Sequence[int],
    arena: RelativeArena,
    clearance_mm: float,
    separation_mm: float,
) -> Tuple[bool, str]:
    local = arena.world_to_local(pose.position)
    if not arena.contains_local(local):
        return False, "car is outside the relative field/cage union"
    if arena.field_contains_local(local):
        wall_distances = [
            local[0],
            arena.width_mm - local[0],
            arena.height_mm - local[1],
        ]
        if local[0] >= arena.cage_width_mm:
            wall_distances.append(local[1])
    else:
        wall_distances = [
            local[0],
            arena.cage_width_mm - local[0],
            local[1] + arena.cage_depth_mm,
        ]
    boundary_probe = min(wall_distances)
    if boundary_probe < clearance_mm:
        return False, f"car is only {boundary_probe:.1f}mm from a virtual outer boundary"
    for other_id in active_ids:
        if other_id == marker_id:
            continue
        other = snapshot.poses.get(car_key(other_id))
        if other is None:
            continue
        distance = vec_norm(vec_sub(pose.position, other.position))
        if distance < separation_mm:
            return False, f"car{other_id} is only {distance:.1f}mm away"
    return True, ""


def calibrate_car(
    marker_id: int,
    controller: SafeTcpCarController,
    tracker: MultiSubjectViconTracker,
    active_ids: Sequence[int],
    arena: RelativeArena,
    frozen_corners: FieldCorners,
    args: argparse.Namespace,
) -> Dict[str, CommandCalibration]:
    key = car_key(marker_id)
    commands: Dict[str, CommandCalibration] = {}
    max_corner_shift = args.corner_max_shift_ratio * arena.scale_mm
    min_displacement = args.min_calibration_displacement_ratio * arena.scale_mm
    max_displacement = args.max_calibration_displacement_ratio * arena.scale_mm
    clearance = args.calibration_clearance_ratio * arena.scale_mm
    separation = args.calibration_separation_ratio * arena.scale_mm

    print(f"Sequentially calibrating car{marker_id} at PWM {args.calibration_pwm} ...")
    for command in MOTION_COMMANDS:
        controller.stop(force=True)
        time.sleep(args.calibration_settle_sec)
        start = collect_pose_median(
            tracker,
            key,
            args.calibration_samples,
            timeout_sec=2.0,
        )
        if start is None:
            print(f"  {command}: no fresh start pose")
            continue
        snapshot = tracker.get_snapshot()
        healthy, reason = corners_health(snapshot, frozen_corners, max_corner_shift)
        if not healthy:
            raise RuntimeError(f"Field corner failure during calibration: {reason}")
        safe, reason = calibration_position_safe(
            marker_id,
            start,
            snapshot,
            active_ids,
            arena,
            clearance,
            separation,
        )
        if not safe:
            print(f"  {command}: skipped because {reason}")
            continue

        deadline = time.monotonic() + args.calibration_move_sec
        pulse_ok = True
        while time.monotonic() < deadline:
            if not controller.set_motion(command, args.calibration_pwm, force=True):
                pulse_ok = False
                break
            snapshot = tracker.get_snapshot()
            healthy, reason = corners_health(snapshot, frozen_corners, max_corner_shift)
            if not healthy:
                controller.stop(force=True)
                raise RuntimeError(f"Field corner failure during calibration: {reason}")
            time.sleep(0.04)
        controller.stop(force=True)
        time.sleep(args.calibration_settle_sec)
        if not pulse_ok:
            print(f"  {command}: TCP failed: {controller.failure_reason}")
            break

        end = collect_pose_median(
            tracker,
            key,
            args.calibration_samples,
            timeout_sec=2.0,
        )
        if end is None:
            print(f"  {command}: no fresh end pose")
            continue
        displacement = vec_sub(end.position, start.position)
        distance = vec_norm(displacement)
        yaw_change_deg = abs(math.degrees(angle_difference(end.yaw, start.yaw)))
        warning = ""
        if distance < min_displacement:
            warning = f"movement too small ({distance:.1f}mm)"
        elif distance > max_displacement:
            warning = f"movement too large ({distance:.1f}mm)"
        elif yaw_change_deg > args.max_calibration_yaw_change_deg:
            warning = f"yaw changed {yaw_change_deg:.1f}deg"
        if warning:
            print(f"  {command}: rejected: {warning}")
            continue

        value = CommandCalibration(
            direction_world=vec_normalize(displacement),
            yaw_world=start.yaw,
            displacement_mm=distance,
            speed_mm_s=distance / max(args.calibration_move_sec, 1e-6),
            yaw_change_deg=yaw_change_deg,
        )
        commands[command] = value
        print(
            f"  {command}: dist={distance:.1f}mm speed={value.speed_mm_s:.1f}mm/s "
            f"dir=({value.direction_world[0]:+.3f},{value.direction_world[1]:+.3f}) "
            f"yaw_change={yaw_change_deg:.1f}deg"
        )
    controller.stop(force=True)
    return commands


def choose_calibrated_command(
    calibrations: Mapping[str, CommandCalibration],
    desired_world_velocity: Vec2,
    current_yaw: float,
    current_command: str,
    min_score: float,
    hysteresis: float,
) -> Tuple[str, float]:
    desired = vec_normalize(desired_world_velocity)
    if vec_norm(desired) < 1e-9 or not calibrations:
        return "STOP", 0.0
    scores = {
        command: vec_dot(
            rotate(value.direction_world, current_yaw - value.yaw_world),
            desired,
        )
        for command, value in calibrations.items()
    }
    best_command, best_score = max(scores.items(), key=lambda item: item[1])
    if current_command in scores and scores[current_command] >= best_score - hysteresis:
        best_command = current_command
        best_score = scores[current_command]
    if best_score < min_score:
        return "STOP", best_score
    return best_command, best_score


def velocity_to_pwm(
    velocity: Vec2,
    limits: MotionLimits,
    min_pwm: int,
    max_pwm: int,
    stop_speed_ratio: float,
) -> int:
    speed_ratio = vec_norm(velocity) / max(limits.max_speed_mm_s, 1e-6)
    if speed_ratio < stop_speed_ratio:
        return 0
    speed_ratio = min(1.0, speed_ratio)
    return int(round(min_pwm + (max_pwm - min_pwm) * speed_ratio))


def slew_pwm(previous: int, desired: int, dt_sec: float, rate_per_sec: float) -> int:
    if desired <= 0:
        return 0
    if previous <= 0:
        return desired
    max_change = max(1.0, rate_per_sec * dt_sec)
    return int(round(clamp_number(desired, previous - max_change, previous + max_change)))


def clamp_number(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def role_for(marker_id: int, herder_ids: Sequence[int], evader_ids: Sequence[int]) -> str:
    if marker_id in herder_ids:
        return "herder"
    if marker_id in evader_ids:
        return "evader"
    return "ignored"


def sanitized_args(args: argparse.Namespace) -> Dict[str, object]:
    values = dict(vars(args))
    values.pop("wifi_password", None)
    return values


def main() -> None:
    args = parse_args()
    requested_herders, requested_evaders = validate_args(args)
    requested_ids = tuple(dict.fromkeys(requested_herders + requested_evaders))
    subject_overrides = parse_id_overrides(args.subject, "subject", requested_ids)
    ip_overrides = parse_id_overrides(args.car_ip, "car-ip", requested_ids)
    specs = {
        marker_id: CarSpec(
            marker_id=marker_id,
            subject_name=subject_overrides.get(marker_id, f"kedaya{marker_id}"),
            ip=ip_overrides.get(marker_id, ""),
        )
        for marker_id in requested_ids
    }

    if not args.skip_wifi:
        connect_wifi(
            args.wifi_ssid,
            args.wifi_wait,
            password=args.wifi_password,
        )

    subjects = make_subject_map(args, specs if args.arm else {})
    tracker = MultiSubjectViconTracker(
        args.vicon_host,
        subjects,
        max_pose_age_sec=args.max_pose_age,
    )
    controllers: Dict[int, SafeTcpCarController] = {}
    logger = ExperimentLogger(Path(args.log_root), enabled=args.arm and not args.no_log)
    try:
        tracker.connect(args.vicon_connect_timeout)
        frozen_corners = sample_field_corners(
            tracker,
            args.corner_samples,
            args.corner_sample_timeout,
        )
        arena = build_relative_arena(
            frozen_corners,
            cage_width_ratio=args.cage_width_ratio,
            cage_depth_ratio=args.cage_depth_ratio,
        )
        print_arena(arena)
        capacity_ok, usable_area, required_area = cage_capacity_ok(
            arena,
            len(requested_evaders),
            args.robot_radius_mm,
        )
        print(
            f"Cage capacity check: usable={usable_area / 1e6:.2f}m^2 "
            f"estimated_required={required_area / 1e6:.2f}m^2"
        )
        if not capacity_ok and not args.allow_tight_cage:
            raise RuntimeError(
                "The relative cage is too small for the configured evader count. "
                "Adjust the cage ratios or pass --allow-tight-cage after a physical safety review."
            )

        if not args.arm:
            print(
                "Geometry validation complete. No car discovery, TCP command, or motion calibration was performed. "
                "Use --arm --confirm-external-cage only after checking the printed cage area is physically clear "
                "and covered by Vicon."
            )
            return

        discovered: Dict[int, str] = {}
        if not args.skip_discovery:
            discovered = discover_car_ips(
                requested_ids,
                args.discovery_timeout,
                bind_ips=args.bind_ip,
                broadcast_ips=args.broadcast_ip,
            )
        ignored_reasons: Dict[int, str] = {}
        connected_ids: List[int] = []
        for marker_id in requested_ids:
            ip = specs[marker_id].ip or discovered.get(marker_id, "")
            if not ip:
                ignored_reasons[marker_id] = "not discovered and no --car-ip override"
                continue
            spec = CarSpec(marker_id, specs[marker_id].subject_name, ip)
            specs[marker_id] = spec
            controller = SafeTcpCarController(
                spec,
                connect_timeout_sec=args.tcp_connect_timeout,
                io_timeout_sec=args.tcp_io_timeout,
            )
            if not controller.connect():
                ignored_reasons[marker_id] = controller.failure_reason
                continue
            if not controller.stop(force=True):
                ignored_reasons[marker_id] = controller.failure_reason
                controller.close(send_stop=False)
                continue
            controllers[marker_id] = controller
            connected_ids.append(marker_id)
            print(f"car{marker_id}: TCP connected at {ip}:{CONTROL_PORT}")

        visible_ids = sample_initial_visibility(
            tracker,
            connected_ids,
            args.initial_visibility_sec,
            args.initial_visibility_ratio,
        )
        for marker_id in connected_ids:
            if marker_id not in visible_ids:
                ignored_reasons[marker_id] = "Vicon pose was not fresh often enough during preflight"
                controllers[marker_id].close()
                controllers.pop(marker_id, None)

        initially_available = set(controllers) & set(visible_ids)
        active_herders, active_evaders = select_participating_roles(
            requested_herders,
            requested_evaders,
            initially_available,
        )
        if len(active_herders) < 3:
            raise RuntimeError(f"Only {len(active_herders)} usable herders remain; at least three are required")
        if not active_evaders:
            raise RuntimeError("No usable evader remains after network and Vicon preflight")

        calibration_path = Path(args.calibration_file)
        calibrations = load_calibrations(calibration_path, specs, args)
        active_ids = active_herders + active_evaders
        for marker_id in active_ids:
            if marker_id in calibrations:
                print(f"car{marker_id}: using saved calibration ({len(calibrations[marker_id])} commands)")
                continue
            commands = calibrate_car(
                marker_id,
                controllers[marker_id],
                tracker,
                active_ids,
                arena,
                frozen_corners,
                args,
            )
            valid, reason = calibration_valid(
                commands,
                args.min_calibrated_commands,
                args.max_calibration_angle_gap_deg,
            )
            if valid and controllers[marker_id].active:
                calibrations[marker_id] = commands
            else:
                ignored_reasons[marker_id] = f"calibration rejected: {reason or controllers[marker_id].failure_reason}"
                controllers[marker_id].close()
                controllers.pop(marker_id, None)

        save_calibrations(calibration_path, calibrations, specs, args.vicon_host)
        calibrated_ids = set(calibrations) & set(controllers)
        active_herders, active_evaders = select_participating_roles(
            active_herders,
            active_evaders,
            calibrated_ids,
        )
        if len(active_herders) < 3:
            raise RuntimeError(f"Only {len(active_herders)} calibrated herders remain; at least three are required")
        if not active_evaders:
            raise RuntimeError("No calibrated evader remains")

        active_ids = active_herders + active_evaders
        for marker_id in tuple(controllers):
            if marker_id not in active_ids:
                controllers[marker_id].close()
                controllers.pop(marker_id, None)

        parameters = ControlParameters.for_arena(arena, args.robot_radius_mm)
        control = CageHerdingController(
            arena,
            active_herders,
            active_evaders,
            parameters,
            seed=args.seed,
        )
        capture_monitor = CaptureHoldMonitor(args.capture_hold_sec)
        pwm_by_id = {marker_id: 0 for marker_id in active_ids}
        logger.write_metadata(
            {
                "started_at": dt.datetime.now().isoformat(timespec="seconds"),
                "arguments": sanitized_args(args),
                "requested_herders": requested_herders,
                "requested_evaders": requested_evaders,
                "active_herders": active_herders,
                "active_evaders": active_evaders,
                "ignored_reasons": {str(key): value for key, value in ignored_reasons.items()},
                "arena": arena.to_dict(),
                "control_parameters": asdict(parameters),
                "calibration_file": str(calibration_path.resolve()),
            }
        )
        for marker_id, reason in sorted(ignored_reasons.items()):
            print(f"Ignoring car{marker_id} for this experiment: {reason}")
            logger.event("car_ignored", f"car{marker_id}: {reason}")
        print(
            f"Active experiment: herders={active_herders} evaders={active_evaders}. "
            "Press Ctrl+C to stop."
        )

        started_at = time.monotonic()
        last_loop_at = started_at
        last_status_at = 0.0
        last_corner_problem = ""
        while True:
            loop_at = time.monotonic()
            loop_dt = clamp_number(loop_at - last_loop_at, 0.01, 0.15)
            last_loop_at = loop_at
            snapshot = tracker.get_snapshot()

            healthy_corners, corner_problem = corners_health(
                snapshot,
                frozen_corners,
                args.corner_max_shift_ratio * arena.scale_mm,
            )
            if not healthy_corners:
                stop_all(controllers)
                if corner_problem != last_corner_problem:
                    print(f"Safety STOP: {corner_problem}")
                    logger.event("corner_safety_stop", corner_problem)
                    last_corner_problem = corner_problem
                if corner_problem.startswith("corner moved"):
                    break
                time.sleep(0.05)
                continue
            last_corner_problem = ""

            active_ids = active_herders + active_evaders
            local_positions: Dict[int, Vec2] = {}
            for marker_id in active_ids:
                pose = snapshot.poses.get(car_key(marker_id))
                if pose is not None:
                    local_positions[marker_id] = arena.world_to_local(pose.position)

            severe_outside = [
                marker_id
                for marker_id, position in local_positions.items()
                if arena.outside_distance_local(position)
                > args.max_outside_distance_ratio * arena.scale_mm
            ]
            if severe_outside:
                stop_all(controllers)
                reason = f"participating cars too far outside the L-shaped arena: {severe_outside}"
                print(f"Safety STOP: {reason}")
                logger.event("outside_safety_stop", reason)
                break

            obstacle_positions = [
                arena.world_to_local(pose.position)
                for marker_id in requested_ids
                if marker_id not in active_ids
                for pose in [snapshot.poses.get(car_key(marker_id))]
                if pose is not None
            ]
            result = control.step(
                local_positions,
                obstacle_positions,
                loop_dt,
                require_all_visible=not args.allow_partial_vicon,
            )

            all_visible = (
                len(result.visible_herders) == len(active_herders)
                and len(result.visible_evaders) == len(active_evaders)
            )
            all_held = all_visible and len(result.held_evaders) == len(active_evaders)
            capture_confirmed = capture_monitor.update(loop_at, all_held, all_visible)
            commands: Dict[int, str] = {}
            scores: Dict[int, float] = {}
            desired_pwms: Dict[int, int] = {}

            for marker_id in active_ids:
                pose = snapshot.poses.get(car_key(marker_id))
                local_velocity = result.velocities.get(marker_id, (0.0, 0.0))
                if (
                    capture_confirmed
                    or not result.safe_to_move
                    or pose is None
                    or vec_norm(local_velocity) < 1e-6
                ):
                    commands[marker_id] = "STOP"
                    scores[marker_id] = 0.0
                    desired_pwms[marker_id] = 0
                    continue

                world_velocity = arena.local_vector_to_world(local_velocity)
                command, score = choose_calibrated_command(
                    calibrations[marker_id],
                    world_velocity,
                    pose.yaw,
                    controllers[marker_id].last_motion_command,
                    args.min_direction_score,
                    args.command_hysteresis,
                )
                limits = parameters.herder_limits if marker_id in active_herders else parameters.evader_limits
                min_pwm = args.herder_min_pwm if marker_id in active_herders else args.evader_min_pwm
                max_pwm = args.herder_max_pwm if marker_id in active_herders else args.evader_max_pwm
                pwm = velocity_to_pwm(
                    local_velocity,
                    limits,
                    min_pwm,
                    max_pwm,
                    args.stop_speed_ratio,
                )
                if command == "STOP" or pwm == 0:
                    command = "STOP"
                    pwm = 0
                pwm = slew_pwm(
                    pwm_by_id.get(marker_id, 0),
                    pwm,
                    loop_dt,
                    args.pwm_slew_per_sec,
                )
                commands[marker_id] = command
                scores[marker_id] = score
                desired_pwms[marker_id] = pwm

            failed_ids: List[int] = []
            for marker_id in active_ids:
                controller = controllers[marker_id]
                if not controller.set_motion(
                    commands.get(marker_id, "STOP"),
                    desired_pwms.get(marker_id, 0),
                ):
                    failed_ids.append(marker_id)
                else:
                    pwm_by_id[marker_id] = desired_pwms.get(marker_id, 0)

            wall_time = dt.datetime.now().isoformat(timespec="milliseconds")
            elapsed = loop_at - started_at
            in_cage_set = set(result.evaders_in_cage)
            held_set = set(result.held_evaders)
            hull_set = set(result.herder_hull_ids)
            for marker_id in requested_ids:
                pose = snapshot.poses.get(car_key(marker_id))
                local = arena.world_to_local(pose.position) if pose is not None else ("", "")
                velocity = result.velocities.get(marker_id, (0.0, 0.0))
                acceleration = result.accelerations.get(marker_id, (0.0, 0.0))
                target = result.targets.get(marker_id, ("", ""))
                logger.state(
                    (
                        wall_time,
                        f"{elapsed:.3f}",
                        snapshot.frame_number,
                        marker_id,
                        role_for(marker_id, active_herders, active_evaders),
                        marker_id in active_ids,
                        pose is not None,
                        f"{snapshot.received_at - pose.seen_at:.4f}" if pose is not None else "",
                        f"{pose.position[0]:.3f}" if pose is not None else "",
                        f"{pose.position[1]:.3f}" if pose is not None else "",
                        f"{local[0]:.3f}" if pose is not None else "",
                        f"{local[1]:.3f}" if pose is not None else "",
                        f"{math.degrees(pose.yaw):.3f}" if pose is not None else "",
                        f"{velocity[0]:.3f}",
                        f"{velocity[1]:.3f}",
                        f"{acceleration[0]:.3f}",
                        f"{acceleration[1]:.3f}",
                        f"{target[0]:.3f}" if target[0] != "" else "",
                        f"{target[1]:.3f}" if target[1] != "" else "",
                        commands.get(marker_id, "IGNORED"),
                        desired_pwms.get(marker_id, ""),
                        f"{scores.get(marker_id, 0.0):.4f}" if marker_id in active_ids else "",
                        marker_id in in_cage_set,
                        marker_id in held_set,
                        marker_id in hull_set,
                    )
                )

            if failed_ids:
                for marker_id in failed_ids:
                    reason = controllers[marker_id].failure_reason or "TCP became inactive"
                    print(f"Removing car{marker_id} from this experiment: {reason}")
                    logger.event("runtime_car_removed", f"car{marker_id}: {reason}")
                    controllers[marker_id].close(send_stop=False)
                    controllers.pop(marker_id, None)
                    control.reset(marker_id)
                    pwm_by_id.pop(marker_id, None)
                active_herders, active_evaders = select_participating_roles(
                    active_herders,
                    active_evaders,
                    controllers,
                )
                if len(active_herders) < 3 or not active_evaders:
                    stop_all(controllers)
                    reason = "too few active role members remain after TCP failure"
                    print(f"Safety STOP: {reason}")
                    logger.event("insufficient_roles", reason)
                    break
                control = CageHerdingController(
                    arena,
                    active_herders,
                    active_evaders,
                    parameters,
                    seed=args.seed,
                )
                capture_monitor.reset()
                continue

            if capture_confirmed:
                stop_all(controllers)
                message = (
                    f"Capture confirmed: all {len(active_evaders)} active evaders remained inside "
                    f"the cage for {args.capture_hold_sec:.1f}s"
                )
                print(message)
                logger.event("capture_confirmed", message)
                break

            if loop_at - last_status_at >= 0.5:
                status = (
                    f"H={len(result.visible_herders)}/{len(active_herders)} "
                    f"E={len(result.visible_evaders)}/{len(active_evaders)} "
                    f"hull={result.herder_hull_ids} outside={result.outside_evaders} "
                    f"in_cage={result.evaders_in_cage} held={result.held_evaders} "
                    f"hold={capture_monitor.elapsed(loop_at):.1f}/{args.capture_hold_sec:.1f}s"
                )
                if result.stop_reason:
                    status += f" STOP={result.stop_reason}"
                print(status)
                last_status_at = loop_at

            if args.max_runtime_sec > 0.0 and elapsed >= args.max_runtime_sec:
                stop_all(controllers)
                logger.event("runtime_limit", f"Reached {args.max_runtime_sec:.1f}s")
                print("Maximum runtime reached; stopping all cars.")
                break
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("Interrupted; stopping all cars.")
        logger.event("keyboard_interrupt", "operator interrupted the run")
    except (ArenaGeometryError, RuntimeError, ValueError) as exc:
        stop_all(controllers)
        logger.event("fatal_error", str(exc))
        raise
    finally:
        stop_all(controllers)
        for controller in controllers.values():
            controller.close(send_stop=False)
        tracker.close()
        logger.close()


if __name__ == "__main__":
    main()
