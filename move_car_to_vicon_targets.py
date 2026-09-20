import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


VICON_SDK_PATH = Path(r"D:\ViconDataStream\Win64\Python\vicon_dssdk")
if VICON_SDK_PATH.exists():
    sys.path.insert(0, str(VICON_SDK_PATH))

from vicon_dssdk import ViconDataStream

from six_car_repel import (
    CAR_CONFIGS,
    CAR_IDS,
    DEFAULT_WIFI_SSID,
    DEFAULT_WIFI_WAIT_SEC,
    VICON_HOST,
    CarConfig,
    TcpCarController,
    connect_wifi,
    discover_car_ips,
)


MOTION_COMMANDS = ("F", "RF", "RB", "B", "LB", "LF")


@dataclass
class SubjectPose:
    position: Tuple[float, float]
    yaw: float
    seen_at: float


@dataclass
class CommandCalibration:
    direction: Tuple[float, float]
    yaw: float


class ViconPoseTracker:
    def __init__(self, host: str):
        self.host = host
        self.client = ViconDataStream.Client()
        self.root_segments: Dict[str, str] = {}
        self.cached_poses: Dict[str, SubjectPose] = {}

    def connect(self) -> None:
        print(f"Connecting to Vicon server: {self.host} ...")
        self.client.SetConnectionTimeout(1000)
        while not self.client.IsConnected():
            try:
                self.client.Connect(self.host)
            except ViconDataStream.DataStreamException as exc:
                print("Vicon connect failed:", exc)
                time.sleep(1.0)

        print("Vicon connected.")
        self.client.EnableSegmentData()
        self.client.SetStreamMode(ViconDataStream.Client.StreamMode.EClientPull)
        self.client.SetAxisMapping(
            ViconDataStream.Client.AxisMapping.EForward,
            ViconDataStream.Client.AxisMapping.ELeft,
            ViconDataStream.Client.AxisMapping.EUp,
        )

    def update_frame(self) -> None:
        try:
            self.client.GetFrame()
        except ViconDataStream.DataStreamException as exc:
            print("Vicon GetFrame failed:", exc)

    def get_pose(self, subject_name: str, max_age_sec: float = 0.25) -> Optional[SubjectPose]:
        self.update_frame()
        root_segment = self.root_segments.get(subject_name)
        if root_segment is None:
            try:
                root_segment = self.client.GetSubjectRootSegmentName(subject_name)
            except ViconDataStream.DataStreamException:
                return self.active_cached_pose(subject_name, max_age_sec)
            self.root_segments[subject_name] = root_segment
            print(f"Vicon subject: {subject_name}")

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
            return self.active_cached_pose(subject_name, max_age_sec)

        if translation_occluded or rotation_occluded:
            return self.active_cached_pose(subject_name, max_age_sec)

        x_mm, y_mm, _z_mm = translation
        _rx, _ry, yaw = rotation
        pose = SubjectPose(
            position=(float(x_mm), float(y_mm)),
            yaw=float(yaw),
            seen_at=time.monotonic(),
        )
        self.cached_poses[subject_name] = pose
        return pose

    def active_cached_pose(self, subject_name: str, max_age_sec: float) -> Optional[SubjectPose]:
        pose = self.cached_poses.get(subject_name)
        if pose is not None and time.monotonic() - pose.seen_at <= max_age_sec:
            return pose
        return None


def vec_sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def vec_norm(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def vec_normalize(v: Tuple[float, float]) -> Tuple[float, float]:
    norm = vec_norm(v)
    if norm < 1e-6:
        return (0.0, 0.0)
    return (v[0] / norm, v[1] / norm)


def dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def rotate(v: Tuple[float, float], angle_rad: float) -> Tuple[float, float]:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return (v[0] * cos_a - v[1] * sin_a, v[0] * sin_a + v[1] * cos_a)


def wait_for_pose(
    tracker: ViconPoseTracker,
    subject_name: str,
    timeout_sec: float,
) -> Optional[SubjectPose]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        pose = tracker.get_pose(subject_name)
        if pose is not None:
            return pose
        time.sleep(0.02)
    return None


def drive_for_duration(controller: TcpCarController, command: str, duration_sec: float) -> None:
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        controller.send(command, force=True)
        time.sleep(0.12)


def sample_motion_vector(
    controller: TcpCarController,
    tracker: ViconPoseTracker,
    car_subject: str,
    command: str,
    move_sec: float,
    settle_sec: float,
) -> Tuple[CommandCalibration, float]:
    controller.send("STOP", force=True)
    time.sleep(settle_sec)

    start = wait_for_pose(tracker, car_subject, 2.0)
    if start is None:
        raise RuntimeError(f"No Vicon pose for {car_subject} before command {command}")

    drive_for_duration(controller, command, move_sec)
    end = wait_for_pose(tracker, car_subject, 2.0)
    controller.send("STOP", force=True)
    time.sleep(settle_sec)

    if end is None:
        raise RuntimeError(f"No Vicon pose for {car_subject} after command {command}")

    displacement = vec_sub(end.position, start.position)
    return CommandCalibration(direction=vec_normalize(displacement), yaw=start.yaw), vec_norm(displacement)


def calibrate_commands(
    controller: TcpCarController,
    tracker: ViconPoseTracker,
    car_subject: str,
    move_sec: float,
    settle_sec: float,
    min_displacement_mm: float,
) -> Dict[str, CommandCalibration]:
    calibrations: Dict[str, CommandCalibration] = {}
    print("Measuring command directions in Vicon world coordinates...")

    for command in MOTION_COMMANDS:
        calibration, distance = sample_motion_vector(
            controller,
            tracker,
            car_subject,
            command,
            move_sec,
            settle_sec,
        )
        calibrations[command] = calibration
        warning = "  LOW MOTION" if distance < min_displacement_mm else ""
        direction = calibration.direction
        print(
            f"{command:>2}: dist={distance:7.1f}mm "
            f"unit=({direction[0]:+.3f}, {direction[1]:+.3f}) "
            f"yaw={math.degrees(calibration.yaw):+.1f}deg{warning}"
        )

    return calibrations


def best_command_for_vector(
    calibrations: Dict[str, CommandCalibration],
    desired_world_vector: Tuple[float, float],
    current_yaw: float,
) -> Tuple[str, float]:
    desired_direction = vec_normalize(desired_world_vector)
    scored_commands = []
    for command, calibration in calibrations.items():
        current_direction = rotate(calibration.direction, current_yaw - calibration.yaw)
        scored_commands.append((command, dot(current_direction, desired_direction)))
    return max(scored_commands, key=lambda item: item[1])


def set_speed(controller: TcpCarController, speed: int) -> None:
    controller.send(f"SPD {max(0, min(255, speed))}", force=True)


def move_to_target(
    controller: TcpCarController,
    tracker: ViconPoseTracker,
    calibrations: Dict[str, CommandCalibration],
    car_subject: str,
    target_subject: str,
    speed: int,
    near_speed: int,
    slow_radius_mm: float,
    tolerance_mm: float,
    max_time_sec: float,
) -> bool:
    print(f"Moving {car_subject} to target {target_subject}...")
    deadline = time.monotonic() + max_time_sec if max_time_sec > 0 else None
    last_status_time = 0.0
    current_speed = None

    while deadline is None or time.monotonic() < deadline:
        car_pose = tracker.get_pose(car_subject)
        target_pose = tracker.get_pose(target_subject)
        if car_pose is None or target_pose is None:
            controller.send("STOP", force=True)
            missing = []
            if car_pose is None:
                missing.append(car_subject)
            if target_pose is None:
                missing.append(target_subject)
            print("Waiting for Vicon subject: " + ", ".join(missing))
            time.sleep(0.1)
            continue

        to_target = vec_sub(target_pose.position, car_pose.position)
        distance = vec_norm(to_target)
        if distance <= tolerance_mm:
            controller.send("STOP", force=True)
            print(f"Reached {target_subject}: distance={distance:.1f}mm")
            return True

        requested_speed = near_speed if distance <= slow_radius_mm else speed
        if requested_speed != current_speed:
            set_speed(controller, requested_speed)
            current_speed = requested_speed

        command, alignment = best_command_for_vector(calibrations, to_target, car_pose.yaw)
        controller.send(command, force=True)

        now = time.monotonic()
        if now - last_status_time >= 0.5:
            print(
                f"{target_subject}: dist={distance:7.1f}mm "
                f"cmd={command} align={alignment:+.3f} "
                f"car=({car_pose.position[0]:.0f},{car_pose.position[1]:.0f}) "
                f"target=({target_pose.position[0]:.0f},{target_pose.position[1]:.0f})"
            )
            last_status_time = now

        time.sleep(0.08)

    controller.send("STOP", force=True)
    print(f"Timeout before reaching {target_subject}.")
    return False


def make_single_config(args: argparse.Namespace) -> CarConfig:
    if args.car_id not in CAR_IDS:
        raise ValueError(f"--car-id must be one of {CAR_IDS}")

    name, default_ip, default_subject = CAR_CONFIGS[args.car_id]
    ip = args.car_ip or default_ip
    subject_name = args.subject or default_subject

    if not ip and not args.skip_discovery:
        discovered = discover_car_ips(
            (args.car_id,),
            args.discovery_timeout,
            bind_ips=args.bind_ip,
            broadcast_ips=args.broadcast_ip,
        )
        ip = discovered.get(args.car_id, "")

    if not ip:
        raise RuntimeError(
            f"No IP for car{args.car_id}. Use --car-ip, or make sure UDP discovery can find it."
        )

    return CarConfig(
        marker_id=args.car_id,
        name=name,
        ip=ip,
        subject_name=subject_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move one Vicon-tracked car to two Vicon target subjects in sequence."
    )
    parser.add_argument("--car-id", type=int, default=1)
    parser.add_argument("--car-ip", default="")
    parser.add_argument("--subject", default="", help="Car Vicon subject name, default comes from CAR_CONFIGS.")
    parser.add_argument("--target", action="append", default=[], help="Target Vicon subject. Use twice for sequence.")
    parser.add_argument("--vicon-host", default=VICON_HOST)
    parser.add_argument("--wifi-ssid", default=DEFAULT_WIFI_SSID)
    parser.add_argument("--skip-wifi", action="store_true")
    parser.add_argument("--wifi-wait", type=float, default=DEFAULT_WIFI_WAIT_SEC)
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--discovery-timeout", type=float, default=5.0)
    parser.add_argument("--bind-ip", action="append", default=[])
    parser.add_argument("--broadcast-ip", action="append", default=[])
    parser.add_argument("--speed", type=int, default=115)
    parser.add_argument("--near-speed", type=int, default=80)
    parser.add_argument("--slow-radius-mm", type=float, default=350.0)
    parser.add_argument("--tolerance-mm", type=float, default=120.0)
    parser.add_argument("--max-time-sec", type=float, default=0.0, help="0 means no timeout.")
    parser.add_argument("--calibration-move-sec", type=float, default=0.45)
    parser.add_argument("--settle-sec", type=float, default=0.35)
    parser.add_argument("--min-displacement-mm", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = tuple(args.target) if args.target else ("left-up", "right-down")

    if not args.skip_wifi:
        connect_wifi(args.wifi_ssid, args.wifi_wait)

    config = make_single_config(args)
    controller = TcpCarController(config)
    tracker = ViconPoseTracker(args.vicon_host)

    try:
        if not controller.connect():
            raise RuntimeError(f"Could not connect to {config.name} at {config.ip}")
        controller.send("STOP", force=True)
        set_speed(controller, args.speed)

        tracker.connect()
        print(f"Waiting for car subject: {config.subject_name}")
        if wait_for_pose(tracker, config.subject_name, 5.0) is None:
            raise RuntimeError(f"No Vicon pose for car subject {config.subject_name}")
        for target in targets:
            print(f"Waiting for target subject: {target}")
            if wait_for_pose(tracker, target, 5.0) is None:
                raise RuntimeError(f"No Vicon pose for target subject {target}")

        calibrations = calibrate_commands(
            controller,
            tracker,
            config.subject_name,
            args.calibration_move_sec,
            args.settle_sec,
            args.min_displacement_mm,
        )

        for target in targets:
            reached = move_to_target(
                controller,
                tracker,
                calibrations,
                config.subject_name,
                target,
                args.speed,
                args.near_speed,
                args.slow_radius_mm,
                args.tolerance_mm,
                args.max_time_sec,
            )
            if not reached:
                break
            time.sleep(args.settle_sec)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
