import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


VICON_SDK_PATH = Path(r"D:\ViconDataStream\Win64\Python\vicon_dssdk")
if VICON_SDK_PATH.exists():
    sys.path.insert(0, str(VICON_SDK_PATH))

from six_car_repel import (
    CAR_CONFIGS,
    CAR_IDS,
    DEFAULT_WIFI_SSID,
    DEFAULT_WIFI_WAIT_SEC,
    VICON_HOST,
    CarConfig,
    TcpCarController,
    ViconTracker,
    connect_wifi,
    discover_car_ips,
)


MOTION_COMMANDS = ("F", "RF", "RB", "B", "LB", "LF")
TARGET_AXES = (("X+", (1.0, 0.0)), ("Y+", (0.0, 1.0)))


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


def wait_for_position(
    tracker: ViconTracker,
    car_id: int,
    timeout_sec: float,
    min_seen_at: float = 0.0,
) -> Optional[Tuple[float, float]]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        observations = tracker.get_observations()
        observation = observations.get(car_id)
        if observation is not None and observation.seen_at >= min_seen_at:
            return observation.position
        time.sleep(0.02)
    return None


def drive_for_duration(controller: TcpCarController, command: str, duration_sec: float) -> None:
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        controller.send(command, force=True)
        time.sleep(0.12)


def sample_motion_vector(
    controller: TcpCarController,
    tracker: ViconTracker,
    car_id: int,
    command: str,
    move_sec: float,
    settle_sec: float,
) -> Tuple[Tuple[float, float], float]:
    controller.send("STOP", force=True)
    time.sleep(settle_sec)

    start = wait_for_position(tracker, car_id, 2.0, min_seen_at=time.monotonic())
    if start is None:
        raise RuntimeError(f"No Vicon position for car{car_id} before command {command}")

    drive_for_duration(controller, command, move_sec)
    end = wait_for_position(tracker, car_id, 2.0, min_seen_at=time.monotonic())
    controller.send("STOP", force=True)
    time.sleep(settle_sec)

    if end is None:
        raise RuntimeError(f"No Vicon position for car{car_id} after command {command}")

    displacement = vec_sub(end, start)
    return displacement, vec_norm(displacement)


def calibrate_commands(
    controller: TcpCarController,
    tracker: ViconTracker,
    car_id: int,
    move_sec: float,
    settle_sec: float,
    min_displacement_mm: float,
) -> Dict[str, Tuple[float, float]]:
    command_directions: Dict[str, Tuple[float, float]] = {}
    print("Measuring command directions in Vicon world coordinates...")

    for command in MOTION_COMMANDS:
        displacement, distance = sample_motion_vector(
            controller,
            tracker,
            car_id,
            command,
            move_sec,
            settle_sec,
        )
        direction = vec_normalize(displacement)
        command_directions[command] = direction
        warning = "  LOW MOTION" if distance < min_displacement_mm else ""
        print(
            f"{command:>2}: dx={displacement[0]:8.1f}mm "
            f"dy={displacement[1]:8.1f}mm "
            f"dist={distance:7.1f}mm "
            f"unit=({direction[0]:+.3f}, {direction[1]:+.3f}){warning}"
        )

    return command_directions


def best_command_for_axis(
    command_directions: Dict[str, Tuple[float, float]],
    target_axis: Tuple[float, float],
) -> Tuple[str, float]:
    return max(
        ((command, dot(direction, target_axis)) for command, direction in command_directions.items()),
        key=lambda item: item[1],
    )


def run_axis_sequence(
    controller: TcpCarController,
    command_directions: Dict[str, Tuple[float, float]],
    axis_move_sec: float,
    settle_sec: float,
) -> None:
    for axis_name, target_axis in TARGET_AXES:
        command, score = best_command_for_axis(command_directions, target_axis)
        angle_error = math.degrees(math.acos(max(-1.0, min(1.0, score))))
        print(
            f"Moving along Vicon {axis_name}: command={command}, "
            f"alignment={score:+.3f}, angle_error={angle_error:.1f}deg"
        )
        drive_for_duration(controller, command, axis_move_sec)
        controller.send("STOP", force=True)
        time.sleep(settle_sec)


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
        description=(
            "Measure one car's real command directions with Vicon, then move it along "
            "Vicon X+ followed by Vicon Y+."
        )
    )
    parser.add_argument("--car-id", type=int, default=1, help="Car ID to test, default: 1.")
    parser.add_argument("--car-ip", default="", help="Override car IP address.")
    parser.add_argument("--subject", default="", help="Override Vicon subject name.")
    parser.add_argument("--vicon-host", default=VICON_HOST)
    parser.add_argument("--wifi-ssid", default=DEFAULT_WIFI_SSID)
    parser.add_argument("--skip-wifi", action="store_true")
    parser.add_argument("--wifi-wait", type=float, default=DEFAULT_WIFI_WAIT_SEC)
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--discovery-timeout", type=float, default=5.0)
    parser.add_argument("--bind-ip", action="append", default=[])
    parser.add_argument("--broadcast-ip", action="append", default=[])
    parser.add_argument("--speed", type=int, default=120, help="ESP32 PWM speed, 0-255.")
    parser.add_argument("--calibration-move-sec", type=float, default=0.45)
    parser.add_argument("--axis-move-sec", type=float, default=1.0)
    parser.add_argument("--settle-sec", type=float, default=0.35)
    parser.add_argument("--min-displacement-mm", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_wifi:
        connect_wifi(args.wifi_ssid, args.wifi_wait)

    config = make_single_config(args)
    config_by_id = {args.car_id: config}

    controller = TcpCarController(config)
    tracker = ViconTracker(args.vicon_host, config_by_id)

    try:
        if not controller.connect():
            raise RuntimeError(f"Could not connect to {config.name} at {config.ip}")
        controller.send("STOP", force=True)
        controller.send(f"SPD {max(0, min(255, args.speed))}", force=True)

        tracker.connect()
        print(f"Waiting for Vicon subject car{args.car_id}: {config.subject_name}")
        if wait_for_position(tracker, args.car_id, 5.0) is None:
            raise RuntimeError(f"No Vicon position for subject {config.subject_name}")

        command_directions = calibrate_commands(
            controller,
            tracker,
            args.car_id,
            args.calibration_move_sec,
            args.settle_sec,
            args.min_displacement_mm,
        )
        run_axis_sequence(controller, command_directions, args.axis_move_sec, args.settle_sec)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
