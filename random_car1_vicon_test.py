import argparse
import math
import random
import time
from typing import Optional

from six_car_repel import (
    CAR_CONFIGS,
    DEFAULT_DISCOVERY_TIMEOUT_SEC,
    DEFAULT_WIFI_WAIT_SEC,
    MOTION_COMMANDS,
    CarConfig,
    CarObservation,
    TcpCarController,
    ViconTracker,
    connect_wifi,
    discover_car_ips,
    set_speed,
)


CAR_ID = 4
DEFAULT_WIFI_SSID = "YAYA"
DEFAULT_VICON_HOST = "192.168.33.103"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the selected car (1-50) and its kedayaN Vicon tracking by "
            "sending random planar motion commands."
        )
    )
    parser.add_argument("--car-id", type=int, choices=range(1, 51), default=CAR_ID,
                        help="Car number, 1-50; defaults to CAR_ID near the top of this file.")
    parser.add_argument("--car-ip", default="", help="Override the selected car's IP address.")
    parser.add_argument("--subject", default="", help="Override Vicon subject name.")
    parser.add_argument("--vicon-host", default=DEFAULT_VICON_HOST, help="Vicon server IP address.")
    parser.add_argument("--wifi-ssid", default=DEFAULT_WIFI_SSID)
    parser.add_argument("--skip-wifi", action="store_true")
    parser.add_argument("--wifi-wait", type=float, default=DEFAULT_WIFI_WAIT_SEC)
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--discovery-timeout", type=float, default=DEFAULT_DISCOVERY_TIMEOUT_SEC)
    parser.add_argument("--bind-ip", action="append", default=[])
    parser.add_argument("--broadcast-ip", action="append", default=[])
    parser.add_argument("--speed", type=int, default=120, help="ESP32 PWM speed, 0-255.")
    parser.add_argument("--duration", type=float, default=20.0, help="Test duration in seconds.")
    parser.add_argument("--move-sec", type=float, default=0.65, help="Duration of each random move.")
    parser.add_argument("--settle-sec", type=float, default=0.35, help="STOP interval between moves.")
    parser.add_argument("--status-interval", type=float, default=0.10, help="Vicon coordinate print interval.")
    parser.add_argument("--command-refresh-sec", type=float, default=0.12, help="TCP command resend interval while moving.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for repeatable moves.")
    return parser.parse_args(argv)


def make_car1_config(args: argparse.Namespace) -> CarConfig:
    car_id = args.car_id
    if not 1 <= car_id <= 50:
        raise ValueError("Car ID must be from 1..50.")
    name, default_ip, default_subject = CAR_CONFIGS.get(
        car_id, (f"car{car_id}_sta", "", f"kedaya{car_id}")
    )
    ip = args.car_ip or default_ip

    if not ip and not args.skip_discovery:
        discovered = discover_car_ips(
            (car_id,),
            args.discovery_timeout,
            bind_ips=args.bind_ip,
            broadcast_ips=args.broadcast_ip,
        )
        ip = discovered.get(car_id, "")

    if not ip:
        raise RuntimeError(
            f"No IP for car{car_id}. Check that car{car_id} is powered and connected to {args.wifi_ssid}, "
            f"or pass --car-ip with car{car_id}'s current IP address."
        )

    return CarConfig(
        marker_id=car_id,
        name=name,
        ip=ip,
        subject_name=args.subject or default_subject,
    )


def print_position(observation: CarObservation) -> None:
    x_mm, y_mm = observation.position
    print(
        f"Vicon kedaya{observation.marker_id}: x={x_mm:8.1f} mm y={y_mm:8.1f} mm "
        f"yaw={math.degrees(observation.yaw):+6.1f} deg"
    )


def print_move_result(
    command: str,
    start: Optional[CarObservation],
    end: Optional[CarObservation],
) -> None:
    if start is None or end is None:
        print(f"Move {command}: Vicon position unavailable for displacement check.")
        return

    dx = end.position[0] - start.position[0]
    dy = end.position[1] - start.position[1]
    distance = math.hypot(dx, dy)
    print(f"Move {command}: dx={dx:+7.1f} mm dy={dy:+7.1f} mm dist={distance:6.1f} mm")


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.move_sec <= 0 or args.settle_sec < 0:
        raise ValueError("--duration and --move-sec must be positive; --settle-sec cannot be negative.")
    if args.status_interval <= 0 or args.command_refresh_sec <= 0:
        raise ValueError("--status-interval and --command-refresh-sec must be positive.")

    if not args.skip_wifi:
        connect_wifi(args.wifi_ssid, args.wifi_wait)

    config = make_car1_config(args)
    controller = TcpCarController(config)
    tracker = ViconTracker(args.vicon_host, {config.marker_id: config})
    rng = random.Random(args.seed)

    try:
        if not controller.connect():
            raise RuntimeError(f"Could not connect to {config.name} at {config.ip}")

        controller.send("STOP", force=True)
        set_speed(controller, args.speed)
        tracker.connect()

        print(f"Waiting for Vicon subject car{config.marker_id}: {config.subject_name}")
        initial = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and initial is None:
            initial = tracker.get_observations().get(config.marker_id)
            time.sleep(0.02)
        if initial is None:
            raise RuntimeError(f"No Vicon position for subject {config.subject_name}; car will not move.")

        print_position(initial)
        print(
            f"Starting {args.duration:.1f}s random car{config.marker_id} test at PWM "
            f"{max(0, min(255, args.speed))}. Keep the area clear; press Ctrl+C to stop."
        )

        test_end = time.monotonic() + args.duration
        current_command: Optional[str] = None
        move_start: Optional[CarObservation] = None
        next_transition = time.monotonic()
        next_refresh = 0.0
        next_status = 0.0
        latest_observation: Optional[CarObservation] = initial

        while time.monotonic() < test_end:
            now = time.monotonic()
            observation = tracker.get_observations().get(config.marker_id)
            if observation is not None:
                latest_observation = observation
                if now >= next_status:
                    print_position(observation)
                    next_status = now + args.status_interval
            elif now >= next_status:
                print(f"Vicon {config.subject_name}: no fresh position")
                next_status = now + args.status_interval

            if current_command is None and now >= next_transition:
                current_command = rng.choice(MOTION_COMMANDS)
                move_start = latest_observation
                controller.send(current_command, force=True)
                next_refresh = now + args.command_refresh_sec
                next_transition = now + args.move_sec
                print(f"Command: {current_command}")
            elif current_command is not None and now >= next_transition:
                controller.send("STOP", force=True)
                print_move_result(current_command, move_start, latest_observation)
                current_command = None
                move_start = None
                next_transition = now + args.settle_sec
                print("Command: STOP")
            elif current_command is not None and now >= next_refresh:
                controller.send(current_command, force=True)
                next_refresh = now + args.command_refresh_sec

            time.sleep(0.02)

        controller.send("STOP", force=True)
        print(f"Test complete. car{config.marker_id} stopped.")
    except KeyboardInterrupt:
        print(f"Interrupted. Stopping car{config.marker_id}...")
    finally:
        controller.close()


if __name__ == "__main__":
    main()
