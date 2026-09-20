import argparse
import math
import time
from typing import Dict, Tuple

from six_car_repel import (
    ACTIVE_REPULSION_RADIUS_MM,
    CAR_IDS,
    DEFAULT_CALIBRATION_MOVE_SEC,
    DEFAULT_CALIBRATION_SPEED,
    DEFAULT_DISCOVERY_TIMEOUT_SEC,
    DEFAULT_MIN_DISPLACEMENT_MM,
    DEFAULT_SETTLE_SEC,
    DEFAULT_SPEED,
    DEFAULT_TARGET_STEP_MM,
    DEFAULT_WIFI_SSID,
    DEFAULT_WIFI_WAIT_SEC,
    MIN_DRIVE_VECTOR_NORM,
    STATUS_INTERVAL_SEC,
    VICON_HOST,
    CalibrationMap,
    CarConfig,
    CarObservation,
    TcpCarController,
    ViconTracker,
    best_command_for_vector,
    calibrate_all_cars,
    connect_wifi,
    make_config_map,
    min_pair_distance,
    resolve_car_ips,
    send_commands,
    set_speed,
    vec_add,
    vec_norm,
    vec_normalize,
    vec_scale,
    vec_sub,
)


HERDER_ID = 0
DEFAULT_HERDER_SUBJECT = "herder1"
DEFAULT_HERDER_ACTIVE_RADIUS_MM = 1200.0
DEFAULT_HERDER_WEIGHT = 1.0
DEFAULT_CAR_REPEL_WEIGHT = 0.35


def compute_car_repel_vector(
    marker_id: int,
    observations: Dict[int, CarObservation],
    active_radius_mm: float,
    weight_scale: float,
) -> Tuple[float, float]:
    car = observations[marker_id]
    total = (0.0, 0.0)

    for other_id in CAR_IDS:
        if other_id == marker_id or other_id not in observations:
            continue

        other = observations[other_id]
        offset = vec_sub(car.position, other.position)
        distance = vec_norm(offset)
        if distance < 1e-6 or distance > active_radius_mm:
            continue

        closeness = max(0.0, active_radius_mm - distance) / active_radius_mm
        weight = weight_scale * (0.2 + closeness * closeness * 2.0)
        total = vec_add(total, vec_scale(vec_normalize(offset), weight))

    return total


def compute_herder_escape_vector(
    car: CarObservation,
    herder: CarObservation,
    args: argparse.Namespace,
) -> Tuple[Tuple[float, float], float]:
    offset = vec_sub(car.position, herder.position)
    distance = vec_norm(offset)
    if distance < 1e-6:
        return car.forward, distance

    if not args.always_avoid_herder and distance > args.herder_active_radius_mm:
        return (0.0, 0.0), distance

    away = vec_normalize(offset)
    if args.always_avoid_herder:
        closeness = 1.0
    else:
        closeness = max(0.0, args.herder_active_radius_mm - distance) / args.herder_active_radius_mm
    weight = args.herder_weight * (0.35 + closeness * closeness * 2.5)
    return vec_scale(away, weight), distance


def compute_commands(
    observations: Dict[int, CarObservation],
    config_by_id: Dict[int, CarConfig],
    calibration_by_id: CalibrationMap,
    args: argparse.Namespace,
) -> Tuple[str, Dict[int, str], Dict[int, float], Dict[int, Tuple[float, float]]]:
    commands = {marker_id: "STOP" for marker_id in config_by_id}
    herder_distances = {marker_id: float("inf") for marker_id in config_by_id}
    targets = {marker_id: observations[marker_id].position for marker_id in observations if marker_id in CAR_IDS}

    herder = observations.get(HERDER_ID)
    visible_car_ids = [marker_id for marker_id in CAR_IDS if marker_id in observations]
    missing_car_ids = [marker_id for marker_id in CAR_IDS if marker_id not in observations]
    uncalibrated_ids = [
        marker_id
        for marker_id in visible_car_ids
        if marker_id in config_by_id and marker_id not in calibration_by_id
    ]

    if herder is None:
        summary = f"Missing Vicon herder subject: {args.herder_subject}"
        return summary, commands, herder_distances, targets

    if args.require_all_visible and missing_car_ids:
        summary = "Missing Vicon car ID " + ",".join(str(marker_id) for marker_id in missing_car_ids)
        return summary, commands, herder_distances, targets

    for marker_id in visible_car_ids:
        if marker_id not in config_by_id or marker_id not in calibration_by_id:
            continue

        car = observations[marker_id]
        herder_vector, herder_distance = compute_herder_escape_vector(car, herder, args)
        car_repel_vector = compute_car_repel_vector(
            marker_id,
            observations,
            args.car_repel_radius_mm,
            args.car_repel_weight,
        )
        escape_vector = vec_add(herder_vector, car_repel_vector)
        herder_distances[marker_id] = herder_distance

        if vec_norm(escape_vector) < MIN_DRIVE_VECTOR_NORM:
            commands[marker_id] = "STOP"
            continue

        escape_direction = vec_normalize(escape_vector)
        targets[marker_id] = vec_add(car.position, vec_scale(escape_direction, args.target_step_mm))
        command, _score = best_command_for_vector(
            calibration_by_id[marker_id],
            escape_direction,
            car.yaw,
        )
        commands[marker_id] = command

    missing_text = "none" if not missing_car_ids else ",".join(str(marker_id) for marker_id in missing_car_ids)
    uncalibrated_text = "none" if not uncalibrated_ids else ",".join(str(marker_id) for marker_id in uncalibrated_ids)
    summary = (
        f"Visible={len(visible_car_ids)}/{len(CAR_IDS)} "
        f"herder=({herder.position[0]:.0f},{herder.position[1]:.0f}) "
        f"min pair={min_pair_distance({k: v for k, v in observations.items() if k in CAR_IDS}):.0f}mm "
        f"missing={missing_text} "
        f"uncalibrated={uncalibrated_text}"
    )
    return summary, commands, herder_distances, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate Vicon-tracked cars, then make them avoid a Vicon herder subject."
    )
    parser.add_argument("--vicon-host", default=VICON_HOST)
    parser.add_argument("--herder-subject", default=DEFAULT_HERDER_SUBJECT)
    parser.add_argument("--wifi-ssid", default=DEFAULT_WIFI_SSID)
    parser.add_argument("--skip-wifi", action="store_true")
    parser.add_argument("--wifi-wait", type=float, default=DEFAULT_WIFI_WAIT_SEC)
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--discovery-timeout", type=float, default=DEFAULT_DISCOVERY_TIMEOUT_SEC)
    parser.add_argument("--bind-ip", action="append", default=[])
    parser.add_argument("--broadcast-ip", action="append", default=[])
    parser.add_argument(
        "--car-ip",
        action="append",
        default=[],
        metavar="ID=IP",
        help="Override a discovered IP, e.g. --car-ip 1=192.168.1.201. Can be used multiple times.",
    )
    parser.add_argument(
        "--subject",
        action="append",
        default=[],
        metavar="ID=NAME",
        help="Override a car Vicon subject, e.g. --subject 1=kedaya1. Can be used multiple times.",
    )
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED)
    parser.add_argument("--calibration-speed", type=int, default=DEFAULT_CALIBRATION_SPEED)
    parser.add_argument("--calibration-move-sec", type=float, default=DEFAULT_CALIBRATION_MOVE_SEC)
    parser.add_argument("--settle-sec", type=float, default=DEFAULT_SETTLE_SEC)
    parser.add_argument("--min-displacement-mm", type=float, default=DEFAULT_MIN_DISPLACEMENT_MM)
    parser.add_argument("--herder-active-radius-mm", type=float, default=DEFAULT_HERDER_ACTIVE_RADIUS_MM)
    parser.add_argument("--herder-weight", type=float, default=DEFAULT_HERDER_WEIGHT)
    parser.add_argument("--always-avoid-herder", action="store_true")
    parser.add_argument("--car-repel-radius-mm", type=float, default=ACTIVE_REPULSION_RADIUS_MM)
    parser.add_argument("--car-repel-weight", type=float, default=DEFAULT_CAR_REPEL_WEIGHT)
    parser.add_argument("--target-step-mm", type=float, default=DEFAULT_TARGET_STEP_MM)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--require-all-visible", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_wifi:
        connect_wifi(args.wifi_ssid, args.wifi_wait)

    config_by_id = make_config_map(args)
    resolve_car_ips(config_by_id, args)
    controllers = {
        marker_id: TcpCarController(config)
        for marker_id, config in config_by_id.items()
        if config.ip
    }

    for controller in controllers.values():
        controller.connect()
        controller.send("STOP", force=True)
        set_speed(controller, args.speed)

    tracker_config_by_id = dict(config_by_id)
    tracker_config_by_id[HERDER_ID] = CarConfig(
        marker_id=HERDER_ID,
        name="herder",
        ip="",
        subject_name=args.herder_subject,
    )
    tracker = ViconTracker(args.vicon_host, tracker_config_by_id)
    tracker.connect()

    if args.skip_calibration:
        calibration_by_id: CalibrationMap = {}
        print("Skipping calibration. Cars without calibration will stay stopped.")
    else:
        calibration_by_id = calibrate_all_cars(controllers, tracker, args)

    calibrated_text = ",".join(str(marker_id) for marker_id in sorted(calibration_by_id)) or "none"
    print(f"Calibrated car IDs: {calibrated_text}")
    print(f"Running {len(CAR_IDS)}-car herder avoidance. Herder subject: {args.herder_subject}. Press Ctrl+C to stop.")

    last_status_time = 0.0
    try:
        while True:
            observations = tracker.get_observations()
            summary, commands, herder_distances, targets = compute_commands(
                observations,
                config_by_id,
                calibration_by_id,
                args,
            )
            send_commands(commands, controllers)

            now = time.monotonic()
            if now - last_status_time >= STATUS_INTERVAL_SEC:
                command_text = " ".join(f"ID{marker_id}:{commands[marker_id]}" for marker_id in CAR_IDS)
                distance_text = " ".join(
                    f"ID{marker_id}:{herder_distances[marker_id]:.0f}mm"
                    for marker_id in CAR_IDS
                    if herder_distances.get(marker_id, float("inf")) != float("inf")
                )
                target_text = " ".join(
                    f"ID{marker_id}:({targets[marker_id][0]:.0f},{targets[marker_id][1]:.0f})"
                    for marker_id in CAR_IDS
                    if marker_id in targets and marker_id in calibration_by_id
                )
                connected_count = sum(1 for controller in controllers.values() if controller.sock is not None)
                print(
                    f"{summary} | TCP={connected_count}/{len(CAR_IDS)} | {command_text} "
                    f"| herder_dist {distance_text} | target {target_text}"
                )
                last_status_time = now

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Stopping cars...")
    finally:
        for controller in controllers.values():
            controller.close()


if __name__ == "__main__":
    main()
