import argparse
import time
from typing import Dict, Optional, Tuple

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


HERDER_ID = 6
FLOCK_IDS = tuple(marker_id for marker_id in CAR_IDS if marker_id != HERDER_ID)
DEFAULT_HERDER_ACTIVE_RADIUS_MM = 1300.0
DEFAULT_HERDER_WEIGHT = 1.0
DEFAULT_CAR_REPEL_WEIGHT = 0.35
DEFAULT_HERDER_SPEED = DEFAULT_SPEED


def nearest_flock_car(
    herder: CarObservation,
    observations: Dict[int, CarObservation],
) -> Tuple[Optional[int], float]:
    nearest_id: Optional[int] = None
    nearest_distance = float("inf")

    for marker_id in FLOCK_IDS:
        car = observations.get(marker_id)
        if car is None:
            continue

        distance = vec_norm(vec_sub(car.position, herder.position))
        if distance < nearest_distance:
            nearest_id = marker_id
            nearest_distance = distance

    return nearest_id, nearest_distance


def compute_flock_repel_vector(
    marker_id: int,
    observations: Dict[int, CarObservation],
    active_radius_mm: float,
    weight_scale: float,
) -> Tuple[float, float]:
    car = observations[marker_id]
    total = (0.0, 0.0)

    for other_id in FLOCK_IDS:
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


def compute_herder_avoid_vector(
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


def choose_command(
    marker_id: int,
    desired_vector: Tuple[float, float],
    observations: Dict[int, CarObservation],
    calibration_by_id: CalibrationMap,
) -> str:
    if marker_id not in observations or marker_id not in calibration_by_id:
        return "STOP"
    if vec_norm(desired_vector) < MIN_DRIVE_VECTOR_NORM:
        return "STOP"

    command, _score = best_command_for_vector(
        calibration_by_id[marker_id],
        desired_vector,
        observations[marker_id].yaw,
    )
    return command


def compute_commands(
    observations: Dict[int, CarObservation],
    config_by_id: Dict[int, CarConfig],
    calibration_by_id: CalibrationMap,
    args: argparse.Namespace,
) -> Tuple[str, Dict[int, str], Dict[int, float], Dict[int, Tuple[float, float]]]:
    commands = {marker_id: "STOP" for marker_id in config_by_id}
    herder_distances = {marker_id: float("inf") for marker_id in FLOCK_IDS}
    targets = {
        marker_id: observations[marker_id].position
        for marker_id in observations
        if marker_id in CAR_IDS
    }

    visible_ids = [marker_id for marker_id in CAR_IDS if marker_id in observations]
    missing_ids = [marker_id for marker_id in CAR_IDS if marker_id not in observations]
    uncalibrated_ids = [
        marker_id
        for marker_id in visible_ids
        if marker_id in config_by_id and marker_id not in calibration_by_id
    ]

    herder = observations.get(HERDER_ID)
    if herder is None:
        return "Missing Vicon subject ID 6/kedaya6 herder", commands, herder_distances, targets

    if args.require_all_visible and missing_ids:
        summary = "Missing Vicon subject ID " + ",".join(str(marker_id) for marker_id in missing_ids)
        return summary, commands, herder_distances, targets

    nearest_id, nearest_distance = nearest_flock_car(herder, observations)
    if nearest_id is not None and HERDER_ID in calibration_by_id:
        target_car = observations[nearest_id]
        chase_vector = vec_sub(target_car.position, herder.position)
        if args.herder_stop_distance_mm > 0 and nearest_distance <= args.herder_stop_distance_mm:
            commands[HERDER_ID] = "STOP"
        else:
            commands[HERDER_ID] = choose_command(HERDER_ID, chase_vector, observations, calibration_by_id)
            targets[HERDER_ID] = target_car.position

    for marker_id in FLOCK_IDS:
        if marker_id not in observations or marker_id not in calibration_by_id:
            continue

        car = observations[marker_id]
        herder_vector, herder_distance = compute_herder_avoid_vector(car, herder, args)
        flock_repel_vector = compute_flock_repel_vector(
            marker_id,
            observations,
            args.flock_repel_radius_mm,
            args.flock_repel_weight,
        )
        escape_vector = vec_add(herder_vector, flock_repel_vector)
        herder_distances[marker_id] = herder_distance

        if vec_norm(escape_vector) >= MIN_DRIVE_VECTOR_NORM:
            escape_direction = vec_normalize(escape_vector)
            commands[marker_id] = choose_command(marker_id, escape_direction, observations, calibration_by_id)
            targets[marker_id] = vec_add(car.position, vec_scale(escape_direction, args.target_step_mm))

    flock_observations = {
        marker_id: observation
        for marker_id, observation in observations.items()
        if marker_id in FLOCK_IDS
    }
    missing_text = "none" if not missing_ids else ",".join(str(marker_id) for marker_id in missing_ids)
    uncalibrated_text = "none" if not uncalibrated_ids else ",".join(str(marker_id) for marker_id in uncalibrated_ids)
    nearest_text = "none" if nearest_id is None else f"ID{nearest_id}:{nearest_distance:.0f}mm"
    summary = (
        f"Visible={len(visible_ids)}/{len(CAR_IDS)} "
        f"herder=ID6({herder.position[0]:.0f},{herder.position[1]:.0f}) "
        f"nearest={nearest_text} "
        f"flock min pair={min_pair_distance(flock_observations):.0f}mm "
        f"missing={missing_text} "
        f"uncalibrated={uncalibrated_text}"
    )
    return summary, commands, herder_distances, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use car6/kedaya6 as a herder that chases the nearest non-herder car while the flock avoids it."
    )
    parser.add_argument("--vicon-host", default=VICON_HOST)
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
        help="Override a Vicon subject, e.g. --subject 6=kedaya6. Can be used multiple times.",
    )
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED)
    parser.add_argument("--herder-speed", type=int, default=DEFAULT_HERDER_SPEED)
    parser.add_argument("--calibration-speed", type=int, default=DEFAULT_CALIBRATION_SPEED)
    parser.add_argument("--calibration-move-sec", type=float, default=DEFAULT_CALIBRATION_MOVE_SEC)
    parser.add_argument("--settle-sec", type=float, default=DEFAULT_SETTLE_SEC)
    parser.add_argument("--min-displacement-mm", type=float, default=DEFAULT_MIN_DISPLACEMENT_MM)
    parser.add_argument("--herder-active-radius-mm", type=float, default=DEFAULT_HERDER_ACTIVE_RADIUS_MM)
    parser.add_argument("--herder-weight", type=float, default=DEFAULT_HERDER_WEIGHT)
    parser.add_argument("--always-avoid-herder", action="store_true")
    parser.add_argument("--herder-stop-distance-mm", type=float, default=0.0, help="0 means the herder never stops while a target is visible.")
    parser.add_argument("--flock-repel-radius-mm", type=float, default=ACTIVE_REPULSION_RADIUS_MM)
    parser.add_argument("--flock-repel-weight", type=float, default=DEFAULT_CAR_REPEL_WEIGHT)
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

    for marker_id, controller in controllers.items():
        controller.connect()
        controller.send("STOP", force=True)
        set_speed(controller, args.herder_speed if marker_id == HERDER_ID else args.speed)

    tracker = ViconTracker(args.vicon_host, config_by_id)
    tracker.connect()

    if args.skip_calibration:
        calibration_by_id: CalibrationMap = {}
        print("Skipping calibration. Cars without calibration will stay stopped.")
    else:
        calibration_by_id = calibrate_all_cars(controllers, tracker, args)

    for marker_id, controller in controllers.items():
        set_speed(controller, args.herder_speed if marker_id == HERDER_ID else args.speed)
        controller.send("STOP", force=True)

    calibrated_text = ",".join(str(marker_id) for marker_id in sorted(calibration_by_id)) or "none"
    print(f"Calibrated car IDs: {calibrated_text}")
    flock_text = ",".join(f"ID{marker_id}" for marker_id in FLOCK_IDS)
    print(f"Running car6 herder chase. ID6/kedaya6 chases nearest {flock_text}. Press Ctrl+C to stop.")

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
                    for marker_id in FLOCK_IDS
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
