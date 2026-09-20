import argparse
import time
from typing import Dict, Optional, Tuple

from six_car_repel import (
    ACTIVE_REPULSION_RADIUS_MM,
    CAR_IDS,
    CONTROL_LOOP_SEC,
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


HERDER_IDS = (1, 2, 3, 4)
EVADER_IDS = tuple(marker_id for marker_id in CAR_IDS if marker_id not in HERDER_IDS)

DEFAULT_HERDER_CHASE_WEIGHT = 1.0
DEFAULT_HERDER_REPEL_RADIUS_MM = 700.0
DEFAULT_HERDER_REPEL_WEIGHT = 1.25
DEFAULT_EVADER_HERDER_ACTIVE_RADIUS_MM = 1300.0
DEFAULT_EVADER_HERDER_WEIGHT = 1.0
DEFAULT_EVADER_REPEL_WEIGHT = 0.35


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


def nearest_visible_evader(
    herder: CarObservation,
    observations: Dict[int, CarObservation],
) -> Tuple[Optional[int], float]:
    nearest_id: Optional[int] = None
    nearest_distance = float("inf")

    for marker_id in EVADER_IDS:
        evader = observations.get(marker_id)
        if evader is None:
            continue

        distance = vec_norm(vec_sub(evader.position, herder.position))
        if distance < nearest_distance:
            nearest_id = marker_id
            nearest_distance = distance

    return nearest_id, nearest_distance


def compute_pair_repel_vector(
    marker_id: int,
    peer_ids: Tuple[int, ...],
    observations: Dict[int, CarObservation],
    active_radius_mm: float,
    weight_scale: float,
) -> Tuple[float, float]:
    car = observations[marker_id]
    total = (0.0, 0.0)

    for other_id in peer_ids:
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


def compute_evader_avoid_herders_vector(
    evader: CarObservation,
    observations: Dict[int, CarObservation],
    args: argparse.Namespace,
) -> Tuple[Tuple[float, float], float]:
    total = (0.0, 0.0)
    nearest_distance = float("inf")

    for herder_id in HERDER_IDS:
        herder = observations.get(herder_id)
        if herder is None:
            continue

        offset = vec_sub(evader.position, herder.position)
        distance = vec_norm(offset)
        nearest_distance = min(nearest_distance, distance)
        if distance < 1e-6:
            total = vec_add(total, evader.forward)
            continue
        if not args.always_avoid_herders and distance > args.evader_herder_active_radius_mm:
            continue

        away = vec_normalize(offset)
        if args.always_avoid_herders:
            closeness = 1.0
        else:
            closeness = max(0.0, args.evader_herder_active_radius_mm - distance) / args.evader_herder_active_radius_mm
        weight = args.evader_herder_weight * (0.35 + closeness * closeness * 2.5)
        total = vec_add(total, vec_scale(away, weight))

    return total, nearest_distance


def compute_herder_commands(
    observations: Dict[int, CarObservation],
    calibration_by_id: CalibrationMap,
    args: argparse.Namespace,
) -> Tuple[Dict[int, str], Dict[int, Tuple[float, float]], Dict[int, Optional[int]], Dict[int, float]]:
    commands = {marker_id: "STOP" for marker_id in HERDER_IDS}
    targets: Dict[int, Tuple[float, float]] = {}
    chased_ids: Dict[int, Optional[int]] = {marker_id: None for marker_id in HERDER_IDS}
    chase_distances = {marker_id: float("inf") for marker_id in HERDER_IDS}

    for herder_id in HERDER_IDS:
        herder = observations.get(herder_id)
        if herder is None or herder_id not in calibration_by_id:
            continue

        desired_vector = compute_pair_repel_vector(
            herder_id,
            HERDER_IDS,
            observations,
            args.herder_repel_radius_mm,
            args.herder_repel_weight,
        )

        nearest_id, nearest_distance = nearest_visible_evader(herder, observations)
        chased_ids[herder_id] = nearest_id
        chase_distances[herder_id] = nearest_distance
        if nearest_id is not None:
            target_evader = observations[nearest_id]
            targets[herder_id] = target_evader.position
            if args.herder_stop_distance_mm <= 0 or nearest_distance > args.herder_stop_distance_mm:
                chase_direction = vec_normalize(vec_sub(target_evader.position, herder.position))
                desired_vector = vec_add(
                    desired_vector,
                    vec_scale(chase_direction, args.herder_chase_weight),
                )

        commands[herder_id] = choose_command(herder_id, desired_vector, observations, calibration_by_id)

    return commands, targets, chased_ids, chase_distances


def compute_evader_commands(
    observations: Dict[int, CarObservation],
    calibration_by_id: CalibrationMap,
    args: argparse.Namespace,
) -> Tuple[Dict[int, str], Dict[int, Tuple[float, float]], Dict[int, float]]:
    commands = {marker_id: "STOP" for marker_id in EVADER_IDS}
    targets: Dict[int, Tuple[float, float]] = {}
    nearest_herder_distances = {marker_id: float("inf") for marker_id in EVADER_IDS}

    for evader_id in EVADER_IDS:
        evader = observations.get(evader_id)
        if evader is None or evader_id not in calibration_by_id:
            continue

        herder_vector, nearest_distance = compute_evader_avoid_herders_vector(evader, observations, args)
        evader_repel_vector = compute_pair_repel_vector(
            evader_id,
            EVADER_IDS,
            observations,
            args.evader_repel_radius_mm,
            args.evader_repel_weight,
        )
        desired_vector = vec_add(herder_vector, evader_repel_vector)
        nearest_herder_distances[evader_id] = nearest_distance

        if vec_norm(desired_vector) >= MIN_DRIVE_VECTOR_NORM:
            direction = vec_normalize(desired_vector)
            commands[evader_id] = choose_command(evader_id, direction, observations, calibration_by_id)
            targets[evader_id] = vec_add(evader.position, vec_scale(direction, args.target_step_mm))

    return commands, targets, nearest_herder_distances


def compute_commands(
    observations: Dict[int, CarObservation],
    config_by_id: Dict[int, CarConfig],
    calibration_by_id: CalibrationMap,
    args: argparse.Namespace,
) -> Tuple[
    str,
    Dict[int, str],
    Dict[int, Tuple[float, float]],
    Dict[int, Optional[int]],
    Dict[int, float],
    Dict[int, float],
]:
    commands = {marker_id: "STOP" for marker_id in config_by_id}
    targets = {
        marker_id: observations[marker_id].position
        for marker_id in observations
        if marker_id in CAR_IDS
    }

    visible_ids = [marker_id for marker_id in CAR_IDS if marker_id in observations]
    missing_ids = [marker_id for marker_id in CAR_IDS if marker_id not in observations]
    visible_herders = [marker_id for marker_id in HERDER_IDS if marker_id in observations]
    visible_evaders = [marker_id for marker_id in EVADER_IDS if marker_id in observations]
    uncalibrated_ids = [
        marker_id
        for marker_id in visible_ids
        if marker_id in config_by_id and marker_id not in calibration_by_id
    ]

    if args.require_all_visible and missing_ids:
        summary = "Missing Vicon subject ID " + ",".join(str(marker_id) for marker_id in missing_ids)
        return summary, commands, targets, {}, {}, {}

    herder_commands, herder_targets, chased_ids, chase_distances = compute_herder_commands(
        observations,
        calibration_by_id,
        args,
    )
    evader_commands, evader_targets, nearest_herder_distances = compute_evader_commands(
        observations,
        calibration_by_id,
        args,
    )
    commands.update(herder_commands)
    commands.update(evader_commands)
    targets.update(herder_targets)
    targets.update(evader_targets)

    herder_observations = {
        marker_id: observation
        for marker_id, observation in observations.items()
        if marker_id in HERDER_IDS
    }
    evader_observations = {
        marker_id: observation
        for marker_id, observation in observations.items()
        if marker_id in EVADER_IDS
    }
    missing_text = "none" if not missing_ids else ",".join(str(marker_id) for marker_id in missing_ids)
    uncalibrated_text = "none" if not uncalibrated_ids else ",".join(str(marker_id) for marker_id in uncalibrated_ids)
    chase_text = " ".join(
        f"H{herder_id}->E{evader_id}:{chase_distances[herder_id]:.0f}mm"
        for herder_id, evader_id in chased_ids.items()
        if evader_id is not None and chase_distances.get(herder_id, float("inf")) != float("inf")
    ) or "none"
    summary = (
        f"Visible={len(visible_ids)}/{len(CAR_IDS)} "
        f"herders={len(visible_herders)}/{len(HERDER_IDS)} "
        f"evaders={len(visible_evaders)}/{len(EVADER_IDS)} "
        f"herder min pair={min_pair_distance(herder_observations):.0f}mm "
        f"evader min pair={min_pair_distance(evader_observations):.0f}mm "
        f"chase {chase_text} "
        f"missing={missing_text} "
        f"uncalibrated={uncalibrated_text}"
    )
    return summary, commands, targets, chased_ids, chase_distances, nearest_herder_distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use cars 1-4 as herders. Each herder chases its nearest evader while herders keep distance."
    )
    parser.add_argument("--vicon-host", default=VICON_HOST)
    parser.add_argument("--wifi-ssid", default=DEFAULT_WIFI_SSID)
    parser.add_argument("--skip-wifi", action="store_true")
    parser.add_argument("--wifi-wait", type=float, default=DEFAULT_WIFI_WAIT_SEC)
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--discovery-timeout", type=float, default=DEFAULT_DISCOVERY_TIMEOUT_SEC)
    parser.add_argument("--bind-ip", action="append", default=[])
    parser.add_argument("--broadcast-ip", action="append", default=[])
    parser.add_argument("--car-ip", action="append", default=[], metavar="ID=IP")
    parser.add_argument("--subject", action="append", default=[], metavar="ID=NAME")
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED)
    parser.add_argument("--herder-speed", type=int, default=None)
    parser.add_argument("--evader-speed", type=int, default=None)
    parser.add_argument("--calibration-speed", type=int, default=DEFAULT_CALIBRATION_SPEED)
    parser.add_argument("--calibration-move-sec", type=float, default=DEFAULT_CALIBRATION_MOVE_SEC)
    parser.add_argument("--settle-sec", type=float, default=DEFAULT_SETTLE_SEC)
    parser.add_argument("--min-displacement-mm", type=float, default=DEFAULT_MIN_DISPLACEMENT_MM)
    parser.add_argument("--herder-chase-weight", type=float, default=DEFAULT_HERDER_CHASE_WEIGHT)
    parser.add_argument("--herder-repel-radius-mm", type=float, default=DEFAULT_HERDER_REPEL_RADIUS_MM)
    parser.add_argument("--herder-repel-weight", type=float, default=DEFAULT_HERDER_REPEL_WEIGHT)
    parser.add_argument("--herder-stop-distance-mm", type=float, default=0.0)
    parser.add_argument("--evader-herder-active-radius-mm", type=float, default=DEFAULT_EVADER_HERDER_ACTIVE_RADIUS_MM)
    parser.add_argument("--evader-herder-weight", type=float, default=DEFAULT_EVADER_HERDER_WEIGHT)
    parser.add_argument("--always-avoid-herders", action="store_true")
    parser.add_argument("--evader-repel-radius-mm", type=float, default=ACTIVE_REPULSION_RADIUS_MM)
    parser.add_argument("--evader-repel-weight", type=float, default=DEFAULT_EVADER_REPEL_WEIGHT)
    parser.add_argument("--target-step-mm", type=float, default=DEFAULT_TARGET_STEP_MM)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--require-all-visible", action="store_true")
    return parser.parse_args()


def speed_for_id(marker_id: int, args: argparse.Namespace) -> int:
    if marker_id in HERDER_IDS:
        return args.herder_speed if args.herder_speed is not None else args.speed
    return args.evader_speed if args.evader_speed is not None else args.speed


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
        set_speed(controller, speed_for_id(marker_id, args))

    tracker = ViconTracker(args.vicon_host, config_by_id)
    tracker.connect()

    if args.skip_calibration:
        calibration_by_id: CalibrationMap = {}
        print("Skipping calibration. Cars without calibration will stay stopped.")
    else:
        calibration_by_id = calibrate_all_cars(controllers, tracker, args)

    for marker_id, controller in controllers.items():
        set_speed(controller, speed_for_id(marker_id, args))
        controller.send("STOP", force=True)

    calibrated_text = ",".join(str(marker_id) for marker_id in sorted(calibration_by_id)) or "none"
    herder_text = ",".join(f"ID{marker_id}" for marker_id in HERDER_IDS)
    evader_text = ",".join(f"ID{marker_id}" for marker_id in EVADER_IDS)
    print(f"Calibrated car IDs: {calibrated_text}")
    print(f"Running 16-car four-herder chase. Herders: {herder_text}. Evaders: {evader_text}. Press Ctrl+C to stop.")

    last_status_time = 0.0
    try:
        while True:
            observations = tracker.get_observations()
            summary, commands, targets, chased_ids, chase_distances, nearest_herder_distances = compute_commands(
                observations,
                config_by_id,
                calibration_by_id,
                args,
            )
            send_commands(commands, controllers)

            now = time.monotonic()
            if now - last_status_time >= STATUS_INTERVAL_SEC:
                command_text = " ".join(f"ID{marker_id}:{commands[marker_id]}" for marker_id in CAR_IDS)
                evader_dist_text = " ".join(
                    f"ID{marker_id}:{nearest_herder_distances[marker_id]:.0f}mm"
                    for marker_id in EVADER_IDS
                    if nearest_herder_distances.get(marker_id, float("inf")) != float("inf")
                )
                target_text = " ".join(
                    f"ID{marker_id}:({targets[marker_id][0]:.0f},{targets[marker_id][1]:.0f})"
                    for marker_id in CAR_IDS
                    if marker_id in targets and marker_id in calibration_by_id
                )
                connected_count = sum(1 for controller in controllers.values() if controller.sock is not None)
                print(
                    f"{summary} | TCP={connected_count}/{len(CAR_IDS)} | {command_text} "
                    f"| evader_near_herder {evader_dist_text} | target {target_text}"
                )
                last_status_time = now

            time.sleep(CONTROL_LOOP_SEC)
    except KeyboardInterrupt:
        print("Stopping cars...")
    finally:
        for controller in controllers.values():
            controller.close()


if __name__ == "__main__":
    main()
