import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from six_car_repel import (
    CONTROL_LOOP_SEC,
    DEFAULT_CALIBRATION_MOVE_SEC,
    DEFAULT_CALIBRATION_SPEED,
    DEFAULT_DISCOVERY_TIMEOUT_SEC,
    DEFAULT_MIN_DISPLACEMENT_MM,
    DEFAULT_SETTLE_SEC,
    DEFAULT_SPEED,
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
    discover_car_ips,
    send_commands,
    set_speed,
    vec_add,
    vec_norm,
    vec_normalize,
    vec_scale,
    vec_sub,
)


HERDER_IDS = (2, 4, 5, 6)
EVADER_ID = 10
TRACKED_IDS = HERDER_IDS + (EVADER_ID,)

# The fourth point in the note was repeated as (13600, 5800). For a rectangle,
# this script uses the implied opposite corner: (13600, 850).
FIELD_CORNERS = (
    (-590.0, 2900.0),
    (5650.0, 2900.0),
    (-590.0, -2200.0),
    (5650.0, -2200.0),
)

DEFAULT_CAPTURE_RADIUS_MM = 950.0
DEFAULT_TARGET_DEADBAND_MM = 110.0
DEFAULT_RING_TARGET_WEIGHT = 1.25
DEFAULT_SURROUND_WEIGHT = 0.45
DEFAULT_HERDER_REPEL_RADIUS_MM = 700.0
DEFAULT_HERDER_REPEL_WEIGHT = 1.25
DEFAULT_EVADER_KEEP_OUT_RADIUS_MM = 430.0
DEFAULT_EVADER_KEEP_OUT_WEIGHT = 1.8
DEFAULT_BOUNDARY_MARGIN_MM = 550.0
DEFAULT_BOUNDARY_WEIGHT = 0.95
DEFAULT_EVADER_ACTIVE_RADIUS_MM = 1300.0
DEFAULT_EVADER_AVOID_WEIGHT = 1.0


@dataclass
class FieldBounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) * 0.5, (self.ymin + self.ymax) * 0.5)

    def contains(self, position: Tuple[float, float]) -> bool:
        x_mm, y_mm = position
        return self.xmin <= x_mm <= self.xmax and self.ymin <= y_mm <= self.ymax


@dataclass
class ControlResult:
    commands: Dict[int, str]
    targets: Dict[int, Tuple[float, float]]
    visible_herders: Tuple[int, ...]
    hull_ids: Tuple[int, ...]
    evader_inside_hull: bool
    captured: bool
    herder_distances: Dict[int, float]


def bounds_from_corners(corners: Sequence[Tuple[float, float]]) -> FieldBounds:
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return FieldBounds(min(xs), max(xs), min(ys), max(ys))


def parse_args() -> argparse.Namespace:
    default_field = bounds_from_corners(FIELD_CORNERS)

    parser = argparse.ArgumentParser(
        description=(
            "Use car1-car4 as convex-hull herders around car6. "
            "Only Vicon observations inside the virtual rectangular field are used."
        )
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
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--require-all-visible", action="store_true")
    parser.add_argument("--drive-evader", action="store_true")
    parser.add_argument("--stop-when-captured", action="store_true")

    parser.add_argument("--xmin", type=float, default=default_field.xmin)
    parser.add_argument("--xmax", type=float, default=default_field.xmax)
    parser.add_argument("--ymin", type=float, default=default_field.ymin)
    parser.add_argument("--ymax", type=float, default=default_field.ymax)
    parser.add_argument("--boundary-margin-mm", type=float, default=DEFAULT_BOUNDARY_MARGIN_MM)
    parser.add_argument("--boundary-weight", type=float, default=DEFAULT_BOUNDARY_WEIGHT)

    parser.add_argument("--capture-radius-mm", type=float, default=DEFAULT_CAPTURE_RADIUS_MM)
    parser.add_argument("--target-deadband-mm", type=float, default=DEFAULT_TARGET_DEADBAND_MM)
    parser.add_argument("--ring-target-weight", type=float, default=DEFAULT_RING_TARGET_WEIGHT)
    parser.add_argument("--surround-weight", type=float, default=DEFAULT_SURROUND_WEIGHT)
    parser.add_argument("--herder-repel-radius-mm", type=float, default=DEFAULT_HERDER_REPEL_RADIUS_MM)
    parser.add_argument("--herder-repel-weight", type=float, default=DEFAULT_HERDER_REPEL_WEIGHT)
    parser.add_argument("--evader-keep-out-radius-mm", type=float, default=DEFAULT_EVADER_KEEP_OUT_RADIUS_MM)
    parser.add_argument("--evader-keep-out-weight", type=float, default=DEFAULT_EVADER_KEEP_OUT_WEIGHT)
    parser.add_argument("--evader-active-radius-mm", type=float, default=DEFAULT_EVADER_ACTIVE_RADIUS_MM)
    parser.add_argument("--evader-avoid-weight", type=float, default=DEFAULT_EVADER_AVOID_WEIGHT)
    return parser.parse_args()


def parse_overrides(overrides: Sequence[str], label: str) -> Dict[int, str]:
    valid_ids = set(TRACKED_IDS)
    result: Dict[int, str] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid --{label} value '{override}', expected ID=VALUE")

        marker_id_text, value = override.split("=", 1)
        marker_id = int(marker_id_text)
        if marker_id not in valid_ids:
            raise ValueError(f"Invalid car id {marker_id}; expected one of {TRACKED_IDS}")
        result[marker_id] = value.strip()
    return result


def make_config_map(args: argparse.Namespace) -> Dict[int, CarConfig]:
    subject_overrides = parse_overrides(args.subject, "subject")
    car_ip_overrides = parse_overrides(args.car_ip, "car-ip")
    configs: Dict[int, CarConfig] = {}

    for marker_id in TRACKED_IDS:
        configs[marker_id] = CarConfig(
            marker_id=marker_id,
            name=f"car{marker_id}_sta",
            ip=car_ip_overrides.get(marker_id, ""),
            subject_name=subject_overrides.get(marker_id, f"kedaya{marker_id}"),
        )
    return configs


def control_ids(args: argparse.Namespace) -> Tuple[int, ...]:
    if args.drive_evader:
        return TRACKED_IDS
    return HERDER_IDS


def resolve_car_ips(
    config_by_id: Dict[int, CarConfig],
    ids_to_control: Tuple[int, ...],
    args: argparse.Namespace,
) -> None:
    if args.skip_discovery:
        return

    discovered = discover_car_ips(
        ids_to_control,
        args.discovery_timeout,
        bind_ips=args.bind_ip,
        broadcast_ips=args.broadcast_ip,
    )
    for marker_id in ids_to_control:
        config = config_by_id[marker_id]
        if not config.ip and marker_id in discovered:
            config.ip = discovered[marker_id]

    missing = [marker_id for marker_id in ids_to_control if not config_by_id[marker_id].ip]
    if missing:
        print(
            "No IP for controlled car ID "
            + ",".join(str(marker_id) for marker_id in missing)
            + "; those cars will stay disconnected."
        )


def field_from_args(args: argparse.Namespace) -> FieldBounds:
    return FieldBounds(
        xmin=min(args.xmin, args.xmax),
        xmax=max(args.xmin, args.xmax),
        ymin=min(args.ymin, args.ymax),
        ymax=max(args.ymin, args.ymax),
    )


def filter_inside_field(
    observations: Dict[int, CarObservation],
    field: FieldBounds,
) -> Dict[int, CarObservation]:
    return {
        marker_id: observation
        for marker_id, observation in observations.items()
        if marker_id in TRACKED_IDS and field.contains(observation.position)
    }


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


def angle_around(center: Tuple[float, float], position: Tuple[float, float]) -> float:
    return math.atan2(position[1] - center[1], position[0] - center[0])


def circular_mean(angles: Sequence[float]) -> float:
    angles = tuple(angles)
    if not angles:
        return 0.0
    sin_sum = sum(math.sin(angle) for angle in angles)
    cos_sum = sum(math.cos(angle) for angle in angles)
    return math.atan2(sin_sum, cos_sum)


def ring_targets(
    visible_herders: Sequence[int],
    observations: Dict[int, CarObservation],
    evader_position: Tuple[float, float],
    radius_mm: float,
) -> Dict[int, Tuple[float, float]]:
    if not visible_herders:
        return {}

    sorted_ids = sorted(
        visible_herders,
        key=lambda marker_id: angle_around(evader_position, observations[marker_id].position),
    )
    step = 2.0 * math.pi / len(sorted_ids)
    current_angles = [
        angle_around(evader_position, observations[marker_id].position)
        for marker_id in sorted_ids
    ]
    base_angle = circular_mean(
        current_angle - index * step
        for index, current_angle in enumerate(current_angles)
    )

    targets: Dict[int, Tuple[float, float]] = {}
    for index, marker_id in enumerate(sorted_ids):
        target_angle = base_angle + index * step
        targets[marker_id] = (
            evader_position[0] + radius_mm * math.cos(target_angle),
            evader_position[1] + radius_mm * math.sin(target_angle),
        )
    return targets


def angular_spacing_vector(
    marker_id: int,
    visible_herders: Sequence[int],
    observations: Dict[int, CarObservation],
    evader_position: Tuple[float, float],
    weight: float,
) -> Tuple[float, float]:
    if len(visible_herders) < 3 or marker_id not in visible_herders:
        return (0.0, 0.0)

    sorted_ids = sorted(
        visible_herders,
        key=lambda herder_id: angle_around(evader_position, observations[herder_id].position),
    )
    index = sorted_ids.index(marker_id)
    prev_id = sorted_ids[index - 1]
    next_id = sorted_ids[(index + 1) % len(sorted_ids)]

    current_angle = angle_around(evader_position, observations[marker_id].position)
    prev_angle = angle_around(evader_position, observations[prev_id].position)
    next_angle = angle_around(evader_position, observations[next_id].position)

    gap_prev = (current_angle - prev_angle) % (2.0 * math.pi)
    gap_next = (next_angle - current_angle) % (2.0 * math.pi)
    tangent = (-math.sin(current_angle), math.cos(current_angle))
    return vec_scale(tangent, weight * (gap_next - gap_prev))


def pair_repel_vector(
    marker_id: int,
    peer_ids: Tuple[int, ...],
    observations: Dict[int, CarObservation],
    active_radius_mm: float,
    weight_scale: float,
) -> Tuple[float, float]:
    if marker_id not in observations:
        return (0.0, 0.0)

    total = (0.0, 0.0)
    car_position = observations[marker_id].position
    for other_id in peer_ids:
        if other_id == marker_id or other_id not in observations:
            continue

        offset = vec_sub(car_position, observations[other_id].position)
        distance = vec_norm(offset)
        if distance < 1e-6 or distance > active_radius_mm:
            continue

        closeness = (active_radius_mm - distance) / active_radius_mm
        weight = weight_scale * (0.25 + closeness * closeness * 2.5)
        total = vec_add(total, vec_scale(vec_normalize(offset), weight))
    return total


def boundary_push_vector(
    position: Tuple[float, float],
    field: FieldBounds,
    margin_mm: float,
    weight: float,
) -> Tuple[float, float]:
    x_mm, y_mm = position
    total = (0.0, 0.0)
    distances = (
        (x_mm - field.xmin, (1.0, 0.0)),
        (field.xmax - x_mm, (-1.0, 0.0)),
        (y_mm - field.ymin, (0.0, 1.0)),
        (field.ymax - y_mm, (0.0, -1.0)),
    )
    for distance, direction in distances:
        if distance < margin_mm:
            closeness = max(0.0, margin_mm - distance) / margin_mm
            total = vec_add(total, vec_scale(direction, weight * (0.35 + closeness * closeness * 2.5)))
    return total


def evader_keep_out_vector(
    herder_position: Tuple[float, float],
    evader_position: Tuple[float, float],
    keep_out_radius_mm: float,
    weight: float,
) -> Tuple[float, float]:
    offset = vec_sub(herder_position, evader_position)
    distance = vec_norm(offset)
    if distance < 1e-6:
        return (weight, 0.0)
    if distance >= keep_out_radius_mm:
        return (0.0, 0.0)

    closeness = (keep_out_radius_mm - distance) / keep_out_radius_mm
    return vec_scale(vec_normalize(offset), weight * (0.4 + closeness * closeness * 2.4))


def cross(
    origin: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def convex_hull_indices(points: Sequence[Tuple[float, float]]) -> List[int]:
    if len(points) <= 1:
        return list(range(len(points)))

    ordered = sorted(range(len(points)), key=lambda index: (points[index][0], points[index][1], index))
    lower: List[int] = []
    for index in ordered:
        while len(lower) >= 2 and cross(points[lower[-2]], points[lower[-1]], points[index]) <= 0:
            lower.pop()
        lower.append(index)

    upper: List[int] = []
    for index in reversed(ordered):
        while len(upper) >= 2 and cross(points[upper[-2]], points[upper[-1]], points[index]) <= 0:
            upper.pop()
        upper.append(index)

    return lower[:-1] + upper[:-1]


def herder_hull(
    observations: Dict[int, CarObservation],
) -> Tuple[Tuple[int, ...], Tuple[Tuple[float, float], ...]]:
    visible_ids = [marker_id for marker_id in HERDER_IDS if marker_id in observations]
    points = [observations[marker_id].position for marker_id in visible_ids]
    hull_local_indices = convex_hull_indices(points)
    hull_ids = tuple(visible_ids[index] for index in hull_local_indices)
    hull_points = tuple(points[index] for index in hull_local_indices)
    return hull_ids, hull_points


def point_in_convex_polygon(
    point: Tuple[float, float],
    polygon: Sequence[Tuple[float, float]],
    eps: float = 1e-6,
) -> bool:
    if len(polygon) < 3:
        return False

    sign: Optional[bool] = None
    for index, origin in enumerate(polygon):
        edge_end = polygon[(index + 1) % len(polygon)]
        value = cross(origin, edge_end, point)
        if abs(value) <= eps:
            continue
        current_sign = value > 0
        if sign is None:
            sign = current_sign
        elif sign != current_sign:
            return False
    return True


def evader_avoid_vector(
    observations: Dict[int, CarObservation],
    field: FieldBounds,
    args: argparse.Namespace,
) -> Tuple[float, float]:
    evader = observations.get(EVADER_ID)
    if evader is None:
        return (0.0, 0.0)

    total = boundary_push_vector(
        evader.position,
        field,
        args.boundary_margin_mm,
        args.boundary_weight,
    )
    for herder_id in HERDER_IDS:
        herder = observations.get(herder_id)
        if herder is None:
            continue

        offset = vec_sub(evader.position, herder.position)
        distance = vec_norm(offset)
        if distance < 1e-6:
            total = vec_add(total, evader.forward)
            continue
        if distance > args.evader_active_radius_mm:
            continue

        closeness = (args.evader_active_radius_mm - distance) / args.evader_active_radius_mm
        weight = args.evader_avoid_weight * (0.35 + closeness * closeness * 2.5)
        total = vec_add(total, vec_scale(vec_normalize(offset), weight))
    return total


def compute_commands(
    observations: Dict[int, CarObservation],
    calibration_by_id: CalibrationMap,
    ids_to_control: Tuple[int, ...],
    field: FieldBounds,
    args: argparse.Namespace,
) -> ControlResult:
    commands = {marker_id: "STOP" for marker_id in ids_to_control}
    targets: Dict[int, Tuple[float, float]] = {}
    visible_herders = tuple(marker_id for marker_id in HERDER_IDS if marker_id in observations)
    evader = observations.get(EVADER_ID)
    herder_distances: Dict[int, float] = {}

    hull_ids, hull_points = herder_hull(observations)
    evader_inside_hull = bool(
        evader is not None and point_in_convex_polygon(evader.position, hull_points)
    )
    captured = (
        evader_inside_hull
        and len(visible_herders) == len(HERDER_IDS)
        and set(hull_ids) == set(HERDER_IDS)
    )

    if evader is None:
        return ControlResult(commands, targets, visible_herders, hull_ids, False, False, herder_distances)
    if args.require_all_visible and len(visible_herders) != len(HERDER_IDS):
        return ControlResult(commands, targets, visible_herders, hull_ids, evader_inside_hull, captured, herder_distances)
    if args.stop_when_captured and captured:
        return ControlResult(commands, targets, visible_herders, hull_ids, evader_inside_hull, captured, herder_distances)

    target_by_herder = ring_targets(
        visible_herders,
        observations,
        evader.position,
        args.capture_radius_mm,
    )

    for herder_id in HERDER_IDS:
        herder = observations.get(herder_id)
        if herder is None:
            continue

        evader_offset = vec_sub(herder.position, evader.position)
        herder_distances[herder_id] = vec_norm(evader_offset)
        desired = (0.0, 0.0)

        target = target_by_herder.get(herder_id)
        if target is not None:
            targets[herder_id] = target
            target_offset = vec_sub(target, herder.position)
            target_distance = vec_norm(target_offset)
            if target_distance > args.target_deadband_mm:
                distance_scale = min(1.6, target_distance / max(1.0, args.capture_radius_mm))
                desired = vec_add(
                    desired,
                    vec_scale(vec_normalize(target_offset), args.ring_target_weight * distance_scale),
                )

        desired = vec_add(
            desired,
            angular_spacing_vector(
                herder_id,
                visible_herders,
                observations,
                evader.position,
                args.surround_weight,
            ),
        )
        desired = vec_add(
            desired,
            pair_repel_vector(
                herder_id,
                HERDER_IDS,
                observations,
                args.herder_repel_radius_mm,
                args.herder_repel_weight,
            ),
        )
        desired = vec_add(
            desired,
            evader_keep_out_vector(
                herder.position,
                evader.position,
                args.evader_keep_out_radius_mm,
                args.evader_keep_out_weight,
            ),
        )
        desired = vec_add(
            desired,
            boundary_push_vector(
                herder.position,
                field,
                args.boundary_margin_mm,
                args.boundary_weight,
            ),
        )
        commands[herder_id] = choose_command(herder_id, desired, observations, calibration_by_id)

    if args.drive_evader and EVADER_ID in ids_to_control:
        commands[EVADER_ID] = choose_command(
            EVADER_ID,
            evader_avoid_vector(observations, field, args),
            observations,
            calibration_by_id,
        )

    return ControlResult(
        commands=commands,
        targets=targets,
        visible_herders=visible_herders,
        hull_ids=hull_ids,
        evader_inside_hull=evader_inside_hull,
        captured=captured,
        herder_distances=herder_distances,
    )


def speed_for_id(marker_id: int, args: argparse.Namespace) -> int:
    if marker_id == EVADER_ID:
        return args.evader_speed if args.evader_speed is not None else args.speed
    return args.herder_speed if args.herder_speed is not None else args.speed


def format_ids(ids: Sequence[int]) -> str:
    return ",".join(str(marker_id) for marker_id in ids) or "none"


def main() -> None:
    args = parse_args()
    field = field_from_args(args)

    if not args.skip_wifi:
        connect_wifi(args.wifi_ssid, args.wifi_wait)

    config_by_id = make_config_map(args)
    ids_to_control = control_ids(args)
    resolve_car_ips(config_by_id, ids_to_control, args)

    controllers = {
        marker_id: TcpCarController(config_by_id[marker_id])
        for marker_id in ids_to_control
        if config_by_id[marker_id].ip
    }

    for marker_id, controller in controllers.items():
        controller.connect()
        controller.send("STOP", force=True)
        set_speed(controller, speed_for_id(marker_id, args))

    tracker = ViconTracker(args.vicon_host, config_by_id)
    tracker.connect()

    if args.skip_calibration:
        calibration_by_id: CalibrationMap = {}
        print("Skipping calibration. Controlled cars without calibration will stay stopped.")
    else:
        calibration_by_id = calibrate_all_cars(controllers, tracker, args)

    for marker_id, controller in controllers.items():
        set_speed(controller, speed_for_id(marker_id, args))
        controller.send("STOP", force=True)

    print(
        "Virtual field: "
        f"x=[{field.xmin:.0f},{field.xmax:.0f}] "
        f"y=[{field.ymin:.0f},{field.ymax:.0f}]"
    )
    print(
        "Running four-herder convex capture. "
        f"Herders={format_ids(HERDER_IDS)} evader={EVADER_ID} "
        f"controlled={format_ids(ids_to_control)}. Press Ctrl+C to stop."
    )
    calibrated_text = format_ids(sorted(calibration_by_id))
    print(f"Calibrated controlled car IDs: {calibrated_text}")

    last_status_time = 0.0
    try:
        while True:
            raw_observations = tracker.get_observations()
            observations = filter_inside_field(raw_observations, field)
            result = compute_commands(
                observations,
                calibration_by_id,
                ids_to_control,
                field,
                args,
            )
            send_commands(result.commands, controllers)

            now = time.monotonic()
            if now - last_status_time >= STATUS_INTERVAL_SEC:
                missing_ids = [marker_id for marker_id in TRACKED_IDS if marker_id not in raw_observations]
                outside_ids = [
                    marker_id
                    for marker_id in TRACKED_IDS
                    if marker_id in raw_observations and marker_id not in observations
                ]
                uncalibrated_ids = [
                    marker_id
                    for marker_id in ids_to_control
                    if marker_id in observations and marker_id not in calibration_by_id
                ]
                command_text = " ".join(
                    f"ID{marker_id}:{result.commands.get(marker_id, 'STOP')}"
                    for marker_id in ids_to_control
                )
                distance_text = " ".join(
                    f"H{marker_id}:{distance:.0f}mm"
                    for marker_id, distance in sorted(result.herder_distances.items())
                ) or "none"
                target_text = " ".join(
                    f"H{marker_id}:({target[0]:.0f},{target[1]:.0f})"
                    for marker_id, target in sorted(result.targets.items())
                ) or "none"
                evader_text = "missing"
                if EVADER_ID in observations:
                    evader_pos = observations[EVADER_ID].position
                    evader_text = f"({evader_pos[0]:.0f},{evader_pos[1]:.0f})"
                connected_count = sum(1 for controller in controllers.values() if controller.sock is not None)
                print(
                    f"Visible herders={len(result.visible_herders)}/{len(HERDER_IDS)} "
                    f"evader={evader_text} "
                    f"hull={format_ids(result.hull_ids)} "
                    f"inside_hull={result.evader_inside_hull} "
                    f"captured={result.captured} "
                    f"missing={format_ids(missing_ids)} "
                    f"outside={format_ids(outside_ids)} "
                    f"uncalibrated={format_ids(uncalibrated_ids)} "
                    f"TCP={connected_count}/{len(ids_to_control)} | "
                    f"{command_text} | dist {distance_text} | target {target_text}"
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
