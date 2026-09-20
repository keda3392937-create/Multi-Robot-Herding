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
    MOTION_COMMANDS,
    STATUS_INTERVAL_SEC,
    VICON_HOST,
    CalibrationMap,
    CarConfig,
    CarObservation,
    TcpCarController,
    ViconTracker,
    best_command_for_vector,
    calibrate_command_for_all_cars,
    connect_wifi,
    discover_car_ips,
    send_commands,
    set_speed,
    stop_all,
    vec_add,
    vec_norm,
    vec_normalize,
    vec_scale,
    vec_sub,
    wait_for_observations,
)


DEFAULT_HERDERS = "1-6"
DEFAULT_EVADERS = "7-16"

FIELD_CORNERS = (
    (8000.0, 6600.0),
    (14000.0, 6600.0),
    (8000.0, 1300.0),
    (14000.0, 1300.0),
)

DEFAULT_CAPTURE_RADIUS_MM = 1700.0
DEFAULT_MAX_CAPTURE_RADIUS_MM = 2800.0
DEFAULT_CAPTURE_MARGIN_MM = 650.0
DEFAULT_TARGET_DEADBAND_MM = 120.0
DEFAULT_RING_TARGET_WEIGHT = 1.20
DEFAULT_SURROUND_WEIGHT = 0.42
DEFAULT_HERDER_HUNT_WEIGHT = 0.85
DEFAULT_HERDER_SENSE_RADIUS_MM = 2800.0
DEFAULT_HERDER_REPEL_RADIUS_MM = 720.0
DEFAULT_HERDER_REPEL_WEIGHT = 1.25
DEFAULT_EVADER_KEEP_OUT_RADIUS_MM = 430.0
DEFAULT_EVADER_KEEP_OUT_WEIGHT = 1.5
DEFAULT_BOUNDARY_MARGIN_MM = 550.0
DEFAULT_BOUNDARY_WEIGHT = 0.95
DEFAULT_EVADER_ACTIVE_RADIUS_MM = 1300.0
DEFAULT_EVADER_AVOID_WEIGHT = 1.0
DEFAULT_EVADER_REPEL_RADIUS_MM = 450.0
DEFAULT_EVADER_REPEL_WEIGHT = 0.35
DEFAULT_EVADER_AGGREGATION_WEIGHT = 0.45
DEFAULT_EVADER_VELOCITY_DECAY = 0.58
DEFAULT_MIN_CALIBRATED_COMMANDS = 3


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
    visible_evaders: Tuple[int, ...]
    hull_ids: Tuple[int, ...]
    captured_evaders: Tuple[int, ...]
    outside_evaders: Tuple[int, ...]
    target_evader_by_herder: Dict[int, Optional[int]]
    formation_center: Tuple[float, float]
    formation_radius: float


class EvaderMotionModel:
    def __init__(self):
        self.velocity_by_id: Dict[int, Tuple[float, float]] = {}

    def reset(self, marker_id: int) -> None:
        self.velocity_by_id.pop(marker_id, None)

    def desired_vector(
        self,
        evader_id: int,
        observations: Dict[int, CarObservation],
        herder_ids: Tuple[int, ...],
        evader_ids: Tuple[int, ...],
        field: FieldBounds,
        args: argparse.Namespace,
    ) -> Tuple[float, float]:
        acceleration = evader_avoid_vector(
            evader_id,
            observations,
            herder_ids,
            evader_ids,
            field,
            args,
        )
        previous = self.velocity_by_id.get(evader_id, (0.0, 0.0))
        velocity = vec_add(
            vec_scale(previous, args.evader_velocity_decay),
            acceleration,
        )

        if vec_norm(velocity) < MIN_DRIVE_VECTOR_NORM:
            self.velocity_by_id[evader_id] = (0.0, 0.0)
            return (0.0, 0.0)

        velocity = vec_normalize(velocity)
        self.velocity_by_id[evader_id] = velocity
        return velocity


def bounds_from_corners(corners: Sequence[Tuple[float, float]]) -> FieldBounds:
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return FieldBounds(min(xs), max(xs), min(ys), max(ys))


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
            ids = range(start_id, end_id + step, step)
        else:
            ids = (int(token),)

        for marker_id in ids:
            if marker_id <= 0:
                raise ValueError("Car IDs must be positive integers.")
            if marker_id not in seen:
                result.append(marker_id)
                seen.add(marker_id)

    if not result:
        raise ValueError("ID list cannot be empty.")
    return tuple(result)


def parse_args() -> argparse.Namespace:
    default_field = bounds_from_corners(FIELD_CORNERS)

    parser = argparse.ArgumentParser(
        description=(
            "Use configurable herder cars to form a convex hull around multiple evaders. "
            "Default setup is six herders and ten evaders."
        )
    )
    parser.add_argument("--herders", default=DEFAULT_HERDERS, help="Herder IDs, e.g. 1-6 or 1,2,4,8.")
    parser.add_argument("--evaders", default=DEFAULT_EVADERS, help="Evader IDs, e.g. 7-16 or 10,12,13.")
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
    parser.add_argument("--min-calibrated-commands", type=int, default=DEFAULT_MIN_CALIBRATED_COMMANDS)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--require-all-visible", action="store_true")
    parser.add_argument("--drive-evaders", action="store_true")
    parser.add_argument("--keep-driving-captured-evaders", action="store_true")
    parser.add_argument("--stop-when-all-captured", action="store_true")

    parser.add_argument("--xmin", type=float, default=default_field.xmin)
    parser.add_argument("--xmax", type=float, default=default_field.xmax)
    parser.add_argument("--ymin", type=float, default=default_field.ymin)
    parser.add_argument("--ymax", type=float, default=default_field.ymax)
    parser.add_argument("--boundary-margin-mm", type=float, default=DEFAULT_BOUNDARY_MARGIN_MM)
    parser.add_argument("--boundary-weight", type=float, default=DEFAULT_BOUNDARY_WEIGHT)

    parser.add_argument("--capture-radius-mm", type=float, default=DEFAULT_CAPTURE_RADIUS_MM)
    parser.add_argument("--max-capture-radius-mm", type=float, default=DEFAULT_MAX_CAPTURE_RADIUS_MM)
    parser.add_argument("--capture-margin-mm", type=float, default=DEFAULT_CAPTURE_MARGIN_MM)
    parser.add_argument("--target-deadband-mm", type=float, default=DEFAULT_TARGET_DEADBAND_MM)
    parser.add_argument("--ring-target-weight", type=float, default=DEFAULT_RING_TARGET_WEIGHT)
    parser.add_argument("--surround-weight", type=float, default=DEFAULT_SURROUND_WEIGHT)
    parser.add_argument("--herder-hunt-weight", type=float, default=DEFAULT_HERDER_HUNT_WEIGHT)
    parser.add_argument("--herder-sense-radius-mm", type=float, default=DEFAULT_HERDER_SENSE_RADIUS_MM)
    parser.add_argument("--herder-repel-radius-mm", type=float, default=DEFAULT_HERDER_REPEL_RADIUS_MM)
    parser.add_argument("--herder-repel-weight", type=float, default=DEFAULT_HERDER_REPEL_WEIGHT)
    parser.add_argument("--evader-keep-out-radius-mm", type=float, default=DEFAULT_EVADER_KEEP_OUT_RADIUS_MM)
    parser.add_argument("--evader-keep-out-weight", type=float, default=DEFAULT_EVADER_KEEP_OUT_WEIGHT)
    parser.add_argument("--evader-active-radius-mm", type=float, default=DEFAULT_EVADER_ACTIVE_RADIUS_MM)
    parser.add_argument("--evader-avoid-weight", type=float, default=DEFAULT_EVADER_AVOID_WEIGHT)
    parser.add_argument("--evader-repel-radius-mm", type=float, default=DEFAULT_EVADER_REPEL_RADIUS_MM)
    parser.add_argument("--evader-repel-weight", type=float, default=DEFAULT_EVADER_REPEL_WEIGHT)
    parser.add_argument("--evader-aggregation-weight", type=float, default=DEFAULT_EVADER_AGGREGATION_WEIGHT)
    parser.add_argument("--evader-velocity-decay", type=float, default=DEFAULT_EVADER_VELOCITY_DECAY)
    return parser.parse_args()


def parsed_id_sets(args: argparse.Namespace) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    herder_ids = parse_id_spec(args.herders)
    evader_ids = parse_id_spec(args.evaders)
    overlap = sorted(set(herder_ids) & set(evader_ids))
    if overlap:
        raise ValueError("The same car cannot be both herder and evader: " + format_ids(overlap))

    tracked_ids = tuple(dict.fromkeys(herder_ids + evader_ids))
    return herder_ids, evader_ids, tracked_ids


def parse_overrides(overrides: Sequence[str], label: str, valid_ids: Sequence[int]) -> Dict[int, str]:
    valid_set = set(valid_ids)
    result: Dict[int, str] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid --{label} value '{override}', expected ID=VALUE")

        marker_id_text, value = override.split("=", 1)
        marker_id = int(marker_id_text)
        if marker_id not in valid_set:
            raise ValueError(f"Invalid car id {marker_id}; expected one of {tuple(valid_ids)}")
        result[marker_id] = value.strip()
    return result


def make_config_map(args: argparse.Namespace, tracked_ids: Tuple[int, ...]) -> Dict[int, CarConfig]:
    subject_overrides = parse_overrides(args.subject, "subject", tracked_ids)
    car_ip_overrides = parse_overrides(args.car_ip, "car-ip", tracked_ids)
    configs: Dict[int, CarConfig] = {}

    for marker_id in tracked_ids:
        configs[marker_id] = CarConfig(
            marker_id=marker_id,
            name=f"car{marker_id}_sta",
            ip=car_ip_overrides.get(marker_id, ""),
            subject_name=subject_overrides.get(marker_id, f"kedaya{marker_id}"),
        )
    return configs


def control_ids(
    herder_ids: Tuple[int, ...],
    evader_ids: Tuple[int, ...],
    args: argparse.Namespace,
) -> Tuple[int, ...]:
    if args.drive_evaders:
        return tuple(dict.fromkeys(herder_ids + evader_ids))
    return herder_ids


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
            + format_ids(missing)
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
    tracked_ids: Tuple[int, ...],
) -> Dict[int, CarObservation]:
    return {
        marker_id: observation
        for marker_id, observation in observations.items()
        if marker_id in tracked_ids and field.contains(observation.position)
    }


def calibrate_controlled_cars(
    marker_ids: Tuple[int, ...],
    controllers: Dict[int, TcpCarController],
    tracker: ViconTracker,
    args: argparse.Namespace,
) -> CalibrationMap:
    candidate_ids = tuple(marker_id for marker_id in marker_ids if marker_id in controllers)
    calibration_by_id: CalibrationMap = {marker_id: {} for marker_id in candidate_ids}

    print("Starting simultaneous motion calibration. Keep the area clear.")
    for controller in controllers.values():
        set_speed(controller, args.calibration_speed)
        controller.send("STOP", force=True)

    visible_before_calibration = wait_for_observations(tracker, candidate_ids, 5.0)
    visible_ids = tuple(marker_id for marker_id in candidate_ids if marker_id in visible_before_calibration)
    missing_ids = [marker_id for marker_id in candidate_ids if marker_id not in visible_before_calibration]
    if missing_ids:
        print("Skipping calibration for no-Vicon car ID " + format_ids(missing_ids))

    for command in MOTION_COMMANDS:
        print(f"Calibrating command {command} on all visible controlled cars...")
        command_calibrations = calibrate_command_for_all_cars(
            command,
            visible_ids,
            controllers,
            tracker,
            args,
        )
        for marker_id, calibration in command_calibrations.items():
            calibration_by_id.setdefault(marker_id, {})[command] = calibration

    min_command_count = max(1, min(len(MOTION_COMMANDS), args.min_calibrated_commands))
    stopped_ids = [
        marker_id
        for marker_id, calibrations in calibration_by_id.items()
        if len(calibrations) < min_command_count
    ]
    partial_ids = [
        marker_id
        for marker_id, calibrations in calibration_by_id.items()
        if min_command_count <= len(calibrations) < len(MOTION_COMMANDS)
    ]
    for marker_id in partial_ids:
        count = len(calibration_by_id[marker_id])
        print(f"car{marker_id} has partial calibration ({count}/{len(MOTION_COMMANDS)}) and will use available commands.")
    for marker_id in stopped_ids:
        count = len(calibration_by_id[marker_id])
        print(f"car{marker_id} has too little calibration ({count}/{len(MOTION_COMMANDS)}) and will stay stopped.")
        calibration_by_id.pop(marker_id, None)

    stop_all(controllers)
    return calibration_by_id


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


def mean_position(positions: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    if not positions:
        return (0.0, 0.0)
    return (
        sum(position[0] for position in positions) / len(positions),
        sum(position[1] for position in positions) / len(positions),
    )


def clamp(value: float, low: float, high: float) -> float:
    if low > high:
        return (low + high) * 0.5
    return max(low, min(high, value))


def angle_around(center: Tuple[float, float], position: Tuple[float, float]) -> float:
    return math.atan2(position[1] - center[1], position[0] - center[0])


def circular_mean(angles: Sequence[float]) -> float:
    angles = tuple(angles)
    if not angles:
        return 0.0
    sin_sum = sum(math.sin(angle) for angle in angles)
    cos_sum = sum(math.cos(angle) for angle in angles)
    return math.atan2(sin_sum, cos_sum)


def formation_center_and_radius(
    observations: Dict[int, CarObservation],
    evader_ids: Tuple[int, ...],
    outside_evaders: Tuple[int, ...],
    field: FieldBounds,
    args: argparse.Namespace,
) -> Tuple[Tuple[float, float], float]:
    reference_ids = outside_evaders or tuple(marker_id for marker_id in evader_ids if marker_id in observations)
    positions = [observations[marker_id].position for marker_id in reference_ids]
    if not positions:
        return field.center, args.capture_radius_mm

    center = mean_position(positions)
    spread = max(vec_norm(vec_sub(position, center)) for position in positions)
    radius = max(args.capture_radius_mm, spread + args.capture_margin_mm)
    radius = min(radius, args.max_capture_radius_mm)
    return center, radius


def ring_targets(
    visible_herders: Sequence[int],
    observations: Dict[int, CarObservation],
    center: Tuple[float, float],
    radius_mm: float,
) -> Dict[int, Tuple[float, float]]:
    if not visible_herders:
        return {}

    sorted_ids = sorted(
        visible_herders,
        key=lambda marker_id: angle_around(center, observations[marker_id].position),
    )
    step = 2.0 * math.pi / len(sorted_ids)
    current_angles = [
        angle_around(center, observations[marker_id].position)
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
            center[0] + radius_mm * math.cos(target_angle),
            center[1] + radius_mm * math.sin(target_angle),
        )
    return targets


def angular_spacing_vector(
    marker_id: int,
    visible_herders: Sequence[int],
    observations: Dict[int, CarObservation],
    center: Tuple[float, float],
    weight: float,
) -> Tuple[float, float]:
    if len(visible_herders) < 3 or marker_id not in visible_herders:
        return (0.0, 0.0)

    sorted_ids = sorted(
        visible_herders,
        key=lambda herder_id: angle_around(center, observations[herder_id].position),
    )
    index = sorted_ids.index(marker_id)
    prev_id = sorted_ids[index - 1]
    next_id = sorted_ids[(index + 1) % len(sorted_ids)]

    current_angle = angle_around(center, observations[marker_id].position)
    prev_angle = angle_around(center, observations[prev_id].position)
    next_angle = angle_around(center, observations[next_id].position)
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
    evader_ids: Tuple[int, ...],
    observations: Dict[int, CarObservation],
    keep_out_radius_mm: float,
    weight_scale: float,
) -> Tuple[float, float]:
    total = (0.0, 0.0)
    for evader_id in evader_ids:
        evader = observations.get(evader_id)
        if evader is None:
            continue

        offset = vec_sub(herder_position, evader.position)
        distance = vec_norm(offset)
        if distance < 1e-6:
            total = vec_add(total, (weight_scale, 0.0))
            continue
        if distance > keep_out_radius_mm:
            continue

        closeness = (keep_out_radius_mm - distance) / keep_out_radius_mm
        total = vec_add(
            total,
            vec_scale(vec_normalize(offset), weight_scale * (0.4 + closeness * closeness * 2.4)),
        )
    return total


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
    herder_ids: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], Tuple[Tuple[float, float], ...]]:
    visible_ids = [marker_id for marker_id in herder_ids if marker_id in observations]
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


def classify_evaders(
    observations: Dict[int, CarObservation],
    evader_ids: Tuple[int, ...],
    hull_points: Sequence[Tuple[float, float]],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    captured: List[int] = []
    outside: List[int] = []
    for evader_id in evader_ids:
        evader = observations.get(evader_id)
        if evader is None:
            continue
        if point_in_convex_polygon(evader.position, hull_points):
            captured.append(evader_id)
        else:
            outside.append(evader_id)
    return tuple(captured), tuple(outside)


def choose_herder_target_evader(
    herder_id: int,
    outside_evaders: Tuple[int, ...],
    observations: Dict[int, CarObservation],
    center: Tuple[float, float],
    args: argparse.Namespace,
) -> Optional[int]:
    if herder_id not in observations or not outside_evaders:
        return None

    herder = observations[herder_id]
    nearby = [
        evader_id
        for evader_id in outside_evaders
        if vec_norm(vec_sub(observations[evader_id].position, herder.position)) <= args.herder_sense_radius_mm
    ]
    if nearby:
        return min(
            nearby,
            key=lambda evader_id: vec_norm(vec_sub(observations[evader_id].position, herder.position)),
        )

    return max(
        outside_evaders,
        key=lambda evader_id: vec_norm(vec_sub(observations[evader_id].position, center)),
    )


def evader_avoid_vector(
    evader_id: int,
    observations: Dict[int, CarObservation],
    herder_ids: Tuple[int, ...],
    evader_ids: Tuple[int, ...],
    field: FieldBounds,
    args: argparse.Namespace,
) -> Tuple[float, float]:
    evader = observations.get(evader_id)
    if evader is None:
        return (0.0, 0.0)

    total = boundary_push_vector(
        evader.position,
        field,
        args.boundary_margin_mm,
        args.boundary_weight,
    )

    neighbor_ids = [
        other_id
        for other_id in evader_ids
        if other_id != evader_id
        and other_id in observations
        and vec_norm(vec_sub(observations[other_id].position, evader.position)) <= args.evader_repel_radius_mm
    ]
    total = vec_add(
        total,
        pair_repel_vector(
            evader_id,
            evader_ids,
            observations,
            args.evader_repel_radius_mm,
            args.evader_repel_weight,
        ),
    )

    herder_detected = False
    for herder_id in herder_ids:
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

        herder_detected = True
        closeness = (args.evader_active_radius_mm - distance) / args.evader_active_radius_mm
        weight = args.evader_avoid_weight * (0.35 + closeness * closeness * 2.5)
        total = vec_add(total, vec_scale(vec_normalize(offset), weight))

    if herder_detected and neighbor_ids:
        neighbor_center = mean_position(
            [observations[neighbor_id].position for neighbor_id in neighbor_ids]
        )
        total = vec_add(
            total,
            vec_scale(
                vec_normalize(vec_sub(neighbor_center, evader.position)),
                args.evader_aggregation_weight,
            ),
        )
    return total


def compute_commands(
    observations: Dict[int, CarObservation],
    calibration_by_id: CalibrationMap,
    ids_to_control: Tuple[int, ...],
    herder_ids: Tuple[int, ...],
    evader_ids: Tuple[int, ...],
    field: FieldBounds,
    evader_motion: EvaderMotionModel,
    args: argparse.Namespace,
) -> ControlResult:
    commands = {marker_id: "STOP" for marker_id in ids_to_control}
    targets: Dict[int, Tuple[float, float]] = {}
    target_evader_by_herder: Dict[int, Optional[int]] = {
        marker_id: None for marker_id in herder_ids
    }
    visible_herders = tuple(marker_id for marker_id in herder_ids if marker_id in observations)
    visible_evaders = tuple(marker_id for marker_id in evader_ids if marker_id in observations)
    hull_ids, hull_points = herder_hull(observations, herder_ids)
    captured_evaders, outside_evaders = classify_evaders(observations, evader_ids, hull_points)
    center, formation_radius = formation_center_and_radius(
        observations,
        evader_ids,
        outside_evaders,
        field,
        args,
    )

    if args.require_all_visible and (
        len(visible_herders) != len(herder_ids) or len(visible_evaders) != len(evader_ids)
    ):
        return ControlResult(
            commands,
            targets,
            visible_herders,
            visible_evaders,
            hull_ids,
            captured_evaders,
            outside_evaders,
            target_evader_by_herder,
            center,
            formation_radius,
        )

    all_captured = len(captured_evaders) == len(evader_ids)
    if args.stop_when_all_captured and all_captured:
        return ControlResult(
            commands,
            targets,
            visible_herders,
            visible_evaders,
            hull_ids,
            captured_evaders,
            outside_evaders,
            target_evader_by_herder,
            center,
            formation_radius,
        )

    target_by_herder = ring_targets(visible_herders, observations, center, formation_radius)

    for herder_id in herder_ids:
        herder = observations.get(herder_id)
        if herder is None:
            continue

        desired = (0.0, 0.0)
        target = target_by_herder.get(herder_id)
        if target is not None:
            targets[herder_id] = target
            target_offset = vec_sub(target, herder.position)
            target_distance = vec_norm(target_offset)
            if target_distance > args.target_deadband_mm:
                distance_scale = min(1.7, target_distance / max(1.0, formation_radius))
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
                center,
                args.surround_weight,
            ),
        )
        desired = vec_add(
            desired,
            pair_repel_vector(
                herder_id,
                herder_ids,
                observations,
                args.herder_repel_radius_mm,
                args.herder_repel_weight,
            ),
        )
        desired = vec_add(
            desired,
            evader_keep_out_vector(
                herder.position,
                evader_ids,
                observations,
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

        target_evader_id = choose_herder_target_evader(
            herder_id,
            outside_evaders,
            observations,
            center,
            args,
        )
        target_evader_by_herder[herder_id] = target_evader_id
        if target_evader_id is not None:
            target_evader = observations[target_evader_id]
            hunt_vector = vec_normalize(vec_sub(target_evader.position, herder.position))
            desired = vec_add(desired, vec_scale(hunt_vector, args.herder_hunt_weight))

        commands[herder_id] = choose_command(herder_id, desired, observations, calibration_by_id)

    if args.drive_evaders:
        captured_set = set(captured_evaders)
        for evader_id in evader_ids:
            if evader_id not in ids_to_control:
                continue
            if evader_id in captured_set and not args.keep_driving_captured_evaders:
                evader_motion.reset(evader_id)
                commands[evader_id] = "STOP"
                continue

            commands[evader_id] = choose_command(
                evader_id,
                evader_motion.desired_vector(
                    evader_id,
                    observations,
                    herder_ids,
                    evader_ids,
                    field,
                    args,
                ),
                observations,
                calibration_by_id,
            )

    return ControlResult(
        commands=commands,
        targets=targets,
        visible_herders=visible_herders,
        visible_evaders=visible_evaders,
        hull_ids=hull_ids,
        captured_evaders=captured_evaders,
        outside_evaders=outside_evaders,
        target_evader_by_herder=target_evader_by_herder,
        formation_center=center,
        formation_radius=formation_radius,
    )


def speed_for_id(
    marker_id: int,
    herder_ids: Tuple[int, ...],
    args: argparse.Namespace,
) -> int:
    if marker_id in herder_ids:
        return args.herder_speed if args.herder_speed is not None else args.speed
    return args.evader_speed if args.evader_speed is not None else args.speed


def format_ids(ids: Sequence[int]) -> str:
    return ",".join(str(marker_id) for marker_id in ids) or "none"


def main() -> None:
    args = parse_args()
    herder_ids, evader_ids, tracked_ids = parsed_id_sets(args)
    field = field_from_args(args)

    if not args.skip_wifi:
        connect_wifi(args.wifi_ssid, args.wifi_wait)

    config_by_id = make_config_map(args, tracked_ids)
    ids_to_control = control_ids(herder_ids, evader_ids, args)
    evader_motion = EvaderMotionModel()
    resolve_car_ips(config_by_id, ids_to_control, args)

    controllers = {
        marker_id: TcpCarController(config_by_id[marker_id])
        for marker_id in ids_to_control
        if config_by_id[marker_id].ip
    }

    for marker_id, controller in controllers.items():
        controller.connect()
        controller.send("STOP", force=True)
        set_speed(controller, speed_for_id(marker_id, herder_ids, args))

    tracker = ViconTracker(args.vicon_host, config_by_id)
    tracker.connect()

    if args.skip_calibration:
        calibration_by_id: CalibrationMap = {}
        print("Skipping calibration. Controlled cars without calibration will stay stopped.")
    else:
        calibration_by_id = calibrate_controlled_cars(ids_to_control, controllers, tracker, args)

    for marker_id, controller in controllers.items():
        set_speed(controller, speed_for_id(marker_id, herder_ids, args))
        controller.send("STOP", force=True)

    print(
        "Virtual field: "
        f"x=[{field.xmin:.0f},{field.xmax:.0f}] "
        f"y=[{field.ymin:.0f},{field.ymax:.0f}]"
    )
    print(
        "Running configurable convex capture. "
        f"Herders={format_ids(herder_ids)} evaders={format_ids(evader_ids)} "
        f"controlled={format_ids(ids_to_control)}. Press Ctrl+C to stop."
    )
    print(f"Calibrated controlled car IDs: {format_ids(sorted(calibration_by_id))}")

    last_status_time = 0.0
    try:
        while True:
            raw_observations = tracker.get_observations()
            observations = filter_inside_field(raw_observations, field, tracked_ids)
            result = compute_commands(
                observations,
                calibration_by_id,
                ids_to_control,
                herder_ids,
                evader_ids,
                field,
                evader_motion,
                args,
            )
            send_commands(result.commands, controllers)

            now = time.monotonic()
            if now - last_status_time >= STATUS_INTERVAL_SEC:
                missing_ids = [marker_id for marker_id in tracked_ids if marker_id not in raw_observations]
                outside_field_ids = [
                    marker_id
                    for marker_id in tracked_ids
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
                chase_text = " ".join(
                    f"H{herder_id}->E{evader_id}"
                    for herder_id, evader_id in sorted(result.target_evader_by_herder.items())
                    if evader_id is not None
                ) or "none"
                target_text = " ".join(
                    f"H{herder_id}:({target[0]:.0f},{target[1]:.0f})"
                    for herder_id, target in sorted(result.targets.items())
                ) or "none"
                connected_count = sum(1 for controller in controllers.values() if controller.sock is not None)
                print(
                    f"Visible H={len(result.visible_herders)}/{len(herder_ids)} "
                    f"E={len(result.visible_evaders)}/{len(evader_ids)} "
                    f"hull={format_ids(result.hull_ids)} "
                    f"captured={len(result.captured_evaders)}/{len(evader_ids)}:{format_ids(result.captured_evaders)} "
                    f"outside_hull={format_ids(result.outside_evaders)} "
                    f"center=({result.formation_center[0]:.0f},{result.formation_center[1]:.0f}) "
                    f"radius={result.formation_radius:.0f}mm "
                    f"missing={format_ids(missing_ids)} "
                    f"outside_field={format_ids(outside_field_ids)} "
                    f"uncalibrated={format_ids(uncalibrated_ids)} "
                    f"TCP={connected_count}/{len(ids_to_control)} | "
                    f"{command_text} | chase {chase_text} | target {target_text}"
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
