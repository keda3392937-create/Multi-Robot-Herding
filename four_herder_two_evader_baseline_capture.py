import argparse
import math
import random
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


DEFAULT_HERDERS = "1-4"
DEFAULT_EVADERS = "5-6"
DEFAULT_HERDER_SPEED = 170
DEFAULT_EVADER_SPEED = 230
DEFAULT_HERDER_MODEL_SPEED_LIMIT = 2.0
DEFAULT_EVADER_MODEL_SPEED_LIMIT = 3.0
DEFAULT_EVADER_INITIAL_SPEED = 1.6
DEFAULT_HERDER_VELOCITY_DECAY = 0.72
DEFAULT_EVADER_VELOCITY_DECAY = 1.0

FIELD_CORNERS = (
    (-500.0, 2900.0),
    (5600.0, 2900.0),
    (-500.0, -2100.0),
    (5600.0, -2100.0),
)

DEFAULT_CAPTURE_RADIUS_MM = 1050.0
DEFAULT_CAPTURE_MARGIN_MM = 650.0
DEFAULT_MAX_CAPTURE_RADIUS_MM = 2500.0
DEFAULT_TARGET_DEADBAND_MM = 120.0
DEFAULT_RING_TARGET_WEIGHT = 1.25
DEFAULT_SURROUND_WEIGHT = 0.55
DEFAULT_HERDER_REPEL_RADIUS_MM = 720.0
DEFAULT_HERDER_REPEL_WEIGHT = 1.35
DEFAULT_EVADER_KEEP_OUT_RADIUS_MM = 430.0
DEFAULT_EVADER_KEEP_OUT_WEIGHT = 1.4

DEFAULT_BOUNDARY_MARGIN_MM = 550.0
DEFAULT_BOUNDARY_WEIGHT = 1.15
DEFAULT_EVADER_NEIGHBOR_RADIUS_MM = 950.0
DEFAULT_EVADER_THREAT_RADIUS_MM = 1300.0
DEFAULT_EVADER_DISPERSION_WEIGHT = 1.25
DEFAULT_EVADER_ESCAPE_WEIGHT = 2.8
DEFAULT_EVADER_AGGREGATION_WEIGHT = 0.35
DEFAULT_MIN_CALIBRATED_COMMANDS = 4
DEFAULT_MAX_CALIBRATION_DISPLACEMENT_MM = 1800.0


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
    herder_targets: Dict[int, Tuple[float, float]]
    visible_herders: Tuple[int, ...]
    visible_evaders: Tuple[int, ...]
    hull_ids: Tuple[int, ...]
    inside_hull_evaders: Tuple[int, ...]
    formation_center: Tuple[float, float]
    formation_radius: float
    evader_accelerations: Dict[int, Tuple[float, float]]
    evader_velocities: Dict[int, Tuple[float, float]]


class MotionModel:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.velocity_by_id: Dict[int, Tuple[float, float]] = {}

    def random_vector(self, speed: float) -> Tuple[float, float]:
        angle = self.rng.uniform(0.0, 2.0 * math.pi)
        return (math.cos(angle) * speed, math.sin(angle) * speed)

    def reset(self, marker_id: int) -> None:
        self.velocity_by_id.pop(marker_id, None)

    def update(
        self,
        marker_id: int,
        acceleration: Tuple[float, float],
        dt: float,
        speed_limit: float,
        velocity_decay: float,
        initial_velocity: Tuple[float, float],
        keep_initial_motion: bool,
    ) -> Tuple[float, float]:
        previous = self.velocity_by_id.get(marker_id)
        if previous is None:
            previous = initial_velocity

        velocity = vec_add(
            vec_scale(previous, velocity_decay),
            vec_scale(acceleration, dt),
        )

        if vec_norm(velocity) < MIN_DRIVE_VECTOR_NORM:
            if keep_initial_motion:
                velocity = initial_velocity
            elif vec_norm(acceleration) >= MIN_DRIVE_VECTOR_NORM:
                velocity = vec_scale(
                    vec_normalize(acceleration),
                    min(speed_limit, max(MIN_DRIVE_VECTOR_NORM, vec_norm(acceleration))),
                )
            else:
                velocity = (0.0, 0.0)

        speed = vec_norm(velocity)
        if speed > speed_limit:
            velocity = vec_scale(vec_normalize(velocity), speed_limit)

        self.velocity_by_id[marker_id] = velocity
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
            "Clean four-herder/two-evader real-car capture script. "
            "Evaders use baseline-style second-order forces and do not stop inside the herder hull."
        )
    )
    parser.add_argument("--herders", default=DEFAULT_HERDERS, help="Exactly four herder IDs, e.g. 1-4.")
    parser.add_argument("--evaders", default=DEFAULT_EVADERS, help="Exactly two evader IDs, e.g. 5-6.")
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
    parser.add_argument("--herder-speed", type=int, default=DEFAULT_HERDER_SPEED)
    parser.add_argument("--evader-speed", type=int, default=DEFAULT_EVADER_SPEED)
    parser.add_argument("--calibration-speed", type=int, default=DEFAULT_CALIBRATION_SPEED)
    parser.add_argument("--calibration-move-sec", type=float, default=DEFAULT_CALIBRATION_MOVE_SEC)
    parser.add_argument("--settle-sec", type=float, default=DEFAULT_SETTLE_SEC)
    parser.add_argument("--min-displacement-mm", type=float, default=DEFAULT_MIN_DISPLACEMENT_MM)
    parser.add_argument("--min-calibrated-commands", type=int, default=DEFAULT_MIN_CALIBRATED_COMMANDS)
    parser.add_argument("--max-calibration-displacement-mm", type=float, default=DEFAULT_MAX_CALIBRATION_DISPLACEMENT_MM)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--require-all-visible", action="store_true")
    parser.add_argument("--passive-evaders", action="store_true", help="Track evaders but do not send commands to them.")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--xmin", type=float, default=default_field.xmin)
    parser.add_argument("--xmax", type=float, default=default_field.xmax)
    parser.add_argument("--ymin", type=float, default=default_field.ymin)
    parser.add_argument("--ymax", type=float, default=default_field.ymax)

    parser.add_argument("--capture-radius-mm", type=float, default=DEFAULT_CAPTURE_RADIUS_MM)
    parser.add_argument("--capture-margin-mm", type=float, default=DEFAULT_CAPTURE_MARGIN_MM)
    parser.add_argument("--max-capture-radius-mm", type=float, default=DEFAULT_MAX_CAPTURE_RADIUS_MM)
    parser.add_argument("--target-deadband-mm", type=float, default=DEFAULT_TARGET_DEADBAND_MM)
    parser.add_argument("--ring-target-weight", type=float, default=DEFAULT_RING_TARGET_WEIGHT)
    parser.add_argument("--surround-weight", type=float, default=DEFAULT_SURROUND_WEIGHT)
    parser.add_argument("--herder-repel-radius-mm", type=float, default=DEFAULT_HERDER_REPEL_RADIUS_MM)
    parser.add_argument("--herder-repel-weight", type=float, default=DEFAULT_HERDER_REPEL_WEIGHT)
    parser.add_argument("--evader-keep-out-radius-mm", type=float, default=DEFAULT_EVADER_KEEP_OUT_RADIUS_MM)
    parser.add_argument("--evader-keep-out-weight", type=float, default=DEFAULT_EVADER_KEEP_OUT_WEIGHT)

    parser.add_argument("--boundary-margin-mm", type=float, default=DEFAULT_BOUNDARY_MARGIN_MM)
    parser.add_argument("--boundary-weight", type=float, default=DEFAULT_BOUNDARY_WEIGHT)
    parser.add_argument("--evader-neighbor-radius-mm", type=float, default=DEFAULT_EVADER_NEIGHBOR_RADIUS_MM)
    parser.add_argument("--evader-threat-radius-mm", type=float, default=DEFAULT_EVADER_THREAT_RADIUS_MM)
    parser.add_argument("--evader-dispersion-weight", type=float, default=DEFAULT_EVADER_DISPERSION_WEIGHT)
    parser.add_argument("--evader-escape-weight", type=float, default=DEFAULT_EVADER_ESCAPE_WEIGHT)
    parser.add_argument("--evader-aggregation-weight", type=float, default=DEFAULT_EVADER_AGGREGATION_WEIGHT)
    parser.add_argument("--herder-model-speed-limit", type=float, default=DEFAULT_HERDER_MODEL_SPEED_LIMIT)
    parser.add_argument("--evader-model-speed-limit", type=float, default=DEFAULT_EVADER_MODEL_SPEED_LIMIT)
    parser.add_argument("--evader-initial-speed", type=float, default=DEFAULT_EVADER_INITIAL_SPEED)
    parser.add_argument("--herder-velocity-decay", type=float, default=DEFAULT_HERDER_VELOCITY_DECAY)
    parser.add_argument("--evader-velocity-decay", type=float, default=DEFAULT_EVADER_VELOCITY_DECAY)
    return parser.parse_args()


def parse_id_sets(args: argparse.Namespace) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    herder_ids = parse_id_spec(args.herders)
    evader_ids = parse_id_spec(args.evaders)
    if len(herder_ids) != 4:
        raise ValueError("--herders must contain exactly four car IDs.")
    if len(evader_ids) != 2:
        raise ValueError("--evaders must contain exactly two car IDs.")
    overlap = sorted(set(herder_ids) & set(evader_ids))
    if overlap:
        raise ValueError("The same car cannot be both herder and evader: " + format_ids(overlap))
    return herder_ids, evader_ids, tuple(dict.fromkeys(herder_ids + evader_ids))


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
    return {
        marker_id: CarConfig(
            marker_id=marker_id,
            name=f"car{marker_id}_sta",
            ip=car_ip_overrides.get(marker_id, ""),
            subject_name=subject_overrides.get(marker_id, f"kedaya{marker_id}"),
        )
        for marker_id in tracked_ids
    }


def field_from_args(args: argparse.Namespace) -> FieldBounds:
    return FieldBounds(
        xmin=min(args.xmin, args.xmax),
        xmax=max(args.xmin, args.xmax),
        ymin=min(args.ymin, args.ymax),
        ymax=max(args.ymin, args.ymax),
    )


def control_ids(herder_ids: Tuple[int, ...], evader_ids: Tuple[int, ...], args: argparse.Namespace) -> Tuple[int, ...]:
    if args.passive_evaders:
        return herder_ids
    return tuple(dict.fromkeys(herder_ids + evader_ids))


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
        if not config_by_id[marker_id].ip and marker_id in discovered:
            config_by_id[marker_id].ip = discovered[marker_id]

    missing = [marker_id for marker_id in ids_to_control if not config_by_id[marker_id].ip]
    if missing:
        print(
            "No IP for controlled car ID "
            + format_ids(missing)
            + "; those cars will stay disconnected."
        )


def filter_tracked_observations(
    observations: Dict[int, CarObservation],
    tracked_ids: Tuple[int, ...],
) -> Dict[int, CarObservation]:
    return {
        marker_id: observation
        for marker_id, observation in observations.items()
        if marker_id in tracked_ids
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
            if calibration.displacement_mm > args.max_calibration_displacement_mm:
                print(
                    f"car{marker_id} {command}: ignoring unrealistic calibration "
                    f"dist={calibration.displacement_mm:.1f}mm"
                )
                continue
            calibration_by_id.setdefault(marker_id, {})[command] = calibration

    min_command_count = max(1, min(len(MOTION_COMMANDS), args.min_calibrated_commands))
    incomplete_ids = [
        marker_id
        for marker_id, calibrations in calibration_by_id.items()
        if len(calibrations) < min_command_count
    ]
    for marker_id in incomplete_ids:
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


def angle_around(center: Tuple[float, float], position: Tuple[float, float]) -> float:
    return math.atan2(position[1] - center[1], position[0] - center[0])


def circular_mean(angles: Sequence[float]) -> float:
    angles = tuple(angles)
    if not angles:
        return 0.0
    return math.atan2(
        sum(math.sin(angle) for angle in angles),
        sum(math.cos(angle) for angle in angles),
    )


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


def formation_center_and_radius(
    observations: Dict[int, CarObservation],
    evader_ids: Tuple[int, ...],
    field: FieldBounds,
    args: argparse.Namespace,
) -> Tuple[Tuple[float, float], float]:
    evader_positions = [
        observations[evader_id].position
        for evader_id in evader_ids
        if evader_id in observations
    ]
    if not evader_positions:
        return field.center, args.capture_radius_mm

    center = mean_position(evader_positions)
    spread = max(vec_norm(vec_sub(position, center)) for position in evader_positions)
    radius = max(args.capture_radius_mm, spread + args.capture_margin_mm)
    return center, min(args.max_capture_radius_mm, radius)


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

    return {
        marker_id: (
            center[0] + radius_mm * math.cos(base_angle + index * step),
            center[1] + radius_mm * math.sin(base_angle + index * step),
        )
        for index, marker_id in enumerate(sorted_ids)
    }


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


def normalized_inverse_square(
    offset: Tuple[float, float],
    radius_mm: float,
) -> Tuple[float, float]:
    distance = vec_norm(offset)
    if distance < 1e-6:
        return (0.0, 0.0)
    normalized_distance = max(0.05, distance / max(1.0, radius_mm))
    return vec_scale(vec_normalize(offset), 1.0 / (normalized_distance * normalized_distance + 1e-6))


def pair_repel_vector(
    marker_id: int,
    peer_ids: Tuple[int, ...],
    observations: Dict[int, CarObservation],
    active_radius_mm: float,
    weight: float,
) -> Tuple[float, float]:
    if marker_id not in observations:
        return (0.0, 0.0)
    total = (0.0, 0.0)
    position = observations[marker_id].position
    for other_id in peer_ids:
        if other_id == marker_id or other_id not in observations:
            continue
        offset = vec_sub(position, observations[other_id].position)
        distance = vec_norm(offset)
        if distance < 1e-6 or distance > active_radius_mm:
            continue
        total = vec_add(total, vec_scale(normalized_inverse_square(offset, active_radius_mm), weight))
    return total


def boundary_force(
    position: Tuple[float, float],
    field: FieldBounds,
    margin_mm: float,
    weight: float,
) -> Tuple[float, float]:
    x_mm, y_mm = position
    total = (0.0, 0.0)
    edges = (
        (x_mm - field.xmin, (1.0, 0.0)),
        (field.xmax - x_mm, (-1.0, 0.0)),
        (y_mm - field.ymin, (0.0, 1.0)),
        (field.ymax - y_mm, (0.0, -1.0)),
    )
    for distance, direction in edges:
        if distance >= margin_mm:
            continue
        normalized_distance = max(0.05, distance / max(1.0, margin_mm))
        total = vec_add(total, vec_scale(direction, weight / (normalized_distance * normalized_distance + 1e-6)))
    return total


def evader_baseline_acceleration(
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

    neighbor_ids = [
        other_id
        for other_id in evader_ids
        if other_id != evader_id
        and other_id in observations
        and vec_norm(vec_sub(observations[other_id].position, evader.position)) <= args.evader_neighbor_radius_mm
    ]

    dispersion = (0.0, 0.0)
    for other_id in neighbor_ids:
        offset = vec_sub(evader.position, observations[other_id].position)
        dispersion = vec_add(
            dispersion,
            normalized_inverse_square(offset, args.evader_neighbor_radius_mm),
        )
    if neighbor_ids:
        dispersion = vec_scale(dispersion, 1.0 / len(neighbor_ids))

    escape = (0.0, 0.0)
    herder_detected = False
    for herder_id in herder_ids:
        herder = observations.get(herder_id)
        if herder is None:
            continue
        offset = vec_sub(evader.position, herder.position)
        distance = vec_norm(offset)
        if distance > args.evader_threat_radius_mm:
            continue
        herder_detected = True
        escape = vec_add(
            escape,
            normalized_inverse_square(offset, args.evader_threat_radius_mm),
        )

    aggregation = (0.0, 0.0)
    if herder_detected and neighbor_ids:
        neighbor_center = mean_position([observations[other_id].position for other_id in neighbor_ids])
        aggregation = vec_normalize(vec_sub(neighbor_center, evader.position))

    total = vec_add(
        boundary_force(evader.position, field, args.boundary_margin_mm, args.boundary_weight),
        vec_scale(dispersion, args.evader_dispersion_weight),
    )
    if herder_detected:
        total = vec_add(total, vec_scale(escape, args.evader_escape_weight))
        total = vec_add(total, vec_scale(aggregation, args.evader_aggregation_weight))
    return total


def evader_keep_out_vector(
    herder_position: Tuple[float, float],
    evader_ids: Tuple[int, ...],
    observations: Dict[int, CarObservation],
    keep_out_radius_mm: float,
    weight: float,
) -> Tuple[float, float]:
    total = (0.0, 0.0)
    for evader_id in evader_ids:
        evader = observations.get(evader_id)
        if evader is None:
            continue
        offset = vec_sub(herder_position, evader.position)
        distance = vec_norm(offset)
        if distance < 1e-6:
            total = vec_add(total, (weight, 0.0))
            continue
        if distance > keep_out_radius_mm:
            continue
        total = vec_add(total, vec_scale(normalized_inverse_square(offset, keep_out_radius_mm), weight))
    return total


def compute_commands(
    observations: Dict[int, CarObservation],
    calibration_by_id: CalibrationMap,
    ids_to_control: Tuple[int, ...],
    herder_ids: Tuple[int, ...],
    evader_ids: Tuple[int, ...],
    field: FieldBounds,
    motion_model: MotionModel,
    dt: float,
    args: argparse.Namespace,
) -> ControlResult:
    commands = {marker_id: "STOP" for marker_id in ids_to_control}
    herder_targets: Dict[int, Tuple[float, float]] = {}
    evader_accelerations: Dict[int, Tuple[float, float]] = {}
    evader_velocities: Dict[int, Tuple[float, float]] = {}

    visible_herders = tuple(marker_id for marker_id in herder_ids if marker_id in observations)
    visible_evaders = tuple(marker_id for marker_id in evader_ids if marker_id in observations)
    hull_ids, hull_points = herder_hull(observations, herder_ids)
    inside_hull_evaders = tuple(
        evader_id
        for evader_id in visible_evaders
        if point_in_convex_polygon(observations[evader_id].position, hull_points)
    )
    center, formation_radius = formation_center_and_radius(observations, evader_ids, field, args)

    if args.require_all_visible and (
        len(visible_herders) != len(herder_ids) or len(visible_evaders) != len(evader_ids)
    ):
        return ControlResult(
            commands,
            herder_targets,
            visible_herders,
            visible_evaders,
            hull_ids,
            inside_hull_evaders,
            center,
            formation_radius,
            evader_accelerations,
            evader_velocities,
        )

    target_by_herder = ring_targets(visible_herders, observations, center, formation_radius)

    for herder_id in herder_ids:
        herder = observations.get(herder_id)
        if herder is None:
            motion_model.reset(herder_id)
            continue

        desired = (0.0, 0.0)
        target = target_by_herder.get(herder_id)
        if target is not None:
            herder_targets[herder_id] = target
            target_offset = vec_sub(target, herder.position)
            target_distance = vec_norm(target_offset)
            if target_distance > args.target_deadband_mm:
                distance_scale = min(1.6, target_distance / max(1.0, formation_radius))
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

        herder_velocity = motion_model.update(
            herder_id,
            desired,
            dt,
            args.herder_model_speed_limit,
            args.herder_velocity_decay,
            (0.0, 0.0),
            keep_initial_motion=False,
        )
        commands[herder_id] = choose_command(herder_id, herder_velocity, observations, calibration_by_id)

    if not args.passive_evaders:
        for evader_id in evader_ids:
            if evader_id not in ids_to_control:
                continue
            evader = observations.get(evader_id)
            if evader is None:
                motion_model.reset(evader_id)
                commands[evader_id] = "STOP"
                continue

            acceleration = evader_baseline_acceleration(
                evader_id,
                observations,
                herder_ids,
                evader_ids,
                field,
                args,
            )
            initial_velocity = motion_model.random_vector(args.evader_initial_speed)
            evader_velocity = motion_model.update(
                evader_id,
                acceleration,
                dt,
                args.evader_model_speed_limit,
                args.evader_velocity_decay,
                initial_velocity,
                keep_initial_motion=True,
            )
            evader_accelerations[evader_id] = acceleration
            evader_velocities[evader_id] = evader_velocity
            commands[evader_id] = choose_command(evader_id, evader_velocity, observations, calibration_by_id)

    return ControlResult(
        commands=commands,
        herder_targets=herder_targets,
        visible_herders=visible_herders,
        visible_evaders=visible_evaders,
        hull_ids=hull_ids,
        inside_hull_evaders=inside_hull_evaders,
        formation_center=center,
        formation_radius=formation_radius,
        evader_accelerations=evader_accelerations,
        evader_velocities=evader_velocities,
    )


def speed_for_id(marker_id: int, herder_ids: Tuple[int, ...], args: argparse.Namespace) -> int:
    if marker_id in herder_ids:
        return args.herder_speed
    return args.evader_speed


def format_ids(ids: Sequence[int]) -> str:
    return ",".join(str(marker_id) for marker_id in ids) or "none"


def main() -> None:
    args = parse_args()
    herder_ids, evader_ids, tracked_ids = parse_id_sets(args)
    ids_to_control = control_ids(herder_ids, evader_ids, args)
    field = field_from_args(args)
    motion_model = MotionModel(args.seed)

    if not args.skip_wifi:
        connect_wifi(args.wifi_ssid, args.wifi_wait)

    config_by_id = make_config_map(args, tracked_ids)
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
        "Virtual field limits evaders: "
        f"x=[{field.xmin:.0f},{field.xmax:.0f}] y=[{field.ymin:.0f},{field.ymax:.0f}]"
    )
    print(
        "Running clean four-herder/two-evader baseline capture. "
        f"Herders={format_ids(herder_ids)} evaders={format_ids(evader_ids)} "
        f"controlled={format_ids(ids_to_control)} "
        f"speed(H/E)=({args.herder_speed}/{args.evader_speed}) "
        f"model_limit(H/E)=({args.herder_model_speed_limit:.1f}/{args.evader_model_speed_limit:.1f}). "
        "Press Ctrl+C to stop."
    )
    print(f"Calibrated controlled car IDs: {format_ids(sorted(calibration_by_id))}")

    last_status_time = 0.0
    last_loop_time = time.monotonic()
    try:
        while True:
            loop_time = time.monotonic()
            dt = max(0.01, min(0.25, loop_time - last_loop_time))
            last_loop_time = loop_time

            raw_observations = tracker.get_observations()
            observations = filter_tracked_observations(raw_observations, tracked_ids)
            result = compute_commands(
                observations,
                calibration_by_id,
                ids_to_control,
                herder_ids,
                evader_ids,
                field,
                motion_model,
                dt,
                args,
            )
            send_commands(result.commands, controllers)

            now = time.monotonic()
            if now - last_status_time >= STATUS_INTERVAL_SEC:
                last_status_time = now
                evader_outside_field = [
                    evader_id
                    for evader_id in evader_ids
                    if evader_id in observations and not field.contains(observations[evader_id].position)
                ]
                command_text = " ".join(
                    f"ID{marker_id}:{command}"
                    for marker_id, command in sorted(result.commands.items())
                )
                velocity_text = " ".join(
                    f"E{evader_id}:v={vec_norm(velocity):.2f}"
                    for evader_id, velocity in sorted(result.evader_velocities.items())
                )
                print(
                    f"Visible H={len(result.visible_herders)}/{len(herder_ids)} "
                    f"E={len(result.visible_evaders)}/{len(evader_ids)} "
                    f"hull={format_ids(result.hull_ids)} "
                    f"inside_hull={format_ids(result.inside_hull_evaders)} "
                    f"evader_outside_field={format_ids(evader_outside_field)} "
                    f"center=({result.formation_center[0]:.0f},{result.formation_center[1]:.0f}) "
                    f"radius={result.formation_radius:.0f} "
                    f"| {command_text} | {velocity_text}"
                )

            time.sleep(CONTROL_LOOP_SEC)
    except KeyboardInterrupt:
        print("\nStopping cars.")
    finally:
        stop_all(controllers)
        for controller in controllers.values():
            controller.close(send_stop=False)


if __name__ == "__main__":
    main()
