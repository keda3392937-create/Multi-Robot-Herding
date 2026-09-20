import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from six_car_repel import (
    ACTIVE_REPULSION_RADIUS_MM,
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
    MOTION_COMMANDS,
    STATUS_INTERVAL_SEC,
    VICON_HOST,
    CalibrationMap,
    CarConfig,
    CarObservation,
    CommandCalibration,
    TcpCarController,
    ViconTracker,
    best_command_for_vector,
    calibrate_command_for_all_cars,
    connect_wifi,
    discover_car_ips,
    min_pair_distance,
    send_commands,
    set_speed,
    vec_add,
    vec_norm,
    vec_normalize,
    vec_scale,
    vec_sub,
    wait_for_observations,
)


HERDER_IDS = (1, 2, 3, 4, 5, 6)
DEFAULT_MAX_CAR_ID = 50
DEFAULT_BOUNDARY_MARGIN_MM = 1200.0
DEFAULT_CAGE_MIN_SIZE_MM = 900.0
DEFAULT_CAGE_MAX_SIZE_MM = 2600.0
DEFAULT_CAGE_SIZE_FRACTION = 0.32
DEFAULT_CAGE_OFFSET_MM = 700.0
DEFAULT_BOUNDARY_PUSH_MARGIN_MM = 650.0
DEFAULT_BOUNDARY_PUSH_WEIGHT = 0.85
DEFAULT_CAGE_AVOID_WEIGHT = 1.25
DEFAULT_HERDER_HUNT_WEIGHT = 1.0
DEFAULT_HERDER_SURROUND_WEIGHT = 0.65
DEFAULT_HERDER_REPEL_RADIUS_MM = 800.0
DEFAULT_HERDER_REPEL_WEIGHT = 1.35
DEFAULT_HERDER_SENSE_RADIUS_MM = 2800.0
DEFAULT_EVADER_HERDER_RADIUS_MM = 1300.0
DEFAULT_EVADER_HERDER_WEIGHT = 1.0
DEFAULT_EVADER_REPEL_WEIGHT = 0.35
DEFAULT_CAPTURE_HOLD_MARGIN_MM = 120.0


@dataclass
class VirtualArena:
    left: float
    right: float
    bottom: float
    top: float
    cage_left: float
    cage_right: float
    cage_bottom: float
    cage_top: float

    @property
    def cage_center(self) -> Tuple[float, float]:
        return (
            (self.cage_left + self.cage_right) * 0.5,
            (self.cage_bottom + self.cage_top) * 0.5,
        )

    @property
    def cage_width(self) -> float:
        return self.cage_right - self.cage_left

    @property
    def cage_height(self) -> float:
        return self.cage_top - self.cage_bottom

    def cage_vertices(self) -> Tuple[Tuple[float, float], ...]:
        return (
            (self.cage_left, self.cage_bottom),
            (self.cage_right, self.cage_bottom),
            (self.cage_right, self.cage_top),
            (self.cage_left, self.cage_top),
        )

    def is_in_cage(self, position: Tuple[float, float], margin: float = 0.0) -> bool:
        x, y = position
        return (
            self.cage_left + margin <= x <= self.cage_right - margin
            and self.cage_bottom + margin <= y <= self.cage_top - margin
        )


def configured_car_ids(args: argparse.Namespace) -> Tuple[int, ...]:
    return tuple(range(1, args.max_car_id + 1))


def evader_ids_for(car_ids: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(marker_id for marker_id in car_ids if marker_id not in HERDER_IDS)


def parse_overrides(overrides: Sequence[str], label: str, valid_ids: Tuple[int, ...]) -> Dict[int, str]:
    result: Dict[int, str] = {}
    valid_set = set(valid_ids)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid --{label} value '{override}', expected ID=VALUE")
        marker_id_text, value = override.split("=", 1)
        marker_id = int(marker_id_text)
        if marker_id not in valid_set:
            raise ValueError(f"Invalid car id {marker_id}; expected one of {valid_ids[0]}..{valid_ids[-1]}")
        result[marker_id] = value.strip()
    return result


def make_config_map(args: argparse.Namespace, car_ids: Tuple[int, ...]) -> Dict[int, CarConfig]:
    subject_overrides = parse_overrides(args.subject, "subject", car_ids)
    car_ip_overrides = parse_overrides(args.car_ip, "car-ip", car_ids)
    configs: Dict[int, CarConfig] = {}
    for marker_id in car_ids:
        configs[marker_id] = CarConfig(
            marker_id=marker_id,
            name=f"car{marker_id}_sta",
            ip=car_ip_overrides.get(marker_id, ""),
            subject_name=subject_overrides.get(marker_id, f"kedaya{marker_id}"),
        )
    return configs


def resolve_car_ips(config_by_id: Dict[int, CarConfig], car_ids: Tuple[int, ...], args: argparse.Namespace) -> None:
    if args.skip_discovery:
        return

    discovered = discover_car_ips(
        car_ids,
        args.discovery_timeout,
        bind_ips=args.bind_ip,
        broadcast_ips=args.broadcast_ip,
    )
    for marker_id, config in config_by_id.items():
        if not config.ip and marker_id in discovered:
            config.ip = discovered[marker_id]

    missing = [marker_id for marker_id, config in config_by_id.items() if not config.ip]
    if missing:
        print(f"No IP for {len(missing)} car(s): {format_ids(missing)}; those cars will stay disconnected.")


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


def make_virtual_arena(
    initial_observations: Dict[int, CarObservation],
    car_ids: Tuple[int, ...],
    evader_ids: Tuple[int, ...],
    args: argparse.Namespace,
) -> VirtualArena:
    positions = [
        initial_observations[marker_id].position
        for marker_id in car_ids
        if marker_id in initial_observations
    ]
    herder_positions = [
        initial_observations[marker_id].position
        for marker_id in HERDER_IDS
        if marker_id in initial_observations
    ]
    evader_positions = [
        initial_observations[marker_id].position
        for marker_id in evader_ids
        if marker_id in initial_observations
    ]

    if not positions:
        raise RuntimeError("No Vicon observations were available to initialize the virtual arena.")
    if not herder_positions:
        raise RuntimeError("No herder observations were available to initialize the virtual arena.")
    if not evader_positions:
        raise RuntimeError("No evader observations were available to initialize the virtual arena.")

    xs = [position[0] for position in positions]
    ys = [position[1] for position in positions]
    left = min(xs) - args.boundary_margin_mm
    right = max(xs) + args.boundary_margin_mm
    bottom = min(ys) - args.boundary_margin_mm
    top = max(ys) + args.boundary_margin_mm

    evader_center = mean_position(evader_positions)
    herder_center = mean_position(herder_positions)
    cage_direction = vec_normalize(vec_sub(evader_center, herder_center))
    if vec_norm(cage_direction) < 1e-6:
        cage_direction = (1.0, 0.0)

    evader_radius = max(vec_norm(vec_sub(position, evader_center)) for position in evader_positions)
    field_span = max(right - left, top - bottom)
    cage_size = max(
        args.cage_min_size_mm,
        args.cage_size_fraction * field_span,
        0.75 * evader_radius,
    )
    cage_size = min(args.cage_max_size_mm, cage_size)

    desired_center = vec_add(
        evader_center,
        vec_scale(cage_direction, evader_radius + args.cage_offset_mm + cage_size * 0.35),
    )
    half = cage_size * 0.5
    cage_center = (
        clamp(desired_center[0], left + half, right - half),
        clamp(desired_center[1], bottom + half, top - half),
    )

    return VirtualArena(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        cage_left=cage_center[0] - half,
        cage_right=cage_center[0] + half,
        cage_bottom=cage_center[1] - half,
        cage_top=cage_center[1] + half,
    )


def print_virtual_arena(arena: VirtualArena) -> None:
    print(
        "Virtual boundary: "
        f"x=({arena.left:.0f},{arena.right:.0f}) "
        f"y=({arena.bottom:.0f},{arena.top:.0f})"
    )
    print(
        "Virtual cage: "
        f"x=({arena.cage_left:.0f},{arena.cage_right:.0f}) "
        f"y=({arena.cage_bottom:.0f},{arena.cage_top:.0f}) "
        f"center=({arena.cage_center[0]:.0f},{arena.cage_center[1]:.0f}) "
        f"size=({arena.cage_width:.0f},{arena.cage_height:.0f})"
    )


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


def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


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


def augmented_hull_herders(
    observations: Dict[int, CarObservation],
    arena: VirtualArena,
) -> Tuple[int, ...]:
    entities: List[Tuple[str, int]] = []
    points: List[Tuple[float, float]] = []
    for marker_id in HERDER_IDS:
        if marker_id in observations:
            entities.append(("herder", marker_id))
            points.append(observations[marker_id].position)
    for index, vertex in enumerate(arena.cage_vertices()):
        entities.append(("cage", index))
        points.append(vertex)

    hull_indices = convex_hull_indices(points)
    return tuple(
        entities[index][1]
        for index in hull_indices
        if entities[index][0] == "herder"
    )


def angle_around(center: Tuple[float, float], position: Tuple[float, float]) -> float:
    return math.atan2(position[1] - center[1], position[0] - center[0])


def angular_spacing_vector(
    marker_id: int,
    observations: Dict[int, CarObservation],
    arena: VirtualArena,
    weight: float,
) -> Tuple[float, float]:
    visible_herders = [herder_id for herder_id in HERDER_IDS if herder_id in observations]
    if len(visible_herders) < 3 or marker_id not in observations:
        return (0.0, 0.0)

    center = arena.cage_center
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

    car = observations[marker_id]
    total = (0.0, 0.0)
    for other_id in peer_ids:
        if other_id == marker_id or other_id not in observations:
            continue

        offset = vec_sub(car.position, observations[other_id].position)
        distance = vec_norm(offset)
        if distance < 1e-6 or distance > active_radius_mm:
            continue

        closeness = max(0.0, active_radius_mm - distance) / active_radius_mm
        weight = weight_scale * (0.25 + closeness * closeness * 2.5)
        total = vec_add(total, vec_scale(vec_normalize(offset), weight))
    return total


def boundary_push_vector(
    position: Tuple[float, float],
    arena: VirtualArena,
    margin_mm: float,
    weight: float,
) -> Tuple[float, float]:
    x, y = position
    total = (0.0, 0.0)
    distances = (
        (x - arena.left, (1.0, 0.0)),
        (arena.right - x, (-1.0, 0.0)),
        (y - arena.bottom, (0.0, 1.0)),
        (arena.top - y, (0.0, -1.0)),
    )
    for distance, direction in distances:
        if distance < margin_mm:
            closeness = max(0.0, margin_mm - distance) / margin_mm
            total = vec_add(total, vec_scale(direction, weight * (0.35 + closeness * closeness * 2.5)))
    return total


def cage_avoid_vector(
    position: Tuple[float, float],
    arena: VirtualArena,
    weight: float,
) -> Tuple[float, float]:
    if not arena.is_in_cage(position, margin=-100.0):
        return (0.0, 0.0)

    away = vec_normalize(vec_sub(position, arena.cage_center))
    if vec_norm(away) < 1e-6:
        away = (1.0, 0.0)
    return vec_scale(away, weight)


def outside_evaders(
    evader_ids: Tuple[int, ...],
    observations: Dict[int, CarObservation],
    arena: VirtualArena,
) -> List[int]:
    return [
        marker_id
        for marker_id in evader_ids
        if marker_id in observations and not arena.is_in_cage(observations[marker_id].position)
    ]


def choose_herder_target(
    herder_id: int,
    outside_ids: Sequence[int],
    observations: Dict[int, CarObservation],
    arena: VirtualArena,
    args: argparse.Namespace,
) -> Optional[int]:
    if not outside_ids or herder_id not in observations:
        return None

    herder = observations[herder_id]
    nearby = [
        evader_id
        for evader_id in outside_ids
        if vec_norm(vec_sub(observations[evader_id].position, herder.position)) <= args.herder_sense_radius_mm
    ]
    if nearby:
        return min(
            nearby,
            key=lambda evader_id: vec_norm(vec_sub(observations[evader_id].position, herder.position)),
        )

    return max(
        outside_ids,
        key=lambda evader_id: vec_norm(vec_sub(observations[evader_id].position, arena.cage_center)),
    )


def evader_avoid_herders_vector(
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
        if distance > args.evader_herder_radius_mm:
            continue

        closeness = max(0.0, args.evader_herder_radius_mm - distance) / args.evader_herder_radius_mm
        weight = args.evader_herder_weight * (0.35 + closeness * closeness * 2.5)
        total = vec_add(total, vec_scale(vec_normalize(offset), weight))

    return total, nearest_distance


def compute_herder_commands(
    observations: Dict[int, CarObservation],
    calibration_by_id: CalibrationMap,
    arena: VirtualArena,
    evader_ids: Tuple[int, ...],
    args: argparse.Namespace,
) -> Tuple[Dict[int, str], Dict[int, Tuple[float, float]], Dict[int, Optional[int]]]:
    commands = {marker_id: "STOP" for marker_id in HERDER_IDS}
    targets: Dict[int, Tuple[float, float]] = {}
    target_by_herder: Dict[int, Optional[int]] = {marker_id: None for marker_id in HERDER_IDS}
    outside_ids = outside_evaders(evader_ids, observations, arena)
    hull_herders = set(augmented_hull_herders(observations, arena))

    for herder_id in HERDER_IDS:
        herder = observations.get(herder_id)
        if herder is None or herder_id not in calibration_by_id:
            continue

        desired = pair_repel_vector(
            herder_id,
            HERDER_IDS,
            observations,
            args.herder_repel_radius_mm,
            args.herder_repel_weight,
        )
        desired = vec_add(
            desired,
            boundary_push_vector(
                herder.position,
                arena,
                args.boundary_push_margin_mm,
                args.boundary_push_weight,
            ),
        )
        desired = vec_add(desired, cage_avoid_vector(herder.position, arena, args.cage_avoid_weight))

        if herder_id in hull_herders:
            desired = vec_add(
                desired,
                angular_spacing_vector(herder_id, observations, arena, args.herder_surround_weight),
            )

        target_id = choose_herder_target(herder_id, outside_ids, observations, arena, args)
        target_by_herder[herder_id] = target_id
        if target_id is not None:
            target_position = observations[target_id].position
            targets[herder_id] = target_position
            chase_vector = vec_normalize(vec_sub(target_position, herder.position))
            desired = vec_add(desired, vec_scale(chase_vector, args.herder_hunt_weight))
        elif args.guard_cage_when_done:
            desired = vec_add(desired, vec_scale(vec_normalize(vec_sub(arena.cage_center, herder.position)), 0.25))

        commands[herder_id] = choose_command(herder_id, desired, observations, calibration_by_id)

    return commands, targets, target_by_herder


def compute_evader_commands(
    observations: Dict[int, CarObservation],
    calibration_by_id: CalibrationMap,
    arena: VirtualArena,
    evader_ids: Tuple[int, ...],
    args: argparse.Namespace,
) -> Tuple[Dict[int, str], Dict[int, Tuple[float, float]], Dict[int, float]]:
    commands = {marker_id: "STOP" for marker_id in evader_ids}
    targets: Dict[int, Tuple[float, float]] = {}
    nearest_herder_distances = {marker_id: float("inf") for marker_id in evader_ids}

    if args.passive_evaders:
        return commands, targets, nearest_herder_distances

    for evader_id in evader_ids:
        evader = observations.get(evader_id)
        if evader is None or evader_id not in calibration_by_id:
            continue

        if arena.is_in_cage(evader.position, margin=args.capture_hold_margin_mm):
            commands[evader_id] = "STOP"
            targets[evader_id] = arena.cage_center
            continue

        if arena.is_in_cage(evader.position):
            desired = vec_normalize(vec_sub(arena.cage_center, evader.position))
            commands[evader_id] = choose_command(evader_id, desired, observations, calibration_by_id)
            targets[evader_id] = arena.cage_center
            continue

        herder_vector, nearest_distance = evader_avoid_herders_vector(evader, observations, args)
        nearest_herder_distances[evader_id] = nearest_distance
        repel_vector = pair_repel_vector(
            evader_id,
            evader_ids,
            observations,
            args.evader_repel_radius_mm,
            args.evader_repel_weight,
        )
        boundary_vector = boundary_push_vector(
            evader.position,
            arena,
            args.boundary_push_margin_mm,
            args.boundary_push_weight,
        )
        desired = vec_add(vec_add(herder_vector, repel_vector), boundary_vector)

        if vec_norm(desired) >= MIN_DRIVE_VECTOR_NORM:
            direction = vec_normalize(desired)
            commands[evader_id] = choose_command(evader_id, direction, observations, calibration_by_id)
            targets[evader_id] = vec_add(evader.position, vec_scale(direction, args.target_step_mm))

    return commands, targets, nearest_herder_distances


def compute_commands(
    observations: Dict[int, CarObservation],
    config_by_id: Dict[int, CarConfig],
    calibration_by_id: CalibrationMap,
    arena: VirtualArena,
    car_ids: Tuple[int, ...],
    evader_ids: Tuple[int, ...],
    args: argparse.Namespace,
) -> Tuple[str, Dict[int, str], Dict[int, Tuple[float, float]], Dict[int, Optional[int]], Dict[int, float]]:
    commands = {marker_id: "STOP" for marker_id in config_by_id}
    targets = {
        marker_id: observations[marker_id].position
        for marker_id in observations
        if marker_id in config_by_id
    }
    visible_ids = [marker_id for marker_id in car_ids if marker_id in observations]
    visible_herders = [marker_id for marker_id in HERDER_IDS if marker_id in observations]
    visible_evaders = [marker_id for marker_id in evader_ids if marker_id in observations]
    missing_ids = [marker_id for marker_id in car_ids if marker_id not in observations]
    uncalibrated_ids = [
        marker_id
        for marker_id in visible_ids
        if marker_id in config_by_id and marker_id not in calibration_by_id
    ]

    if args.require_all_visible and missing_ids:
        summary = f"Missing Vicon subject ID {format_ids(missing_ids)}"
        return summary, commands, targets, {}, {}

    herder_commands, herder_targets, target_by_herder = compute_herder_commands(
        observations,
        calibration_by_id,
        arena,
        evader_ids,
        args,
    )
    evader_commands, evader_targets, nearest_herder_distances = compute_evader_commands(
        observations,
        calibration_by_id,
        arena,
        evader_ids,
        args,
    )
    commands.update(herder_commands)
    commands.update(evader_commands)
    targets.update(herder_targets)
    targets.update(evader_targets)

    evader_observations = {
        marker_id: observation
        for marker_id, observation in observations.items()
        if marker_id in evader_ids
    }
    herder_observations = {
        marker_id: observation
        for marker_id, observation in observations.items()
        if marker_id in HERDER_IDS
    }
    captured_ids = [
        marker_id
        for marker_id in visible_evaders
        if arena.is_in_cage(observations[marker_id].position, margin=args.capture_hold_margin_mm)
    ]
    chase_text = " ".join(
        f"H{herder_id}->E{target_id}"
        for herder_id, target_id in target_by_herder.items()
        if target_id is not None
    ) or "none"
    summary = (
        f"Visible={len(visible_ids)}/{len(car_ids)} "
        f"herders={len(visible_herders)}/{len(HERDER_IDS)} "
        f"evaders={len(visible_evaders)}/{len(evader_ids)} "
        f"captured={len(captured_ids)}/{len(visible_evaders)} "
        f"herder min pair={min_pair_distance(herder_observations):.0f}mm "
        f"evader min pair={min_pair_distance(evader_observations):.0f}mm "
        f"chase {chase_text} "
        f"missing={format_ids(missing_ids)} "
        f"uncalibrated={format_ids(uncalibrated_ids)}"
    )
    return summary, commands, targets, target_by_herder, nearest_herder_distances


def calibrate_configured_cars(
    car_ids: Tuple[int, ...],
    controllers: Dict[int, TcpCarController],
    tracker: ViconTracker,
    args: argparse.Namespace,
) -> CalibrationMap:
    candidate_ids = tuple(marker_id for marker_id in car_ids if marker_id in controllers)
    calibration_by_id: CalibrationMap = {marker_id: {} for marker_id in candidate_ids}

    print("Starting simultaneous motion calibration. Keep the area clear.")
    for controller in controllers.values():
        set_speed(controller, args.calibration_speed)
        controller.send("STOP", force=True)

    visible_before_calibration = wait_for_observations(tracker, candidate_ids, 5.0)
    marker_ids = tuple(marker_id for marker_id in candidate_ids if marker_id in visible_before_calibration)
    missing_ids = [marker_id for marker_id in candidate_ids if marker_id not in visible_before_calibration]
    if missing_ids:
        print(f"Skipping calibration for no-Vicon car ID {format_ids(missing_ids)}")

    for command in MOTION_COMMANDS:
        print(f"Calibrating command {command} on all visible cars...")
        command_calibrations = calibrate_command_for_all_cars(
            command,
            marker_ids,
            controllers,
            tracker,
            args,
        )
        for marker_id, calibration in command_calibrations.items():
            calibration_by_id.setdefault(marker_id, {})[command] = calibration

    incomplete_ids = [
        marker_id
        for marker_id, calibrations in calibration_by_id.items()
        if len(calibrations) != len(MOTION_COMMANDS)
    ]
    for marker_id in incomplete_ids:
        print(f"car{marker_id} has incomplete calibration and will stay stopped.")
        calibration_by_id.pop(marker_id, None)

    return calibration_by_id


def speed_for_id(marker_id: int, args: argparse.Namespace) -> int:
    if marker_id in HERDER_IDS:
        return args.herder_speed if args.herder_speed is not None else args.speed
    return args.evader_speed if args.evader_speed is not None else args.speed


def format_ids(ids: Sequence[int], limit: int = 12) -> str:
    if not ids:
        return "none"
    sorted_ids = sorted(ids)
    if len(sorted_ids) <= limit:
        return ",".join(str(marker_id) for marker_id in sorted_ids)
    visible = ",".join(str(marker_id) for marker_id in sorted_ids[:limit])
    return f"{visible},...(+{len(sorted_ids) - limit})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline-inspired cage capture: cars 1-6 are herders, the remaining configured cars are evaders."
    )
    parser.add_argument("--max-car-id", type=int, default=DEFAULT_MAX_CAR_ID)
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
    parser.add_argument("--initial-observation-timeout", type=float, default=5.0)
    parser.add_argument("--boundary-margin-mm", type=float, default=DEFAULT_BOUNDARY_MARGIN_MM)
    parser.add_argument("--boundary-push-margin-mm", type=float, default=DEFAULT_BOUNDARY_PUSH_MARGIN_MM)
    parser.add_argument("--boundary-push-weight", type=float, default=DEFAULT_BOUNDARY_PUSH_WEIGHT)
    parser.add_argument("--cage-min-size-mm", type=float, default=DEFAULT_CAGE_MIN_SIZE_MM)
    parser.add_argument("--cage-max-size-mm", type=float, default=DEFAULT_CAGE_MAX_SIZE_MM)
    parser.add_argument("--cage-size-fraction", type=float, default=DEFAULT_CAGE_SIZE_FRACTION)
    parser.add_argument("--cage-offset-mm", type=float, default=DEFAULT_CAGE_OFFSET_MM)
    parser.add_argument("--cage-avoid-weight", type=float, default=DEFAULT_CAGE_AVOID_WEIGHT)
    parser.add_argument("--herder-hunt-weight", type=float, default=DEFAULT_HERDER_HUNT_WEIGHT)
    parser.add_argument("--herder-surround-weight", type=float, default=DEFAULT_HERDER_SURROUND_WEIGHT)
    parser.add_argument("--herder-repel-radius-mm", type=float, default=DEFAULT_HERDER_REPEL_RADIUS_MM)
    parser.add_argument("--herder-repel-weight", type=float, default=DEFAULT_HERDER_REPEL_WEIGHT)
    parser.add_argument("--herder-sense-radius-mm", type=float, default=DEFAULT_HERDER_SENSE_RADIUS_MM)
    parser.add_argument("--evader-herder-radius-mm", type=float, default=DEFAULT_EVADER_HERDER_RADIUS_MM)
    parser.add_argument("--evader-herder-weight", type=float, default=DEFAULT_EVADER_HERDER_WEIGHT)
    parser.add_argument("--evader-repel-radius-mm", type=float, default=ACTIVE_REPULSION_RADIUS_MM)
    parser.add_argument("--evader-repel-weight", type=float, default=DEFAULT_EVADER_REPEL_WEIGHT)
    parser.add_argument("--capture-hold-margin-mm", type=float, default=DEFAULT_CAPTURE_HOLD_MARGIN_MM)
    parser.add_argument("--target-step-mm", type=float, default=DEFAULT_TARGET_STEP_MM)
    parser.add_argument("--passive-evaders", action="store_true")
    parser.add_argument("--guard-cage-when-done", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--require-all-visible", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_car_id <= max(HERDER_IDS):
        raise ValueError(f"--max-car-id must be greater than {max(HERDER_IDS)}")

    car_ids = configured_car_ids(args)
    evader_ids = evader_ids_for(car_ids)

    if not args.skip_wifi:
        connect_wifi(args.wifi_ssid, args.wifi_wait)

    config_by_id = make_config_map(args, car_ids)
    resolve_car_ips(config_by_id, car_ids, args)
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

    initial_observations = wait_for_observations(
        tracker,
        car_ids,
        args.initial_observation_timeout,
    )
    arena = make_virtual_arena(initial_observations, car_ids, evader_ids, args)
    print_virtual_arena(arena)

    if args.skip_calibration:
        calibration_by_id: CalibrationMap = {}
        print("Skipping calibration. Cars without calibration will stay stopped.")
    else:
        calibration_by_id = calibrate_configured_cars(car_ids, controllers, tracker, args)

    for marker_id, controller in controllers.items():
        set_speed(controller, speed_for_id(marker_id, args))
        controller.send("STOP", force=True)

    calibrated_text = format_ids(sorted(calibration_by_id))
    print(f"Calibrated car IDs: {calibrated_text}")
    print(
        "Running six-herder cage capture. "
        f"Herders: {format_ids(HERDER_IDS)}. "
        f"Evaders: {format_ids(evader_ids)}. "
        "Press Ctrl+C to stop."
    )

    last_status_time = 0.0
    try:
        while True:
            observations = tracker.get_observations()
            summary, commands, targets, target_by_herder, nearest_herder_distances = compute_commands(
                observations,
                config_by_id,
                calibration_by_id,
                arena,
                car_ids,
                evader_ids,
                args,
            )
            send_commands(commands, controllers)

            now = time.monotonic()
            if now - last_status_time >= STATUS_INTERVAL_SEC:
                status_ids = [
                    marker_id
                    for marker_id in car_ids
                    if marker_id in controllers or marker_id in observations
                ]
                command_text = " ".join(f"ID{marker_id}:{commands[marker_id]}" for marker_id in status_ids)
                target_text = " ".join(
                    f"ID{marker_id}:({targets[marker_id][0]:.0f},{targets[marker_id][1]:.0f})"
                    for marker_id in status_ids
                    if marker_id in targets and marker_id in calibration_by_id
                )
                evader_dist_text = " ".join(
                    f"ID{marker_id}:{nearest_herder_distances[marker_id]:.0f}mm"
                    for marker_id in evader_ids
                    if nearest_herder_distances.get(marker_id, float("inf")) != float("inf")
                )
                connected_count = sum(1 for controller in controllers.values() if controller.sock is not None)
                print(
                    f"{summary} | TCP={connected_count}/{len(car_ids)} | {command_text} "
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
