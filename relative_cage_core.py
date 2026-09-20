from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Vec2 = Tuple[float, float]
EPSILON = 1e-9


def vec_add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def vec_sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def vec_scale(v: Vec2, scale: float) -> Vec2:
    return (v[0] * scale, v[1] * scale)


def vec_dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def vec_cross(a: Vec2, b: Vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


def vec_norm(v: Vec2) -> float:
    return math.hypot(v[0], v[1])


def vec_normalize(v: Vec2) -> Vec2:
    norm = vec_norm(v)
    if norm <= EPSILON:
        return (0.0, 0.0)
    return (v[0] / norm, v[1] / norm)


def vec_limit(v: Vec2, maximum: float) -> Vec2:
    if maximum < 0.0:
        raise ValueError("maximum must be non-negative")
    norm = vec_norm(v)
    if norm <= maximum or norm <= EPSILON:
        return v
    return vec_scale(v, maximum / norm)


def vec_mean(points: Iterable[Vec2]) -> Vec2:
    values = tuple(points)
    if not values:
        return (0.0, 0.0)
    return (
        sum(point[0] for point in values) / len(values),
        sum(point[1] for point in values) / len(values),
    )


def angle_wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ArenaGeometryError(ValueError):
    pass


@dataclass(frozen=True)
class FieldCorners:
    left_up: Vec2
    left_down: Vec2
    right_up: Vec2
    right_down: Vec2

    def as_dict(self) -> Dict[str, Vec2]:
        return {
            "left-up": self.left_up,
            "left-down": self.left_down,
            "right-up": self.right_up,
            "right-down": self.right_down,
        }


@dataclass(frozen=True)
class RelativeArena:
    origin_world: Vec2
    x_axis_world: Vec2
    y_axis_world: Vec2
    width_mm: float
    height_mm: float
    cage_width_mm: float
    cage_height_mm: float
    corner_fit_error_mm: float
    corner_angle_deg: float

    @property
    def scale_mm(self) -> float:
        return min(self.width_mm, self.height_mm)

    @property
    def field_center_local(self) -> Vec2:
        return (self.width_mm * 0.5, self.height_mm * 0.5)

    @property
    def cage_left(self) -> float:
        return 0.0

    @property
    def cage_right(self) -> float:
        return self.cage_width_mm

    @property
    def cage_bottom(self) -> float:
        return self.height_mm - self.cage_height_mm

    @property
    def cage_top(self) -> float:
        return self.height_mm

    @property
    def cage_center_local(self) -> Vec2:
        return (
            self.cage_width_mm * 0.5,
            self.height_mm - self.cage_height_mm * 0.5,
        )

    @property
    def cage_top_left_local(self) -> Vec2:
        return (0.0, self.height_mm)

    @property
    def cage_right_down_local(self) -> Vec2:
        return (self.cage_width_mm, self.height_mm - self.cage_height_mm)

    def world_to_local(self, point: Vec2) -> Vec2:
        relative = vec_sub(point, self.origin_world)
        return (
            vec_dot(relative, self.x_axis_world),
            vec_dot(relative, self.y_axis_world),
        )

    def local_to_world(self, point: Vec2) -> Vec2:
        return vec_add(
            self.origin_world,
            vec_add(
                vec_scale(self.x_axis_world, point[0]),
                vec_scale(self.y_axis_world, point[1]),
            ),
        )

    def world_vector_to_local(self, vector: Vec2) -> Vec2:
        return (
            vec_dot(vector, self.x_axis_world),
            vec_dot(vector, self.y_axis_world),
        )

    def local_vector_to_world(self, vector: Vec2) -> Vec2:
        return vec_add(
            vec_scale(self.x_axis_world, vector[0]),
            vec_scale(self.y_axis_world, vector[1]),
        )

    def contains_field_local(self, point: Vec2, margin_mm: float = 0.0) -> bool:
        x_mm, y_mm = point
        return (
            margin_mm <= x_mm <= self.width_mm - margin_mm
            and margin_mm <= y_mm <= self.height_mm - margin_mm
        )

    def contains_cage_local(self, point: Vec2, margin_mm: float = 0.0) -> bool:
        x_mm, y_mm = point
        return (
            self.cage_left + margin_mm <= x_mm <= self.cage_right - margin_mm
            and self.cage_bottom + margin_mm <= y_mm <= self.cage_top - margin_mm
        )

    def contains_field_world(self, point: Vec2, margin_mm: float = 0.0) -> bool:
        return self.contains_field_local(self.world_to_local(point), margin_mm)

    def contains_cage_world(self, point: Vec2, margin_mm: float = 0.0) -> bool:
        return self.contains_cage_local(self.world_to_local(point), margin_mm)

    def distance_outside_field_local(self, point: Vec2) -> float:
        x_mm, y_mm = point
        closest = (
            clamp(x_mm, 0.0, self.width_mm),
            clamp(y_mm, 0.0, self.height_mm),
        )
        return vec_norm(vec_sub(point, closest))

    def cage_anchor_points_local(self) -> Tuple[Vec2, Vec2, Vec2]:
        # baseline.py uses three cage vertices as fixed convex-hull anchors.
        return (
            self.cage_top_left_local,
            (self.cage_left, self.cage_bottom),
            (self.cage_right, self.cage_top),
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "origin_world": self.origin_world,
            "x_axis_world": self.x_axis_world,
            "y_axis_world": self.y_axis_world,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "cage_width_mm": self.cage_width_mm,
            "cage_height_mm": self.cage_height_mm,
            "cage_top_left_world": self.local_to_world(self.cage_top_left_local),
            "cage_right_down_world": self.local_to_world(self.cage_right_down_local),
            "corner_fit_error_mm": self.corner_fit_error_mm,
            "corner_angle_deg": self.corner_angle_deg,
        }


def fit_relative_arena(
    corners: FieldCorners,
    cage_width_ratio: float = 0.40,
    cage_height_ratio: float = 0.40,
    min_field_size_mm: float = 2000.0,
    max_opposite_edge_error_ratio: float = 0.10,
    max_angle_error_deg: float = 15.0,
    max_corner_fit_error_ratio: float = 0.05,
    min_corner_fit_error_mm: float = 150.0,
) -> RelativeArena:
    if not 0.05 <= cage_width_ratio <= 0.80:
        raise ArenaGeometryError("cage_width_ratio must be between 0.05 and 0.80")
    if not 0.05 <= cage_height_ratio <= 0.80:
        raise ArenaGeometryError("cage_height_ratio must be between 0.05 and 0.80")

    left_center = vec_mean((corners.left_up, corners.left_down))
    right_center = vec_mean((corners.right_up, corners.right_down))
    up_center = vec_mean((corners.left_up, corners.right_up))
    down_center = vec_mean((corners.left_down, corners.right_down))
    horizontal = vec_sub(right_center, left_center)
    vertical = vec_sub(up_center, down_center)
    x_axis = vec_normalize(horizontal)
    if vec_norm(x_axis) <= EPSILON or vec_norm(vertical) <= EPSILON:
        raise ArenaGeometryError("field corners are degenerate")

    corner_angle_deg = math.degrees(
        math.acos(clamp(vec_dot(x_axis, vec_normalize(vertical)), -1.0, 1.0))
    )
    if abs(corner_angle_deg - 90.0) > max_angle_error_deg:
        raise ArenaGeometryError(
            f"field axes are not close to perpendicular: {corner_angle_deg:.1f} deg"
        )

    vertical_orthogonal = vec_sub(vertical, vec_scale(x_axis, vec_dot(vertical, x_axis)))
    y_axis = vec_normalize(vertical_orthogonal)
    if vec_norm(y_axis) <= EPSILON:
        raise ArenaGeometryError("field vertical axis is degenerate")
    if vec_dot(y_axis, vertical) < 0.0:
        y_axis = vec_scale(y_axis, -1.0)

    center = vec_mean(corners.as_dict().values())

    def projection(point: Vec2) -> Vec2:
        relative = vec_sub(point, center)
        return (vec_dot(relative, x_axis), vec_dot(relative, y_axis))

    projected = {name: projection(point) for name, point in corners.as_dict().items()}
    u_left = 0.5 * (projected["left-up"][0] + projected["left-down"][0])
    u_right = 0.5 * (projected["right-up"][0] + projected["right-down"][0])
    v_bottom = 0.5 * (projected["left-down"][1] + projected["right-down"][1])
    v_top = 0.5 * (projected["left-up"][1] + projected["right-up"][1])
    width_mm = u_right - u_left
    height_mm = v_top - v_bottom
    if width_mm < min_field_size_mm or height_mm < min_field_size_mm:
        raise ArenaGeometryError(
            f"field is too small or labels are invalid: width={width_mm:.1f}, height={height_mm:.1f}"
        )

    top_width = vec_norm(vec_sub(corners.right_up, corners.left_up))
    bottom_width = vec_norm(vec_sub(corners.right_down, corners.left_down))
    left_height = vec_norm(vec_sub(corners.left_up, corners.left_down))
    right_height = vec_norm(vec_sub(corners.right_up, corners.right_down))
    width_error = abs(top_width - bottom_width) / max(EPSILON, 0.5 * (top_width + bottom_width))
    height_error = abs(left_height - right_height) / max(EPSILON, 0.5 * (left_height + right_height))
    if max(width_error, height_error) > max_opposite_edge_error_ratio:
        raise ArenaGeometryError(
            "opposite field edges differ too much: "
            f"width_error={width_error:.3f}, height_error={height_error:.3f}"
        )

    origin_world = vec_add(center, vec_add(vec_scale(x_axis, u_left), vec_scale(y_axis, v_bottom)))
    expected = {
        "left-up": (0.0, height_mm),
        "left-down": (0.0, 0.0),
        "right-up": (width_mm, height_mm),
        "right-down": (width_mm, 0.0),
    }
    residuals = []
    for name, point in corners.as_dict().items():
        relative = vec_sub(point, origin_world)
        actual = (vec_dot(relative, x_axis), vec_dot(relative, y_axis))
        residuals.append(vec_norm(vec_sub(actual, expected[name])))
    corner_fit_error_mm = max(residuals)
    allowed_error = max(min_corner_fit_error_mm, max_corner_fit_error_ratio * min(width_mm, height_mm))
    if corner_fit_error_mm > allowed_error:
        raise ArenaGeometryError(
            f"corner labels do not fit a rectangle: max residual={corner_fit_error_mm:.1f}mm, "
            f"allowed={allowed_error:.1f}mm"
        )

    return RelativeArena(
        origin_world=origin_world,
        x_axis_world=x_axis,
        y_axis_world=y_axis,
        width_mm=width_mm,
        height_mm=height_mm,
        cage_width_mm=width_mm * cage_width_ratio,
        cage_height_mm=height_mm * cage_height_ratio,
        corner_fit_error_mm=corner_fit_error_mm,
        corner_angle_deg=corner_angle_deg,
    )


def field_boundary_force(
    point: Vec2,
    arena: RelativeArena,
    margin_mm: float,
    weight: float,
) -> Vec2:
    if margin_mm <= 0.0:
        return (0.0, 0.0)

    x_mm, y_mm = point
    closest = (
        clamp(x_mm, 0.0, arena.width_mm),
        clamp(y_mm, 0.0, arena.height_mm),
    )
    outside_offset = vec_sub(closest, point)
    outside_distance = vec_norm(outside_offset)
    if outside_distance > EPSILON:
        scale = weight * min(3.0, 1.0 + outside_distance / margin_mm)
        return vec_scale(vec_normalize(outside_offset), scale)

    total = (0.0, 0.0)
    edges = (
        (x_mm, (1.0, 0.0)),
        (arena.width_mm - x_mm, (-1.0, 0.0)),
        (y_mm, (0.0, 1.0)),
        (arena.height_mm - y_mm, (0.0, -1.0)),
    )
    for distance, inward in edges:
        if distance >= margin_mm:
            continue
        closeness = clamp(1.0 - distance / margin_mm, 0.0, 1.0)
        total = vec_add(total, vec_scale(inward, weight * closeness * closeness))
    return total


def smooth_repulsion(offset: Vec2, radius_mm: float, weight: float) -> Vec2:
    distance = vec_norm(offset)
    if radius_mm <= 0.0 or distance >= radius_mm:
        return (0.0, 0.0)
    if distance <= EPSILON:
        return (weight, 0.0)
    closeness = 1.0 - distance / radius_mm
    return vec_scale(vec_normalize(offset), weight * closeness * closeness)


def cage_exit_force(point: Vec2, arena: RelativeArena, weight: float) -> Vec2:
    if not arena.contains_cage_local(point):
        return (0.0, 0.0)
    distance_to_right = arena.cage_right - point[0]
    distance_to_bottom = point[1] - arena.cage_bottom
    if distance_to_right <= distance_to_bottom:
        return (weight, 0.0)
    return (0.0, -weight)


def cross_points(origin: Vec2, a: Vec2, b: Vec2) -> float:
    return vec_cross(vec_sub(a, origin), vec_sub(b, origin))


def convex_hull_indices(points: Sequence[Vec2]) -> List[int]:
    if len(points) <= 1:
        return list(range(len(points)))

    ordered = sorted(range(len(points)), key=lambda index: (points[index][0], points[index][1], index))
    lower: List[int] = []
    for index in ordered:
        while len(lower) >= 2 and cross_points(points[lower[-2]], points[lower[-1]], points[index]) <= 0.0:
            lower.pop()
        lower.append(index)

    upper: List[int] = []
    for index in reversed(ordered):
        while len(upper) >= 2 and cross_points(points[upper[-2]], points[upper[-1]], points[index]) <= 0.0:
            upper.pop()
        upper.append(index)

    hull = lower[:-1] + upper[:-1]
    result: List[int] = []
    seen_points = set()
    for index in hull:
        point = points[index]
        if point not in seen_points:
            result.append(index)
            seen_points.add(point)
    return result


def point_in_convex_polygon(point: Vec2, polygon: Sequence[Vec2], epsilon: float = 1e-6) -> bool:
    if len(polygon) < 3:
        return False
    sign: Optional[bool] = None
    for index, origin in enumerate(polygon):
        edge_end = polygon[(index + 1) % len(polygon)]
        value = cross_points(origin, edge_end, point)
        if abs(value) <= epsilon:
            continue
        current_sign = value > 0.0
        if sign is None:
            sign = current_sign
        elif sign != current_sign:
            return False
    return sign is not None


def hull_for_ids(ids: Sequence[int], positions: Mapping[int, Vec2]) -> Tuple[Tuple[int, ...], Tuple[Vec2, ...]]:
    visible_ids = [marker_id for marker_id in ids if marker_id in positions]
    points = [positions[marker_id] for marker_id in visible_ids]
    local_indices = convex_hull_indices(points)
    return (
        tuple(visible_ids[index] for index in local_indices),
        tuple(points[index] for index in local_indices),
    )


def augmented_hull(
    herder_ids: Sequence[int],
    positions: Mapping[int, Vec2],
    arena: RelativeArena,
) -> Tuple[Tuple[int, ...], Tuple[Vec2, ...]]:
    hull_herders, hull_points, _index_by_id = augmented_hull_details(
        herder_ids,
        positions,
        arena,
    )
    return hull_herders, hull_points


def augmented_hull_details(
    herder_ids: Sequence[int],
    positions: Mapping[int, Vec2],
    arena: RelativeArena,
) -> Tuple[Tuple[int, ...], Tuple[Vec2, ...], Dict[int, int]]:
    entities: List[Tuple[str, int]] = []
    points: List[Vec2] = []
    for marker_id in herder_ids:
        if marker_id in positions:
            entities.append(("herder", marker_id))
            points.append(positions[marker_id])
    for index, point in enumerate(arena.cage_anchor_points_local()):
        entities.append(("cage", index))
        points.append(point)
    hull_indices = convex_hull_indices(points)
    hull_herders: List[int] = []
    index_by_id: Dict[int, int] = {}
    for hull_index, source_index in enumerate(hull_indices):
        entity_type, entity_id = entities[source_index]
        if entity_type == "herder":
            hull_herders.append(entity_id)
            index_by_id[entity_id] = hull_index
    return (
        tuple(hull_herders),
        tuple(points[index] for index in hull_indices),
        index_by_id,
    )


@dataclass(frozen=True)
class LocalObservation:
    marker_id: int
    position: Vec2
    yaw_world: float
    seen_at: float


@dataclass(frozen=True)
class DynamicsLimits:
    """Physical limits in mm/s, mm/s^2, and 1/s respectively."""

    max_speed: float
    max_acceleration: float
    drag_per_second: float


class SecondOrderDynamics:
    def __init__(self, seed: Optional[int] = None):
        self.velocity_by_id: Dict[int, Vec2] = {}
        self.wander_angle_by_id: Dict[int, float] = {}
        self.wander_elapsed_by_id: Dict[int, float] = {}
        self.formation_target_by_herder: Dict[int, Vec2] = {}
        self.shared_target_evader_id: Optional[int] = None
        self.rng = random.Random(seed)

    def reset(self, marker_id: int) -> None:
        self.velocity_by_id.pop(marker_id, None)
        self.wander_angle_by_id.pop(marker_id, None)
        self.wander_elapsed_by_id.pop(marker_id, None)
        self.formation_target_by_herder.pop(marker_id, None)

    def observe_velocity(
        self,
        marker_id: int,
        velocity_mm_s: Vec2,
        limits: DynamicsLimits,
    ) -> Vec2:
        """Synchronize model state with a measured Vicon velocity."""
        if not all(math.isfinite(component) for component in velocity_mm_s):
            raise ValueError("observed velocity must be finite")
        self._validate_limits(limits)
        observed = vec_limit(velocity_mm_s, limits.max_speed)
        self.velocity_by_id[marker_id] = observed
        return observed

    def set_velocity(
        self,
        marker_id: int,
        velocity_mm_s: Vec2,
        limits: DynamicsLimits,
    ) -> Vec2:
        return self.observe_velocity(marker_id, velocity_mm_s, limits)

    @staticmethod
    def _validate_limits(limits: DynamicsLimits) -> None:
        if limits.max_speed <= 0.0 or limits.max_acceleration <= 0.0:
            raise ValueError("speed and acceleration limits must be positive")
        if limits.drag_per_second < 0.0:
            raise ValueError("drag_per_second must be non-negative")

    def step(
        self,
        marker_id: int,
        acceleration: Vec2,
        dt: float,
        limits: DynamicsLimits,
    ) -> Vec2:
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        self._validate_limits(limits)
        dt = min(dt, 0.25)
        previous = self.velocity_by_id.get(marker_id, (0.0, 0.0))
        used_acceleration = vec_limit(acceleration, limits.max_acceleration)
        drag = math.exp(-limits.drag_per_second * dt)
        acceleration_dt = (
            dt
            if limits.drag_per_second <= EPSILON
            else (1.0 - drag) / limits.drag_per_second
        )
        velocity = vec_add(
            vec_scale(previous, drag),
            vec_scale(used_acceleration, acceleration_dt),
        )
        velocity = vec_limit(velocity, limits.max_speed)
        maximum_delta = limits.max_acceleration * dt
        velocity = vec_add(previous, vec_limit(vec_sub(velocity, previous), maximum_delta))
        velocity = vec_limit(velocity, limits.max_speed)
        if vec_norm(velocity) < 1e-6:
            velocity = (0.0, 0.0)
        self.velocity_by_id[marker_id] = velocity
        return velocity

    def wander_vector(self, marker_id: int, dt: float, change_interval_sec: float = 1.8) -> Vec2:
        elapsed = self.wander_elapsed_by_id.get(marker_id, change_interval_sec) + max(0.0, dt)
        angle = self.wander_angle_by_id.get(marker_id)
        if angle is None or elapsed >= change_interval_sec:
            if angle is None:
                angle = self.rng.uniform(-math.pi, math.pi)
            else:
                angle += self.rng.uniform(-0.75, 0.75)
            elapsed = 0.0
        self.wander_angle_by_id[marker_id] = angle
        self.wander_elapsed_by_id[marker_id] = elapsed
        return (math.cos(angle), math.sin(angle))


@dataclass(frozen=True)
class ControlParameters:
    """Distances are millimeters and force weights are accelerations in mm/s^2."""

    boundary_margin_mm: float
    boundary_weight: float
    herder_repel_radius_mm: float
    herder_repel_weight: float
    evader_keepout_radius_mm: float
    evader_keepout_weight: float
    herder_standoff_mm: float
    herder_formation_weight: float
    herder_hunt_weight: float
    herder_arc_span_rad: float
    evader_neighbor_radius_mm: float
    evader_threat_radius_mm: float
    evader_dispersion_weight: float
    evader_escape_weight: float
    evader_aggregation_weight: float
    evader_wander_weight: float
    cage_center_weight: float
    cage_hold_margin_mm: float
    robot_radius_mm: float
    obstacle_radius_mm: float
    obstacle_weight: float
    herder_dynamics: DynamicsLimits
    evader_dynamics: DynamicsLimits

    @classmethod
    def from_arena(
        cls,
        arena: RelativeArena,
        robot_radius_mm: float = 150.0,
    ) -> "ControlParameters":
        if robot_radius_mm <= 0.0:
            raise ValueError("robot_radius_mm must be positive")
        scale = arena.scale_mm
        body_clearance = 2.0 * robot_radius_mm + 40.0
        return cls(
            boundary_margin_mm=max(0.06 * scale, robot_radius_mm + 40.0),
            boundary_weight=0.22 * scale,
            herder_repel_radius_mm=max(0.11 * scale, body_clearance),
            herder_repel_weight=0.20 * scale,
            evader_keepout_radius_mm=max(0.07 * scale, body_clearance),
            evader_keepout_weight=0.22 * scale,
            herder_standoff_mm=max(0.13 * scale, body_clearance + 40.0),
            herder_formation_weight=0.18 * scale,
            herder_hunt_weight=0.10 * scale,
            herder_arc_span_rad=math.radians(190.0),
            evader_neighbor_radius_mm=max(0.16 * scale, body_clearance),
            evader_threat_radius_mm=0.22 * scale,
            evader_dispersion_weight=0.115 * scale,
            evader_escape_weight=0.28 * scale,
            evader_aggregation_weight=0.035 * scale,
            evader_wander_weight=0.018 * scale,
            cage_center_weight=0.14 * scale,
            cage_hold_margin_mm=max(
                robot_radius_mm + 20.0,
                0.06 * min(arena.cage_width_mm, arena.cage_height_mm),
            ),
            robot_radius_mm=robot_radius_mm,
            obstacle_radius_mm=max(0.09 * scale, body_clearance),
            obstacle_weight=0.24 * scale,
            herder_dynamics=DynamicsLimits(
                max_speed=0.08 * scale,
                max_acceleration=0.24 * scale,
                drag_per_second=1.0,
            ),
            evader_dynamics=DynamicsLimits(
                max_speed=0.11 * scale,
                max_acceleration=0.30 * scale,
                drag_per_second=0.35,
            ),
        )


@dataclass(frozen=True)
class ControlResult:
    velocities: Dict[int, Vec2]
    accelerations: Dict[int, Vec2]
    targets: Dict[int, Vec2]
    target_evader_by_herder: Dict[int, Optional[int]]
    visible_herders: Tuple[int, ...]
    visible_evaders: Tuple[int, ...]
    herder_hull_ids: Tuple[int, ...]
    augmented_hull_herder_ids: Tuple[int, ...]
    contained_evaders: Tuple[int, ...]
    inside_cage_evaders: Tuple[int, ...]
    outside_cage_evaders: Tuple[int, ...]
    missing_herders: Tuple[int, ...]
    missing_evaders: Tuple[int, ...]


def pair_repulsion(
    marker_id: int,
    peer_ids: Sequence[int],
    positions: Mapping[int, Vec2],
    radius_mm: float,
    weight: float,
) -> Vec2:
    position = positions.get(marker_id)
    if position is None:
        return (0.0, 0.0)
    total = (0.0, 0.0)
    for other_id in peer_ids:
        if other_id == marker_id or other_id not in positions:
            continue
        offset = vec_sub(position, positions[other_id])
        if vec_norm(offset) <= EPSILON:
            offset = (1.0, 0.0) if marker_id < other_id else (-1.0, 0.0)
        total = vec_add(
            total,
            smooth_repulsion(offset, radius_mm, weight),
        )
    return total


def obstacle_repulsion(
    position: Vec2,
    obstacle_ids: Sequence[int],
    positions: Mapping[int, Vec2],
    radius_mm: float,
    weight: float,
) -> Vec2:
    total = (0.0, 0.0)
    for obstacle_id in obstacle_ids:
        obstacle = positions.get(obstacle_id)
        if obstacle is None:
            continue
        total = vec_add(total, smooth_repulsion(vec_sub(position, obstacle), radius_mm, weight))
    return total


def select_shared_target_evader(
    outside_evaders: Sequence[int],
    contained_evaders: Sequence[int],
    positions: Mapping[int, Vec2],
    cage_center: Vec2,
) -> Optional[int]:
    visible_outside = [marker_id for marker_id in outside_evaders if marker_id in positions]
    if not visible_outside:
        return None
    contained = set(contained_evaders)
    candidates = [marker_id for marker_id in visible_outside if marker_id not in contained]
    if not candidates:
        candidates = visible_outside
    return max(
        candidates,
        key=lambda marker_id: (
            vec_norm(vec_sub(positions[marker_id], cage_center)),
            -marker_id,
        ),
    )


def minimum_cost_assignment(costs: Sequence[Sequence[float]]) -> Tuple[int, ...]:
    """Return one distinct column per row using the rectangular Hungarian algorithm."""
    row_count = len(costs)
    if row_count == 0:
        return ()
    column_count = len(costs[0])
    if column_count < row_count or any(len(row) != column_count for row in costs):
        raise ValueError("assignment matrix must be rectangular with at least as many columns as rows")

    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)

    for row_index in range(1, row_count + 1):
        matched_row[0] = row_index
        current_column = 0
        minimum = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[current_column] = True
            current_row = matched_row[current_column]
            delta = float("inf")
            next_column = 0
            for column_index in range(1, column_count + 1):
                if used[column_index]:
                    continue
                reduced = (
                    costs[current_row - 1][column_index - 1]
                    - row_potential[current_row]
                    - column_potential[column_index]
                )
                if reduced < minimum[column_index]:
                    minimum[column_index] = reduced
                    previous_column[column_index] = current_column
                if minimum[column_index] < delta:
                    delta = minimum[column_index]
                    next_column = column_index
            for column_index in range(column_count + 1):
                if used[column_index]:
                    row_potential[matched_row[column_index]] += delta
                    column_potential[column_index] -= delta
                else:
                    minimum[column_index] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break

        while True:
            next_column = previous_column[current_column]
            matched_row[current_column] = matched_row[next_column]
            current_column = next_column
            if current_column == 0:
                break

    assignment = [-1] * row_count
    for column_index in range(1, column_count + 1):
        if matched_row[column_index] != 0:
            assignment[matched_row[column_index] - 1] = column_index - 1
    if any(column_index < 0 for column_index in assignment):
        raise ValueError("could not construct a complete assignment")
    return tuple(assignment)


def safe_axis_grid(low: float, high: float, spacing: float) -> Tuple[float, ...]:
    if spacing <= 0.0 or high < low:
        return ()
    count = int(math.floor((high - low) / spacing)) + 1
    used = (count - 1) * spacing
    offset = 0.5 * ((high - low) - used)
    return tuple(low + offset + index * spacing for index in range(count))


def formation_targets(
    herder_ids: Sequence[int],
    positions: Mapping[int, Vec2],
    evader_ids: Sequence[int],
    arena: RelativeArena,
    params: ControlParameters,
    previous_targets: Optional[Mapping[int, Vec2]] = None,
) -> Dict[int, Vec2]:
    visible_herders = [marker_id for marker_id in herder_ids if marker_id in positions]
    visible_evaders = [marker_id for marker_id in evader_ids if marker_id in positions]
    if not visible_herders or not visible_evaders:
        return {}

    center = positions[visible_evaders[0]]
    radius = params.herder_standoff_mm
    direction_to_cage = vec_normalize(vec_sub(arena.cage_center_local, center))
    if vec_norm(direction_to_cage) <= EPSILON:
        direction_to_cage = (-1.0, 1.0)
        direction_to_cage = vec_normalize(direction_to_cage)
    rear_angle = math.atan2(-direction_to_cage[1], -direction_to_cage[0])

    if len(visible_herders) == 1:
        slot_angles = [rear_angle]
    else:
        start = rear_angle - params.herder_arc_span_rad * 0.5
        step = params.herder_arc_span_rad / (len(visible_herders) - 1)
        slot_angles = [start + index * step for index in range(len(visible_herders))]

    safety_margin = min(
        max(params.boundary_margin_mm, params.cage_hold_margin_mm),
        0.49 * min(arena.width_mm, arena.height_mm),
    )
    nominal_slots = tuple(
        (
            clamp(
                center[0] + radius * math.cos(angle),
                safety_margin,
                arena.width_mm - safety_margin,
            ),
            clamp(
                center[1] + radius * math.sin(angle),
                safety_margin,
                arena.height_mm - safety_margin,
            ),
        )
        for angle in slot_angles
    )

    herder_tuple = tuple(visible_herders)
    minimum_spacing = 2.0 * params.robot_radius_mm + 40.0
    previous_targets = previous_targets or {}
    previous_slots = set(previous_targets.values())
    slot_switch_penalty = 0.55 * minimum_spacing
    x_grid = safe_axis_grid(safety_margin, arena.width_mm - safety_margin, minimum_spacing)
    y_grid = safe_axis_grid(safety_margin, arena.height_mm - safety_margin, minimum_spacing)
    candidates = tuple(
        (x_mm, y_mm)
        for x_mm in x_grid
        for y_mm in y_grid
        if not arena.contains_cage_local(
            (x_mm, y_mm),
            margin_mm=-params.robot_radius_mm,
        )
    )
    if len(candidates) < len(herder_tuple):
        raise ValueError(
            "field has too little collision-free space for the active herder formation"
        )

    slot_candidate_indices = minimum_cost_assignment(
        tuple(
            tuple(
                vec_norm(vec_sub(nominal_slot, candidate))
                + (0.0 if candidate in previous_slots else slot_switch_penalty)
                for candidate in candidates
            )
            for nominal_slot in nominal_slots
        )
    )
    slots = tuple(candidates[index] for index in slot_candidate_indices)
    assignment = minimum_cost_assignment(
        tuple(
            tuple(
                vec_norm(vec_sub(positions[marker_id], slot))
                + (
                    0.0
                    if previous_targets.get(marker_id) == slot
                    else 0.35 * minimum_spacing
                )
                for slot in slots
            )
            for marker_id in herder_tuple
        )
    )

    return {
        marker_id: slots[assignment[index]]
        for index, marker_id in enumerate(herder_tuple)
    }


def hull_gap_surround_force(
    marker_id: int,
    hull_herder_ids: Sequence[int],
    hull_points: Sequence[Vec2],
    positions: Mapping[int, Vec2],
    cage_center: Vec2,
    maximum_force: float,
    hull_index_by_herder: Optional[Mapping[int, int]] = None,
) -> Vec2:
    if marker_id not in hull_herder_ids or marker_id not in positions or len(hull_points) < 3:
        return (0.0, 0.0)
    marker_position = positions[marker_id]
    if hull_index_by_herder is not None:
        index = hull_index_by_herder.get(marker_id)
        if index is None or not 0 <= index < len(hull_points):
            return (0.0, 0.0)
    else:
        try:
            index = next(
                index
                for index, point in enumerate(hull_points)
                if vec_norm(vec_sub(point, marker_position)) <= EPSILON
            )
        except StopIteration:
            return (0.0, 0.0)

    previous = hull_points[(index - 1) % len(hull_points)]
    following = hull_points[(index + 1) % len(hull_points)]
    current_angle = math.atan2(
        marker_position[1] - cage_center[1],
        marker_position[0] - cage_center[0],
    )
    previous_angle = math.atan2(
        previous[1] - cage_center[1],
        previous[0] - cage_center[0],
    )
    following_angle = math.atan2(
        following[1] - cage_center[1],
        following[0] - cage_center[0],
    )
    previous_gap = (current_angle - previous_angle) % (2.0 * math.pi)
    following_gap = (following_angle - current_angle) % (2.0 * math.pi)
    imbalance = clamp((following_gap - previous_gap) / math.pi, -1.0, 1.0)
    tangent_ccw = (-math.sin(current_angle), math.cos(current_angle))
    return vec_scale(tangent_ccw, max(0.0, maximum_force) * imbalance)


def select_target_evader(
    herder_id: int,
    outside_evaders: Sequence[int],
    positions: Mapping[int, Vec2],
    cage_center: Vec2,
) -> Optional[int]:
    if herder_id not in positions or not outside_evaders:
        return None
    herder_position = positions[herder_id]
    return min(
        outside_evaders,
        key=lambda evader_id: (
            vec_norm(vec_sub(positions[evader_id], herder_position))
            + 0.20 * vec_norm(vec_sub(positions[evader_id], cage_center))
        ),
    )


def evader_acceleration(
    evader_id: int,
    active_herders: Sequence[int],
    active_evaders: Sequence[int],
    obstacle_ids: Sequence[int],
    positions: Mapping[int, Vec2],
    arena: RelativeArena,
    params: ControlParameters,
    dynamics: SecondOrderDynamics,
    dt: float,
) -> Vec2:
    evader_position = positions[evader_id]
    boundary = field_boundary_force(
        evader_position,
        arena,
        params.boundary_margin_mm,
        params.boundary_weight,
    )
    separation = pair_repulsion(
        evader_id,
        active_evaders,
        positions,
        params.evader_neighbor_radius_mm,
        params.evader_dispersion_weight,
    )
    obstacles = obstacle_repulsion(
        evader_position,
        obstacle_ids,
        positions,
        params.obstacle_radius_mm,
        params.obstacle_weight,
    )

    if arena.contains_cage_local(evader_position, params.cage_hold_margin_mm):
        dynamics.reset(evader_id)
        return (0.0, 0.0)
    if arena.contains_cage_local(evader_position):
        to_center = vec_normalize(vec_sub(arena.cage_center_local, evader_position))
        return vec_limit(
            vec_add(vec_add(boundary, separation), vec_scale(to_center, params.cage_center_weight)),
            params.evader_dynamics.max_acceleration,
        )

    escape = (0.0, 0.0)
    threatened = False
    for herder_id in active_herders:
        herder_position = positions.get(herder_id)
        if herder_position is None:
            continue
        offset = vec_sub(evader_position, herder_position)
        if vec_norm(offset) < params.evader_threat_radius_mm:
            threatened = True
        escape = vec_add(
            escape,
            smooth_repulsion(offset, params.evader_threat_radius_mm, params.evader_escape_weight),
        )

    aggregation = (0.0, 0.0)
    neighbors = [
        other_id
        for other_id in active_evaders
        if other_id != evader_id
        and other_id in positions
        and vec_norm(vec_sub(positions[other_id], evader_position)) < params.evader_neighbor_radius_mm
    ]
    if threatened and neighbors:
        neighbor_center = vec_mean(positions[other_id] for other_id in neighbors)
        aggregation = vec_scale(
            vec_normalize(vec_sub(neighbor_center, evader_position)),
            params.evader_aggregation_weight,
        )

    wander = vec_scale(dynamics.wander_vector(evader_id, dt), params.evader_wander_weight)
    total = boundary
    for component in (separation, obstacles, escape, aggregation, wander):
        total = vec_add(total, component)
    return vec_limit(total, params.evader_dynamics.max_acceleration)


def compute_control_step(
    observations: Mapping[int, LocalObservation],
    active_herders: Sequence[int],
    active_evaders: Sequence[int],
    obstacle_ids: Sequence[int],
    arena: RelativeArena,
    params: ControlParameters,
    dynamics: SecondOrderDynamics,
    dt: float,
    drive_evaders: bool = True,
) -> ControlResult:
    positions = {marker_id: observation.position for marker_id, observation in observations.items()}
    visible_herders = tuple(marker_id for marker_id in active_herders if marker_id in positions)
    visible_evaders = tuple(marker_id for marker_id in active_evaders if marker_id in positions)
    missing_herders = tuple(marker_id for marker_id in active_herders if marker_id not in positions)
    missing_evaders = tuple(marker_id for marker_id in active_evaders if marker_id not in positions)
    inside_cage = tuple(
        marker_id
        for marker_id in visible_evaders
        if arena.contains_cage_local(positions[marker_id], params.cage_hold_margin_mm)
    )
    outside_cage = tuple(marker_id for marker_id in visible_evaders if marker_id not in inside_cage)
    herder_hull_ids, _herder_hull_points = hull_for_ids(visible_herders, positions)
    augmented_hull_ids, augmented_hull_points, augmented_hull_index_by_id = augmented_hull_details(
        visible_herders,
        positions,
        arena,
    )
    contained_evaders = tuple(
        marker_id
        for marker_id in outside_cage
        if point_in_convex_polygon(positions[marker_id], augmented_hull_points)
    )

    accelerations: Dict[int, Vec2] = {}
    velocities: Dict[int, Vec2] = {}
    previous_shared_target = dynamics.shared_target_evader_id
    shared_target_evader = previous_shared_target
    if shared_target_evader not in outside_cage:
        shared_target_evader = select_shared_target_evader(
            outside_cage,
            contained_evaders,
            positions,
            arena.cage_center_local,
        )
        dynamics.shared_target_evader_id = shared_target_evader
    if shared_target_evader != previous_shared_target:
        dynamics.formation_target_by_herder.clear()
    target_ids = () if shared_target_evader is None else (shared_target_evader,)
    targets = formation_targets(
        visible_herders,
        positions,
        target_ids,
        arena,
        params,
        dynamics.formation_target_by_herder,
    )
    dynamics.formation_target_by_herder = dict(targets)
    target_evader_by_herder: Dict[int, Optional[int]] = {
        marker_id: None for marker_id in active_herders
    }

    if len(visible_herders) >= 3 and outside_cage:
        for herder_id in visible_herders:
            herder_position = positions[herder_id]
            acceleration = field_boundary_force(
                herder_position,
                arena,
                params.boundary_margin_mm,
                params.boundary_weight,
            )
            acceleration = vec_add(
                acceleration,
                pair_repulsion(
                    herder_id,
                    visible_herders,
                    positions,
                    params.herder_repel_radius_mm,
                    params.herder_repel_weight,
                ),
            )
            acceleration = vec_add(
                acceleration,
                obstacle_repulsion(
                    herder_position,
                    obstacle_ids,
                    positions,
                    params.obstacle_radius_mm,
                    params.obstacle_weight,
                ),
            )
            acceleration = vec_add(
                acceleration,
                cage_exit_force(herder_position, arena, params.evader_keepout_weight),
            )
            acceleration = vec_add(
                acceleration,
                hull_gap_surround_force(
                    herder_id,
                    augmented_hull_ids,
                    augmented_hull_points,
                    positions,
                    arena.cage_center_local,
                    0.35 * params.herder_formation_weight,
                    augmented_hull_index_by_id,
                ),
            )

            formation_target = targets.get(herder_id)
            if formation_target is not None:
                acceleration = vec_add(
                    acceleration,
                    vec_scale(
                        vec_normalize(vec_sub(formation_target, herder_position)),
                        params.herder_formation_weight,
                    ),
                )

            target_evader = shared_target_evader
            target_evader_by_herder[herder_id] = target_evader
            if target_evader is not None:
                evader_position = positions[target_evader]
                away_from_cage = vec_normalize(vec_sub(evader_position, arena.cage_center_local))
                push_point = vec_add(
                    evader_position,
                    vec_scale(away_from_cage, 0.70 * params.herder_standoff_mm),
                )
                acceleration = vec_add(
                    acceleration,
                    vec_scale(
                        vec_normalize(vec_sub(push_point, herder_position)),
                        params.herder_hunt_weight,
                    ),
                )

            for evader_id in visible_evaders:
                acceleration = vec_add(
                    acceleration,
                    smooth_repulsion(
                        vec_sub(herder_position, positions[evader_id]),
                        params.evader_keepout_radius_mm,
                        params.evader_keepout_weight,
                    ),
                )

            acceleration = vec_limit(acceleration, params.herder_dynamics.max_acceleration)
            accelerations[herder_id] = acceleration
            velocities[herder_id] = dynamics.step(
                herder_id,
                acceleration,
                dt,
                params.herder_dynamics,
            )
    else:
        for herder_id in active_herders:
            dynamics.reset(herder_id)
            velocities[herder_id] = (0.0, 0.0)
            accelerations[herder_id] = (0.0, 0.0)

    for evader_id in active_evaders:
        if evader_id not in positions or not drive_evaders:
            dynamics.reset(evader_id)
            velocities[evader_id] = (0.0, 0.0)
            accelerations[evader_id] = (0.0, 0.0)
            continue
        acceleration = evader_acceleration(
            evader_id,
            visible_herders,
            active_evaders,
            obstacle_ids,
            positions,
            arena,
            params,
            dynamics,
            dt,
        )
        accelerations[evader_id] = acceleration
        if vec_norm(acceleration) <= EPSILON and arena.contains_cage_local(
            positions[evader_id], params.cage_hold_margin_mm
        ):
            velocities[evader_id] = (0.0, 0.0)
        else:
            velocities[evader_id] = dynamics.step(
                evader_id,
                acceleration,
                dt,
                params.evader_dynamics,
            )

    return ControlResult(
        velocities=velocities,
        accelerations=accelerations,
        targets=targets,
        target_evader_by_herder=target_evader_by_herder,
        visible_herders=visible_herders,
        visible_evaders=visible_evaders,
        herder_hull_ids=herder_hull_ids,
        augmented_hull_herder_ids=augmented_hull_ids,
        contained_evaders=contained_evaders,
        inside_cage_evaders=inside_cage,
        outside_cage_evaders=outside_cage,
        missing_herders=missing_herders,
        missing_evaders=missing_evaders,
    )


@dataclass
class CaptureMonitor:
    hold_sec: float = 2.0
    started_at: Optional[float] = None
    confirmed: bool = False

    def reset(self) -> None:
        self.started_at = None
        self.confirmed = False

    def update(
        self,
        now: float,
        active_evaders: Sequence[int],
        visible_evaders: Sequence[int],
        inside_cage_evaders: Sequence[int],
    ) -> bool:
        active_set = set(active_evaders)
        ready = bool(active_set) and set(visible_evaders) == active_set and set(inside_cage_evaders) == active_set
        if not ready:
            self.reset()
            return False
        if self.started_at is None:
            self.started_at = now
        self.confirmed = now - self.started_at >= self.hold_sec
        return self.confirmed


def active_role_ids(
    requested_herders: Sequence[int],
    requested_evaders: Sequence[int],
    usable_ids: Iterable[int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    usable = set(usable_ids)
    return (
        tuple(marker_id for marker_id in requested_herders if marker_id in usable),
        tuple(marker_id for marker_id in requested_evaders if marker_id in usable),
    )
