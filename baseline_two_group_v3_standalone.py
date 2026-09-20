"""Standalone fixed-two-group V3 herding baseline.

This file is a deployment-oriented consolidation of:

* baseline_two_group.py
* legacy_baseline_safety_v2.py
* herding_safety_v2.py
* target_region_v3.py
* baseline_two_group_target_region_v3.py

It keeps the current simulation protocol while avoiding local-module imports.
Headless execution is the default and performs no Matplotlib pause/show calls.
Use ``--render`` for the interactive visualization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import numbers
import random
import time
from typing import Any, Sequence

import numpy as np
from scipy.spatial import ConvexHull


PHYSICS_VARIANT = "crowding_target_region_v3"
PROTOCOL_VERSION = 3
METHOD_NAME = "baseline_two_group_target_region_v3"


@dataclass(frozen=True)
class DensitySlowdownState:
    visible_count: int
    sector_area: float
    observed_density: float
    start_density: float
    full_density: float
    activation: float
    speed_limit: float


@dataclass(frozen=True)
class OverlapResolution:
    pair_correction_operations: int
    iterations: int
    min_distance_before: float
    min_distance_after: float
    max_penetration_before: float
    max_penetration_after: float


def validate_safety_parameters(
    *,
    evader_radius: float,
    slowdown_start_count: float,
    slowdown_full_count: float,
    min_speed_factor: float,
    collision_iterations: int,
    separation_margin: float,
) -> None:
    if not math.isfinite(evader_radius) or evader_radius <= 0:
        raise ValueError("evader_radius must be finite and positive.")
    if not math.isfinite(slowdown_start_count) or slowdown_start_count < 0:
        raise ValueError("density_slowdown_start_count must be finite and non-negative.")
    if not math.isfinite(slowdown_full_count) or slowdown_full_count <= slowdown_start_count:
        raise ValueError(
            "density_slowdown_full_count must be finite and greater than "
            "density_slowdown_start_count."
        )
    if not math.isfinite(min_speed_factor) or not 0 < min_speed_factor <= 1:
        raise ValueError("min_speed_factor must be in (0, 1].")
    if (
        isinstance(collision_iterations, bool)
        or not isinstance(collision_iterations, numbers.Integral)
        or collision_iterations < 1
    ):
        raise ValueError("collision_resolution_iterations must be a positive integer.")
    if not math.isfinite(separation_margin) or separation_margin < 0:
        raise ValueError("collision_separation_margin must be finite and non-negative.")


def density_slowdown_state(
    *,
    visible_count: int,
    sensing_radius: float,
    fov_radians: float,
    slowdown_start_count: float,
    slowdown_full_count: float,
    base_speed_limit: float,
    min_speed_factor: float,
) -> DensitySlowdownState:
    """Compute a dimensionally consistent density-dependent speed limit."""
    if sensing_radius <= 0 or fov_radians <= 0:
        raise ValueError("sensing_radius and fov_radians must be positive.")
    if slowdown_full_count <= slowdown_start_count:
        raise ValueError("slowdown_full_count must be greater than slowdown_start_count.")
    if base_speed_limit <= 0:
        raise ValueError("base_speed_limit must be positive.")
    if not 0 < min_speed_factor <= 1:
        raise ValueError("min_speed_factor must be in (0, 1].")

    sector_area = 0.5 * float(sensing_radius) ** 2 * float(fov_radians)
    observed_density = max(0, int(visible_count)) / sector_area
    start_density = float(slowdown_start_count) / sector_area
    full_density = float(slowdown_full_count) / sector_area
    activation = float(
        np.clip(
            (observed_density - start_density) / (full_density - start_density),
            0.0,
            1.0,
        )
    )
    speed_factor = 1.0 - activation * (1.0 - float(min_speed_factor))
    return DensitySlowdownState(
        visible_count=max(0, int(visible_count)),
        sector_area=sector_area,
        observed_density=observed_density,
        start_density=start_density,
        full_density=full_density,
        activation=activation,
        speed_limit=float(base_speed_limit) * speed_factor,
    )


@lru_cache(maxsize=None)
def _pair_indices(count: int) -> tuple[np.ndarray, np.ndarray]:
    first, second = np.triu_indices(count, k=1)
    first.setflags(write=False)
    second.setflags(write=False)
    return first, second


def _pairwise_min_and_pairs(
    positions: np.ndarray,
    threshold: float,
) -> tuple[float, np.ndarray]:
    if len(positions) < 2:
        return float("inf"), np.empty((0, 2), dtype=np.int64)
    positions64 = np.asarray(positions, dtype=np.float64)
    first, second = _pair_indices(len(positions64))
    relative = positions64[first] - positions64[second]
    distances_sq = np.sum(relative * relative, axis=1)
    minimum = float(math.sqrt(float(np.min(distances_sq))))
    selected = distances_sq < float(threshold) ** 2
    pairs = np.column_stack((first[selected], second[selected])).astype(
        np.int64,
        copy=False,
    )
    return minimum, pairs


def pairwise_min_distance(positions: np.ndarray) -> float:
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2).")
    minimum, _ = _pairwise_min_and_pairs(positions, threshold=0.0)
    return minimum


def count_overlapping_pairs(
    positions: np.ndarray,
    evader_radius: float,
    tolerance: float = 1e-6,
) -> int:
    required = max(0.0, 2.0 * float(evader_radius) - float(tolerance))
    _, pairs = _pairwise_min_and_pairs(np.asarray(positions), threshold=required)
    return int(len(pairs))


def _deterministic_pair_normal(first: int, second: int) -> np.ndarray:
    phase = ((first + 1) * 0.754877666 + (second + 1) * 0.569840296) % 1.0
    angle = 2.0 * math.pi * phase
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)


def _directional_capacity(
    position: np.ndarray,
    direction: np.ndarray,
    lower: float,
    upper: float,
) -> float:
    capacity = float("inf")
    for coordinate, component in zip(position, direction):
        if component > 1e-15:
            capacity = min(capacity, (upper - coordinate) / component)
        elif component < -1e-15:
            capacity = min(capacity, (lower - coordinate) / component)
    return max(0.0, float(capacity))


def _candidate_separation_directions(
    relative: np.ndarray,
    first: int,
    second: int,
    midpoint: np.ndarray,
    arena_center: np.ndarray,
    tolerance: float,
) -> list[np.ndarray]:
    if np.linalg.norm(relative) > tolerance:
        primary = relative / np.linalg.norm(relative)
    else:
        primary = _deterministic_pair_normal(first, second)
    perpendicular = np.array([-primary[1], primary[0]], dtype=np.float64)
    toward_center = arena_center - midpoint
    if np.linalg.norm(toward_center) > tolerance:
        toward_center = toward_center / np.linalg.norm(toward_center)
    else:
        toward_center = np.array([1.0, 0.0], dtype=np.float64)
    return [
        primary,
        -primary,
        perpendicular,
        -perpendicular,
        toward_center,
        -toward_center,
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([-1.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([0.0, -1.0], dtype=np.float64),
    ]


def _required_combined_move(
    relative: np.ndarray,
    direction: np.ndarray,
    target_distance: float,
) -> float:
    projection = float(np.dot(relative, direction))
    distance_sq = float(np.dot(relative, relative))
    discriminant = max(
        0.0,
        projection * projection + target_distance**2 - distance_sq,
    )
    return max(0.0, -projection + math.sqrt(discriminant))


def _choose_separation_direction(
    *,
    relative: np.ndarray,
    first: int,
    second: int,
    first_position: np.ndarray,
    second_position: np.ndarray,
    lower: float,
    upper: float,
    target_distance: float,
    tolerance: float,
) -> tuple[np.ndarray, float, float, float]:
    arena_center = np.array([(lower + upper) / 2.0] * 2, dtype=np.float64)
    candidates = _candidate_separation_directions(
        relative=relative,
        first=first,
        second=second,
        midpoint=0.5 * (first_position + second_position),
        arena_center=arena_center,
        tolerance=tolerance,
    )
    choices = []
    for order, direction in enumerate(candidates):
        required_move = _required_combined_move(relative, direction, target_distance)
        first_capacity = _directional_capacity(first_position, direction, lower, upper)
        second_capacity = _directional_capacity(second_position, -direction, lower, upper)
        total_capacity = first_capacity + second_capacity
        feasible = total_capacity + tolerance >= required_move
        choices.append(
            (
                0 if feasible else 1,
                required_move if feasible else -total_capacity,
                order,
                direction,
                required_move,
                first_capacity,
                second_capacity,
            )
        )
    _, _, _, direction, required_move, first_capacity, second_capacity = min(
        choices,
        key=lambda item: item[:3],
    )
    return direction, required_move, first_capacity, second_capacity


def resolve_disk_overlaps(
    positions: np.ndarray,
    *,
    evader_radius: float,
    arena_size: float,
    max_iterations: int = 512,
    separation_margin: float = 1e-4,
    strict: bool = True,
    tolerance: float = 1e-6,
) -> OverlapResolution:
    """Deterministically project disk centers to a non-overlapping state."""
    positions_array = np.asarray(positions)
    if positions_array.ndim != 2 or positions_array.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2).")
    if (
        not np.issubdtype(positions_array.dtype, np.floating)
        or positions_array.dtype.itemsize < 4
    ):
        raise TypeError("positions must use a floating dtype with at least float32 precision.")
    if not np.all(np.isfinite(positions_array)):
        raise ValueError("positions contain non-finite values.")
    if not math.isfinite(evader_radius) or evader_radius <= 0:
        raise ValueError("evader_radius must be finite and positive.")
    if not math.isfinite(arena_size) or arena_size <= 2.0 * evader_radius:
        raise ValueError("arena_size must be greater than the evader diameter.")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, numbers.Integral)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer.")
    if not math.isfinite(separation_margin) or separation_margin < 0:
        raise ValueError("separation_margin must be finite and non-negative.")
    if not math.isfinite(tolerance) or tolerance < 0 or tolerance >= 2.0 * evader_radius:
        raise ValueError("tolerance must be finite and in [0, evader diameter).")

    exact_lower = float(evader_radius)
    exact_upper = float(arena_size) - float(evader_radius)
    dtype = positions_array.dtype
    lower_value = dtype.type(exact_lower)
    if float(lower_value) < exact_lower:
        lower_value = np.nextafter(lower_value, dtype.type(np.inf), dtype=dtype)
    upper_value = dtype.type(exact_upper)
    if float(upper_value) > exact_upper:
        upper_value = np.nextafter(upper_value, dtype.type(-np.inf), dtype=dtype)
    lower = float(lower_value)
    upper = float(upper_value)
    required_distance = 2.0 * float(evader_radius)
    rounding_guard = (
        np.finfo(positions_array.dtype).eps
        * max(1.0, abs(float(arena_size)), abs(required_distance))
        * 8.0
    )
    target_distance = (
        required_distance
        + max(float(separation_margin), float(rounding_guard))
        + float(tolerance)
    )
    working = positions_array.astype(np.float64, copy=True)
    np.clip(working, lower, upper, out=working)

    min_before, pairs = _pairwise_min_and_pairs(
        working,
        threshold=target_distance - tolerance,
    )
    max_penetration_before = (
        0.0 if math.isinf(min_before) else max(0.0, required_distance - min_before)
    )
    pair_correction_operations = 0
    completed_iterations = 0

    for iteration in range(1, int(max_iterations) + 1):
        if len(pairs) == 0:
            break
        completed_iterations = iteration
        for first, second in pairs:
            relative = working[first] - working[second]
            distance = float(np.linalg.norm(relative))
            if distance >= target_distance - tolerance:
                continue
            if distance > tolerance:
                normal = relative / distance
                correction_needed = target_distance - distance
                first_capacity = _directional_capacity(
                    working[first], normal, lower, upper
                )
                second_capacity = _directional_capacity(
                    working[second], -normal, lower, upper
                )
            else:
                correction_needed = float("inf")
                first_capacity = 0.0
                second_capacity = 0.0
            if first_capacity + second_capacity + tolerance < correction_needed:
                (
                    normal,
                    correction_needed,
                    first_capacity,
                    second_capacity,
                ) = _choose_separation_direction(
                    relative=relative,
                    first=int(first),
                    second=int(second),
                    first_position=working[first],
                    second_position=working[second],
                    lower=lower,
                    upper=upper,
                    target_distance=target_distance,
                    tolerance=tolerance,
                )

            first_move = min(0.5 * correction_needed, first_capacity)
            second_move = min(0.5 * correction_needed, second_capacity)
            remaining = correction_needed - first_move - second_move
            if remaining > 0:
                extra_first = min(remaining, first_capacity - first_move)
                first_move += extra_first
                remaining -= extra_first
            if remaining > 0:
                extra_second = min(remaining, second_capacity - second_move)
                second_move += extra_second

            working[first] += normal * first_move
            working[second] -= normal * second_move
            np.clip(working[first], lower, upper, out=working[first])
            np.clip(working[second], lower, upper, out=working[second])
            pair_correction_operations += 1

        if iteration < int(max_iterations):
            _, pairs = _pairwise_min_and_pairs(
                working,
                threshold=target_distance - tolerance,
            )

    cast_working = working.astype(positions_array.dtype, copy=False)
    if pair_correction_operations == 0:
        min_after = min_before
        unresolved = 0
    else:
        min_after, unresolved_pairs = _pairwise_min_and_pairs(
            cast_working,
            threshold=required_distance,
        )
        unresolved = int(len(unresolved_pairs))
    max_penetration_after = (
        0.0 if math.isinf(min_after) else max(0.0, required_distance - min_after)
    )
    cast_float64 = cast_working.astype(np.float64, copy=False)
    bounds_valid = bool(
        np.all(cast_float64 >= exact_lower) and np.all(cast_float64 <= exact_upper)
    )
    if strict and unresolved:
        raise RuntimeError(
            "Disk projection did not converge: "
            f"unresolved_pairs={unresolved}, min_distance={min_after:.9f}, "
            f"required_distance={required_distance:.9f}, iterations={max_iterations}."
        )
    if strict and not bounds_valid:
        raise RuntimeError(
            "Disk projection produced a radius-aware arena-boundary violation."
        )

    positions_array[...] = cast_working
    return OverlapResolution(
        pair_correction_operations=pair_correction_operations,
        iterations=completed_iterations,
        min_distance_before=min_before,
        min_distance_after=min_after,
        max_penetration_before=max_penetration_before,
        max_penetration_after=max_penetration_after,
    )


def target_region_protocol() -> dict[str, Any]:
    return {
        "physics_variant": PHYSICS_VARIANT,
        "protocol_version": PROTOCOL_VERSION,
        "goal_geometry": "soft_repulsive_axis_aligned_target_region",
        "target_boundary_style": "complete_dashed_rectangle",
        "target_openings_enabled": False,
        "external_target_boundary_force_enabled": True,
        "target_boundary_force": (
            "inverse_square_repulsion_from_nearest_point_on_complete_rectangle"
        ),
        "target_entry_mechanism": "herder_pressure_overcomes_boundary_repulsion",
        "herder_target_boundary_collision_enabled": False,
        "captured_evader_retention_enabled": True,
        "captured_evader_retention_mechanism": "low_dispersion_and_center_force",
    }


def soft_target_boundary_force(
    position: np.ndarray,
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    interaction_distance: float,
    coefficient: float,
    dtype: np.dtype | type = float,
) -> np.ndarray:
    """Return outward inverse-square repulsion from a complete rectangle."""
    x, y = np.asarray(position, dtype=float)
    force = np.zeros(2, dtype=dtype)
    if left <= x <= right and bottom <= y <= top:
        return force
    closest = np.array([np.clip(x, left, right), np.clip(y, bottom, top)])
    outward = np.array([x, y], dtype=float) - closest
    distance = float(np.linalg.norm(outward))
    if distance <= 0.0 or distance > interaction_distance:
        return force
    magnitude = coefficient / (distance**2 + 1e-10)
    return np.asarray(outward / distance * magnitude, dtype=dtype)


class HerdingSimulation:
    """Fixed 4+4 herder baseline with per-step PCA evader task assignment."""

    def __init__(
        self,
        random_seed: int | None = None,
        *,
        evader_radius: float = 0.5,
        density_slowdown_start_count: float = 3.0,
        density_slowdown_full_count: float = 20.0,
        min_speed_factor: float = 0.3,
        collision_resolution_iterations: int = 512,
        collision_separation_margin: float = 1e-4,
        render: bool = False,
    ) -> None:
        validate_safety_parameters(
            evader_radius=evader_radius,
            slowdown_start_count=density_slowdown_start_count,
            slowdown_full_count=density_slowdown_full_count,
            min_speed_factor=min_speed_factor,
            collision_iterations=collision_resolution_iterations,
            separation_margin=collision_separation_margin,
        )
        if random_seed is None:
            random_seed = int(time.time() * 1_000_000) % (2**32)
        if isinstance(random_seed, bool) or not isinstance(random_seed, numbers.Integral):
            raise ValueError("random_seed must be an integer or None.")
        self.randomSeed = int(random_seed) % (2**32)
        np.random.seed(self.randomSeed)
        random.seed(self.randomSeed)

        self.renderEnabled = bool(render)
        self._plt = None
        self._patches = None

        self.numEvaders = 30
        self.numHerders = 8
        self.arenaSize = 80.0
        self.Re = 6.0
        self.Rh = 12.0
        self.speedLimitEvader = 3.0
        self.baseSpeedLimitHerder = 2.0
        self.dt = 0.1
        self.totalTime = 1500.0
        self.edgeRepulsionDist = 5.0
        self.edgeRepulsionCoeff = 50.0
        self.dispersionCoeff = 150.0
        self.aggregationCoeff = 1.0

        self.herder_repulsion_dist = 8.0
        self.herder_repulsion_coeff = 100.0
        self.min_herder_distance = 6.0
        self.lambda_val = self.baseSpeedLimitHerder / self.speedLimitEvader
        self.theta_i = 2.0 * np.arcsin(self.lambda_val)
        self.h_i = 0.1
        self.k_i = 0.5
        self.dc = 2.0
        self.waitTime = 1.0

        self.fov_angle = 120.0
        self.fov_rad = np.radians(self.fov_angle)
        self.rotation_speed = 0.5
        self.numGroups = 2
        self.group_colors = ["deepskyblue", "darkorange"]

        self.evaderRadius = float(evader_radius)
        self.densitySlowdownStartCount = float(density_slowdown_start_count)
        self.densitySlowdownFullCount = float(density_slowdown_full_count)
        self.minSpeedFactor = float(min_speed_factor)
        self.collisionResolutionIterations = int(collision_resolution_iterations)
        self.collisionSeparationMargin = float(collision_separation_margin)

        self._activeDensitySpeedLimits: np.ndarray | None = None
        self._overlapPairCorrectionOperations = 0
        self._overlapCorrectionSteps = 0
        self._totalCollisionResolutionIterations = 0
        self._maxCollisionResolutionIterations = 0
        self._densityEvaluationCount = 0
        self._densityActivationCount = 0
        self._maxVisibleCount = 0
        self._minAppliedHerderSpeedLimit = float(self.baseSpeedLimitHerder)
        self._taskAssignmentCount = 0
        self._lastTaskSizes = [0, 0]

        self.setup_fixed_target_region()
        self.initialize_positions()
        initial_resolution = self.lastOverlapResolution
        self.initialMinEvaderDistanceBeforeProjection = (
            initial_resolution.min_distance_before
        )
        self.initialOverlapPairCorrectionOperations = (
            initial_resolution.pair_correction_operations
        )
        self.initialMaxOverlapPenetrationBefore = (
            initial_resolution.max_penetration_before
        )
        self._overlapPairCorrectionOperations = 0
        self._overlapCorrectionSteps = 0
        self._totalCollisionResolutionIterations = 0
        self._maxCollisionResolutionIterations = 0

        self.initialize_herder_groups()
        self.group_target_evaders = {
            group_id: np.array([], dtype=np.int64)
            for group_id in range(self.numGroups)
        }
        self.group_target_points = {
            group_id: self.cageCenter.copy() for group_id in range(self.numGroups)
        }
        self._initialInTargetCount = self.count_evaders_in_target()
        self.validate_safety_timestep()
        if self.renderEnabled:
            self.setup_visualization()

    # ------------------------------------------------------------------
    # Initialization and protocol
    # ------------------------------------------------------------------
    def setup_fixed_target_region(self) -> None:
        width = 16.0
        height = 16.0
        margin = 0.0
        self.cageLeft = margin
        self.cageRight = self.cageLeft + width
        self.cageBottom = self.arenaSize - margin - height
        self.cageTop = self.cageBottom + height
        self.cageCenter = np.array(
            [self.cageLeft + width / 2.0, self.cageBottom + height / 2.0],
            dtype=float,
        )

    def initialize_positions(self) -> None:
        self.positionsEvader = np.random.rand(self.numEvaders, 2) * self.arenaSize
        self.velocitiesEvader = np.random.randn(self.numEvaders, 2)
        self.positionsHerder = np.random.rand(self.numHerders, 2) * self.arenaSize
        self.velocitiesHerder = np.zeros((self.numHerders, 2), dtype=float)
        self.herder_directions = np.random.rand(self.numHerders) * 2.0 * np.pi
        self._apply_evader_non_overlap_constraint()

    def initialize_herder_groups(self) -> None:
        """Assign a fixed angularly alternating 4+4 partition."""
        if self.numHerders != 8 or self.numGroups != 2:
            raise ValueError("This baseline protocol requires 8 herders and 2 groups.")
        relative = self.positionsHerder - self.cageCenter
        angles = np.arctan2(relative[:, 1], relative[:, 0])
        sorted_indices = np.argsort(angles)
        groups = np.zeros(self.numHerders, dtype=np.int64)
        for order, herder_idx in enumerate(sorted_indices):
            groups[int(herder_idx)] = order % self.numGroups
        self.herderGroups = groups.astype(int).tolist()
        if [self.herderGroups.count(0), self.herderGroups.count(1)] != [4, 4]:
            raise RuntimeError("Fixed angular grouping failed to produce 4+4 groups.")

    def validate_safety_timestep(self) -> None:
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive.")
        if self.speedLimitEvader * self.dt >= 2.0 * self.evaderRadius:
            raise ValueError(
                "Safety V2 requires speedLimitEvader * dt < 2 * evaderRadius "
                "to prevent endpoint collision tunneling."
            )

    def is_in_cage(self, position: Sequence[float]) -> bool:
        x, y = position
        return bool(
            self.cageLeft <= x <= self.cageRight
            and self.cageBottom <= y <= self.cageTop
        )

    def target_mask(self) -> np.ndarray:
        return np.asarray(
            [self.is_in_cage(position) for position in self.positionsEvader],
            dtype=bool,
        )

    def count_evaders_in_target(self) -> int:
        return int(np.count_nonzero(self.target_mask()))

    # ------------------------------------------------------------------
    # Strict evader disk geometry and density slowdown
    # ------------------------------------------------------------------
    def _apply_evader_non_overlap_constraint(self) -> OverlapResolution:
        result = resolve_disk_overlaps(
            self.positionsEvader,
            evader_radius=self.evaderRadius,
            arena_size=self.arenaSize,
            max_iterations=self.collisionResolutionIterations,
            separation_margin=self.collisionSeparationMargin,
            strict=True,
        )
        self._overlapPairCorrectionOperations += result.pair_correction_operations
        if result.pair_correction_operations:
            self._overlapCorrectionSteps += 1
            self._totalCollisionResolutionIterations += result.iterations
            self._maxCollisionResolutionIterations = max(
                self._maxCollisionResolutionIterations,
                result.iterations,
            )
        self.lastOverlapResolution = result
        return result

    def _density_state_for_herder(self, herder_idx: int) -> DensitySlowdownState:
        visible_count = len(self.get_visible_evaders(herder_idx))
        return density_slowdown_state(
            visible_count=visible_count,
            sensing_radius=self.Rh,
            fov_radians=self.fov_rad,
            slowdown_start_count=self.densitySlowdownStartCount,
            slowdown_full_count=self.densitySlowdownFullCount,
            base_speed_limit=self.baseSpeedLimitHerder,
            min_speed_factor=self.minSpeedFactor,
        )

    def _snapshot_density_speed_limits(self) -> np.ndarray:
        states = [
            self._density_state_for_herder(index) for index in range(self.numHerders)
        ]
        self.lastVisibleEvaderCounts = np.asarray(
            [state.visible_count for state in states],
            dtype=np.int64,
        )
        self.lastDensityActivations = np.asarray(
            [state.activation for state in states],
            dtype=float,
        )
        self.lastDensitySpeedLimits = np.asarray(
            [state.speed_limit for state in states],
            dtype=float,
        )
        self._densityEvaluationCount += len(states)
        self._densityActivationCount += int(
            np.count_nonzero(self.lastDensityActivations > 0.0)
        )
        if states:
            self._maxVisibleCount = max(
                self._maxVisibleCount,
                int(np.max(self.lastVisibleEvaderCounts)),
            )
            self._minAppliedHerderSpeedLimit = min(
                self._minAppliedHerderSpeedLimit,
                float(np.min(self.lastDensitySpeedLimits)),
            )
        return self.lastDensitySpeedLimits.copy()

    def calculate_speed_limit(
        self,
        herder_idx: int,
        candidate_indices: Sequence[int] | None = None,
    ) -> float:
        # Safety V2 deliberately measures total visible crowding, not task-only crowding.
        del candidate_indices
        if self._activeDensitySpeedLimits is not None:
            return float(self._activeDensitySpeedLimits[herder_idx])
        return float(self._density_state_for_herder(herder_idx).speed_limit)

    def get_safety_metrics(self) -> dict[str, Any]:
        evaluation_count = max(1, self._densityEvaluationCount)
        metrics: dict[str, Any] = {
            "physics_variant": "crowding_safety_v2",
            "evader_radius": self.evaderRadius,
            "required_evader_center_distance": 2.0 * self.evaderRadius,
            "initial_min_evader_distance_before_projection": (
                self.initialMinEvaderDistanceBeforeProjection
            ),
            "initial_overlap_pair_correction_operations": (
                self.initialOverlapPairCorrectionOperations
            ),
            "initial_max_overlap_penetration_before": (
                self.initialMaxOverlapPenetrationBefore
            ),
            "current_min_evader_distance": pairwise_min_distance(
                self.positionsEvader
            ),
            "remaining_overlap_pairs": count_overlapping_pairs(
                self.positionsEvader,
                self.evaderRadius,
            ),
            "overlap_pair_correction_operations": (
                self._overlapPairCorrectionOperations
            ),
            "overlap_correction_steps": self._overlapCorrectionSteps,
            "max_collision_resolution_iterations": (
                self._maxCollisionResolutionIterations
            ),
            "mean_collision_resolution_iterations_when_active": (
                self._totalCollisionResolutionIterations
                / max(1, self._overlapCorrectionSteps)
            ),
            "max_visible_evader_count": self._maxVisibleCount,
            "min_applied_herder_speed_limit": self._minAppliedHerderSpeedLimit,
            "density_slowdown_activation_rate": (
                self._densityActivationCount / evaluation_count
            ),
        }
        metrics.update(target_region_protocol())
        return metrics

    # ------------------------------------------------------------------
    # Evader dynamics and V3 target-region forces
    # ------------------------------------------------------------------
    def calculate_cage_force(self, evader_idx: int) -> np.ndarray:
        return soft_target_boundary_force(
            self.positionsEvader[evader_idx],
            left=self.cageLeft,
            right=self.cageRight,
            bottom=self.cageBottom,
            top=self.cageTop,
            interaction_distance=self.edgeRepulsionDist,
            coefficient=self.edgeRepulsionCoeff,
            dtype=float,
        )

    def calculate_boundary_force(self, evader_idx: int) -> np.ndarray:
        x, y = self.positionsEvader[evader_idx]
        force = np.zeros(2, dtype=float)
        if x < self.edgeRepulsionDist:
            force[0] = self.edgeRepulsionCoeff / (x**2 + 1e-10)
        elif x > self.arenaSize - self.edgeRepulsionDist:
            force[0] = -self.edgeRepulsionCoeff / (
                (self.arenaSize - x) ** 2 + 1e-10
            )
        if y < self.edgeRepulsionDist:
            force[1] = self.edgeRepulsionCoeff / (y**2 + 1e-10)
        elif y > self.arenaSize - self.edgeRepulsionDist:
            force[1] = -self.edgeRepulsionCoeff / (
                (self.arenaSize - y) ** 2 + 1e-10
            )
        return force

    def calculate_cage_center_force(
        self,
        evader_idx: int,
        is_in_target: np.ndarray,
    ) -> np.ndarray:
        x, y = self.positionsEvader[evader_idx]
        force = np.zeros(2, dtype=float)
        if is_in_target[evader_idx]:
            if x < self.cageLeft + self.edgeRepulsionDist:
                force[0] = self.edgeRepulsionCoeff / (
                    (x - self.cageLeft) ** 2 + 1e-10
                )
            elif x > self.cageRight - self.edgeRepulsionDist:
                force[0] = -self.edgeRepulsionCoeff / (
                    (self.cageRight - x) ** 2 + 1e-10
                )
            if y < self.cageBottom + self.edgeRepulsionDist:
                force[1] = self.edgeRepulsionCoeff / (
                    (y - self.cageBottom) ** 2 + 1e-10
                )
            elif y > self.cageTop - self.edgeRepulsionDist:
                force[1] = -self.edgeRepulsionCoeff / (
                    (self.cageTop - y) ** 2 + 1e-10
                )
        return force

    def update_evaders(self) -> None:
        self.validate_safety_timestep()
        is_in_target = self.target_mask()
        for evader_idx in range(self.numEvaders):
            relative_all = self.positionsEvader - self.positionsEvader[evader_idx]
            distances = np.linalg.norm(relative_all, axis=1)
            neighbors = np.where((distances < self.Re) & (distances > 0.0))[0]

            if len(neighbors) > 0:
                delta = (
                    self.positionsEvader[evader_idx]
                    - self.positionsEvader[neighbors]
                )
                dispersion_force = np.sum(
                    delta / (distances[neighbors, np.newaxis] ** 3 + 1e-10),
                    axis=0,
                ) / len(neighbors)
            else:
                dispersion_force = np.zeros(2, dtype=float)

            aggregation_force = np.zeros(2, dtype=float)
            herder_detected = False
            for herder_idx in range(self.numHerders):
                distance = np.linalg.norm(
                    self.positionsEvader[evader_idx]
                    - self.positionsHerder[herder_idx]
                )
                if distance < self.Re:
                    herder_detected = True
                    if len(neighbors) > 0:
                        aggregation_force += np.mean(
                            self.positionsEvader[neighbors]
                            - self.positionsEvader[evader_idx],
                            axis=0,
                        )

            escape_force = np.zeros(2, dtype=float)
            for herder_idx in range(self.numHerders):
                delta = (
                    self.positionsEvader[evader_idx]
                    - self.positionsHerder[herder_idx]
                )
                distance = np.linalg.norm(delta)
                if distance < self.Re:
                    escape_force += delta / (distance**3 + 1e-10)

            target_force = self.calculate_cage_force(evader_idx)
            arena_force = self.calculate_boundary_force(evader_idx)
            retention_force = self.calculate_cage_center_force(
                evader_idx,
                is_in_target,
            )
            if is_in_target[evader_idx]:
                acceleration = (
                    0.01 * self.dispersionCoeff * dispersion_force
                    + 10.0 * retention_force
                )
            elif herder_detected:
                acceleration = (
                    1e6 * escape_force
                    + target_force
                    + arena_force
                    + self.dispersionCoeff * dispersion_force
                    + self.aggregationCoeff * aggregation_force
                )
            else:
                acceleration = (
                    self.dispersionCoeff * dispersion_force
                    + target_force
                    + arena_force
                )

            self.velocitiesEvader[evader_idx] += acceleration * self.dt
            speed = np.linalg.norm(self.velocitiesEvader[evader_idx])
            if speed > self.speedLimitEvader:
                self.velocitiesEvader[evader_idx] *= self.speedLimitEvader / speed
            new_position = (
                self.positionsEvader[evader_idx]
                + self.velocitiesEvader[evader_idx] * self.dt
            )
            self.positionsEvader[evader_idx] = np.clip(
                new_position,
                0.0,
                self.arenaSize,
            )
        self._apply_evader_non_overlap_constraint()

    # ------------------------------------------------------------------
    # Fixed herder groups and per-step PCA evader tasks
    # ------------------------------------------------------------------
    def get_group_indices(self, group_id: int) -> list[int]:
        return [
            index for index, assigned in enumerate(self.herderGroups)
            if assigned == group_id
        ]

    def get_outside_evader_indices(self) -> np.ndarray:
        return np.asarray(
            [
                index
                for index in range(self.numEvaders)
                if not self.is_in_cage(self.positionsEvader[index])
            ],
            dtype=np.int64,
        )

    def split_outside_evaders_into_groups(self) -> list[np.ndarray]:
        """Median-split outside evaders along their first PCA/SVD axis."""
        outside = self.get_outside_evader_indices()
        empty = np.array([], dtype=np.int64)
        if len(outside) == 0:
            return [empty.copy(), empty.copy()]
        if len(outside) == 1:
            return [outside.copy(), empty.copy()]

        points = self.positionsEvader[outside]
        centered = points - np.mean(points, axis=0)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            split_axis = vh[0]
        except np.linalg.LinAlgError:
            split_axis = np.array([1.0, 0.0])
        projections = centered @ split_axis
        order = np.argsort(projections)
        split = min(max(len(order) // 2, 1), len(order) - 1)
        first = outside[order[:split]]
        second = outside[order[split:]]
        if len(first) == 0 or len(second) == 0:
            fallback_order = np.argsort(points[:, 1])
            split = min(max(len(fallback_order) // 2, 1), len(fallback_order) - 1)
            first = outside[fallback_order[:split]]
            second = outside[fallback_order[split:]]
        return [
            np.asarray(first, dtype=np.int64),
            np.asarray(second, dtype=np.int64),
        ]

    def get_evader_group_center(self, indices: Sequence[int]) -> np.ndarray:
        if len(indices) == 0:
            return self.cageCenter.copy()
        return np.mean(self.positionsEvader[np.asarray(indices, dtype=int)], axis=0)

    def assign_target_clusters_to_herder_groups(self) -> None:
        target_groups = self.split_outside_evaders_into_groups()
        if all(len(group) > 0 for group in target_groups[:2]):
            herder_centers = []
            target_centers = []
            for group_id in range(self.numGroups):
                indices = self.get_group_indices(group_id)
                herder_centers.append(
                    np.mean(self.positionsHerder[indices], axis=0)
                    if indices
                    else self.cageCenter.copy()
                )
                target_centers.append(
                    self.get_evader_group_center(target_groups[group_id])
                )
            direct_cost = np.linalg.norm(
                herder_centers[0] - target_centers[0]
            ) + np.linalg.norm(herder_centers[1] - target_centers[1])
            swapped_cost = np.linalg.norm(
                herder_centers[0] - target_centers[1]
            ) + np.linalg.norm(herder_centers[1] - target_centers[0])
            if swapped_cost < direct_cost:
                target_groups[0], target_groups[1] = (
                    target_groups[1],
                    target_groups[0],
                )

        for group_id in range(self.numGroups):
            assigned = np.asarray(target_groups[group_id], dtype=np.int64)
            self.group_target_evaders[group_id] = assigned
            self.group_target_points[group_id] = self.get_evader_group_center(assigned)
        self._taskAssignmentCount += 1
        self._lastTaskSizes = [len(target_groups[0]), len(target_groups[1])]

    def get_visible_evaders(
        self,
        herder_idx: int,
        candidate_indices: Sequence[int] | None = None,
    ) -> np.ndarray:
        if candidate_indices is None:
            candidate_indices = range(self.numEvaders)
        herder_position = self.positionsHerder[herder_idx]
        herder_direction = self.herder_directions[herder_idx]
        visible = []
        for raw_index in candidate_indices:
            evader_idx = int(raw_index)
            if self.is_in_cage(self.positionsEvader[evader_idx]):
                continue
            relative = self.positionsEvader[evader_idx] - herder_position
            distance = np.linalg.norm(relative)
            if distance > self.Rh or distance == 0.0:
                continue
            angle = np.arctan2(relative[1], relative[0])
            angle_difference = (
                (angle - herder_direction + np.pi) % (2.0 * np.pi) - np.pi
            )
            if abs(angle_difference) <= self.fov_rad / 2.0:
                visible.append(evader_idx)
        return np.asarray(visible, dtype=np.int64)

    def select_target_for_herder(
        self,
        herder_idx: int,
        preferred_evaders: Sequence[int],
        fallback_point: np.ndarray,
    ) -> np.ndarray:
        visible = self.get_visible_evaders(herder_idx, preferred_evaders)
        if len(visible) > 0:
            distances = np.linalg.norm(
                self.positionsEvader[visible] - self.positionsHerder[herder_idx],
                axis=1,
            )
            return self.positionsEvader[int(visible[np.argmin(distances)])]
        if len(preferred_evaders) > 0:
            return np.mean(
                self.positionsEvader[np.asarray(preferred_evaders, dtype=int)],
                axis=0,
            )
        outside = self.get_outside_evader_indices()
        if len(outside) > 0:
            farthest = int(
                outside[
                    np.argmax(
                        np.linalg.norm(
                            self.positionsEvader[outside] - self.cageCenter,
                            axis=1,
                        )
                    )
                ]
            )
            return self.positionsEvader[farthest]
        return np.asarray(fallback_point, dtype=float)

    def update_herder_direction(
        self,
        herder_idx: int,
        candidate_indices: Sequence[int] | None = None,
        fallback_target: np.ndarray | None = None,
    ) -> None:
        visible = self.get_visible_evaders(herder_idx, candidate_indices)
        if len(visible) > 0:
            distances = np.linalg.norm(
                self.positionsEvader[visible] - self.positionsHerder[herder_idx],
                axis=1,
            )
            closest = int(visible[np.argmin(distances)])
            relative = (
                self.positionsEvader[closest] - self.positionsHerder[herder_idx]
            )
            self.herder_directions[herder_idx] = np.arctan2(
                relative[1],
                relative[0],
            )
        elif fallback_target is not None:
            relative = fallback_target - self.positionsHerder[herder_idx]
            if np.linalg.norm(relative) > 0.0:
                self.herder_directions[herder_idx] = np.arctan2(
                    relative[1],
                    relative[0],
                )
            else:
                self._rotate_herder_direction(herder_idx)
        else:
            self._rotate_herder_direction(herder_idx)

    def _rotate_herder_direction(self, herder_idx: int) -> None:
        self.herder_directions[herder_idx] += self.rotation_speed * self.dt
        self.herder_directions[herder_idx] %= 2.0 * np.pi

    # ------------------------------------------------------------------
    # Convex-hull herder controller
    # ------------------------------------------------------------------
    def get_cage_hull_vertices(self) -> np.ndarray:
        return np.array(
            [
                [self.cageLeft, self.cageTop],
                [self.cageLeft, self.cageBottom],
                [self.cageRight, self.cageTop],
            ],
            dtype=float,
        )

    def get_augmented_hull_entities(
        self,
        group_indices: Sequence[int],
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        group_indices = list(group_indices)
        augmented = np.vstack(
            [self.positionsHerder[group_indices], self.get_cage_hull_vertices()]
        )
        hull = ConvexHull(augmented)
        points = augmented[hull.vertices]
        entities = []
        for vertex_idx in hull.vertices:
            if vertex_idx < len(group_indices):
                entities.append(
                    {
                        "type": "herder",
                        "index": group_indices[int(vertex_idx)],
                        "position": augmented[vertex_idx],
                    }
                )
            else:
                entities.append(
                    {
                        "type": "cage_vertex",
                        "index": int(vertex_idx) - len(group_indices),
                        "position": augmented[vertex_idx],
                    }
                )
        return entities, points

    def update_herders(self, current_time: float) -> None:
        speed_limits = self._snapshot_density_speed_limits()
        self._activeDensitySpeedLimits = speed_limits
        try:
            accelerations = np.zeros((self.numHerders, 2), dtype=float)
            if current_time >= self.waitTime:
                accelerations = self.update_herders_with_convex_hull(current_time)
            self.velocitiesHerder += accelerations * self.dt
            for herder_idx in range(self.numHerders):
                dynamic_limit = float(speed_limits[herder_idx])
                speed = np.linalg.norm(self.velocitiesHerder[herder_idx])
                if speed > dynamic_limit:
                    self.velocitiesHerder[herder_idx] *= dynamic_limit / speed
                new_position = (
                    self.positionsHerder[herder_idx]
                    + self.velocitiesHerder[herder_idx] * self.dt
                )
                self.positionsHerder[herder_idx] = np.clip(
                    new_position,
                    0.0,
                    self.arenaSize,
                )
        finally:
            self._activeDensitySpeedLimits = None

    def update_herders_with_convex_hull(self, current_time: float) -> np.ndarray:
        accelerations = np.zeros((self.numHerders, 2), dtype=float)
        if current_time >= self.waitTime and self.numHerders >= 1:
            self.assign_target_clusters_to_herder_groups()
            for group_id in range(self.numGroups):
                group_indices = self.get_group_indices(group_id)
                if group_indices:
                    accelerations += self.update_herders_group(
                        group_indices,
                        group_id,
                    )
        return accelerations

    def update_herders_group(
        self,
        group_indices: Sequence[int],
        group_id: int,
    ) -> np.ndarray:
        accelerations = np.zeros((self.numHerders, 2), dtype=float)
        if not group_indices:
            return accelerations
        target_evaders = self.group_target_evaders.get(
            group_id,
            np.array([], dtype=np.int64),
        )
        target_point = self.group_target_points.get(group_id, self.cageCenter)
        try:
            hull_entities, _ = self.get_augmented_hull_entities(group_indices)
            hull_herders = [
                entity["index"]
                for entity in hull_entities
                if entity["type"] == "herder"
            ]
        except Exception:
            hull_entities = [
                {
                    "type": "herder",
                    "index": index,
                    "position": self.positionsHerder[index],
                }
                for index in group_indices
            ]
            hull_herders = list(group_indices)
        if hull_herders:
            self.update_convex_hull_herders(
                hull_entities,
                accelerations,
                target_evaders,
                target_point,
            )
        non_hull = [index for index in group_indices if index not in hull_herders]
        self.update_non_convex_hull_herders(
            non_hull,
            accelerations,
            target_evaders,
            target_point,
        )
        return accelerations

    def update_convex_hull_herders(
        self,
        hull_entities: Sequence[dict[str, Any]],
        accelerations: np.ndarray,
        target_evaders: Sequence[int],
        target_point: np.ndarray,
    ) -> None:
        if not hull_entities:
            return
        hull_positions = np.asarray(
            [entity["position"] for entity in hull_entities],
            dtype=float,
        )
        delta = hull_positions - self.cageCenter
        radii = np.sqrt(np.sum(delta**2, axis=1))
        angles = np.arctan2(delta[:, 1], delta[:, 0])
        angles[angles < 0.0] += 2.0 * np.pi
        order = np.argsort(angles)
        sorted_entities = [hull_entities[int(index)] for index in order]
        angles = angles[order]
        radii = radii[order]
        count = len(hull_entities)
        psi = np.zeros(count, dtype=float)
        for index in range(count):
            if index < count - 1:
                psi[index] = angles[index + 1] - angles[index] - self.theta_i / 2.0
            else:
                psi[index] = (
                    angles[0] - angles[-1] + 2.0 * np.pi - self.theta_i / 2.0
                )
        for sorted_index, entity in enumerate(sorted_entities):
            if entity["type"] == "herder":
                self.update_single_convex_herder(
                    int(entity["index"]),
                    sorted_index,
                    psi,
                    radii,
                    angles,
                    count,
                    accelerations,
                    target_evaders,
                    target_point,
                )

    def _add_herder_repulsion(
        self,
        herder_idx: int,
        force: np.ndarray,
    ) -> np.ndarray:
        current = self.positionsHerder[herder_idx]
        for other_idx in range(self.numHerders):
            if other_idx == herder_idx:
                continue
            relative = current - self.positionsHerder[other_idx]
            distance = np.linalg.norm(relative)
            if 0.0 < distance < self.herder_repulsion_dist:
                force += (
                    self.herder_repulsion_coeff / (distance**2 + 1e-10)
                ) * (relative / distance)
        return force

    @staticmethod
    def _cap_force(force: np.ndarray, limit: float) -> np.ndarray:
        magnitude = np.linalg.norm(force)
        if magnitude > limit:
            return force / magnitude * limit
        return force

    def update_non_convex_hull_herders(
        self,
        indices: Sequence[int],
        accelerations: np.ndarray,
        target_evaders: Sequence[int],
        target_point: np.ndarray,
    ) -> None:
        for herder_idx in indices:
            current = self.positionsHerder[herder_idx]
            speed_limit = self.calculate_speed_limit(herder_idx, target_evaders)
            target = self.select_target_for_herder(
                herder_idx,
                target_evaders,
                target_point,
            )
            direction = target - current
            direction /= np.linalg.norm(direction) + 1e-10
            force = self.h_i * np.linalg.norm(current - self.cageCenter) * direction
            force = self._add_herder_repulsion(herder_idx, force)
            accelerations[herder_idx] = self._cap_force(force, speed_limit)
            self.update_herder_direction(
                herder_idx,
                target_evaders,
                target_point,
            )

    def update_single_convex_herder(
        self,
        herder_idx: int,
        index: int,
        psi: np.ndarray,
        radii: np.ndarray,
        angles: np.ndarray,
        hull_count: int,
        accelerations: np.ndarray,
        target_evaders: Sequence[int],
        target_point: np.ndarray,
    ) -> None:
        current = self.positionsHerder[herder_idx]
        speed_limit = self.calculate_speed_limit(herder_idx, target_evaders)
        psi_difference = psi[index] - psi[index - 1] if index else psi[0] - psi[-1]
        delta_i = 2.0 * abs(psi_difference) / (4.0 * np.pi - self.theta_i)
        sum_radii = (
            radii[index]
            + radii[(index - 2) % hull_count]
            + radii[index % hull_count]
        )
        gamma_i = np.sin(np.pi * (radii[index] / sum_radii)) * np.log2(3.0)
        beta_i = np.pi / 2.0 * (1.0 - np.exp(-delta_i * gamma_i))
        surround_direction = np.array(
            [-np.sin(angles[index]), np.cos(angles[index])]
        )
        target = self.select_target_for_herder(
            herder_idx,
            target_evaders,
            target_point,
        )
        hunt_direction = target - current
        hunt_direction /= np.linalg.norm(hunt_direction) + 1e-10
        surround_velocity = (
            self.k_i
            * radii[index]
            * psi_difference
            * surround_direction
            * np.sin(beta_i)
        )
        hunt_velocity = (
            self.h_i * radii[index] * hunt_direction * np.cos(beta_i)
        )
        force = self._add_herder_repulsion(
            herder_idx,
            surround_velocity + hunt_velocity,
        )
        accelerations[herder_idx] = self._cap_force(force, speed_limit)
        self.update_herder_direction(
            herder_idx,
            target_evaders,
            target_point,
        )

    # ------------------------------------------------------------------
    # Optional visualization. Headless execution never enters these paths.
    # ------------------------------------------------------------------
    def setup_visualization(self) -> None:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        self._plt = plt
        self._patches = patches
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(0.0, self.arenaSize)
        self.ax.set_ylim(0.0, self.arenaSize)
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.ax.add_patch(
            plt.Rectangle(
                (0.0, 0.0),
                self.arenaSize,
                self.arenaSize,
                fill=False,
                edgecolor="black",
                linewidth=2.0,
            )
        )
        self.targetRegionPatch = patches.Rectangle(
            (self.cageLeft, self.cageBottom),
            self.cageRight - self.cageLeft,
            self.cageTop - self.cageBottom,
            fill=False,
            edgecolor="#16803a",
            linestyle="--",
            linewidth=2.0,
            zorder=2,
        )
        self.ax.add_patch(self.targetRegionPatch)
        self.evader_scatter = self.ax.scatter(
            self.positionsEvader[:, 0],
            self.positionsEvader[:, 1],
            s=30,
            c="yellow",
            marker="o",
            edgecolors="black",
        )
        colors = [self.group_colors[group_id] for group_id in self.herderGroups]
        self.herder_scatter = self.ax.scatter(
            self.positionsHerder[:, 0],
            self.positionsHerder[:, 1],
            s=80,
            c=colors,
            marker="o",
        )
        self.convex_hull_lines = [
            self.ax.plot([], [], color=color, linewidth=2.0)[0]
            for color in self.group_colors
        ]
        self.herder_fov_patches = []
        self.herder_repulsion_circles = []
        self.herder_interaction_circles = []
        for herder_idx in range(self.numHerders):
            fov = self.create_fov_patch(herder_idx)
            self.herder_fov_patches.append(fov)
            self.ax.add_patch(fov)
            sensing = patches.Circle(
                self.positionsHerder[herder_idx],
                self.Re,
                fill=True,
                facecolor="gray",
                alpha=0.3,
                edgecolor="gray",
                linewidth=0.5,
            )
            interaction = patches.Circle(
                self.positionsHerder[herder_idx],
                self.herder_repulsion_dist,
                fill=False,
                edgecolor="red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.5,
            )
            self.herder_repulsion_circles.append(sensing)
            self.herder_interaction_circles.append(interaction)
            self.ax.add_patch(sensing)
            self.ax.add_patch(interaction)

    def create_fov_patch(self, herder_idx: int):
        if self._patches is None:
            raise RuntimeError("Visualization is not initialized.")
        direction = self.herder_directions[herder_idx]
        return self._patches.Wedge(
            self.positionsHerder[herder_idx],
            self.Rh,
            np.degrees(direction - self.fov_rad / 2.0),
            np.degrees(direction + self.fov_rad / 2.0),
            fill=False,
            edgecolor=[0.7, 0.7, 1.0],
            linestyle="--",
            linewidth=1.0,
        )

    def update_convex_hull_visualization(self) -> None:
        for group_id, line in enumerate(self.convex_hull_lines):
            group_indices = self.get_group_indices(group_id)
            if not group_indices:
                line.set_data([], [])
                continue
            try:
                _, points = self.get_augmented_hull_entities(group_indices)
                closed = np.vstack([points, points[0]])
                line.set_data(closed[:, 0], closed[:, 1])
            except Exception:
                line.set_data([], [])

    def update_visualization(self, current_time: float) -> bool:
        if not self.renderEnabled or self._plt is None:
            raise RuntimeError("update_visualization requires render=True.")
        self.evader_scatter.set_offsets(self.positionsEvader)
        self.herder_scatter.set_offsets(self.positionsHerder)
        self.update_convex_hull_visualization()
        for index in range(self.numHerders):
            self.herder_fov_patches[index].remove()
            self.herder_fov_patches[index] = self.create_fov_patch(index)
            self.ax.add_patch(self.herder_fov_patches[index])
            self.herder_repulsion_circles[index].center = tuple(
                self.positionsHerder[index]
            )
            self.herder_interaction_circles[index].center = tuple(
                self.positionsHerder[index]
            )
        count = self.count_evaders_in_target()
        self.ax.set_title(
            f"Time = {current_time:.1f}s | Evaders in target region: "
            f"{count}/{self.numEvaders}"
        )
        self._plt.draw()
        self._plt.pause(0.01)
        return count == self.numEvaders

    # ------------------------------------------------------------------
    # Episode execution and machine-readable reporting
    # ------------------------------------------------------------------
    def episode_summary(
        self,
        *,
        physics_steps: int,
        simulated_seconds: float,
        wall_seconds: float,
    ) -> dict[str, Any]:
        final_count = self.count_evaders_in_target()
        return {
            "schema_version": 1,
            "method": METHOD_NAME,
            "seed": self.randomSeed,
            "dt": float(self.dt),
            "requested_duration_seconds": float(self.totalTime),
            "simulated_seconds": float(simulated_seconds),
            "physics_steps": int(physics_steps),
            "wall_seconds": float(wall_seconds),
            "initial_in_target_count": int(self._initialInTargetCount),
            "final_in_target_count": int(final_count),
            "final_in_target_ratio": float(final_count / self.numEvaders),
            "success_at_horizon": bool(final_count == self.numEvaders),
            "render_enabled": bool(self.renderEnabled),
            "herder_groups": [int(group) for group in self.herderGroups],
            "group_sizes": [
                int(self.herderGroups.count(group_id))
                for group_id in range(self.numGroups)
            ],
            "grouping_membership_dynamic": False,
            "evader_task_allocator": "per_step_pca_median_split",
            "task_assignment_count": int(self._taskAssignmentCount),
            "last_task_sizes": [int(size) for size in self._lastTaskSizes],
            "safety": self.get_safety_metrics(),
        }

    def run_simulation(self) -> dict[str, Any]:
        if not math.isfinite(self.totalTime) or self.totalTime < 0.0:
            raise ValueError("totalTime must be finite and non-negative.")
        self.validate_safety_timestep()
        start_wall = time.perf_counter()
        physics_steps = 0
        simulated_seconds = 0.0
        for current_time in np.arange(0.0, self.totalTime, self.dt):
            self.update_evaders()
            self.update_herders(float(current_time))
            physics_steps += 1
            simulated_seconds = physics_steps * self.dt
            if self.renderEnabled:
                complete = self.update_visualization(float(current_time))
            else:
                complete = self.count_evaders_in_target() == self.numEvaders
            if complete:
                break
        wall_seconds = time.perf_counter() - start_wall
        if self.renderEnabled and self._plt is not None:
            self._plt.show()
        return self.episode_summary(
            physics_steps=physics_steps,
            simulated_seconds=simulated_seconds,
            wall_seconds=wall_seconds,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--duration", type=float, default=1500.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--headless",
        dest="render",
        action="store_false",
        help="Run without creating a GUI (default).",
    )
    mode.add_argument(
        "--render",
        dest="render",
        action="store_true",
        help="Render the interactive Matplotlib visualization.",
    )
    parser.set_defaults(render=False)
    parser.add_argument("--evader-radius", type=float, default=0.5)
    parser.add_argument("--density-slowdown-start-count", type=float, default=3.0)
    parser.add_argument("--density-slowdown-full-count", type=float, default=20.0)
    parser.add_argument("--min-speed-factor", type=float, default=0.3)
    parser.add_argument("--collision-resolution-iterations", type=int, default=512)
    parser.add_argument("--collision-separation-margin", type=float, default=1e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    simulation = HerdingSimulation(
        random_seed=args.seed,
        evader_radius=args.evader_radius,
        density_slowdown_start_count=args.density_slowdown_start_count,
        density_slowdown_full_count=args.density_slowdown_full_count,
        min_speed_factor=args.min_speed_factor,
        collision_resolution_iterations=args.collision_resolution_iterations,
        collision_separation_margin=args.collision_separation_margin,
        render=args.render,
    )
    simulation.dt = float(args.dt)
    simulation.totalTime = float(args.duration)
    summary = simulation.run_simulation()
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
