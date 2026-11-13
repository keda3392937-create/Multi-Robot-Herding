"""Place a fixed virtual field from initial car poses; virtual cage wall geometry.

Two ways to obtain the virtual field:

* ``InitialCarArena`` derives it from the initial Vicon poses (automatic mode);
* ``FixedCornerArena`` uses hard-coded Vicon world coordinates for the four
  field corners and the four cage corners (the mode used on the real robots).
"""
from __future__ import annotations

import math
import time
from statistics import median

from relative_cage_core import (
    ArenaGeometryError, FieldCorners, RelativeArena,
    vec_add, vec_scale, vec_sub, vec_dot, vec_norm, vec_normalize,
)


def point_segment_distance(point, start, end):
    delta = vec_sub(end, start)
    length2 = vec_dot(delta, delta)
    t = 0.0 if length2 < 1e-12 else max(0.0, min(1.0, vec_dot(vec_sub(point, start), delta) / length2))
    closest = vec_add(start, vec_scale(delta, t))
    return vec_norm(vec_sub(point, closest)), closest


def segment_distance(a, b, c, d):
    """Distance between two closed line segments, including mid-path crossing."""
    r, s = vec_sub(b, a), vec_sub(d, c)
    cross = lambda u, v: u[0] * v[1] - u[1] * v[0]
    denom = cross(r, s)
    if abs(denom) > 1e-12:
        offset = vec_sub(c, a)
        t, u = cross(offset, s) / denom, cross(offset, r) / denom
        if 0 <= t <= 1 and 0 <= u <= 1:
            return 0.0
    return min(point_segment_distance(a, c, d)[0], point_segment_distance(b, c, d)[0],
               point_segment_distance(c, a, b)[0], point_segment_distance(d, a, b)[0])


def cage_walls(arena, opening_mm):
    """Left/top walls and bottom/right walls with centred entry openings."""
    if not math.isfinite(opening_mm) or not 0 < opening_mm < min(arena.cage_width_mm, arena.cage_height_mm):
        raise ValueError("Cage opening must be positive and smaller than both cage dimensions")
    l, r, b, t = arena.cage_left, arena.cage_right, arena.cage_bottom, arena.cage_top
    cx, cy = arena.cage_center_local
    half = opening_mm / 2
    return (((l, b), (l, t)), ((l, t), (r, t)),
            ((l, b), (cx - half, b)), ((cx + half, b), (r, b)),
            ((r, b), (r, cy - half)), ((r, cy + half), (r, t)))


def cage_outline(arena):
    """The cage rectangle as four closed walls (no openings)."""
    l, r, b, t = arena.cage_left, arena.cage_right, arena.cage_bottom, arena.cage_top
    return (((l, b), (l, t)), ((l, t), (r, t)), ((r, t), (r, b)), ((r, b), (l, b)))


def cage_exit_target(point, arena, offset_mm=60.0):
    """Nearest point just outside the cage, through the walls facing the field.

    The cage sits in a field corner, so its left and top walls are the field
    boundary itself; the only sensible way out is across the bottom or the
    right wall.
    """
    x_mm, y_mm = point
    to_bottom = abs(y_mm - arena.cage_bottom)
    to_right = abs(arena.cage_right - x_mm)
    if to_bottom <= to_right:
        return (x_mm, arena.cage_bottom - offset_mm)
    return (arena.cage_right + offset_mm, y_mm)


def herder_cage_repulsion(point, arena, margin_mm, weight, body_radius_mm):
    """Keep herders out of the cage; inside, push them back across the nearest wall."""
    if margin_mm <= 0.0 or weight <= 0.0:
        return (0.0, 0.0)
    x_mm, y_mm = point
    if arena.contains_cage_local(point):
        to_bottom = max(0.0, y_mm - arena.cage_bottom)
        to_right = max(0.0, arena.cage_right - x_mm)
        if to_bottom <= to_right:
            return (0.0, -weight * min(3.0, 1.0 + to_bottom / margin_mm))
        return (weight * min(3.0, 1.0 + to_right / margin_mm), 0.0)

    closest = (
        min(max(x_mm, arena.cage_left), arena.cage_right),
        min(max(y_mm, arena.cage_bottom), arena.cage_top),
    )
    offset = vec_sub(point, closest)
    distance = vec_norm(offset)
    gap = max(0.0, distance - body_radius_mm)
    if gap >= margin_mm or distance <= 1e-9:
        return (0.0, 0.0)
    closeness = 1.0 - gap / margin_mm
    return vec_scale(vec_normalize(offset), weight * closeness * closeness)


def evader_membrane_force(point, arena, margin_mm, weight):
    """One-way soft boundary: evaders inside the cage are pushed back to its centre.

    Outside the cage nothing is applied, so herders can push an evader across
    the boundary anywhere; once inside, the membrane keeps it there.
    """
    if margin_mm <= 0.0 or weight <= 0.0 or not arena.contains_cage_local(point):
        return (0.0, 0.0)
    x_mm, y_mm = point
    total = (0.0, 0.0)
    for distance, inward in (
        (y_mm - arena.cage_bottom, (0.0, 1.0)),
        (arena.cage_right - x_mm, (-1.0, 0.0)),
    ):
        if distance >= margin_mm:
            continue
        closeness = 1.0 - max(distance, 0.0) / margin_mm
        total = vec_add(total, vec_scale(inward, weight * closeness * closeness))
    return total


def cage_gate_target(point, arena, opening_mm):
    # Aim a little beyond the right/bottom opening, for a herder exiting a cage.
    cx, cy = arena.cage_center_local
    gates = ((arena.cage_right + opening_mm / 2, cy), (cx, arena.cage_bottom - opening_mm / 2))
    return min(gates, key=lambda p: vec_norm(vec_sub(p, point)))


def virtual_wall_force(point, arena, opening_mm, margin_mm, weight, body_radius_mm):
    total = (0.0, 0.0)
    for start, end in cage_walls(arena, opening_mm):
        distance, closest = point_segment_distance(point, start, end)
        gap = max(0.0, distance - body_radius_mm)
        if gap >= margin_mm:
            continue
        offset = vec_sub(point, closest)
        if distance < 1e-9:
            offset = vec_sub(arena.cage_center_local, point)
            distance = max(vec_norm(offset), 1e-9)
        total = vec_add(total, vec_scale(offset, weight * (1 - gap / margin_mm) ** 2 / distance))
    return total


def initial_position_arena(positions, herder_ids, evader_ids, width_mm=4000.0,
                           height_mm=4000.0, diameter_mm=150.0, gap_mm=80.0,
                           cage_width_ratio=0.4, cage_height_ratio=0.4,
                           opening_mm=700.0, initial_inset_mm=300.0):
    """Choose translation and quarter-turn; never resize to hide outlying cars.

    Prefer a clear cage away from initial herders. The centre is based on the
    robot bounding box, not on a fixed Vicon origin. The returned object is frozen.
    """
    if not positions or not all(math.isfinite(v) for p in positions.values() for v in p):
        raise ValueError("Need finite initial car positions")
    if not all(math.isfinite(v) and v > 0 for v in (width_mm, height_mm, diameter_mm, gap_mm, initial_inset_mm)):
        raise ValueError("Field dimensions and margins must be finite and positive")
    if not 0.05 <= cage_width_ratio <= 0.8 or not 0.05 <= cage_height_ratio <= 0.8:
        raise ValueError("Cage ratios must be in [0.05, 0.8]")
    if opening_mm <= diameter_mm + gap_mm:
        raise ValueError("Cage opening must exceed the car diameter plus body gap")
    inset = max(initial_inset_mm, diameter_mm / 2 + gap_mm / 2)
    rotations = (((1, 0), (0, 1)), ((0, 1), (-1, 0)),
                 ((-1, 0), (0, -1)), ((0, -1), (1, 0)))
    candidates = []
    for axis_x, axis_y in rotations:
        projected = {i: (vec_dot(p, axis_x), vec_dot(p, axis_y)) for i, p in positions.items()}
        xs, ys = [p[0] for p in projected.values()], [p[1] for p in projected.values()]
        xlo, xhi = max(xs) + inset - width_mm, min(xs) - inset
        ylo, yhi = max(ys) + inset - height_mm, min(ys) - inset
        if xlo > xhi or ylo > yhi:
            continue
        centre = ((min(xs) + max(xs) - width_mm) / 2, (min(ys) + max(ys) - height_mm) / 2)
        for ix in range(9):
            for iy in range(9):
                ox, oy = xlo + (xhi - xlo) * ix / 8, ylo + (yhi - ylo) * iy / 8
                origin = vec_add(vec_scale(axis_x, ox), vec_scale(axis_y, oy))
                arena = RelativeArena(origin, axis_x, axis_y, width_mm, height_mm,
                                      width_mm * cage_width_ratio, height_mm * cage_height_ratio, 0.0, 90.0)
                walls = cage_walls(arena, opening_mm)
                local = {i: (p[0] - ox, p[1] - oy) for i, p in projected.items()}
                wall_conflicts = sum(min(point_segment_distance(p, *wall)[0] for wall in walls)
                                     < diameter_mm / 2 + gap_mm / 2 for p in local.values())
                h_in = sum(arena.contains_cage_local(local[i], -diameter_mm / 2) for i in herder_ids if i in local)
                e_in = sum(arena.contains_cage_local(local[i]) for i in evader_ids if i in local)
                shift = math.hypot(ox - centre[0], oy - centre[1])
                score = (wall_conflicts, h_in, e_in, shift)
                candidates.append((score, arena))
    if not candidates:
        raise ValueError("Initial cars do not fit inside the configured field with margins; move them closer or change FIELD_WIDTH_MM/FIELD_HEIGHT_MM")
    score, arena = min(candidates, key=lambda item: item[0])
    if score[0]:
        raise ValueError("No virtual cage placement clears initial cars; spread/reposition cars or adjust cage ratios")
    corners = FieldCorners(arena.local_to_world((0, height_mm)), arena.local_to_world((0, 0)),
                          arena.local_to_world((width_mm, height_mm)), arena.local_to_world((width_mm, 0)))
    return arena, corners


class InitialCarArena:
    def __init__(self, width_mm, height_mm, gap_mm, opening_mm, initial_inset_mm=300.0):
        self.width, self.height, self.gap = width_mm, height_mm, gap_mm
        self.opening, self.inset = opening_mm, initial_inset_mm

    def __call__(self, tracker, args, subjects_by_id, herders, evaders):
        requested = tuple(herders) + tuple(evaders)
        samples = {i: [] for i in requested}
        deadline = time.monotonic() + args.initial_visibility_sec
        frames = 0
        while time.monotonic() < deadline:
            snapshot = tracker.get_snapshot()
            if snapshot.fresh_subjects:
                frames += 1
                for i in requested:
                    name = subjects_by_id[i]
                    if name in snapshot.fresh_subjects:
                        samples[i].append(snapshot.poses[name].position)
            time.sleep(0.01)
        minimum = max(3, math.ceil(frames * args.initial_visibility_ratio))
        positions = {i: (median(p[0] for p in values), median(p[1] for p in values))
                     for i, values in samples.items() if len(values) >= minimum}
        if sum(i in positions for i in herders) < 3 or not any(i in positions for i in evaders):
            raise RuntimeError("Need at least 3 visible herders and 1 visible evader to place the virtual field")
        arena, corners = initial_position_arena(
            positions, herders, evaders, self.width, self.height, args.robot_diameter_mm,
            self.gap, args.cage_width_ratio, args.cage_height_ratio, self.opening, self.inset,
        )
        missing = [i for i in requested if i not in positions]
        args.virtual_field_initial_positions = positions
        print(f"Virtual field frozen from {len(positions)} initial car poses; no corner markers required.")
        if missing:
            print("No stable initial pose for car(s): " + ",".join(map(str, missing)))
        return arena, corners, {}


# ---------------------------------------------------------------------------
# Hard-coded field and cage coordinates (real-robot mode).
# ---------------------------------------------------------------------------


def _unit(vector, label="vector"):
    length = math.hypot(vector[0], vector[1])
    if length < 1e-9:
        raise ArenaGeometryError(f"{label} has zero length; two coordinates are identical")
    return (vector[0] / length, vector[1] / length)


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def order_rectangle(corners, label="field"):
    """Return the four corners in counter-clockwise order around their centre.

    The input order does not matter, so coordinates can be typed in whatever
    order the Vicon system shows them.
    """
    points = []
    for corner in corners:
        try:
            x_mm, y_mm = float(corner[0]), float(corner[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ArenaGeometryError(
                f"{label} corners must be (x, y) millimetre pairs"
            ) from exc
        if not (math.isfinite(x_mm) and math.isfinite(y_mm)):
            raise ArenaGeometryError(f"{label} corner {corner!r} is not finite")
        points.append((x_mm, y_mm))
    if len(points) != 4:
        raise ArenaGeometryError(f"{label} needs exactly four (x, y) corners")
    if len(set(points)) != 4:
        raise ArenaGeometryError(f"{label} corners must be four distinct points")
    centre_x = sum(point[0] for point in points) / 4.0
    centre_y = sum(point[1] for point in points) / 4.0
    points.sort(key=lambda point: math.atan2(point[1] - centre_y, point[0] - centre_x))
    return points


def rectangle_metrics(points, label="field"):
    """Validate one rectangle ordering and return its size, axes and fit error."""
    p0, p1, p2, p3 = points
    width = math.dist(p0, p1)
    height = math.dist(p1, p2)
    if width <= 1e-6 or height <= 1e-6:
        raise ArenaGeometryError(f"{label} corners do not span a rectangle")
    error = max(
        abs(width - math.dist(p2, p3)),
        abs(height - math.dist(p3, p0)),
        abs(math.dist(p0, p2) - math.dist(p1, p3)) / 2.0,
    )
    tolerance = max(20.0, 0.02 * min(width, height))
    if error > tolerance:
        raise ArenaGeometryError(
            f"{label} corners do not form a rectangle "
            f"(sides differ by {error:.1f}mm, tolerance {tolerance:.1f}mm)"
        )
    x_axis = _unit(vec_sub(p1, p0), f"{label} width")
    y_axis = _unit(vec_sub(p3, p0), f"{label} height")
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, vec_dot(x_axis, y_axis)))))
    if abs(angle_deg - 90.0) > 2.0:
        raise ArenaGeometryError(
            f"{label} corners meet at {angle_deg:.1f}deg instead of 90deg"
        )
    return width, height, x_axis, y_axis, error, angle_deg


def fixed_arena(field_corners_world, cage_corners_world, opening_mm,
                robot_diameter_mm=150.0, gap_mm=80.0, corner_tolerance_mm=None):
    """Build ``RelativeArena`` from hard-coded world coordinates.

    The cage is the top-left corner of the local frame, so the local frame is
    chosen automatically among the four possible field orientations to place
    the cage there. Local X points along the field width, local Y along the
    field height, and the two cage openings stay on the walls that face the
    field interior.
    """
    field = order_rectangle(field_corners_world, "field")
    cage = order_rectangle(cage_corners_world, "cage")
    _, _, _, _, error, angle_deg = rectangle_metrics(field, "field")
    rectangle_metrics(cage, "cage")

    candidates = []
    for shift in range(4):
        p0 = field[shift]
        p1 = field[(shift + 1) % 4]
        p3 = field[(shift - 1) % 4]
        width = math.dist(p0, p1)
        height = math.dist(p0, p3)
        x_axis = _unit(vec_sub(p1, p0), "field width")
        y_axis = _unit(vec_sub(p3, p0), "field height")
        if _cross(x_axis, y_axis) <= 0.0:
            continue
        local = [
            (vec_dot(vec_sub(corner, p0), x_axis), vec_dot(vec_sub(corner, p0), y_axis))
            for corner in cage
        ]
        xs = [point[0] for point in local]
        ys = [point[1] for point in local]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        # The cage has to touch the local left (x=0) and top (y=height) edges.
        score = abs(x0) + abs(y1 - height)
        candidates.append((score, p0, width, height, x_axis, y_axis, error, angle_deg,
                           (x0, x1, y0, y1), local))
    if not candidates:
        raise ArenaGeometryError("field corners are degenerate; check FIELD_CORNERS_WORLD")

    candidates.sort(key=lambda item: item[0])
    (score, origin, width, height, x_axis, y_axis, error, angle_deg,
     extents, local_cage) = candidates[0]
    tolerance = (
        float(corner_tolerance_mm)
        if corner_tolerance_mm is not None
        else max(50.0, 0.03 * min(width, height))
    )
    if score > tolerance:
        raise ArenaGeometryError(
            "the cage must touch one corner of the field: cage is "
            f"{score:.0f}mm away from the nearest field corner "
            f"(tolerance {tolerance:.0f}mm); check CAGE_CORNERS_WORLD"
        )
    x0, x1, y0, y1 = extents
    cage_width = x1
    cage_height = height - y0
    if not (0.02 * width <= cage_width <= 0.98 * width
            and 0.02 * height <= cage_height <= 0.98 * height):
        raise ArenaGeometryError(
            f"cage size {cage_width:.0f}x{cage_height:.0f}mm is not a corner region "
            f"of the {width:.0f}x{height:.0f}mm field; check CAGE_CORNERS_WORLD"
        )
    snapped = (
        (0.0, height), (cage_width, height),
        (cage_width, height - cage_height), (0.0, height - cage_height),
    )
    for corner in local_cage:
        nearest = min(math.dist(corner, box_corner) for box_corner in snapped)
        if nearest > tolerance:
            raise ArenaGeometryError(
                "cage edges are not parallel to the field edges "
                f"(corner off by {nearest:.0f}mm); check CAGE_CORNERS_WORLD"
            )
    if opening_mm is not None:
        if not (math.isfinite(opening_mm) and 0.0 < opening_mm < min(cage_width, cage_height)):
            raise ArenaGeometryError(
                "cage opening must be positive and smaller than both cage dimensions"
            )
        if opening_mm <= robot_diameter_mm + gap_mm:
            raise ArenaGeometryError(
                "cage opening must exceed the car diameter plus the body gap"
            )

    arena = RelativeArena(
        origin_world=origin,
        x_axis_world=x_axis,
        y_axis_world=y_axis,
        width_mm=width,
        height_mm=height,
        cage_width_mm=cage_width,
        cage_height_mm=cage_height,
        corner_fit_error_mm=error,
        corner_angle_deg=angle_deg,
    )
    corners = FieldCorners(
        left_up=arena.local_to_world((0.0, height)),
        left_down=arena.local_to_world((0.0, 0.0)),
        right_up=arena.local_to_world((width, height)),
        right_down=arena.local_to_world((width, 0.0)),
    )
    return arena, corners


def sample_initial_positions(tracker, subjects_by_id, marker_ids,
                             duration_sec, minimum_ratio=0.5):
    """Median Vicon position per car over a short window; ignores unstable cars."""
    requested = tuple(marker_ids)
    samples = {marker_id: [] for marker_id in requested}
    frames = 0
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        snapshot = tracker.get_snapshot()
        if snapshot.fresh_subjects:
            frames += 1
            for marker_id in requested:
                subject_name = subjects_by_id[marker_id]
                if subject_name in snapshot.fresh_subjects:
                    samples[marker_id].append(snapshot.poses[subject_name].position)
        time.sleep(0.01)
    minimum = max(3, math.ceil(frames * minimum_ratio))
    return {
        marker_id: (median(point[0] for point in values),
                    median(point[1] for point in values))
        for marker_id, values in samples.items()
        if len(values) >= minimum
    }


class FixedCornerArena:
    """Use hard-coded Vicon world coordinates instead of automatic placement.

    The field and cage coordinates come from the caller (``FIELD_CORNERS_WORLD``
    and ``CAGE_CORNERS_WORLD`` in the experiment entry). Nothing is derived from
    the cars, so the field stays identical between runs.
    """

    def __init__(self, field_corners_world, cage_corners_world, opening_mm=None,
                 robot_diameter_mm=150.0, gap_mm=80.0, minimum_clearance_mm=300.0):
        self.opening = opening_mm
        self.robot_radius = 0.5 * robot_diameter_mm
        self.minimum_clearance = minimum_clearance_mm
        self.field_corners_world = order_rectangle(field_corners_world, "field")
        self.cage_corners_world = order_rectangle(cage_corners_world, "cage")
        self.arena, self.corners = fixed_arena(
            self.field_corners_world, self.cage_corners_world, opening_mm,
            robot_diameter_mm, gap_mm,
        )

    def describe(self):
        arena = self.arena
        origin = arena.origin_world
        print(
            "Fixed field: "
            f"{arena.width_mm:.0f}x{arena.height_mm:.0f}mm, "
            f"origin world=({origin[0]:.1f},{origin[1]:.1f}), "
            f"x axis=({arena.x_axis_world[0]:+.3f},{arena.x_axis_world[1]:+.3f}), "
            f"y axis=({arena.y_axis_world[0]:+.3f},{arena.y_axis_world[1]:+.3f})"
        )
        if self.opening:
            print(
                f"Fixed cage: {arena.cage_width_mm:.0f}x{arena.cage_height_mm:.0f}mm "
                f"in the field corner at local (0,{arena.height_mm:.0f}); "
                f"{self.opening:.0f}mm openings on its bottom and right walls"
            )
        else:
            print(
                f"Fixed cage: {arena.cage_width_mm:.0f}x{arena.cage_height_mm:.0f}mm "
                f"in the field corner at local (0,{arena.height_mm:.0f}); "
                "one-way soft boundary (evaders can be pushed in, cannot leave; "
                "herders stay out)"
            )
        cage_corners_world = {
            "left-up": arena.local_to_world(arena.cage_top_left_local),
            "right-down": arena.local_to_world(arena.cage_right_down_local),
        }
        for name, position in cage_corners_world.items():
            local = arena.world_to_local(position)
            print(
                f"  cage {name:>10}: world=({position[0]:.1f},{position[1]:.1f}) "
                f"local=({local[0]:.1f},{local[1]:.1f})"
            )

    def check_cars(self, positions, herders, evaders, arm):
        """Report every car relative to the fixed field; refuse impossible starts."""
        arena = self.arena
        too_close = []
        outside = []
        for marker_id in sorted(positions):
            local = arena.world_to_local(positions[marker_id])
            clearance = min(
                local[0], arena.width_mm - local[0],
                local[1], arena.height_mm - local[1],
            )
            role = "H" if marker_id in herders else ("E" if marker_id in evaders else "?")
            note = ""
            if not arena.contains_field_local(local, self.robot_radius):
                note = "  <-- OUTSIDE the field body margin"
                outside.append(marker_id)
            elif clearance < self.minimum_clearance:
                note = f"  <-- only {clearance:.0f}mm from the field boundary"
                too_close.append(marker_id)
            print(
                f"  car{marker_id:>2}({role}) world=({positions[marker_id][0]:8.1f},"
                f"{positions[marker_id][1]:8.1f}) local=({local[0]:7.1f},{local[1]:7.1f})"
                f" edge={clearance:6.1f}mm{note}"
            )
        if outside and arm:
            raise ArenaGeometryError(
                "car(s) " + ",".join(map(str, outside))
                + " are outside the configured field (body radius included); "
                "move them inside FIELD_CORNERS_WORLD or correct the coordinates"
            )
        if too_close:
            print(
                "Warning: car(s) " + ",".join(map(str, too_close))
                + f" are closer than {self.minimum_clearance:.0f}mm to the field boundary; "
                "the six-direction calibration pulse may push them across it."
            )
        return outside, too_close

    def __call__(self, tracker, args, subjects_by_id, herders, evaders):
        requested = tuple(herders) + tuple(evaders)
        print("Arena is fixed from FIELD_CORNERS_WORLD / CAGE_CORNERS_WORLD; "
              "initial car positions are only checked, never used for placement.")
        self.describe()
        positions = sample_initial_positions(
            tracker, subjects_by_id, requested,
            getattr(args, "initial_visibility_sec", 1.0),
            getattr(args, "initial_visibility_ratio", 0.5),
        )
        missing = [marker_id for marker_id in requested if marker_id not in positions]
        args.virtual_field_initial_positions = positions
        print(f"Initial Vicon poses: {len(positions)}/{len(requested)} cars.")
        if missing:
            print("No stable initial pose for car(s): " + ",".join(map(str, missing)))
        if positions:
            self.check_cars(positions, herders, evaders, getattr(args, "arm", False))
        return self.arena, self.corners, {}
