"""Baseline force adapter and collision guard, independent of Vicon/network.

Finite-horizon checks depend on measured geometry and calibrated speed accuracy;
they are not a proof of physical collision avoidance or eventual capture.
"""
from __future__ import annotations

import math
from dataclasses import replace
import numpy as np
from baseline_forces import BaselineForces
from baseline_virtual_arena import (
    cage_exit_target, cage_gate_target, cage_walls, evader_membrane_force,
    herder_cage_repulsion, segment_distance, virtual_wall_force,
)
from relative_cage_core import (
    ControlParameters, ControlResult, DynamicsLimits, SecondOrderDynamics,
    augmented_hull, hull_for_ids, point_in_convex_polygon, field_boundary_force,
    vec_add, vec_sub, vec_scale, vec_dot, vec_norm, vec_normalize,
)


def physical_parameters(arena, radius, time_scale=1.6, herder_accel=300.0, evader_accel=450.0):
    if not all(math.isfinite(v) and v > 0 for v in (radius, time_scale, herder_accel, evader_accel)):
        raise ValueError("Physical dimensions, time scale and acceleration must be positive and finite")
    unit = arena.scale_mm / 80.0
    return replace(ControlParameters.from_arena(arena, radius),
                   herder_dynamics=DynamicsLimits(2 * unit * time_scale, herder_accel, 0.0),
                   evader_dynamics=DynamicsLimits(3 * unit * time_scale, evader_accel, 0.0))


class BaselineController:
    """Keep source FOV, density, convex hull and acceleration expressions.

    Isotropic scaling preserves angles and the 2:3 speed limit ratio. Positions
    come from Vicon, with no simulated clipping or simulated position feedback.
    The source's very large escape acceleration gets a physical acceleration cap.
    """
    def __init__(self, time_scale=1.6, body_gap_mm=80.0,
                 cage_opening_mm=None, wall_margin_mm=300.0,
                 cage_membrane_mm=None, herder_cage_margin_mm=200.0):
        self.model = BaselineForces()
        self.time_scale = time_scale
        self.body_gap_mm = body_gap_mm
        self.headings = {}
        self.elapsed = 0.0
        self.dynamics = SecondOrderDynamics()
        self.cage_opening = cage_opening_mm
        self.wall_margin = wall_margin_mm
        self.cage_membrane = cage_membrane_mm
        self.herder_cage_margin = herder_cage_margin_mm

    def reset(self, marker_id):
        self.dynamics.reset(marker_id)

    def __call__(self, observations, herders, evaders, obstacles, arena, params,
                 dynamics, dt, drive_evaders=True):
        unit = arena.scale_mm / 80.0
        h = tuple(i for i in herders if i in observations)
        e = tuple(i for i in evaders if i in observations)
        for i in set(self.dynamics.velocity_by_id) - set(h + e):
            self.reset(i)
        positions = {i: o.position for i, o in observations.items()}
        m = self.model
        m.numHerders, m.numEvaders = len(h), len(e)
        m.arenaWidth, m.arenaHeight = arena.width_mm / unit, arena.height_mm / unit
        m.cageLeft, m.cageRight = arena.cage_left / unit, arena.cage_right / unit
        m.cageBottom, m.cageTop = arena.cage_bottom / unit, arena.cage_top / unit
        m.cageCenter = np.array(arena.cage_center_local) / unit
        m.positionsHerder = np.array([positions[i] for i in h], dtype=float).reshape(-1, 2) / unit
        m.positionsEvader = np.array([positions[i] for i in e], dtype=float).reshape(-1, 2) / unit
        # Detect opponents before a finite chassis exhausts its clearance.
        clearance = 2 * params.robot_radius_mm + self.body_gap_mm
        m.Re = max(6.0, (clearance + 0.08 * arena.scale_mm) / unit)
        m.Rh = max(12.0, 1.6 * m.Re)
        m.herder_repulsion_dist = max(8.0, 1.5 * clearance / unit)
        m.dt = dt * self.time_scale
        m.herder_directions = np.array([
            self.headings.get(i, math.atan2(m.cageCenter[1] - m.positionsHerder[k, 1],
                                          m.cageCenter[0] - m.positionsHerder[k, 0]))
            for k, i in enumerate(h)
        ])
        ah = m.update_herders_with_convex_hull(self.elapsed * self.time_scale)
        ae = m.evader_accelerations()
        self.elapsed += dt
        self.headings = {i: float(m.herder_directions[k]) for k, i in enumerate(h)}
        accelerations, velocities = {}, {}
        gain = unit * self.time_scale ** 2
        for ids, values, limits in ((h, ah, params.herder_dynamics), (e, ae, params.evader_dynamics)):
            for k, i in enumerate(ids):
                if i in e and not drive_evaders:
                    self.reset(i)
                    accelerations[i] = velocities[i] = (0.0, 0.0)
                    continue
                a = tuple(float(v) * gain for v in values[k])
                if not all(math.isfinite(v) for v in a):
                    raise ValueError(f"Non-finite baseline acceleration for car{i}")
                n = vec_norm(a)
                if n > limits.max_acceleration:
                    a = vec_scale(a, limits.max_acceleration / n)
                if self.cage_membrane is not None:
                    # 一向软边界：evader 从任意位置都能被推进笼子，进笼后不能再出来；
                    # herder 被笼子挡在外面，只能把 evader 往里推。
                    boundary = field_boundary_force(
                        positions[i], arena, self.wall_margin + params.robot_radius_mm,
                        2.0 * limits.max_acceleration,
                    )
                    a = vec_add(a, boundary)
                    if i in e:
                        a = vec_add(a, evader_membrane_force(
                            positions[i], arena, self.cage_membrane,
                            2.0 * limits.max_acceleration,
                        ))
                    else:
                        a = vec_add(a, herder_cage_repulsion(
                            positions[i], arena, self.herder_cage_margin,
                            2.0 * limits.max_acceleration, params.robot_radius_mm,
                        ))
                        if arena.contains_cage_local(positions[i]):
                            target = cage_exit_target(positions[i], arena)
                            a = vec_scale(vec_normalize(vec_sub(target, positions[i])),
                                          limits.max_acceleration)
                    n = vec_norm(a)
                    if n > limits.max_acceleration:
                        a = vec_scale(a, limits.max_acceleration / n)
                elif self.cage_opening is not None:
                    # Apply virtual barriers after capping the baseline's huge
                    # escape force, so they can actually redirect a real evader.
                    barrier = virtual_wall_force(
                        positions[i], arena, self.cage_opening, self.wall_margin,
                        2.0 * limits.max_acceleration, params.robot_radius_mm,
                    )
                    boundary = field_boundary_force(
                        positions[i], arena, self.wall_margin + params.robot_radius_mm,
                        2.0 * limits.max_acceleration,
                    )
                    a = vec_add(a, vec_add(barrier, boundary))
                    if i in h and arena.contains_cage_local(positions[i]):
                        target = cage_gate_target(positions[i], arena, self.cage_opening)
                        a = vec_scale(vec_normalize(vec_sub(target, positions[i])), limits.max_acceleration)
                    n = vec_norm(a)
                    if n > limits.max_acceleration:
                        a = vec_scale(a, limits.max_acceleration / n)
                accelerations[i] = a
                # Preserve the simulation's desired velocity state. Replacing
                # it every frame with measured zero prevents acceleration out
                # of the motor deadband. Vicon velocity is used independently
                # by the runtime speed/response monitor and collision guard.
                velocities[i] = self.dynamics.step(i, a, dt, limits)
        inside = tuple(i for i in e if arena.contains_cage_local(positions[i], params.cage_hold_margin_mm))
        outside = tuple(i for i in e if i not in inside)
        hull_h, _ = hull_for_ids(h, positions)
        aug_h, polygon = augmented_hull(h, positions, arena)
        contained = tuple(i for i in outside if point_in_convex_polygon(positions[i], polygon))
        targets, target_ids = {}, {}
        for k, i in enumerate(h):
            candidates = [e[int(j)] for j in m.get_visible_evaders(k)]
            target = (min(candidates, key=lambda j: vec_norm(vec_sub(positions[i], positions[j])))
                      if candidates else max(outside, key=lambda j: vec_norm(vec_sub(positions[j], arena.cage_center_local)), default=None))
            target_ids[i] = target
            if target is not None:
                targets[i] = positions[target]
        return ControlResult(velocities, accelerations, targets, target_ids, h, e, hull_h, aug_h,
                             contained, inside, outside,
                             tuple(i for i in herders if i not in positions),
                             tuple(i for i in evaders if i not in positions))


def swept_distance(relative_position, relative_velocity, duration):
    """Minimum centre distance over continuous time, not just endpoints."""
    vv = vec_dot(relative_velocity, relative_velocity)
    t = 0.0 if vv < 1e-12 else max(0.0, min(duration, -vec_dot(relative_position, relative_velocity) / vv))
    return vec_norm(vec_add(relative_position, vec_scale(relative_velocity, t)))


def rotate(vector, angle):
    c, s = math.cos(angle), math.sin(angle)
    return (c * vector[0] - s * vector[1], s * vector[0] + c * vector[1])


def segment_hits_cage(start, end, arena, margin):
    """Slab intersection with the expanded virtual cage rectangle."""
    lower = (arena.cage_left - margin, arena.cage_bottom - margin)
    upper = (arena.cage_right + margin, arena.cage_top + margin)
    lo, hi = 0.0, 1.0
    for axis in (0, 1):
        delta = end[axis] - start[axis]
        if abs(delta) < 1e-12:
            if not lower[axis] <= start[axis] <= upper[axis]:
                return False
        else:
            a, b = (lower[axis] - start[axis]) / delta, (upper[axis] - start[axis]) / delta
            lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
            if lo > hi:
                return False
    return True


def herder_cage_path_allowed(start, end, arena, margin):
    if arena.contains_cage_local(start):
        # Permit an initially enclosed herder to leave through a gate. Prevent
        # it moving deeper into the top-left target while it is still inside.
        return min(arena.cage_right - end[0], end[1] - arena.cage_bottom) < min(
            arena.cage_right - start[0], start[1] - arena.cage_bottom) - 1e-6
    def outside_distance(p):
        return math.hypot(max(arena.cage_left - p[0], 0, p[0] - arena.cage_right),
                          max(arena.cage_bottom - p[1], 0, p[1] - arena.cage_top))
    distance = outside_distance(start)
    if distance < margin:
        return outside_distance(end) > distance + 1e-6
    return not segment_hits_cage(start, end, arena, margin)


class PhysicalCommandFilter:
    """Check calibrated/PWM commands against all visible cars, including failures.

    Prediction includes a measured coasting segment. Each other car is considered
    stationary, continuing at measured velocity, and following its chosen command.
    No feasible direction means STOP, not passage through a repulsive neighbour.
    """
    def __init__(self, gap_mm=80.0, horizon_sec=1.2, reaction_sec=0.25,
                 speed_factor=1.4, jam_timeout_sec=30.0,
                 herder_min_pwm=90, evader_min_pwm=95, isolate_failures=False,
                 cage_opening_mm=None, cage_membrane_mm=None):
        values = (gap_mm, horizon_sec, reaction_sec, speed_factor, jam_timeout_sec)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Collision guard settings must be finite and positive")
        if horizon_sec <= reaction_sec or speed_factor < 1:
            raise ValueError("Prediction must exceed reaction time; uncertainty must be >= 1")
        self.gap, self.horizon, self.reaction = gap_mm, horizon_sec, reaction_sec
        self.speed_factor, self.jam_timeout = speed_factor, jam_timeout_sec
        self.blocked_since = None
        self.last_status = ""
        self.last_status_print_at = -math.inf
        self.minimum_pwm = (herder_min_pwm, evader_min_pwm)
        self.known_positions = {}
        self.known_speeds = {}
        self.lost_padding = {}
        self.isolate_failures = isolate_failures
        self.cage_opening = cage_opening_mm
        self.cage_membrane = cage_membrane_mm
        self.herder_ids = set()
        # A herder this close to a wall-trapped evader backs away, otherwise the
        # evader has no collision-free direction and stays pinned in the corner.
        self.yield_radius_factor = 3.0

    def safe_velocity(self, i, velocity, positions, measured, chosen, arena, radius):
        p = positions[i]
        coast = measured.get(i, (0.0, 0.0))
        reaction_end = vec_add(p, vec_scale(coast, self.reaction))
        end = vec_add(reaction_end, vec_scale(velocity, self.horizon - self.reaction))
        if any(not arena.contains_field_local(q, radius + self.gap * 0.5)
               for q in (p, reaction_end, end)):
            return False
        if self.cage_opening is not None:
            for wall in cage_walls(arena, self.cage_opening):
                if min(segment_distance(p, reaction_end, *wall),
                       segment_distance(reaction_end, end, *wall)) < radius + self.gap / 2:
                    return False
        if self.cage_membrane is not None and i not in self.herder_ids:
            # One-way soft boundary: an evader inside the cage may not leave it,
            # while an evader outside is free to be pushed in anywhere.
            if arena.contains_cage_local(p) and not arena.contains_cage_local(end):
                return False
        for j, q in positions.items():
            if j == i:
                continue
            separation = 2 * radius + self.gap + self.lost_padding.get(j, 0.0)
            peer_coast = measured.get(j, (0.0, 0.0))
            if swept_distance(vec_sub(p, q), vec_sub(coast, peer_coast), self.reaction) < separation:
                return False
            peer_end = vec_add(q, vec_scale(peer_coast, self.reaction))
            relative_start = vec_sub(reaction_end, peer_end)
            for peer_v in ((0.0, 0.0), peer_coast, chosen.get(j, (0.0, 0.0))):
                if swept_distance(relative_start, vec_sub(velocity, peer_v), self.horizon - self.reaction) < separation:
                    return False
        return True

    def __call__(self, decisions, observations, world_poses, calibrations, arena,
                 params, measured, herders, now):
        radius = params.robot_radius_mm
        self.herder_ids = set(herders)
        positions = {i: o.position for i, o in observations.items()
                     if i in decisions or arena.contains_field_local(o.position, -(radius + self.gap))}
        # An excluded car stays an obstacle. In fault-isolation mode retain its
        # last position plus a watchdog/coasting envelope if tracking vanishes.
        missing_obstacles = set(self.known_positions) - set(observations)
        if missing_obstacles and not self.isolate_failures:
            raise RuntimeError("Lost previously visible obstacle pose(s): " + ",".join(map(str, sorted(missing_obstacles))))
        new_lost = missing_obstacles - set(self.lost_padding)
        for i in missing_obstacles:
            positions[i] = self.known_positions[i]
            maximum = max(params.herder_dynamics.max_speed, params.evader_dynamics.max_speed)
            self.lost_padding[i] = max(self.known_speeds.get(i, 0.0), maximum) * self.horizon
        for i in tuple(self.known_positions):
            if i in observations and i not in positions:
                self.known_positions.pop(i, None)
                self.known_speeds.pop(i, None)
                self.lost_padding.pop(i, None)
        for i, p in positions.items():
            if i in observations:
                self.lost_padding.pop(i, None)
                self.known_positions[i] = p
                self.known_speeds[i] = vec_norm(measured.get(i, (0.0, 0.0)))
        if new_lost:
            print("Retaining lost car(s) as obstacles: " + ",".join(map(str, sorted(new_lost))))
        clearance = 2 * radius + self.gap
        ids = tuple(positions)
        held_ids = set()
        for k, i in enumerate(ids):
            for j in ids[k + 1:]:
                if vec_norm(vec_sub(positions[i], positions[j])) < clearance:
                    if not self.isolate_failures:
                        raise RuntimeError(f"Cars {i}/{j} below {clearance:.0f}mm clearance; stop and separate them")
                    held_ids.update((i, j))
        chosen, output, blocked = {}, {}, []
        moving_requested = any(d.command != "STOP" and d.pwm > 0 for d in decisions.values())
        for i in sorted(decisions, key=lambda i: (i not in herders, i)):
            original = decisions[i]
            output[i] = type(original)("STOP", 0.0, 0)
            chosen[i] = (0.0, 0.0)
            if i in held_ids:
                blocked.append(i)
                continue
            if i not in positions or i not in world_poses or original.command == "STOP" or original.pwm <= 0:
                continue
            calibration = calibrations[i]
            if original.command not in calibration:
                blocked.append(i)
                continue
            yaw = world_poses[i].yaw
            reference = calibration[original.command]
            preferred = arena.world_vector_to_local(rotate(reference.direction_world, yaw - reference.yaw))
            # Yield space to a boundary-trapped evader rather than resisting its
            # inward boundary force with a herder on the same escape path.
            if i in herders:
                if self.cage_opening is not None and arena.contains_cage_local(positions[i]):
                    preferred = vec_normalize(vec_sub(cage_gate_target(positions[i], arena, self.cage_opening), positions[i]))
                elif self.cage_membrane is not None and arena.contains_cage_local(positions[i]):
                    preferred = vec_normalize(vec_sub(cage_exit_target(positions[i], arena), positions[i]))
                for j, p in (() if arena.contains_cage_local(positions[i]) else positions.items()):
                    if j in herders:
                        continue
                    wall = min(p[0], p[1], arena.width_mm - p[0], arena.height_mm - p[1])
                    if (wall < radius + clearance
                            and vec_norm(vec_sub(positions[i], p)) < self.yield_radius_factor * clearance):
                        preferred = vec_normalize(vec_sub(positions[i], p))
                        break
            candidates = []
            for command, ref in calibration.items():
                direction = vec_normalize(arena.world_vector_to_local(rotate(ref.direction_world, yaw - ref.yaw)))
                score = vec_dot(direction, vec_normalize(preferred))
                if score < 0.1:
                    continue
                for fraction in (1.0, 0.65, 0.4):
                    pwm = max(1, int(original.pwm * fraction))
                    # Preserve the runtime's PWM ramp through the motor dead
                    # zone. Turning each sub-minimum ramp step into STOP would
                    # reset PWM to zero forever. Safety reductions of an
                    # already-running command still respect minimum drive PWM.
                    minimum = min(original.pwm, self.minimum_pwm[0 if i in herders else 1])
                    if pwm < minimum:
                        continue
                    nominal = ref.measured_speed_mm_s * pwm / ref.calibration_pwm
                    maximum = (params.herder_dynamics if i in herders else params.evader_dynamics).max_speed
                    if nominal > maximum:
                        continue
                    speed = max(nominal * self.speed_factor, vec_norm(measured.get(i, (0.0, 0.0))))
                    velocity = vec_scale(direction, speed)
                    if i in herders:
                        end = vec_add(positions[i], vec_scale(velocity, self.horizon))
                        if not herder_cage_path_allowed(positions[i], end, arena, radius + self.gap / 2):
                            continue
                    if self.safe_velocity(i, velocity, positions, measured, chosen, arena, radius):
                        candidates.append((score * fraction, command, pwm, velocity, score))
            if candidates:
                _, command, pwm, velocity, score = max(candidates, key=lambda c: c[0])
                output[i] = type(original)(command, score, pwm)
                chosen[i] = velocity
            else:
                blocked.append(i)
        # Replacing a command with STOP can block a peer. Recheck to a fixed point.
        for _ in range(len(output) + 1):
            unsafe = [i for i in output if output[i].command != "STOP"
                      and not self.safe_velocity(i, chosen[i], positions, measured, chosen, arena, radius)]
            if not unsafe:
                break
            for i in unsafe:
                output[i] = type(output[i])("STOP", 0.0, 0)
                chosen[i] = (0.0, 0.0)
                blocked.append(i)
        if moving_requested and not any(d.command != "STOP" for d in output.values()):
            if self.blocked_since is None:
                self.blocked_since = now
            elif now - self.blocked_since >= self.jam_timeout:
                raise RuntimeError("No collision-free command batch; reposition cars or enlarge the field")
        else:
            self.blocked_since = None
        status = ",".join(map(str, sorted(set(blocked))))
        if status != self.last_status:
            if status and now - self.last_status_print_at >= 1.0:
                print(f"Collision guard holding car(s): {status}")
                self.last_status_print_at = now
                self.last_status = status
            elif not status:
                self.last_status = ""
        return output
