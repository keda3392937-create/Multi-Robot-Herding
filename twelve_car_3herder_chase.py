"""Vicon chase using the HERDER_IDS and EVADER_IDS configured below.

Connect the computer to YAYA, then run this Python file. Cars calibrate all six
motion directions together before chasing/escaping. Missing or failed cars are
skipped; each connected car has its own command/connection worker.
"""
import argparse
import ipaddress
import math
import time
from typing import Dict, Optional, Tuple

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
    MIN_DRIVE_VECTOR_NORM,
    VICON_HOST,
    CalibrationMap,
    CarConfig,
    CarObservation,
    ViconTracker,
    best_command_for_vector,
    calibrate_all_cars,
    parse_overrides,
    min_pair_distance,
    send_commands,
    set_speed,
    vec_add,
    vec_norm,
    vec_normalize,
    vec_scale,
    vec_sub,
)


from multi_robot_control import CONTROL_PORT, NETWORK, RobotWorker, discover_fixed


HERDER_IDS = (18,19,20,21,22,23,24,25)
EVADER_IDS = (1, 2, 3, 4, 5, 6,7,8, 9, 10, 11, 12, 13, 14, 15, 16,17)
CAR_IDS = HERDER_IDS + EVADER_IDS
COMMAND_MAX_AGE_SEC = 0.4
CONNECTION_WAIT_SEC = 3.0

DEFAULT_HERDER_CHASE_WEIGHT = 1.0
DEFAULT_HERDER_REPEL_RADIUS_MM = 700.0
DEFAULT_HERDER_REPEL_WEIGHT = 1.25
DEFAULT_EVADER_HERDER_ACTIVE_RADIUS_MM = 1300.0
DEFAULT_EVADER_HERDER_WEIGHT = 1.0
DEFAULT_EVADER_REPEL_WEIGHT = 0.35


def make_config_map(args: argparse.Namespace) -> Dict[int, CarConfig]:
    if len(set(CAR_IDS)) != len(CAR_IDS) or any(car_id not in range(1, 51) for car_id in CAR_IDS):
        raise ValueError("HERDER_IDS and EVADER_IDS must contain distinct car IDs from 1..50.")
    subjects = parse_overrides(args.subject, "subject", valid_ids=CAR_IDS)
    addresses = parse_overrides(args.car_ip, "car-ip", valid_ids=CAR_IDS)
    for address in addresses.values():
        if ipaddress.ip_address(address) not in NETWORK:
            raise ValueError(f"Car IP addresses must be on YAYA's {NETWORK} network.")
    return {
        car_id: CarConfig(car_id, f"car{car_id}", addresses.get(car_id, ""),
                          subjects.get(car_id, f"kedaya{car_id}"))
        for car_id in CAR_IDS
    }


def resolve_endpoints(configs: Dict[int, CarConfig], args: argparse.Namespace) -> dict:
    endpoints = {car_id: (config.ip, CONTROL_PORT)
                 for car_id, config in configs.items() if config.ip}
    wanted = tuple(car_id for car_id in configs if car_id not in endpoints)
    if wanted:
        endpoints.update(discover_fixed(wanted, args.discovery_timeout, args.bind_ip))
    for car_id, (address, _port) in endpoints.items():
        configs[car_id].ip = address
    missing = [f"kedaya{car_id}" for car_id in configs if car_id not in endpoints]
    if missing:
        print("Skipping undiscovered: " + ", ".join(missing))
    return endpoints


class MotionWorker(RobotWorker):
    """Keep each TCP link independent; expired motion requests become STOP."""

    def __init__(self, car_id: int, endpoint: Tuple[str, int], speed: int):
        super().__init__(car_id, endpoint, "STOP", speed)
        self.requested_at = 0.0

    def request(self, command: str) -> None:
        with self.lock:
            if command.startswith("SPD "):
                self.speed = max(0, min(255, int(command[4:])))
            else:
                self.command = command
                self.requested_at = time.monotonic()

    def pending_motion(self, now: float) -> Tuple[int, str]:
        with self.lock:
            command = self.command if now - self.requested_at <= COMMAND_MAX_AGE_SEC else "STOP"
            return self.speed, command

    def _run(self) -> None:
        retry_at = 0.0
        sent_speed = None
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                if self.sock is None:
                    if now < retry_at:
                        self.stop_event.wait(min(0.1, retry_at - now))
                        continue
                    if not self._connect():
                        with self.lock:
                            self.status.reconnects += 1
                        retry_at = time.monotonic() + 0.8
                        continue
                    sent_speed = None
                try:
                    speed, command = self.pending_motion(time.monotonic())
                    payload = (f"SPD {speed}\n" if speed != sent_speed else "") + command + "\n"
                    self.sock.sendall(payload.encode("ascii"))
                    sent_speed = speed
                    with self.lock:
                        self.status.last_tx = time.monotonic()
                        self.status.state = "RUNNING"
                    self.stop_event.wait(0.08)
                except OSError as exc:
                    self._set_state("SEND_FAILED", f"{type(exc).__name__}: {exc}")
                    self._close_socket()
                    with self.lock:
                        self.status.reconnects += 1
                    retry_at = time.monotonic() + 0.8
        finally:
            self._close_socket(send_stop=True)
            self._set_state("STOPPED")


class ChaseCarController:
    """Expose the existing calibration controller API over a per-car worker."""

    def __init__(self, config: CarConfig, endpoint: Tuple[str, int], speed: int):
        self.config = config
        self.worker = MotionWorker(config.marker_id, endpoint, speed)
        self.closed = False

    def connect(self) -> None:
        self.worker.start()

    def send(self, command: str, force: bool = False) -> None:
        if not self.closed:
            self.worker.request(command)

    def connected(self) -> bool:
        status = self.worker.snapshot()
        return status.state == "RUNNING" and time.monotonic() - status.last_tx < 0.5

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.worker.request("STOP")
            self.worker.stop()


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

        active_evaders = {car_id: pose for car_id, pose in observations.items()
                          if car_id in calibration_by_id}
        nearest_id, nearest_distance = nearest_visible_evader(herder, active_evaders)
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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vicon-host", default=VICON_HOST)
    parser.add_argument("--discovery-timeout", type=float, default=DEFAULT_DISCOVERY_TIMEOUT_SEC)
    parser.add_argument("--bind-ip", action="append", default=[])
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
    parser.add_argument("--require-all-visible", action="store_true")
    parser.add_argument("--status-interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Seconds after calibration; 0 runs until Ctrl+C.")
    args = parser.parse_args(argv)
    for name in ("speed", "herder_speed", "evader_speed", "calibration_speed"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 255:
            parser.error(f"--{name.replace('_', '-')} must be 0..255")
    for name in ("discovery_timeout", "calibration_move_sec", "min_displacement_mm",
                 "herder_repel_radius_mm", "evader_herder_active_radius_mm",
                 "evader_repel_radius_mm", "target_step_mm", "status_interval"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("settle_sec", "herder_stop_distance_mm", "duration", "herder_chase_weight",
                 "herder_repel_weight", "evader_herder_weight", "evader_repel_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    return args


def speed_for_id(marker_id: int, args: argparse.Namespace) -> int:
    if marker_id in HERDER_IDS:
        return args.herder_speed if args.herder_speed is not None else args.speed
    return args.evader_speed if args.evader_speed is not None else args.speed


def main() -> None:
    args = parse_args()
    config_by_id = make_config_map(args)
    endpoints = resolve_endpoints(config_by_id, args)
    controllers = {}
    tracker = None
    print("Herders: " + ", ".join(f"kedaya{car_id}" for car_id in HERDER_IDS))
    print("Evaders: " + ", ".join(f"kedaya{car_id}" for car_id in EVADER_IDS))
    try:
        for car_id, endpoint in endpoints.items():
            controller = ChaseCarController(config_by_id[car_id], endpoint, speed_for_id(car_id, args))
            controllers[car_id] = controller
            controller.connect()
        deadline = time.monotonic() + CONNECTION_WAIT_SEC
        while controllers and time.monotonic() < deadline:
            if all(controller.connected() for controller in controllers.values()):
                break
            time.sleep(0.05)
        for car_id in tuple(controllers):
            if not controllers[car_id].connected():
                print(f"Skipping kedaya{car_id}: TCP connection failed.")
                controllers.pop(car_id).close()
        if not controllers:
            print("No requested cars connected. Check YAYA and car power.")
            return

        tracker = ViconTracker(args.vicon_host, config_by_id)
        tracker.connect()
        calibration_by_id = calibrate_all_cars(controllers, tracker, args)
        for car_id, controller in controllers.items():
            set_speed(controller, speed_for_id(car_id, args))
            controller.send("STOP", force=True)
        for car_id in tuple(controllers):
            if car_id not in calibration_by_id:
                print(f"Skipping kedaya{car_id}: no complete motion calibration.")
                controllers.pop(car_id).close()
        if not controllers:
            print("No cars passed motion calibration; stopped.")
            return

        print("Calibrated: " + ", ".join(f"kedaya{car_id}" for car_id in controllers))
        print(f"Running chase: {len(HERDER_IDS)} configured herders, "
              f"{len(EVADER_IDS)} configured evaders. Press Ctrl+C to stop.")
        last_status_time = 0.0
        run_started = time.monotonic()
        while args.duration == 0 or time.monotonic() - run_started < args.duration:
            observations = tracker.get_observations()
            summary, commands, _, _, _, _ = compute_commands(
                observations, config_by_id, calibration_by_id, args,
            )
            send_commands(commands, controllers)
            now = time.monotonic()
            if now - last_status_time >= args.status_interval:
                command_text = " ".join(f"ID{car_id}:{commands[car_id]}" for car_id in CAR_IDS)
                connected = sum(controller.connected() for controller in controllers.values())
                print(f"{summary} | TCP={connected}/{len(controllers)} | {command_text}")
                last_status_time = now
            time.sleep(CONTROL_LOOP_SEC)
    except KeyboardInterrupt:
        print("Interrupted; stopping all cars.")
    finally:
        # Clear every motion request before waiting for any worker to finish.
        for controller in controllers.values():
            controller.send("STOP", force=True)
        for controller in controllers.values():
            controller.close()
        if tracker is not None:
            tracker.client.Disconnect()
        print("All cars stopped; Vicon disconnected.")


if __name__ == "__main__":
    main()
