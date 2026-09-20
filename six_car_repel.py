import argparse
import concurrent.futures
import ipaddress
import math
import os
import re
import select
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from xml.sax.saxutils import escape


VICON_SDK_PATH = Path(r"D:\ViconDataStream\Win64\Python\vicon_dssdk")
if VICON_SDK_PATH.exists():
    sys.path.insert(0, str(VICON_SDK_PATH))

from vicon_dssdk import ViconDataStream


VICON_HOST = "192.168.30.100"
CONTROL_PORT = 23
DISCOVERY_PORT = 4210
DISCOVERY_REQUEST = "BOID_CAR_DISCOVER"
DISCOVERY_PREFIX = "BOID_CAR"
DEFAULT_WIFI_SSID = "YAYA"
DEFAULT_WIFI_PASSWORD = "kedayaya"
DEFAULT_WIFI_WAIT_SEC = 4.0
DEFAULT_DISCOVERY_TIMEOUT_SEC = 5.0
COMMAND_INTERVAL_SEC = 0.08
CONTROL_LOOP_SEC = 0.05
TRACK_HOLD_SEC = 0.25
ACTIVE_REPULSION_RADIUS_MM = 550.0
CENTROID_PUSH_WEIGHT = 0.45
MIN_DRIVE_VECTOR_NORM = 0.08
MIN_VISIBLE_CARS_TO_MOVE = 2
STATUS_INTERVAL_SEC = 0.5
DEFAULT_SPEED = 255
DEFAULT_CALIBRATION_SPEED = 255 
DEFAULT_CALIBRATION_MOVE_SEC = 0.45
DEFAULT_SETTLE_SEC = 0.35
DEFAULT_MIN_DISPLACEMENT_MM = 30.0
DEFAULT_TARGET_STEP_MM = 350.0
MOTION_COMMANDS = ("F", "RF", "RB", "B", "LB", "LF")

CAR_IDS = tuple(range(1, 17))

CAR_CONFIGS = {
    1: ("car1_sta", "", "kedaya1"),
    2: ("car2_sta", "", "kedaya2"),
    3: ("car3_sta", "", "kedaya3"),
    4: ("car4_sta", "", "kedaya4"),
    5: ("car5_sta", "", "kedaya5"),
    6: ("car6_sta", "", "kedaya6"),
    7: ("car7_sta", "", "kedaya7"),
    8: ("car8_sta", "", "kedaya8"),
    9: ("car9_sta", "", "kedaya9"),
    10: ("car10_sta", "", "kedaya10"),
    11: ("car11_sta", "", "kedaya11"),
    12: ("car12_sta", "", "kedaya12"),
    13: ("car13_sta", "", "kedaya13"),
    14: ("car14_sta", "", "kedaya14"),
    15: ("car15_sta", "", "kedaya15"),
    16: ("car16_sta", "", "kedaya16"),
}


@dataclass
class CarConfig:
    marker_id: int
    name: str
    ip: str
    subject_name: str


@dataclass
class CarObservation:
    marker_id: int
    position: Tuple[float, float]
    forward: Tuple[float, float]
    yaw: float
    seen_at: float


@dataclass
class CommandCalibration:
    direction: Tuple[float, float]
    yaw: float
    displacement_mm: float


CalibrationMap = Dict[int, Dict[str, CommandCalibration]]


class TcpCarController:
    def __init__(self, config: CarConfig):
        self.config = config
        self.sock: Optional[socket.socket] = None
        self.last_command: Optional[str] = None
        self.last_send_time = 0.0

    def connect(self) -> bool:
        self.close(send_stop=False)
        if not self.config.ip:
            return False

        try:
            sock = socket.create_connection((self.config.ip, CONTROL_PORT), timeout=1.2)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(1.0)
            self.sock = sock
            self.last_command = None
            self.last_send_time = 0.0
            print(f"Connected to {self.config.name} at {self.config.ip}:{CONTROL_PORT}")
            return True
        except OSError as exc:
            print(f"Connect failed for {self.config.name}: {exc}")
            self.sock = None
            return False

    def send(self, command: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and command == self.last_command and now - self.last_send_time < COMMAND_INTERVAL_SEC:
            return

        if self.sock is None and not self.connect():
            return

        try:
            assert self.sock is not None
            self.sock.sendall((command + "\n").encode("utf-8"))
            self.last_command = command
            self.last_send_time = now
        except OSError as exc:
            print(f"Send failed for {self.config.name}: {exc}; reconnecting and retrying once.")
            self.close(send_stop=False)
            if not self.connect():
                return
            try:
                assert self.sock is not None
                self.sock.sendall((command + "\n").encode("utf-8"))
                self.last_command = command
                self.last_send_time = time.monotonic()
            except OSError as retry_exc:
                print(f"Retry send failed for {self.config.name}: {retry_exc}")
                self.close(send_stop=False)

    def close(self, send_stop: bool = True) -> None:
        if self.sock is not None:
            if send_stop:
                try:
                    self.sock.sendall(b"STOP\n")
                except OSError:
                    pass
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None


class ViconTracker:
    def __init__(self, host: str, config_by_id: Dict[int, CarConfig]):
        self.host = host
        self.config_by_id = config_by_id
        self.client = ViconDataStream.Client()
        self.root_segments: Dict[int, str] = {}
        self.cached_observations: Dict[int, CarObservation] = {}

    def connect(self) -> None:
        print(f"Connecting to Vicon server: {self.host} ...")
        self.client.SetConnectionTimeout(1000)
        while not self.client.IsConnected():
            try:
                self.client.Connect(self.host)
            except ViconDataStream.DataStreamException as exc:
                print("Vicon connect failed:", exc)
                time.sleep(1.0)

        print("Vicon connected.")
        self.client.EnableSegmentData()
        self.client.SetStreamMode(ViconDataStream.Client.StreamMode.EClientPull)
        self.client.SetAxisMapping(
            ViconDataStream.Client.AxisMapping.EForward,
            ViconDataStream.Client.AxisMapping.ELeft,
            ViconDataStream.Client.AxisMapping.EUp,
        )

    def get_observations(self) -> Dict[int, CarObservation]:
        try:
            has_frame = self.client.GetFrame()
        except ViconDataStream.DataStreamException as exc:
            print("Vicon GetFrame failed:", exc)
            return self.active_observations()

        if not has_frame:
            return self.active_observations()

        now = time.monotonic()
        for marker_id, config in self.config_by_id.items():
            root_segment = self.root_segments.get(marker_id)
            if root_segment is None:
                try:
                    root_segment = self.client.GetSubjectRootSegmentName(config.subject_name)
                    self.root_segments[marker_id] = root_segment
                    print(f"Vicon subject car{marker_id}: {config.subject_name}")
                except ViconDataStream.DataStreamException:
                    continue

            try:
                translation, translation_occluded = self.client.GetSegmentGlobalTranslation(
                    config.subject_name,
                    root_segment,
                )
                rotation, rotation_occluded = self.client.GetSegmentGlobalRotationEulerXYZ(
                    config.subject_name,
                    root_segment,
                )
            except ViconDataStream.DataStreamException:
                continue

            if translation_occluded or rotation_occluded:
                continue

            x_mm, y_mm, _z_mm = translation
            _rx, _ry, rz = rotation
            self.cached_observations[marker_id] = CarObservation(
                marker_id=marker_id,
                position=(float(x_mm), float(y_mm)),
                forward=(math.cos(rz), math.sin(rz)),
                yaw=float(rz),
                seen_at=now,
            )

        return self.active_observations()

    def active_observations(self) -> Dict[int, CarObservation]:
        now = time.monotonic()
        return {
            marker_id: obs
            for marker_id, obs in self.cached_observations.items()
            if now - obs.seen_at <= TRACK_HOLD_SEC
        }


def add_wifi_profile(ssid: str, password: str) -> bool:
    profile_path = Path(tempfile.gettempdir()) / f"vicon_car_wifi_{ssid}_{os.getpid()}.xml"
    profile_xml = f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
    <name>{escape(ssid)}</name>
    <SSIDConfig>
        <SSID>
            <name>{escape(ssid)}</name>
        </SSID>
    </SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>auto</connectionMode>
    <MSM>
        <security>
            <authEncryption>
                <authentication>WPA2PSK</authentication>
                <encryption>AES</encryption>
                <useOneX>false</useOneX>
            </authEncryption>
            <sharedKey>
                <keyType>passPhrase</keyType>
                <protected>false</protected>
                <keyMaterial>{escape(password)}</keyMaterial>
            </sharedKey>
        </security>
    </MSM>
</WLANProfile>
"""
    try:
        profile_path.write_text(profile_xml, encoding="utf-8")
        result = subprocess.run(
            ["netsh", "wlan", "add", "profile", f"filename={profile_path}", "user=current"],
            capture_output=True,
            check=False,
        )
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        output = (stdout + stderr).strip()
        if output:
            print(output)
        return result.returncode == 0
    finally:
        try:
            profile_path.unlink(missing_ok=True)
        except OSError:
            pass


def connect_wifi(ssid: str, wait_sec: float, password: str = "") -> None:
    if sys.platform != "win32":
        print("Wi-Fi auto-connect is only implemented for Windows; skipping netsh.")
        return

    print(f"Connecting Wi-Fi profile: {ssid}")
    result = subprocess.run(
        ["netsh", "wlan", "connect", f"name={ssid}"],
        capture_output=True,
        check=False,
    )
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    output = (stdout + stderr).strip()
    if output:
        print(output)
    if result.returncode != 0:
        profile_password = password
        if not profile_password and ssid == DEFAULT_WIFI_SSID:
            profile_password = DEFAULT_WIFI_PASSWORD

        if profile_password:
            print(f"Wi-Fi profile {ssid} was not connected; adding profile and retrying.")
            if add_wifi_profile(ssid, profile_password):
                result = subprocess.run(
                    ["netsh", "wlan", "connect", f"name={ssid}"],
                    capture_output=True,
                    check=False,
                )
                stdout = result.stdout.decode("utf-8", errors="replace")
                stderr = result.stderr.decode("utf-8", errors="replace")
                output = (stdout + stderr).strip()
                if output:
                    print(output)

        if result.returncode != 0:
            print("Wi-Fi connect command failed. Make sure this SSID has been saved in Windows.")
    time.sleep(wait_sec)


def parse_discovery_reply(data: bytes, source_ip: str) -> Optional[Tuple[int, str]]:
    text = data.decode("utf-8", errors="replace").strip()
    if not text.startswith(DISCOVERY_PREFIX):
        return None

    fields = {}
    for key, value in re.findall(r"(\w+)=([^\s]+)", text):
        fields[key.lower()] = value

    try:
        marker_id = int(fields["id"])
    except (KeyError, ValueError):
        return None

    return marker_id, fields.get("ip", source_ip)


def is_usable_local_ipv4(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False
    if ip.version != 4 or ip.is_loopback or ip.is_link_local or ip.is_multicast:
        return False
    first_octet = int(ip_text.split(".")[0])
    if first_octet >= 224 or ip_text.startswith("255."):
        return False
    return ip_text not in {"0.0.0.0", "255.255.255.0", "255.255.0.0", "255.254.0.0"}


def local_ipv4_candidates() -> Tuple[str, ...]:
    candidates = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip_text = info[4][0]
            if is_usable_local_ipv4(ip_text):
                candidates.add(ip_text)
    except OSError:
        pass

    if sys.platform == "win32":
        result = subprocess.run(["ipconfig"], capture_output=True, check=False)
        output = result.stdout.decode("utf-8", errors="replace")
        for line in output.splitlines():
            if "IPv4" not in line:
                continue
            for ip_text in re.findall(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])", line):
                if is_usable_local_ipv4(ip_text):
                    candidates.add(ip_text)

    return tuple(sorted(candidates))


def default_broadcast_targets(bind_ips) -> Tuple[str, ...]:
    targets = {"255.255.255.255"}
    for ip_text in bind_ips:
        parts = ip_text.split(".")
        if len(parts) == 4:
            targets.add(".".join(parts[:3] + ["255"]))
    return tuple(sorted(targets))


def make_discovery_sockets(bind_ips) -> list:
    sockets = []
    bind_targets = [""] + list(bind_ips)
    for bind_ip in bind_targets:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.setblocking(False)
            sock.bind((bind_ip, 0))
            sockets.append(sock)
        except OSError as exc:
            print(f"Discovery socket bind failed for {bind_ip or 'default'}: {exc}")
    return sockets


def discover_car_ips(
    expected_ids,
    timeout_sec: float,
    bind_ips=(),
    broadcast_ips=(),
) -> Dict[int, str]:
    discovered: Dict[int, str] = {}
    deadline = time.monotonic() + timeout_sec
    discovered_bind_ips = tuple(bind_ips) if bind_ips else local_ipv4_candidates()
    targets = tuple(broadcast_ips) if broadcast_ips else default_broadcast_targets(discovered_bind_ips)
    sockets = make_discovery_sockets(discovered_bind_ips)

    print(f"Discovering cars on UDP {DISCOVERY_PORT} for {timeout_sec:.1f}s...")
    if discovered_bind_ips:
        print("Discovery local IPv4 candidates: " + ", ".join(discovered_bind_ips))
    print("Discovery broadcast targets: " + ", ".join(targets))

    try:
        next_broadcast = 0.0
        while time.monotonic() < deadline and set(discovered) < set(expected_ids):
            now = time.monotonic()
            if now >= next_broadcast:
                payload = DISCOVERY_REQUEST.encode("utf-8")
                for sock in sockets:
                    for target in targets:
                        try:
                            sock.sendto(payload, (target, DISCOVERY_PORT))
                        except OSError:
                            pass
                next_broadcast = now + 0.35

            readable, _, _ = select.select(sockets, [], [], 0.15)
            for sock in readable:
                try:
                    data, addr = sock.recvfrom(256)
                except OSError:
                    continue

                parsed = parse_discovery_reply(data, addr[0])
                if parsed is None:
                    continue

                marker_id, ip = parsed
                if marker_id in expected_ids and discovered.get(marker_id) != ip:
                    discovered[marker_id] = ip
                    print(f"Discovered car{marker_id}: {ip}")
    finally:
        for sock in sockets:
            sock.close()

    return discovered


def parse_overrides(overrides, label: str) -> Dict[int, str]:
    result: Dict[int, str] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid --{label} value '{override}', expected ID=VALUE")
        marker_id_text, value = override.split("=", 1)
        marker_id = int(marker_id_text)
        if marker_id not in CAR_IDS:
            raise ValueError(f"Invalid car id {marker_id}; expected one of {CAR_IDS}")
        result[marker_id] = value.strip()
    return result


def make_config_map(args: argparse.Namespace) -> Dict[int, CarConfig]:
    subject_overrides = parse_overrides(args.subject, "subject")
    car_ip_overrides = parse_overrides(args.car_ip, "car-ip")
    configs = {}
    for marker_id, (name, ip, subject_name) in CAR_CONFIGS.items():
        configs[marker_id] = CarConfig(
            marker_id=marker_id,
            name=name,
            ip=car_ip_overrides.get(marker_id, ip),
            subject_name=subject_overrides.get(marker_id, subject_name),
        )
    return configs


def resolve_car_ips(config_by_id: Dict[int, CarConfig], args: argparse.Namespace) -> None:
    if args.skip_discovery:
        return

    discovered = discover_car_ips(
        CAR_IDS,
        args.discovery_timeout,
        bind_ips=args.bind_ip,
        broadcast_ips=args.broadcast_ip,
    )
    for marker_id, config in config_by_id.items():
        if not config.ip and marker_id in discovered:
            config.ip = discovered[marker_id]

    missing = [str(marker_id) for marker_id, config in config_by_id.items() if not config.ip]
    if missing:
        print("No IP for car ID " + ",".join(missing) + "; those cars will stay disconnected.")


def vec_add(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])


def vec_sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def vec_scale(v: Tuple[float, float], scale: float) -> Tuple[float, float]:
    return (v[0] * scale, v[1] * scale)


def vec_norm(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def vec_normalize(v: Tuple[float, float]) -> Tuple[float, float]:
    norm = vec_norm(v)
    if norm < 1e-6:
        return (0.0, 0.0)
    return (v[0] / norm, v[1] / norm)


def dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def rotate(v: Tuple[float, float], angle_rad: float) -> Tuple[float, float]:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return (v[0] * cos_a - v[1] * sin_a, v[0] * sin_a + v[1] * cos_a)


def set_speed(controller: TcpCarController, speed: int) -> None:
    controller.send(f"SPD {max(0, min(255, speed))}", force=True)


def best_command_for_vector(
    calibrations: Dict[str, CommandCalibration],
    desired_world_vector: Tuple[float, float],
    current_yaw: float,
) -> Tuple[str, float]:
    desired_direction = vec_normalize(desired_world_vector)
    if vec_norm(desired_direction) < MIN_DRIVE_VECTOR_NORM:
        return "STOP", 0.0

    best_command = "STOP"
    best_score = -float("inf")
    for command, calibration in calibrations.items():
        current_direction = rotate(calibration.direction, current_yaw - calibration.yaw)
        score = dot(current_direction, desired_direction)
        if score > best_score:
            best_command = command
            best_score = score

    return best_command, best_score


def wait_for_observations(
    tracker: ViconTracker,
    marker_ids: Tuple[int, ...],
    timeout_sec: float,
    min_seen_at: float = 0.0,
) -> Dict[int, CarObservation]:
    deadline = time.monotonic() + timeout_sec
    result: Dict[int, CarObservation] = {}
    while time.monotonic() < deadline and set(result) < set(marker_ids):
        observations = tracker.get_observations()
        for marker_id in marker_ids:
            observation = observations.get(marker_id)
            if observation is not None and observation.seen_at >= min_seen_at:
                result[marker_id] = observation
        time.sleep(0.02)
    return result


def stop_all(controllers: Dict[int, TcpCarController]) -> None:
    for controller in controllers.values():
        controller.send("STOP", force=True)


def send_all_for_duration(
    controllers: Dict[int, TcpCarController],
    command: str,
    duration_sec: float,
) -> None:
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        for controller in controllers.values():
            controller.send(command, force=True)
        time.sleep(0.12)


def calibrate_command_for_all_cars(
    command: str,
    marker_ids: Tuple[int, ...],
    controllers: Dict[int, TcpCarController],
    tracker: ViconTracker,
    args: argparse.Namespace,
) -> Dict[int, CommandCalibration]:
    stop_all(controllers)
    time.sleep(args.settle_sec)

    start_time = time.monotonic()
    starts = wait_for_observations(tracker, marker_ids, 2.0, min_seen_at=start_time)
    missing_start = [marker_id for marker_id in marker_ids if marker_id not in starts]
    if missing_start:
        print(
            f"{command}: no start Vicon observation for car "
            + ",".join(str(marker_id) for marker_id in missing_start)
        )

    active_controllers = {
        marker_id: controllers[marker_id]
        for marker_id in marker_ids
        if marker_id in starts and marker_id in controllers
    }
    send_all_for_duration(active_controllers, command, args.calibration_move_sec)

    end_time = time.monotonic()
    ends = wait_for_observations(tracker, tuple(active_controllers), 2.0, min_seen_at=end_time)
    stop_all(controllers)
    time.sleep(args.settle_sec)

    calibrations: Dict[int, CommandCalibration] = {}
    for marker_id, start in starts.items():
        end = ends.get(marker_id)
        if end is None:
            print(f"{command}: no end Vicon observation for car{marker_id}")
            continue

        displacement = vec_sub(end.position, start.position)
        calibration = CommandCalibration(
            direction=vec_normalize(displacement),
            yaw=start.yaw,
            displacement_mm=vec_norm(displacement),
        )
        warning = " LOW_MOTION" if calibration.displacement_mm < args.min_displacement_mm else ""
        print(
            f"car{marker_id} {command:>2}: "
            f"dist={calibration.displacement_mm:6.1f}mm "
            f"unit=({calibration.direction[0]:+.3f},{calibration.direction[1]:+.3f}) "
            f"yaw={math.degrees(calibration.yaw):+.1f}deg{warning}"
        )
        if calibration.displacement_mm >= args.min_displacement_mm:
            calibrations[marker_id] = calibration

    return calibrations


def calibrate_all_cars(
    controllers: Dict[int, TcpCarController],
    tracker: ViconTracker,
    args: argparse.Namespace,
) -> CalibrationMap:
    candidate_ids = tuple(marker_id for marker_id in CAR_IDS if marker_id in controllers)
    calibration_by_id: CalibrationMap = {marker_id: {} for marker_id in candidate_ids}

    print("Starting simultaneous motion calibration. Keep the area clear.")
    for controller in controllers.values():
        set_speed(controller, args.calibration_speed)
        controller.send("STOP", force=True)

    visible_before_calibration = wait_for_observations(tracker, candidate_ids, 5.0)
    marker_ids = tuple(marker_id for marker_id in candidate_ids if marker_id in visible_before_calibration)
    missing_ids = [marker_id for marker_id in candidate_ids if marker_id not in visible_before_calibration]
    if missing_ids:
        print("Skipping calibration for no-Vicon car ID " + ",".join(str(marker_id) for marker_id in missing_ids))

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

    for controller in controllers.values():
        set_speed(controller, args.speed)
        controller.send("STOP", force=True)

    return calibration_by_id


def compute_repulsion_vector(
    marker_id: int,
    observations: Dict[int, CarObservation],
    active_radius_mm: float,
) -> Tuple[Tuple[float, float], float]:
    car = observations[marker_id]
    total = (0.0, 0.0)
    min_distance = float("inf")
    other_positions = []

    for other_id, other in observations.items():
        if other_id == marker_id:
            continue

        offset = vec_sub(car.position, other.position)
        distance = vec_norm(offset)
        if distance < 1e-6:
            continue

        min_distance = min(min_distance, distance)
        other_positions.append(other.position)
        if distance > active_radius_mm:
            continue

        away = vec_normalize(offset)
        closeness = max(0.0, active_radius_mm - distance) / active_radius_mm
        weight = 0.25 + closeness * closeness * 2.5
        total = vec_add(total, vec_scale(away, weight))

    if other_positions:
        centroid_x = sum(position[0] for position in other_positions) / len(other_positions)
        centroid_y = sum(position[1] for position in other_positions) / len(other_positions)
        centroid_push = vec_normalize(vec_sub(car.position, (centroid_x, centroid_y)))
        total = vec_add(total, vec_scale(centroid_push, CENTROID_PUSH_WEIGHT))

    return total, min_distance


def compute_escape_target(
    marker_id: int,
    observations: Dict[int, CarObservation],
    args: argparse.Namespace,
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    car = observations[marker_id]
    escape_vector, nearest_distance = compute_repulsion_vector(
        marker_id,
        observations,
        args.active_radius_mm,
    )

    if vec_norm(escape_vector) < MIN_DRIVE_VECTOR_NORM:
        return car.position, (0.0, 0.0), nearest_distance

    escape_direction = vec_normalize(escape_vector)
    target = vec_add(car.position, vec_scale(escape_direction, args.target_step_mm))
    return target, escape_direction, nearest_distance


def min_pair_distance(observations: Dict[int, CarObservation]) -> float:
    ids = sorted(observations)
    result = float("inf")
    for index, marker_id in enumerate(ids):
        for other_id in ids[index + 1:]:
            distance = vec_norm(vec_sub(observations[marker_id].position, observations[other_id].position))
            result = min(result, distance)
    return 0.0 if result == float("inf") else result


def compute_commands(
    observations: Dict[int, CarObservation],
    config_by_id: Dict[int, CarConfig],
    calibration_by_id: CalibrationMap,
    args: argparse.Namespace,
) -> Tuple[str, Dict[int, str], Dict[int, float], Dict[int, Tuple[float, float]]]:
    commands = {marker_id: "STOP" for marker_id in config_by_id}
    nearest_distances = {marker_id: float("inf") for marker_id in config_by_id}
    targets = {marker_id: observations[marker_id].position for marker_id in observations}
    visible_ids = [marker_id for marker_id in CAR_IDS if marker_id in observations]
    missing_ids = [marker_id for marker_id in CAR_IDS if marker_id not in observations]
    uncalibrated_ids = [
        marker_id
        for marker_id in visible_ids
        if marker_id in config_by_id and marker_id not in calibration_by_id
    ]

    if args.require_all_visible and missing_ids:
        summary = "Missing Vicon subject ID " + ",".join(str(marker_id) for marker_id in missing_ids)
    elif len(visible_ids) >= MIN_VISIBLE_CARS_TO_MOVE:
        for marker_id in visible_ids:
            if marker_id not in config_by_id or marker_id not in calibration_by_id:
                continue

            car = observations[marker_id]
            target, escape_direction, nearest_distance = compute_escape_target(
                marker_id,
                observations,
                args,
            )
            nearest_distances[marker_id] = nearest_distance
            targets[marker_id] = target

            if vec_norm(escape_direction) < MIN_DRIVE_VECTOR_NORM:
                command = "STOP"
            else:
                command, _score = best_command_for_vector(
                    calibration_by_id[marker_id],
                    escape_direction,
                    car.yaw,
                )

            commands[marker_id] = command

        missing_text = "none" if not missing_ids else ",".join(str(marker_id) for marker_id in missing_ids)
        uncalibrated_text = (
            "none" if not uncalibrated_ids else ",".join(str(marker_id) for marker_id in uncalibrated_ids)
        )
        summary = (
            f"Visible={len(visible_ids)}/{len(CAR_IDS)} "
            f"min pair={min_pair_distance(observations):.0f}mm "
            f"missing={missing_text} "
            f"uncalibrated={uncalibrated_text}"
        )
    else:
        summary = f"Need at least {MIN_VISIBLE_CARS_TO_MOVE} visible Vicon subjects"

    return summary, commands, nearest_distances, targets


def send_commands(commands: Dict[int, str], controllers: Dict[int, TcpCarController]) -> None:
    for marker_id, command in commands.items():
        controller = controllers.get(marker_id)
        if controller is not None:
            controller.send(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repel Vicon-tracked cars over Wi-Fi TCP.")
    parser.add_argument("--vicon-host", default=VICON_HOST)
    parser.add_argument("--wifi-ssid", default=DEFAULT_WIFI_SSID)
    parser.add_argument("--skip-wifi", action="store_true")
    parser.add_argument("--wifi-wait", type=float, default=DEFAULT_WIFI_WAIT_SEC)
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--discovery-timeout", type=float, default=DEFAULT_DISCOVERY_TIMEOUT_SEC)
    parser.add_argument(
        "--bind-ip",
        action="append",
        default=[],
        help="Local computer IPv4 address to send discovery from. Can be used multiple times.",
    )
    parser.add_argument(
        "--broadcast-ip",
        action="append",
        default=[],
        help="Broadcast address to send discovery to, e.g. 192.168.1.255. Can be used multiple times.",
    )
    parser.add_argument(
        "--car-ip",
        action="append",
        default=[],
        metavar="ID=IP",
        help="Override a discovered IP, e.g. --car-ip 1=192.168.1.201. Can be used multiple times.",
    )
    parser.add_argument(
        "--subject",
        action="append",
        default=[],
        metavar="ID=NAME",
        help="Override a Vicon subject, e.g. --subject 1=kedaya1. Can be used multiple times.",
    )
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED, help="ESP32 motor speed, 0-255.")
    parser.add_argument("--calibration-speed", type=int, default=DEFAULT_CALIBRATION_SPEED)
    parser.add_argument("--calibration-move-sec", type=float, default=DEFAULT_CALIBRATION_MOVE_SEC)
    parser.add_argument("--settle-sec", type=float, default=DEFAULT_SETTLE_SEC)
    parser.add_argument("--min-displacement-mm", type=float, default=DEFAULT_MIN_DISPLACEMENT_MM)
    parser.add_argument("--active-radius-mm", type=float, default=ACTIVE_REPULSION_RADIUS_MM)
    parser.add_argument(
        "--target-step-mm",
        type=float,
        default=DEFAULT_TARGET_STEP_MM,
        help="Temporary dynamic target distance in the escape direction.",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Debug only: skip motion calibration. Cars without calibration will not move.",
    )
    parser.add_argument(
        "--require-all-visible",
        action="store_true",
        help="Only move when all configured cars are visible. By default, visible cars repel and missing cars stop.",
    )
    return parser.parse_args()


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

    connected_controllers: Dict[int, TcpCarController] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(controllers))) as executor:
        future_by_id = {
            executor.submit(controller.connect): marker_id
            for marker_id, controller in controllers.items()
        }
        for future in concurrent.futures.as_completed(future_by_id):
            marker_id = future_by_id[future]
            controller = controllers[marker_id]
            try:
                connected = future.result()
            except OSError as exc:
                print(f"Connect failed for {controller.config.name}: {type(exc).__name__}: {exc}")
                connected = False
            if connected:
                connected_controllers[marker_id] = controller
                controller.send("STOP", force=True)
                set_speed(controller, args.speed)
            else:
                controller.close(send_stop=False)
    controllers = connected_controllers
    print(f"TCP connected: {len(controllers)}/{len(config_by_id)}")

    tracker = ViconTracker(args.vicon_host, config_by_id)
    tracker.connect()

    if args.skip_calibration:
        calibration_by_id: CalibrationMap = {}
        print("Skipping calibration. Cars without calibration will stay stopped.")
    else:
        calibration_by_id = calibrate_all_cars(controllers, tracker, args)

    calibrated_text = ",".join(str(marker_id) for marker_id in sorted(calibration_by_id)) or "none"
    print(f"Calibrated car IDs: {calibrated_text}")
    print(f"Running calibrated Vicon {len(CAR_IDS)}-car repel over Wi-Fi TCP. Press Ctrl+C to stop.")
    last_status_time = 0.0
    try:
        while True:
            observations = tracker.get_observations()
            summary, commands, nearest_distances, targets = compute_commands(
                observations,
                config_by_id,
                calibration_by_id,
                args,
            )
            send_commands(commands, controllers)

            now = time.monotonic()
            if now - last_status_time >= STATUS_INTERVAL_SEC:
                command_text = " ".join(f"ID{marker_id}:{commands[marker_id]}" for marker_id in CAR_IDS)
                distance_text = " ".join(
                    f"ID{marker_id}:{nearest_distances[marker_id]:.0f}mm"
                    for marker_id in CAR_IDS
                    if nearest_distances[marker_id] != float("inf")
                )
                target_text = " ".join(
                    f"ID{marker_id}:({targets[marker_id][0]:.0f},{targets[marker_id][1]:.0f})"
                    for marker_id in CAR_IDS
                    if marker_id in targets and marker_id in calibration_by_id
                )
                connected_count = sum(1 for controller in controllers.values() if controller.sock is not None)
                print(
                    f"{summary} | TCP={connected_count}/{len(CAR_IDS)} | {command_text} "
                    f"| near {distance_text} | target {target_text}"
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
