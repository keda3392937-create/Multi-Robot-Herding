#!/usr/bin/env python3
"""Read rigid-body poses from an FZMOTION VRPN stream.

This module implements the small, standard subset of VRPN needed for tracker
position/quaternion reports. It has no third-party Python dependencies.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import ipaddress
import json
import math
import select
import socket
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence, TextIO


VRPN_PORT = 3883
VRPN_ALIGN = 8
VRPN_HEADER_SIZE = 24
VRPN_COOKIE_SIZE = 24
VRPN_MAGIC = b"vrpn: ver. 07.38"
VRPN_MAJOR_PREFIX = b"vrpn: ver. 07."
MAX_MESSAGE_SIZE = 16 * 1024 * 1024

SENDER_DESCRIPTION = -1
TYPE_DESCRIPTION = -2
UDP_DESCRIPTION = -3
TRACKER_POSE_TYPE = "vrpn_Tracker Pos_Quat"

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


class VRPNError(RuntimeError):
    """Base error for VRPN connection and protocol failures."""


class VRPNConnectionError(VRPNError):
    """The VRPN server could not be reached or disconnected."""


class VRPNProtocolError(VRPNError):
    """The peer sent data that does not follow the VRPN wire format."""


@dataclass(frozen=True)
class VRPNMessage:
    timestamp_s: float
    sender_id: int
    type_id: int
    payload: bytes


@dataclass(frozen=True)
class Pose:
    object_name: str
    sensor_id: int
    server_time_s: float
    receive_time_s: float
    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    @property
    def rpy_deg(self) -> tuple[float, float, float]:
        """Return Hamilton-quaternion roll, pitch and yaw in degrees."""
        x, y, z, w = self.quaternion_xyzw
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        pitch = math.asin(pitch_term)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return tuple(math.degrees(value) for value in (roll, pitch, yaw))

    def position(self, units: str) -> tuple[float, float, float]:
        scale = 1000.0 if units == "mm" else 1.0
        return tuple(value * scale for value in self.position_m)

    def as_dict(self, units: str = "m") -> dict[str, Any]:
        px, py, pz = self.position(units)
        qx, qy, qz, qw = self.quaternion_xyzw
        roll, pitch, yaw = self.rpy_deg
        return {
            "receive_time_utc": _iso_utc(self.receive_time_s),
            "server_time_s": self.server_time_s,
            "object_name": self.object_name,
            "sensor_id": self.sensor_id,
            f"x_{units}": px,
            f"y_{units}": py,
            f"z_{units}": pz,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
            "roll_deg": roll,
            "pitch_deg": pitch,
            "yaw_deg": yaw,
        }


def _iso_utc(timestamp_s: float) -> str:
    return datetime.fromtimestamp(timestamp_s, timezone.utc).isoformat(timespec="milliseconds")


def _padded_size(length: int) -> int:
    return (length + VRPN_ALIGN - 1) & ~(VRPN_ALIGN - 1)


def _make_cookie() -> bytes:
    cookie = VRPN_MAGIC + b"  0"
    if len(cookie) > VRPN_COOKIE_SIZE:
        raise AssertionError("VRPN cookie constant is too long")
    return cookie.ljust(VRPN_COOKIE_SIZE, b"\x00")


def _valid_cookie(cookie: bytes) -> bool:
    return len(cookie) == VRPN_COOKIE_SIZE and cookie.startswith(VRPN_MAJOR_PREFIX)


def _pack_message(
    payload: bytes,
    type_id: int,
    sender_id: int,
    *,
    sequence: int = 0,
    timestamp_s: float | None = None,
) -> bytes:
    timestamp_s = time.time() if timestamp_s is None else timestamp_s
    seconds = int(timestamp_s)
    microseconds = int((timestamp_s - seconds) * 1_000_000)
    declared_length = VRPN_HEADER_SIZE + len(payload)
    header = struct.pack(
        "!IIIiiI",
        declared_length,
        seconds,
        microseconds,
        sender_id,
        type_id,
        sequence,
    )
    return header + payload.ljust(_padded_size(len(payload)), b"\x00")


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        try:
            chunk = sock.recv(remaining)
        except (OSError, socket.timeout) as exc:
            raise VRPNConnectionError(f"socket receive failed: {exc}") from exc
        if not chunk:
            raise VRPNConnectionError("VRPN server closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _unpack_header(header: bytes) -> tuple[int, int, int, int, int, int]:
    if len(header) != VRPN_HEADER_SIZE:
        raise VRPNProtocolError(f"invalid VRPN header size: {len(header)}")
    declared_length, seconds, microseconds, sender_id, type_id, sequence = struct.unpack(
        "!IIIiiI", header
    )
    if declared_length < VRPN_HEADER_SIZE or declared_length > MAX_MESSAGE_SIZE:
        raise VRPNProtocolError(f"invalid VRPN message length: {declared_length}")
    if microseconds >= 1_000_000:
        raise VRPNProtocolError(f"invalid VRPN timestamp microseconds: {microseconds}")
    return declared_length, seconds, microseconds, sender_id, type_id, sequence


def _decode_name(payload: bytes) -> str:
    if len(payload) < 5:
        raise VRPNProtocolError("description payload is too short")
    (name_length,) = struct.unpack("!I", payload[:4])
    if name_length < 1 or name_length > len(payload) - 4:
        raise VRPNProtocolError(f"invalid description name length: {name_length}")
    raw_name = payload[4 : 4 + name_length]
    if raw_name[-1] != 0:
        raise VRPNProtocolError("description name is not null terminated")
    raw_name = raw_name[:-1]
    for encoding in ("utf-8", "gb18030", "latin-1"):
        try:
            return raw_name.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_name.decode("utf-8", errors="replace")


class VRPNClient:
    """Minimal VRPN tracker client supporting TCP and negotiated UDP data."""

    def __init__(
        self,
        host: str,
        port: int = VRPN_PORT,
        *,
        transport: str = "tcp",
        local_ip: str | None = None,
        connect_timeout_s: float = 3.0,
    ) -> None:
        if transport not in {"tcp", "udp"}:
            raise ValueError("transport must be 'tcp' or 'udp'")
        self.host = host
        self.port = port
        self.transport = transport
        self.local_ip = local_ip or None
        self.connect_timeout_s = connect_timeout_s
        self.sender_names: dict[int, str] = {}
        self.type_names: dict[int, str] = {}
        self.known_tracks: set[tuple[str, int]] = set()
        self.server_cookie: bytes | None = None
        self._tcp: socket.socket | None = None
        self._udp: socket.socket | None = None
        self._pending_messages: deque[VRPNMessage] = deque(maxlen=128)
        self._sequence = 0

    @property
    def connected(self) -> bool:
        return self._tcp is not None

    @property
    def selected_local_ip(self) -> str | None:
        if self._tcp is None:
            return None
        return str(self._tcp.getsockname()[0])

    @property
    def udp_port(self) -> int | None:
        if self._udp is None:
            return None
        return int(self._udp.getsockname()[1])

    def connect(self) -> None:
        self.close()
        source = (self.local_ip, 0) if self.local_ip else None
        tcp_sock: socket.socket | None = None
        try:
            tcp_sock = socket.create_connection(
                (self.host, self.port),
                timeout=self.connect_timeout_s,
                source_address=source,
            )
            tcp_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            tcp_sock.settimeout(self.connect_timeout_s)
            tcp_sock.sendall(_make_cookie())
            server_cookie = _recv_exact(tcp_sock, VRPN_COOKIE_SIZE)
        except (OSError, VRPNConnectionError) as exc:
            if tcp_sock is not None:
                try:
                    tcp_sock.close()
                except OSError:
                    pass
            raise VRPNConnectionError(
                f"cannot connect to VRPN server {self.host}:{self.port}: {exc}"
            ) from exc

        if not _valid_cookie(server_cookie):
            tcp_sock.close()
            preview = server_cookie.rstrip(b"\x00").decode("ascii", errors="replace")
            raise VRPNProtocolError(
                f"{self.host}:{self.port} is not a compatible VRPN 07 server "
                f"(cookie={preview!r})"
            )

        self._tcp = tcp_sock
        self.server_cookie = server_cookie
        self.sender_names.clear()
        self.type_names.clear()
        self.known_tracks.clear()
        self._pending_messages.clear()
        self._sequence = 0

        if self.transport == "udp":
            try:
                self._enable_udp()
            except BaseException:
                self.close()
                raise

    def _enable_udp(self) -> None:
        if self._tcp is None:
            raise VRPNConnectionError("TCP handshake must finish before UDP setup")
        route_ip = self.local_ip or str(self._tcp.getsockname()[0])
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            udp_sock.bind((route_ip, 0))
            udp_sock.setblocking(False)
            udp_port = int(udp_sock.getsockname()[1])
            payload = route_ip.encode("ascii") + b"\x00"
            self._tcp.sendall(
                _pack_message(
                    payload,
                    UDP_DESCRIPTION,
                    udp_port,
                    sequence=self._next_sequence(),
                )
            )
        except (OSError, UnicodeEncodeError) as exc:
            udp_sock.close()
            raise VRPNConnectionError(f"cannot negotiate VRPN UDP transport: {exc}") from exc
        self._udp = udp_sock

    def _next_sequence(self) -> int:
        value = self._sequence
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF
        return value

    def close(self) -> None:
        for sock in (self._udp, self._tcp):
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
        self._udp = None
        self._tcp = None

    def __enter__(self) -> "VRPNClient":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _read_tcp_message(self) -> VRPNMessage:
        if self._tcp is None:
            raise VRPNConnectionError("client is not connected")
        header = _recv_exact(self._tcp, VRPN_HEADER_SIZE)
        declared_length, seconds, microseconds, sender_id, type_id, _ = _unpack_header(header)
        payload_length = declared_length - VRPN_HEADER_SIZE
        padded_length = _padded_size(payload_length)
        padded_payload = _recv_exact(self._tcp, padded_length) if padded_length else b""
        return VRPNMessage(
            timestamp_s=seconds + microseconds / 1_000_000.0,
            sender_id=sender_id,
            type_id=type_id,
            payload=padded_payload[:payload_length],
        )

    @staticmethod
    def _read_udp_messages(datagram: bytes) -> list[VRPNMessage]:
        messages: list[VRPNMessage] = []
        offset = 0
        while offset < len(datagram):
            if len(datagram) - offset < VRPN_HEADER_SIZE:
                raise VRPNProtocolError("truncated VRPN UDP header")
            header = datagram[offset : offset + VRPN_HEADER_SIZE]
            declared_length, seconds, microseconds, sender_id, type_id, _ = _unpack_header(header)
            payload_length = declared_length - VRPN_HEADER_SIZE
            padded_length = _padded_size(payload_length)
            message_end = offset + VRPN_HEADER_SIZE + padded_length
            if message_end > len(datagram):
                raise VRPNProtocolError("truncated VRPN UDP payload")
            payload_start = offset + VRPN_HEADER_SIZE
            messages.append(
                VRPNMessage(
                    timestamp_s=seconds + microseconds / 1_000_000.0,
                    sender_id=sender_id,
                    type_id=type_id,
                    payload=datagram[payload_start : payload_start + payload_length],
                )
            )
            offset = message_end
        return messages

    def _decode_pose(self, message: VRPNMessage) -> Pose | None:
        type_name = self.type_names.get(message.type_id)
        sender_name = self.sender_names.get(message.sender_id)
        if type_name is None or sender_name is None:
            return None
        if type_name != TRACKER_POSE_TYPE:
            return None
        if len(message.payload) != 64:
            raise VRPNProtocolError(
                f"tracker pose payload has {len(message.payload)} bytes; expected 64"
            )
        sensor_id, _padding, px, py, pz, qx, qy, qz, qw = struct.unpack(
            "!ii7d", message.payload
        )
        pose = Pose(
            object_name=sender_name,
            sensor_id=sensor_id,
            server_time_s=message.timestamp_s,
            receive_time_s=time.time(),
            position_m=(px, py, pz),
            quaternion_xyzw=(qx, qy, qz, qw),
        )
        self.known_tracks.add((sender_name, sensor_id))
        return pose

    def _process_message(self, message: VRPNMessage) -> list[Pose]:
        if message.type_id == SENDER_DESCRIPTION:
            self.sender_names[message.sender_id] = _decode_name(message.payload)
            return self._replay_pending()
        if message.type_id == TYPE_DESCRIPTION:
            self.type_names[message.sender_id] = _decode_name(message.payload)
            return self._replay_pending()
        if message.type_id < 0:
            return []

        pose = self._decode_pose(message)
        if pose is not None:
            return [pose]
        if message.type_id not in self.type_names or message.sender_id not in self.sender_names:
            self._pending_messages.append(message)
        return []

    def _replay_pending(self) -> list[Pose]:
        poses: list[Pose] = []
        still_pending: deque[VRPNMessage] = deque(maxlen=self._pending_messages.maxlen)
        while self._pending_messages:
            message = self._pending_messages.popleft()
            if message.type_id in self.type_names and message.sender_id in self.sender_names:
                pose = self._decode_pose(message)
                if pose is not None:
                    poses.append(pose)
            else:
                still_pending.append(message)
        self._pending_messages = still_pending
        return poses

    def poll_poses(self, timeout_s: float = 0.5) -> list[Pose]:
        if self._tcp is None:
            raise VRPNConnectionError("client is not connected")
        sockets = [self._tcp]
        if self._udp is not None:
            sockets.append(self._udp)
        try:
            ready, _, exceptional = select.select(sockets, [], sockets, max(0.0, timeout_s))
        except (OSError, ValueError) as exc:
            raise VRPNConnectionError(f"socket polling failed: {exc}") from exc
        if exceptional:
            raise VRPNConnectionError("VRPN socket reported an exceptional condition")

        messages: list[VRPNMessage] = []
        if self._tcp in ready:
            messages.append(self._read_tcp_message())
        if self._udp is not None and self._udp in ready:
            try:
                datagram, _peer = self._udp.recvfrom(65535)
            except OSError as exc:
                raise VRPNConnectionError(f"UDP receive failed: {exc}") from exc
            messages.extend(self._read_udp_messages(datagram))

        poses: list[Pose] = []
        for message in messages:
            poses.extend(self._process_message(message))
        return poses

    def iter_poses(self, poll_timeout_s: float = 0.5) -> Iterator[Pose]:
        while self.connected:
            yield from self.poll_poses(poll_timeout_s)


def _probe_vrpn(host: str, port: int, timeout_s: float) -> tuple[str, str] | None:
    sock: socket.socket | None = None
    try:
        sock = socket.create_connection((host, port), timeout=timeout_s)
        sock.settimeout(timeout_s)
        sock.sendall(_make_cookie())
        cookie = _recv_exact(sock, VRPN_COOKIE_SIZE)
        if not _valid_cookie(cookie):
            return None
        version = cookie[:16].decode("ascii", errors="replace")
        return host, version
    except (OSError, VRPNError):
        return None
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def scan_vrpn_servers(
    subnet: str,
    port: int = VRPN_PORT,
    *,
    timeout_s: float = 0.35,
    workers: int = 48,
) -> list[tuple[str, str]]:
    try:
        network = ipaddress.ip_network(subnet, strict=False)
    except ValueError as exc:
        raise ValueError(f"invalid subnet {subnet!r}: {exc}") from exc
    if network.version != 4:
        raise ValueError("only IPv4 subnet scanning is supported")
    if network.num_addresses > 1024:
        raise ValueError("refusing to scan more than 1024 addresses; use a narrower subnet")
    addresses = [str(address) for address in network.hosts()]
    found: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(addresses) or 1)) as pool:
        futures = [pool.submit(_probe_vrpn, address, port, timeout_s) for address in addresses]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                found.append(result)
    return sorted(found, key=lambda item: ipaddress.ip_address(item[0]))


class CsvPoseWriter:
    def __init__(self, path: Path, units: str) -> None:
        self.path = path
        self.units = units
        self._file: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._last_flush = 0.0

    def __enter__(self) -> "CsvPoseWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        sample_fields = list(
            Pose("", 0, 0.0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)).as_dict(
                self.units
            )
        )
        if not is_new:
            with self.path.open("r", newline="", encoding="utf-8-sig") as existing_file:
                existing_fields = next(csv.reader(existing_file), [])
            if existing_fields != sample_fields:
                raise ValueError(
                    f"CSV header in {self.path} does not match {self.units!r} output; "
                    "choose a new file or use the original units"
                )
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=sample_fields)
        if is_new:
            self._writer.writeheader()
        return self

    def write(self, pose: Pose) -> None:
        if self._writer is None or self._file is None:
            raise RuntimeError("CSV writer is not open")
        self._writer.writerow(pose.as_dict(self.units))
        now = time.monotonic()
        if now - self._last_flush >= 1.0:
            self._file.flush()
            self._last_flush = now

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None


class _NullPoseWriter:
    def __enter__(self) -> "_NullPoseWriter":
        return self

    def write(self, pose: Pose) -> None:
        del pose

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read config file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"config file {path} must contain a JSON object")
    return value


def _setting(args: argparse.Namespace, config: dict[str, Any], key: str, default: Any) -> Any:
    value = getattr(args, key, None)
    if value is not None:
        return value
    value = config.get(key, default)
    if value is None or value == "":
        return default
    return value


def _boolean_setting(
    args: argparse.Namespace, config: dict[str, Any], key: str, default: bool
) -> bool:
    value = _setting(args, config, key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def _resolve_host(
    args: argparse.Namespace,
    config: dict[str, Any],
    port: int,
    connect_timeout_s: float,
) -> str:
    host = _setting(args, config, "host", None)
    if host:
        return str(host)
    subnet = str(_setting(args, config, "subnet", ""))
    if not subnet:
        raise ValueError("host is empty and no subnet is configured for automatic discovery")
    print(f"No host configured; scanning {subnet} for VRPN on TCP {port}...", file=sys.stderr)
    found = scan_vrpn_servers(
        subnet,
        port,
        timeout_s=min(0.75, max(0.1, connect_timeout_s)),
    )
    if not found:
        raise VRPNConnectionError(
            f"no VRPN server found in {subnet} on TCP {port}; enable VRPN streaming in FZMOTION"
        )
    if len(found) > 1:
        hosts = ", ".join(item[0] for item in found)
        raise VRPNConnectionError(f"multiple VRPN servers found ({hosts}); set host in config.json")
    print(f"Auto-detected VRPN server {found[0][0]} ({found[0][1]}).", file=sys.stderr)
    return found[0][0]


def _connection_values(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[str, int, str, str | None, float]:
    port = int(_setting(args, config, "port", VRPN_PORT))
    transport = str(_setting(args, config, "transport", "tcp"))
    local_ip_value = _setting(args, config, "local_ip", None)
    local_ip = str(local_ip_value) if local_ip_value else None
    connect_timeout_s = float(_setting(args, config, "connect_timeout_s", 3.0))
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if connect_timeout_s <= 0:
        raise ValueError("connect_timeout_s must be greater than zero")
    host = _resolve_host(args, config, port, connect_timeout_s)
    return host, port, transport, local_ip, connect_timeout_s


def _new_client(values: tuple[str, int, str, str | None, float]) -> VRPNClient:
    host, port, transport, local_ip, connect_timeout_s = values
    return VRPNClient(
        host,
        port,
        transport=transport,
        local_ip=local_ip,
        connect_timeout_s=connect_timeout_s,
    )


def _pose_matches(pose: Pose, object_name: str | None, sensor_id: int | None) -> bool:
    if object_name is not None and pose.object_name != object_name:
        return False
    if sensor_id is not None and pose.sensor_id != sensor_id:
        return False
    return True


def _format_pose(pose: Pose, units: str) -> str:
    px, py, pz = pose.position(units)
    qx, qy, qz, qw = pose.quaternion_xyzw
    roll, pitch, yaw = pose.rpy_deg
    return (
        f"{_iso_utc(pose.receive_time_s)} {pose.object_name}[{pose.sensor_id}] "
        f"pos_{units}=({px:.6f}, {py:.6f}, {pz:.6f}) "
        f"quat_xyzw=({qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f}) "
        f"rpy_deg=({roll:.3f}, {pitch:.3f}, {yaw:.3f})"
    )


def _tracks_text(client: VRPNClient) -> str:
    if client.known_tracks:
        return ", ".join(f"{name}[{sensor}]" for name, sensor in sorted(client.known_tracks))
    if client.sender_names:
        return "senders=" + ", ".join(sorted(set(client.sender_names.values())))
    return "none"


def _run_scan(args: argparse.Namespace, config: dict[str, Any]) -> int:
    subnet = str(_setting(args, config, "subnet", ""))
    if not subnet:
        raise ValueError("scan requires --subnet or a subnet value in config.json")
    port = int(_setting(args, config, "port", VRPN_PORT))
    timeout_s = float(_setting(args, config, "scan_timeout_s", 0.35))
    print(f"Scanning {subnet} for VRPN servers on TCP {port}...", file=sys.stderr)
    found = scan_vrpn_servers(subnet, port, timeout_s=timeout_s)
    if not found:
        print("No compatible VRPN server found.")
        return 2
    for host, version in found:
        print(f"{host}:{port}\t{version}")
    return 0


def _run_discover(args: argparse.Namespace, config: dict[str, Any]) -> int:
    values = _connection_values(args, config)
    duration_s = float(_setting(args, config, "duration_s", 8.0))
    if duration_s <= 0:
        raise ValueError("duration_s must be greater than zero")
    client = _new_client(values)
    seen: set[tuple[str, int]] = set()
    with client:
        print(
            f"Connected to {client.host}:{client.port} via {client.transport}; "
            f"local IP {client.selected_local_ip}.",
            file=sys.stderr,
        )
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            for pose in client.poll_poses(min(0.5, remaining)):
                key = (pose.object_name, pose.sensor_id)
                if key not in seen:
                    seen.add(key)
                    print(_format_pose(pose, "m"))
    if seen:
        print(f"Discovered {len(seen)} tracker target(s).", file=sys.stderr)
        return 0
    print(
        "Connected, but no tracker pose arrived. Check rigid-body streaming; "
        "if transport is udp, retry with --transport tcp.",
        file=sys.stderr,
    )
    if client.sender_names:
        print("Advertised senders: " + ", ".join(client.sender_names.values()), file=sys.stderr)
    return 2


def _run_listen(args: argparse.Namespace, config: dict[str, Any]) -> int:
    values = _connection_values(args, config)
    object_value = _setting(args, config, "object_name", None)
    object_name = str(object_value) if object_value else None
    sensor_value = _setting(args, config, "sensor_id", None)
    sensor_id = int(sensor_value) if sensor_value is not None else None
    units = str(_setting(args, config, "units", "m"))
    print_format = str(_setting(args, config, "print_format", "pretty"))
    print_rate_hz = float(_setting(args, config, "print_rate_hz", 10.0))
    csv_value = _setting(args, config, "csv_path", None)
    csv_path = Path(str(csv_value)) if csv_value else None
    reconnect = _boolean_setting(args, config, "reconnect", False)
    reconnect_delay_s = float(_setting(args, config, "reconnect_delay_s", 2.0))
    data_timeout_s = float(_setting(args, config, "data_timeout_s", 5.0))
    duration_value = _setting(args, config, "duration_s", None)
    duration_s = float(duration_value) if duration_value is not None else None

    if units not in {"m", "mm"}:
        raise ValueError("units must be 'm' or 'mm'")
    if print_format not in {"pretty", "jsonl", "none"}:
        raise ValueError("print_format must be pretty, jsonl, or none")
    if print_rate_hz < 0:
        raise ValueError("print_rate_hz cannot be negative")
    if data_timeout_s < 0:
        raise ValueError("data_timeout_s cannot be negative")
    if reconnect_delay_s < 0:
        raise ValueError("reconnect_delay_s cannot be negative")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be greater than zero")

    writer_context: CsvPoseWriter | _NullPoseWriter
    writer_context = CsvPoseWriter(csv_path, units) if csv_path else _NullPoseWriter()
    started = time.monotonic()
    last_print: dict[tuple[str, int], float] = {}

    with writer_context as writer:
        while True:
            client = _new_client(values)
            try:
                with client:
                    print(
                        f"Connected to {client.host}:{client.port} via {client.transport}; "
                        f"local IP {client.selected_local_ip}. Press Ctrl+C to stop.",
                        file=sys.stderr,
                    )
                    last_matching_pose = time.monotonic()
                    while True:
                        now = time.monotonic()
                        if duration_s is not None and now - started >= duration_s:
                            return 0
                        poses = client.poll_poses(0.5)
                        matched = [
                            pose
                            for pose in poses
                            if _pose_matches(pose, object_name, sensor_id)
                        ]
                        if matched:
                            last_matching_pose = time.monotonic()
                        for pose in matched:
                            writer.write(pose)
                            if print_format == "none":
                                continue
                            key = (pose.object_name, pose.sensor_id)
                            print_now = time.monotonic()
                            interval = 0.0 if print_rate_hz == 0 else 1.0 / print_rate_hz
                            if print_now - last_print.get(key, 0.0) < interval:
                                continue
                            last_print[key] = print_now
                            if print_format == "jsonl":
                                print(json.dumps(pose.as_dict(units), ensure_ascii=False), flush=True)
                            else:
                                print(_format_pose(pose, units), flush=True)
                        if data_timeout_s > 0 and time.monotonic() - last_matching_pose > data_timeout_s:
                            target = object_name or "any object"
                            if sensor_id is not None:
                                target += f" sensor {sensor_id}"
                            raise VRPNConnectionError(
                                f"no pose for {target} in {data_timeout_s:g}s; "
                                f"discovered tracks: {_tracks_text(client)}"
                            )
            except (VRPNError, OSError) as exc:
                if not reconnect:
                    raise
                print(f"Connection lost: {exc}; retrying in {reconnect_delay_s:g}s...", file=sys.stderr)
                time.sleep(reconnect_delay_s)


def _add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", help="FZMOTION/VRPN server IPv4 address")
    parser.add_argument("--port", type=int, help=f"VRPN TCP port (default {VRPN_PORT})")
    parser.add_argument("--transport", choices=("tcp", "udp"), help="pose transport")
    parser.add_argument("--local-ip", help="force a local IPv4 interface")
    parser.add_argument("--connect-timeout-s", type=float, help="connection timeout")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read FZMOTION rigid-body poses through the standard VRPN protocol."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"JSON config file (default: {DEFAULT_CONFIG_PATH.name})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="find VRPN servers on an IPv4 subnet")
    scan_parser.add_argument("--subnet", help="CIDR subnet, for example 192.168.2.0/24")
    scan_parser.add_argument("--port", type=int, help=f"VRPN TCP port (default {VRPN_PORT})")
    scan_parser.add_argument("--scan-timeout-s", type=float, help="timeout per address")

    discover_parser = subparsers.add_parser(
        "discover", help="list object names and sensor IDs from a server"
    )
    _add_connection_arguments(discover_parser)
    discover_parser.add_argument("--duration-s", type=float, help="discovery duration")

    listen_parser = subparsers.add_parser("listen", help="print and/or record tracker poses")
    _add_connection_arguments(listen_parser)
    listen_parser.add_argument("--object", dest="object_name", help="exact VRPN sender name")
    listen_parser.add_argument("--sensor", dest="sensor_id", type=int, help="sensor ID")
    listen_parser.add_argument("--units", choices=("m", "mm"), help="display/CSV units")
    listen_parser.add_argument(
        "--format", dest="print_format", choices=("pretty", "jsonl", "none"), help="stdout format"
    )
    listen_parser.add_argument(
        "--print-rate-hz", type=float, help="maximum console rate per target; 0 means every frame"
    )
    listen_parser.add_argument("--csv", dest="csv_path", help="append every matching pose to CSV")
    listen_parser.add_argument("--duration-s", type=float, help="stop after this many seconds")
    listen_parser.add_argument("--data-timeout-s", type=float, help="fail after no matching pose")
    listen_parser.add_argument(
        "--reconnect", action=argparse.BooleanOptionalAction, default=None, help="reconnect on failure"
    )
    listen_parser.add_argument("--reconnect-delay-s", type=float, help="delay between reconnects")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = _load_config(args.config)
        if args.command == "scan":
            return _run_scan(args, config)
        if args.command == "discover":
            return _run_discover(args, config)
        if args.command == "listen":
            return _run_listen(args, config)
        parser.error(f"unknown command {args.command}")
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 130
    except (TypeError, ValueError, VRPNError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
