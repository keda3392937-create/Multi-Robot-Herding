"""Safely control one ESP32 car from a dedicated keyboard window."""

from __future__ import annotations

import argparse
import ipaddress
import locale
import math
import os
import queue
import re
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Dict, Iterable, Optional, Set, Tuple
from xml.sax.saxutils import escape

from vicon_monitor import DEFAULT_SDK_PATH, DEFAULT_VICON_HOST, ViconMonitor, describe_snapshot


# Defaults shared with the existing single-car tools; the fleet window selects cars interactively.
CAR_ID = 13
CAR_IP = ""  # Leave empty to find the selected car by UDP discovery.
MOTOR_SPEED = 150  # ESP32 PWM value, 0..255.

WIFI_SSID = "YAYA"
WIFI_PASSWORD = "kedayaya"
AUTO_CONNECT_WIFI = False

CONTROL_PORT = 23
DISCOVERY_PORT = 4210
DISCOVERY_REQUEST = b"BOID_CAR_DISCOVER"
DISCOVERY_PREFIX = "BOID_CAR"
DISCOVERY_TIMEOUT_SEC = 5.0
WIFI_WAIT_SEC = 4.0
SOCKET_TIMEOUT_SEC = 1.5
COMMAND_REFRESH_SEC = 0.12
UI_REFRESH_MS = 20
SPEED_STEP = 10
VALID_CAR_IDS = range(1, 51)

MOTION_BY_KEY = {
    "Q": "LF",
    "W": "F",
    "E": "RF",
    "A": "LB",
    "S": "B",
    "D": "RB",
    "UP": "F",
    "DOWN": "B",
}
VALID_MOTION_COMMANDS = frozenset({"F", "RF", "RB", "B", "LB", "LF", "STOP"})

TK_MOTION_KEYS = {
    "q": "Q",
    "w": "W",
    "e": "E",
    "a": "A",
    "s": "S",
    "d": "D",
    "Up": "UP",
    "Down": "DOWN",
}


def motion_command_for_keys(pressed_keys: Iterable[str]) -> str:
    """Return one unambiguous motion command, otherwise STOP."""
    pressed = set(pressed_keys)
    if "SPACE" in pressed:
        return "STOP"

    commands = {MOTION_BY_KEY[key] for key in pressed if key in MOTION_BY_KEY}
    if len(commands) != 1:
        return "STOP"
    return commands.pop()


class KeyboardControlState:
    """Pure safety state shared by Tk events and offline tests."""

    def __init__(self) -> None:
        self.armed = False
        self.pressed_keys: Set[str] = set()
        self.mouse_command: Optional[str] = None

    def arm(self) -> bool:
        if self.pressed_keys or self.mouse_command is not None:
            return False
        self.armed = True
        return True

    def disarm(self) -> None:
        self.armed = False
        self.pressed_keys.clear()
        self.mouse_command = None

    def press_key(self, key: str) -> None:
        if key in MOTION_BY_KEY:
            self.pressed_keys.add(key)

    def release_key(self, key: str) -> None:
        self.pressed_keys.discard(key)

    def press_mouse(self, command: str) -> None:
        if command not in VALID_MOTION_COMMANDS or command == "STOP":
            raise ValueError(f"Invalid mouse motion command: {command}")
        self.mouse_command = command

    def release_mouse(self) -> None:
        self.mouse_command = None

    def desired_command(self) -> str:
        if not self.armed:
            return "STOP"
        keyboard_command = motion_command_for_keys(self.pressed_keys)
        if self.pressed_keys and keyboard_command == "STOP":
            return "STOP"
        commands = set()
        if keyboard_command != "STOP":
            commands.add(keyboard_command)
        if self.mouse_command is not None:
            commands.add(self.mouse_command)
        if len(commands) != 1:
            return "STOP"
        return commands.pop()


def parse_discovery_reply(data: bytes, source_ip: str) -> Optional[Tuple[int, str, int]]:
    text = data.decode("ascii", errors="replace").strip()
    if not text.startswith(DISCOVERY_PREFIX + " "):
        return None

    fields: Dict[str, str] = {
        key.lower(): value for key, value in re.findall(r"(\w+)=([^\s]+)", text)
    }
    try:
        car_id = int(fields["id"])
        port = int(fields.get("tcp", str(CONTROL_PORT)))
        ipaddress.ip_address(source_ip)
    except (KeyError, ValueError):
        return None

    if car_id not in VALID_CAR_IDS or not 1 <= port <= 65535:
        return None
    return car_id, source_ip, port


def parse_pong_id(line: str) -> Optional[int]:
    match = re.fullmatch(r"PONG\s+id=(\d+)", line.strip(), flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def is_usable_local_ipv4(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False
    return bool(
        ip.version == 4
        and not ip.is_loopback
        and not ip.is_link_local
        and not ip.is_multicast
        and ip_text not in {"0.0.0.0", "255.255.255.255"}
        and not ip_text.startswith("255.")
    )


def _decode_console_output(data: bytes) -> str:
    encodings = ("utf-8", locale.getpreferredencoding(False), "mbcs")
    for encoding in dict.fromkeys(encodings):
        try:
            return data.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


def local_ipv4_candidates() -> Tuple[str, ...]:
    candidates: Set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip_text = info[4][0]
            if is_usable_local_ipv4(ip_text):
                candidates.add(ip_text)
    except OSError:
        pass

    if sys.platform == "win32":
        result = subprocess.run(["ipconfig"], capture_output=True, check=False)
        output = _decode_console_output(result.stdout)
        for line in output.splitlines():
            if "IPv4" not in line:
                continue
            addresses = re.findall(
                r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])", line
            )
            for ip_text in addresses:
                if is_usable_local_ipv4(ip_text):
                    candidates.add(ip_text)

    return tuple(sorted(candidates))


def default_broadcast_targets(bind_ips: Iterable[str]) -> Tuple[str, ...]:
    targets = {"255.255.255.255", "192.168.30.255"}
    for ip_text in bind_ips:
        parts = ip_text.split(".")
        if len(parts) == 4:
            targets.add(".".join(parts[:3] + ["255"]))
    return tuple(sorted(targets))


def make_discovery_sockets(bind_ips: Iterable[str]) -> list[socket.socket]:
    sockets: list[socket.socket] = []
    for bind_ip in ("", *tuple(bind_ips)):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.setblocking(False)
            sock.bind((bind_ip, 0))
            sockets.append(sock)
        except OSError as exc:
            print(f"Discovery socket bind failed for {bind_ip or 'default'}: {exc}")
    return sockets


def discover_car(
    car_id: int,
    timeout_sec: float,
    bind_ips: Iterable[str] = (),
    broadcast_ips: Iterable[str] = (),
) -> Tuple[str, int]:
    local_ips = tuple(bind_ips) or local_ipv4_candidates()
    targets = tuple(broadcast_ips) or default_broadcast_targets(local_ips)
    sockets = make_discovery_sockets(local_ips)
    if not sockets:
        raise RuntimeError("Could not create a UDP discovery socket.")

    print(f"Discovering car{car_id} on UDP {DISCOVERY_PORT}...")
    if local_ips:
        print("Local IPv4 candidates: " + ", ".join(local_ips))
    print("Broadcast targets: " + ", ".join(targets))

    deadline = time.monotonic() + timeout_sec
    next_broadcast = 0.0
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_broadcast:
                for sock in sockets:
                    for target in targets:
                        try:
                            sock.sendto(DISCOVERY_REQUEST, (target, DISCOVERY_PORT))
                        except OSError:
                            pass
                next_broadcast = now + 0.35

            readable, _, _ = select.select(sockets, [], [], 0.15)
            for sock in readable:
                try:
                    data, address = sock.recvfrom(256)
                except OSError:
                    continue
                reply = parse_discovery_reply(data, address[0])
                if reply is None:
                    continue
                found_id, found_ip, found_port = reply
                if found_id == car_id:
                    print(f"Discovered car{car_id}: {found_ip}:{found_port}")
                    return found_ip, found_port
    finally:
        for sock in sockets:
            sock.close()

    raise RuntimeError(
        f"car{car_id} was not discovered. Check its power and {WIFI_SSID} connection, "
        "or set CAR_IP near the top of this file."
    )


def add_windows_wifi_profile(ssid: str, password: str) -> None:
    profile_xml = f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
    <name>{escape(ssid)}</name>
    <SSIDConfig><SSID><name>{escape(ssid)}</name></SSID></SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>auto</connectionMode>
    <MSM><security>
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
    </security></MSM>
</WLANProfile>
"""
    profile_path = Path(tempfile.gettempdir()) / f"vicon_car_wifi_{os.getpid()}.xml"
    try:
        profile_path.write_text(profile_xml, encoding="utf-8")
        result = subprocess.run(
            [
                "netsh",
                "wlan",
                "add",
                "profile",
                f"filename={profile_path}",
                "user=current",
            ],
            capture_output=True,
            check=False,
        )
        output = (_decode_console_output(result.stdout) + _decode_console_output(result.stderr)).strip()
        if output:
            print(output)
        if result.returncode != 0:
            raise RuntimeError(f"Could not add the Windows Wi-Fi profile for {ssid}.")
    finally:
        try:
            profile_path.unlink(missing_ok=True)
        except OSError:
            pass


def connect_windows_wifi(ssid: str, password: str, wait_sec: float) -> None:
    if sys.platform != "win32":
        raise RuntimeError("Automatic Wi-Fi connection is only supported on Windows.")

    interfaces = subprocess.run(
        ["netsh", "wlan", "show", "interfaces"], capture_output=True, check=False,
    )
    if interfaces.returncode == 0:
        connected_ssids = re.findall(
            r"^\s*SSID\s*:\s*(.*?)\s*$",
            _decode_console_output(interfaces.stdout), flags=re.MULTILINE,
        )
        if ssid in connected_ssids:
            print(f"Already connected to {ssid}; keeping the current Wi-Fi connection.")
            return

    print(f"Connecting Windows Wi-Fi profile: {ssid}")
    result = subprocess.run(
        ["netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}"],
        capture_output=True,
        check=False,
    )
    output = (_decode_console_output(result.stdout) + _decode_console_output(result.stderr)).strip()
    if output:
        print(output)
    if result.returncode != 0:
        profile = subprocess.run(
            ["netsh", "wlan", "show", "profiles", f"name={ssid}"],
            capture_output=True, check=False,
        )
        if profile.returncode == 0:
            raise RuntimeError(
                f"Windows could not connect using the existing {ssid} profile. "
                "Connect in Windows Wi-Fi settings, then use --skip-wifi."
            )
        print(f"No usable Wi-Fi profile for {ssid}; creating a current-user profile.")
        add_windows_wifi_profile(ssid, password)
        result = subprocess.run(
            ["netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}"],
            capture_output=True,
            check=False,
        )
        output = (
            _decode_console_output(result.stdout) + _decode_console_output(result.stderr)
        ).strip()
        if output:
            print(output)
        if result.returncode != 0:
            raise RuntimeError(f"Windows could not connect to Wi-Fi {ssid}.")
    if wait_sec > 0:
        time.sleep(wait_sec)


class CarConnection:
    def __init__(self, sock: socket.socket, car_id: int, ip: str, port: int,
                 *, discovered_endpoint: Optional[Tuple[int, str, int]] = None):
        self.sock = sock
        self.car_id = car_id
        self.ip = ip
        self.port = port
        self.closed = False
        self.discovery_verified = discovered_endpoint == (car_id, ip, port)
        self.legacy_pong = False

    @classmethod
    def connect(cls, car_id: int, ip: str, port: int,
                *, discovered_endpoint: Optional[Tuple[int, str, int]] = None) -> "CarConnection":
        print(f"Connecting to car{car_id} at {ip}:{port}...")
        sock = socket.create_connection((ip, port), timeout=SOCKET_TIMEOUT_SEC)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(SOCKET_TIMEOUT_SEC)
        connection = cls(sock, car_id, ip, port, discovered_endpoint=discovered_endpoint)
        try:
            connection.verify_identity()
        except BaseException:
            connection.close(send_stop=False)
            raise
        source = "UDP discovery + legacy PONG" if connection.legacy_pong else "PONG id"
        print(f"Verified car{car_id}: {source}")
        return connection

    def send_line(self, line: str) -> None:
        if self.closed:
            raise RuntimeError("The car connection is closed.")
        if "\r" in line or "\n" in line:
            raise ValueError("A command must not contain a line break.")
        self.sock.sendall((line + "\n").encode("ascii"))

    def read_line(self, max_bytes: int = 128) -> str:
        data = bytearray()
        while len(data) < max_bytes:
            chunk = self.sock.recv(min(64, max_bytes - len(data)))
            if not chunk:
                raise ConnectionError("The car closed the connection before replying.")
            data.extend(chunk)
            newline = data.find(b"\n")
            if newline >= 0:
                return bytes(data[:newline]).decode("ascii", errors="replace").strip("\r")
        raise RuntimeError("The car reply exceeded the maximum line length.")

    def verify_identity(self) -> None:
        self.send_line("PING")
        response = self.read_line()
        if response.strip().upper() == "PONG" and self.discovery_verified:
            self.legacy_pong = True
            return
        response_id = parse_pong_id(response)
        if response_id != self.car_id:
            raise RuntimeError(
                f"Identity check failed: selected car{self.car_id}, received {response!r}. "
                "Legacy PONG requires UDP discovery of this car's ID, IP and port."
            )

    def accepts_heartbeat(self, response: str) -> bool:
        return parse_pong_id(response) == self.car_id or (
            self.legacy_pong and response.strip().upper() == "PONG"
        )

    def set_speed(self, speed: int) -> None:
        if not 0 <= speed <= 255:
            raise ValueError("Motor speed must be in the range 0..255.")
        self.send_line(f"SPD {speed}")

    def stop(self) -> None:
        self.send_line("STOP")

    def close(self, send_stop: bool = True) -> None:
        if self.closed:
            return
        if send_stop:
            try:
                self.sock.sendall(b"STOP\n")
            except OSError:
                pass
        try:
            self.sock.close()
        except OSError:
            pass
        self.closed = True


class CommandRefresher:
    def __init__(self, connection: CarConnection, refresh_sec: float):
        self.connection = connection
        self.refresh_sec = refresh_sec
        self.last_command: Optional[str] = None
        self.last_send_at = float("-inf")

    def update(self, command: str, now: float) -> bool:
        if command not in VALID_MOTION_COMMANDS:
            raise ValueError(f"Invalid motion command: {command}")
        changed = command != self.last_command
        due = now - self.last_send_at >= self.refresh_sec
        if not changed and not due:
            return False
        self.connection.send_line(command)
        self.last_command = command
        self.last_send_at = now
        return True

    def force_next(self) -> None:
        self.last_command = None


class CarHeartbeat:
    def __init__(self, connection: CarConnection):
        self.connection = connection
        self.pending_at = None
        self.next_ping = 0.0
        self.buffer = bytearray()

    def poll(self, now: float) -> None:
        readable, _, _ = select.select([self.connection.sock], [], [], 0)
        if readable:
            data = self.connection.sock.recv(256)
            if not data:
                raise ConnectionError("Car closed the connection")
            self.buffer.extend(data)
            while b"\n" in self.buffer:
                line, _, rest = self.buffer.partition(b"\n")
                self.buffer = bytearray(rest)
                if not self.connection.accepts_heartbeat(line.decode("ascii", errors="replace")):
                    raise RuntimeError("Unexpected car heartbeat identity")
                self.pending_at = None
                self.next_ping = now + 1.0
            if len(self.buffer) > 128:
                raise RuntimeError("Oversized car heartbeat")
        if self.pending_at is not None:
            if now - self.pending_at > 0.6:
                raise ConnectionError("Car heartbeat timed out")
        elif now >= self.next_ping:
            self.connection.send_line("PING")
            self.pending_at = now


class RemoteControlWindow:
    def __init__(self, connection: Optional[CarConnection], car_id: int, speed: int,
                 args: Optional[argparse.Namespace] = None) -> None:
        self.connection = connection
        self.args = args
        self.speed = speed
        self.state = KeyboardControlState()
        self.refresher = CommandRefresher(connection, COMMAND_REFRESH_SEC) if connection else None
        self.closed = False
        self.monitor = None
        self.connecting = False
        self.results = queue.Queue()
        self.result_lock = threading.Lock()
        self.heartbeat = None
        self.next_car_id = car_id

        self.root = tk.Tk()
        self.root.title("Vicon Fleet Remote")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.bind("<FocusOut>", self._on_focus_out)
        self.root.bind("<Unmap>", self._on_unmap)

        frame = ttk.Frame(self.root, padding=12)
        frame.grid()
        self.selected_car = tk.StringVar(value=f"kedaya{car_id}")
        self.selector = ttk.Combobox(frame, textvariable=self.selected_car, state="readonly",
                                     values=[f"kedaya{i}" for i in VALID_CAR_IDS], width=16)
        self.selector.grid(row=0, column=0, sticky="ew")
        self.selector.bind("<<ComboboxSelected>>", self._selection_changed)
        self.connect_button = ttk.Button(frame, text="Connect", command=self._connect,
                                         takefocus=False)
        self.connect_button.grid(row=0, column=1, sticky="ew")
        self.car_status = tk.StringVar(value="CAR: DISCONNECTED")
        ttk.Label(frame, textvariable=self.car_status, wraplength=560).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=8)
        self.status = tk.StringVar(value="DISARMED")
        ttk.Label(frame, textvariable=self.status, width=60).grid(
            row=2, column=0, columnspan=2, pady=(8, 10), sticky="w"
        )
        ttk.Button(frame, text="ARM", command=self._arm, takefocus=False).grid(
            row=3, column=0, padx=(0, 6), sticky="ew"
        )
        ttk.Button(frame, text="STOP", command=self._stop_and_disarm, takefocus=False).grid(
            row=3, column=1, padx=(6, 0), sticky="ew"
        )
        self.speed_value = tk.IntVar(value=speed)
        self.speed_label = tk.StringVar(value=f"PWM: {speed}")
        ttk.Label(frame, textvariable=self.speed_label).grid(row=4, column=0, sticky="w", pady=8)
        ttk.Scale(frame, from_=0, to=255, variable=self.speed_value,
                  command=lambda value: self._set_speed(round(float(value))),
                  takefocus=False).grid(row=4, column=1, sticky="ew")
        ttk.Separator(frame).grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)
        self.vicon_status = tk.StringVar(value="VICON: DISABLED")
        self.pose_status = tk.StringVar(value="--")
        host = args.vicon_host if args else DEFAULT_VICON_HOST
        ttk.Label(frame, text=f"Vicon {host} | Wi-Fi {args.wifi_ssid if args else WIFI_SSID}").grid(
            row=6, column=0, columnspan=2, sticky="w")
        ttk.Label(frame, textvariable=self.vicon_status, wraplength=560).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Label(frame, textvariable=self.pose_status, wraplength=560).grid(
            row=8, column=0, columnspan=2, sticky="w")
        if args and not args.no_vicon:
            self.monitor = ViconMonitor(args.vicon_host, args.vicon_sdk_path)
        if connection:
            self._connected(connection)

        self.root.after(UI_REFRESH_MS, self._tick)
        self.root.after(100, self.root.focus_force)

    def _selection_changed(self, _event=None) -> None:
        self._disconnect()
        self.next_car_id = int(self.selected_car.get().removeprefix("kedaya"))

    def _connect(self) -> None:
        if self.connecting or self.args is None:
            return
        self._disconnect()
        self.connecting = True
        self.selector.configure(state="disabled")
        self.connect_button.configure(state="disabled")
        self.car_status.set(f"CAR: CONNECTING {self.selected_car.get()}")
        car_id = self.next_car_id

        def worker():
            connection = None
            try:
                if (AUTO_CONNECT_WIFI or self.args.connect_wifi) and not self.args.skip_wifi:
                    connect_windows_wifi(self.args.wifi_ssid, self.args.wifi_password,
                                         self.args.wifi_wait)
                discovered_endpoint = None
                if self.args.car_ip and car_id == self.args.car_id:
                    ip, port = self.args.car_ip, CONTROL_PORT
                else:
                    ip, port = discover_car(car_id, self.args.discovery_timeout,
                                           self.args.bind_ip, self.args.broadcast_ip)
                    discovered_endpoint = (car_id, ip, port)
                connection = CarConnection.connect(
                    car_id, ip, port, discovered_endpoint=discovered_endpoint)
                connection.stop()
                connection.sock.settimeout(0.15)
                result = connection
            except Exception as exc:
                if connection:
                    connection.close()
                result = exc
            with self.result_lock:
                if self.closed:
                    if isinstance(result, CarConnection):
                        result.close()
                else:
                    self.results.put(result)

        threading.Thread(target=worker, daemon=True).start()

    def _connected(self, connection: CarConnection) -> None:
        self.connection = connection
        self.refresher = CommandRefresher(connection, COMMAND_REFRESH_SEC)
        self.heartbeat = CarHeartbeat(connection)
        self.state.disarm()
        try:
            connection.set_speed(self.speed)
            self.car_status.set(f"CAR: VERIFIED kedaya{connection.car_id} | {connection.ip}:{connection.port}")
            self.root.focus_set()
        except OSError as exc:
            self._network_failed(exc)

    def _disconnect(self) -> None:
        self.state.disarm()
        if self.connection:
            self.connection.close()
        self.connection = None
        self.refresher = None
        self.heartbeat = None
        self.status.set("DISARMED")
        self.car_status.set("CAR: DISCONNECTED")

    @staticmethod
    def _motion_key(keysym: str) -> Optional[str]:
        return TK_MOTION_KEYS.get(keysym, TK_MOTION_KEYS.get(keysym.lower()))

    def _arm(self) -> None:
        if self.connection is None:
            self.status.set("CAR: DISCONNECTED")
            return
        if self.state.arm():
            self.status.set("ARMED - STOP")
        else:
            self.status.set("RELEASE DIRECTION KEYS")
        self.root.focus_set()

    def _stop_and_disarm(self) -> None:
        self.state.disarm()
        self.status.set("DISARMED")
        self._transmit()

    def _on_key_press(self, event) -> Optional[str]:
        if event.keysym == "Escape":
            self.close()
            return "break"
        if event.keysym in {"Return", "KP_Enter"}:
            self._arm()
            return "break"
        if event.keysym == "space":
            self._stop_and_disarm()
            return "break"
        if event.keysym == "bracketleft":
            self._set_speed(max(0, self.speed - SPEED_STEP))
            return "break"
        if event.keysym == "bracketright":
            self._set_speed(min(255, self.speed + SPEED_STEP))
            return "break"

        key = self._motion_key(event.keysym)
        if key is not None:
            self.state.press_key(key)
            self._transmit()
            return "break"
        return None

    def _on_key_release(self, event) -> Optional[str]:
        key = self._motion_key(event.keysym)
        if key is not None:
            self.state.release_key(key)
            self._transmit()
            return "break"
        return None

    def _on_focus_out(self, _event) -> None:
        self.root.after_idle(self._disarm_if_unfocused)

    def _disarm_if_unfocused(self) -> None:
        if self.closed:
            return
        # Combobox popdowns are Tcl widgets without Python widget wrappers.
        try:
            focused = self.root.tk.call("focus")
            in_control_window = bool(focused) and str(
                self.root.tk.call("winfo", "toplevel", focused)
            ) == self.root._w
        except tk.TclError:
            in_control_window = False
        if not in_control_window:
            self._stop_and_disarm()

    def _on_unmap(self, event) -> None:
        if event.widget is self.root:
            self._stop_and_disarm()

    def _set_speed(self, speed: int) -> None:
        if speed == self.speed:
            return
        self.speed = speed
        self.speed_value.set(speed)
        self.speed_label.set(f"PWM: {speed}")
        if self.connection is None:
            return
        try:
            self.connection.set_speed(speed)
            self.refresher.force_next()
            self.status.set(f"SPEED {speed}")
        except OSError as exc:
            self._network_failed(exc)

    def _transmit(self) -> None:
        if self.closed or self.refresher is None:
            return
        command = self.state.desired_command()
        try:
            if self.refresher.update(command, time.monotonic()):
                self.status.set(command if self.state.armed else "DISARMED")
        except OSError as exc:
            self._network_failed(exc)

    def _tick(self) -> None:
        if self.closed:
            return
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            pass
        else:
            self.connecting = False
            self.selector.configure(state="readonly")
            self.connect_button.configure(state="normal")
            if isinstance(result, Exception):
                self.car_status.set(f"CAR ERROR: {result}")
            else:
                self._connected(result)
        self._transmit()
        if self.heartbeat:
            try:
                self.heartbeat.poll(time.monotonic())
            except (OSError, RuntimeError) as exc:
                self._network_failed(exc)
        if self.monitor:
            status, pose = describe_snapshot(self.monitor.poll(), self.selected_car.get(), time.monotonic())
            self.vicon_status.set(f"VICON: {status}")
            self.pose_status.set(pose)
        self.root.after(UI_REFRESH_MS, self._tick)

    def _network_failed(self, exc: Exception) -> None:
        print(f"Car connection lost: {exc}")
        self._disconnect()
        self.car_status.set(f"CAR ERROR: {exc}")
        self.status.set("CONNECTION LOST")

    def close(self) -> None:
        if self.closed:
            return
        with self.result_lock:
            self.closed = True
            try:
                result = self.results.get_nowait()
                if isinstance(result, CarConnection):
                    result.close()
            except queue.Empty:
                pass
        self._disconnect()
        if self.monitor:
            self.monitor.close()
        self.root.destroy()

    def run(self) -> None:
        print("Keyboard: Enter=arm, Q/W/E/A/S/D=move, Space=stop, Esc=quit")
        print("Up/Down also move forward/backward; [ and ] change speed.")
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyboard control for one ESP32 Vicon car.")
    parser.add_argument("--car-id", type=int, default=1, help="Initially selected car; also selectable in the window.")
    parser.add_argument("--car-ip", default=CAR_IP, help="Skip discovery and use this IPv4 address.")
    parser.add_argument("--speed", type=int, default=MOTOR_SPEED, help="ESP32 PWM speed, 0..255.")
    parser.add_argument("--wifi-ssid", default=WIFI_SSID)
    parser.add_argument("--wifi-password", default=WIFI_PASSWORD)
    parser.add_argument("--skip-wifi", action="store_true")
    parser.add_argument("--connect-wifi", action="store_true", help="Opt in to automatic Windows Wi-Fi connection.")
    parser.add_argument("--wifi-wait", type=float, default=WIFI_WAIT_SEC)
    parser.add_argument("--discovery-timeout", type=float, default=DISCOVERY_TIMEOUT_SEC)
    parser.add_argument("--bind-ip", action="append", default=[])
    parser.add_argument("--broadcast-ip", action="append", default=[])
    parser.add_argument("--vicon-host", default=DEFAULT_VICON_HOST)
    parser.add_argument("--vicon-sdk-path", default=DEFAULT_SDK_PATH)
    parser.add_argument("--no-vicon", action="store_true", help="Only test the car control link.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.car_id not in VALID_CAR_IDS:
        raise ValueError("--car-id must be in the range 1..50.")
    if not 0 <= args.speed <= 255:
        raise ValueError("--speed must be in the range 0..255.")
    if (not math.isfinite(args.wifi_wait) or not math.isfinite(args.discovery_timeout)
            or args.wifi_wait < 0 or args.discovery_timeout <= 0):
        raise ValueError("Wi-Fi wait cannot be negative and discovery timeout must be positive.")
    if args.car_ip:
        ip = ipaddress.ip_address(args.car_ip)
        if ip.version != 4:
            raise ValueError("--car-ip must be an IPv4 address.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if sys.platform != "win32":
        raise RuntimeError("This keyboard remote currently requires Windows.")

    window = RemoteControlWindow(None, args.car_id, args.speed, args)
    try:
        window.run()
    except KeyboardInterrupt:
        print("\nInterrupted; stopping car.")
    finally:
        window.close()
    print("Car stopped; connection closed.")


if __name__ == "__main__":
    main()
