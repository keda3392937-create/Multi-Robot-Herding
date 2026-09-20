"""Fault-isolated fixed-ID movement test for multiple ESP32 cars.

The script discovers only the requested IDs on YAYA, opens one persistent TCP
connection per car, and gives every car its own worker thread. A missing or
disconnected car is reported and skipped while the remaining cars continue.
"""

from __future__ import annotations

import argparse
import ipaddress
import re
import select
import socket
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple


DEFAULT_IDS = ( 1,2, 3, 4, 5, 6, 8, 10, 12, 13, 15)
NETWORK = ipaddress.ip_network("192.168.30.0/24")
DISCOVERY_PORT = 4210
CONTROL_PORT = 23
DISCOVERY_REQUEST = b"BOID_CAR_DISCOVER"
COMMANDS = ("F", "B", "LF", "RF", "LB", "RB", "STOP")


def local_yaya_ips(bind_ips: Iterable[str] = ()) -> Tuple[str, ...]:
    if bind_ips:
        values = tuple(bind_ips)
    else:
        values = tuple(
            info[4][0]
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        )
    return tuple(sorted({ip for ip in values if ipaddress.ip_address(ip) in NETWORK}))


def parse_discovery(data: bytes, source_ip: str) -> Optional[Tuple[int, str, int]]:
    text = data.decode("ascii", errors="replace").strip()
    if not text.startswith("BOID_CAR "):
        return None
    fields = {key.lower(): value for key, value in re.findall(r"(\w+)=([^\s]+)", text)}
    try:
        car_id = int(fields["id"])
        port = int(fields.get("tcp", str(CONTROL_PORT)))
        ipaddress.ip_address(source_ip)
    except (KeyError, ValueError):
        return None
    if not 1 <= car_id <= 50 or not 1 <= port <= 65535:
        return None
    return car_id, source_ip, port


def discover_fixed(ids: Sequence[int], timeout: float, bind_ips: Iterable[str] = ()) -> Dict[int, Tuple[str, int]]:
    """Discover all requested IDs in one quiet UDP pass."""
    local_ips = local_yaya_ips(bind_ips)
    if not local_ips:
        raise RuntimeError("No local 192.168.30.x address; connect the computer to YAYA first.")
    sockets = []
    found: Dict[int, Tuple[str, int]] = {}
    try:
        for bind_ip in local_ips:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.setblocking(False)
            sock.bind((bind_ip, 0))
            sockets.append(sock)
        deadline = time.monotonic() + timeout
        next_send = 0.0
        wanted = set(ids)
        while time.monotonic() < deadline and set(found) != wanted:
            now = time.monotonic()
            if now >= next_send:
                for sock in sockets:
                    try:
                        sock.sendto(DISCOVERY_REQUEST, ("192.168.30.255", DISCOVERY_PORT))
                    except OSError:
                        pass
                next_send = now + 0.3
            readable, _, _ = select.select(sockets, [], [], 0.1)
            for sock in readable:
                try:
                    data, address = sock.recvfrom(256)
                except OSError:
                    continue
                reply = parse_discovery(data, address[0])
                if reply and reply[0] in wanted:
                    found[reply[0]] = (reply[1], reply[2])
    finally:
        for sock in sockets:
            sock.close()
    return found


@dataclass
class RobotStatus:
    state: str = "DISCOVERED"
    ip: str = ""
    last_tx: float = 0.0
    last_error: str = ""
    reconnects: int = 0


class RobotWorker:
    def __init__(self, car_id: int, endpoint: Tuple[str, int], command: str, speed: int):
        self.car_id = car_id
        self.ip, self.port = endpoint
        self.command = command
        self.speed = speed
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"car{car_id}", daemon=True)
        self.sock: Optional[socket.socket] = None
        self.lock = threading.Lock()
        self.status = RobotStatus(ip=self.ip)

    def start(self) -> None:
        self.thread.start()

    def _set_state(self, state: str, error: str = "") -> None:
        with self.lock:
            self.status.state = state
            self.status.last_error = error

    def _close_socket(self, send_stop: bool = False) -> None:
        sock, self.sock = self.sock, None
        if sock is None:
            return
        if send_stop:
            try:
                sock.sendall(b"STOP\n")
            except OSError:
                pass
        try:
            sock.close()
        except OSError:
            pass

    def _connect(self) -> bool:
        try:
            sock = socket.create_connection((self.ip, self.port), timeout=1.2)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(0.5)
            sock.sendall(f"SPD {self.speed}\nSTOP\n".encode("ascii"))
            self.sock = sock
            with self.lock:
                self.status.state = "CONNECTED"
                self.status.last_error = ""
            return True
        except OSError as exc:
            self._set_state("CONNECT_FAILED", f"{type(exc).__name__}: {exc}")
            self._close_socket()
            return False

    def _run(self) -> None:
        retry_at = 0.0
        while not self.stop_event.is_set():
            now = time.monotonic()
            if self.sock is None:
                if now < retry_at:
                    self.stop_event.wait(min(0.2, retry_at - now))
                    continue
                if not self._connect():
                    with self.lock:
                        self.status.reconnects += 1
                    retry_at = now + 0.8
                    continue
            try:
                assert self.sock is not None
                self.sock.sendall((self.command + "\n").encode("ascii"))
                with self.lock:
                    self.status.last_tx = time.monotonic()
                    self.status.state = "RUNNING"
                self.stop_event.wait(0.12)
            except OSError as exc:
                self._set_state("SEND_FAILED", f"{type(exc).__name__}: {exc}")
                self._close_socket()
                with self.lock:
                    self.status.reconnects += 1
                retry_at = time.monotonic() + 0.8
        self._close_socket(send_stop=True)
        self._set_state("STOPPED")

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self._close_socket(send_stop=True)

    def snapshot(self) -> RobotStatus:
        with self.lock:
            return RobotStatus(**vars(self.status))


def parse_ids(text: str) -> Tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item) for item in text.split(",") if item.strip()))
    if not values or any(value not in range(1, 51) for value in values):
        raise ValueError("--ids must contain car IDs from 1..50")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", default=",".join(map(str, DEFAULT_IDS)))
    parser.add_argument("--command", choices=COMMANDS, default="F")
    parser.add_argument("--speed", type=int, default=150)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--discovery-timeout", type=float, default=4.0)
    parser.add_argument("--bind-ip", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids = parse_ids(args.ids)
    if not 0 <= args.speed <= 255 or args.duration <= 0:
        raise ValueError("speed must be 0..255 and duration must be positive")
    endpoints = discover_fixed(ids, args.discovery_timeout, args.bind_ip)
    missing = [car_id for car_id in ids if car_id not in endpoints]
    if missing:
        print("Skipped undiscovered: " + ", ".join(f"kedaya{car_id}" for car_id in missing))
    workers = {car_id: RobotWorker(car_id, endpoints[car_id], args.command, args.speed)
               for car_id in ids if car_id in endpoints}
    if not workers:
        raise RuntimeError("No requested cars were discovered.")
    print("Starting: " + ", ".join(f"kedaya{car_id}" for car_id in workers))
    for worker in workers.values():
        worker.start()
    deadline = time.monotonic() + args.duration
    try:
        while time.monotonic() < deadline:
            time.sleep(1.0)
            status = []
            for car_id, worker in workers.items():
                current = worker.snapshot()
                detail = current.state
                if current.last_error:
                    detail += f" ({current.last_error})"
                tx_age = "never" if not current.last_tx else f"{time.monotonic() - current.last_tx:.1f}s ago"
                status.append(f"{car_id}:{detail}; last_tx={tx_age}; reconnects={current.reconnects}")
            print(" | ".join(status))
    except KeyboardInterrupt:
        print("Stopping requested cars...")
    finally:
        for worker in workers.values():
            worker.stop()
        print("All requested workers stopped.")


if __name__ == "__main__":
    main()
