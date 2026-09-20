"""Simple fixed four-car synchronized motion test for kedaya3, 7, 10 and 13."""
from __future__ import annotations
import argparse
import time
from keyboard_remote_control import CarConnection, discover_car, local_ipv4_candidates

CAR_IDS = (3, 7, 10, 13)
VALID_COMMANDS = ("F", "B", "LF", "RF", "LB", "RB", "STOP")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", choices=VALID_COMMANDS, default="F")
    parser.add_argument("--speed", type=int, default=150,
                        help="PWM speed, 0..255; default 150.")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--discovery-timeout", type=float, default=5.0)
    parser.add_argument("--discovery-retries", type=int, default=2)
    parser.add_argument("--bind-ip", action="append", default=[])
    return parser.parse_args()

def connect_car(car_id: int, args: argparse.Namespace) -> CarConnection:
    bind_ips = tuple(args.bind_ip) or tuple(ip for ip in local_ipv4_candidates() if ip.startswith("192.168.30."))
    if not bind_ips:
        raise RuntimeError("No local 192.168.30.x address. Connect the computer to YAYA first.")
    last_error = None
    for attempt in range(args.discovery_retries + 1):
        try:
            ip, port = discover_car(car_id, args.discovery_timeout, bind_ips=bind_ips, broadcast_ips=("192.168.30.255",))
            connection = CarConnection.connect(car_id, ip, port, discovered_endpoint=(car_id, ip, port))
            connection.set_speed(args.speed)
            connection.stop()
            return connection
        except Exception as exc:
            last_error = exc
            if attempt < args.discovery_retries:
                print(f"Retrying kedaya{car_id} ({attempt + 1}/{args.discovery_retries})...")
                time.sleep(0.25)
    raise RuntimeError(f"kedaya{car_id}: {last_error}")

def main() -> None:
    args = parse_args()
    if not 0 <= args.speed <= 255 or args.duration <= 0 or args.discovery_retries < 0:
        raise ValueError("Invalid speed, duration, or retry count")
    connections = {}
    try:
        for car_id in CAR_IDS:
            connections[car_id] = connect_car(car_id, args)
        print("All cars connected: " + ", ".join(f"kedaya{id}" for id in CAR_IDS))
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            for connection in connections.values():
                connection.send_line(args.command)
            time.sleep(0.08)
    except KeyboardInterrupt:
        print("Interrupted; stopping both cars.")
    finally:
        for connection in connections.values():
            connection.close(send_stop=True)
        print("Both cars stopped.")

if __name__ == "__main__":
    main()
