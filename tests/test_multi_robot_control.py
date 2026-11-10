import socket
import threading
import time
from types import SimpleNamespace

import pytest

from multi_robot_control import RobotWorker, discover_fixed, local_yaya_ips, parse_discovery, parse_ids


def test_parse_discovery_uses_packet_source_ip_and_port():
    assert parse_discovery(b"BOID_CAR id=13 name=kedaya13 ip=bad tcp=23", "192.168.30.113") == (
        13, "192.168.30.113", 23)
    assert parse_discovery(b"BOID_CAR id=0 tcp=23", "192.168.30.1") is None


def test_parse_ids_deduplicates_and_validates():
    assert parse_ids("2,3,2,13") == (2, 3, 13)
    with pytest.raises(ValueError):
        parse_ids("2,51")


def test_local_yaya_ips_selects_current_lan_and_excludes_other_adapters(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_: [
        (socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))
        for ip in ("192.168.145.1", "192.168.33.112", "192.168.30.101", "192.168.54.1")
    ])
    assert local_yaya_ips() == ("192.168.33.112",)
    assert local_yaya_ips(("192.168.33.113",)) == ("192.168.33.113",)


def test_discovery_is_restricted_to_requested_ids_and_current_broadcast(monkeypatch):
    created = []

    class FakeSocket:
        def __init__(self):
            self.sent = []
            created.append(self)
        def setsockopt(self, *args): pass
        def setblocking(self, *args): pass
        def bind(self, endpoint): self.bound = endpoint
        def sendto(self, payload, endpoint): self.sent.append((payload, endpoint))
        def recvfrom(self, size):
            return replies.pop(0)
        def close(self): self.closed = True

    replies = [
        (b"BOID_CAR id=13 tcp=23", ("192.168.33.113", 4210)),
        (b"BOID_CAR id=2 ip=192.168.30.102 tcp=23", ("192.168.33.102", 4210)),
        (b"BOID_CAR id=3 tcp=23", ("192.168.33.103", 4210)),
    ]
    monkeypatch.setattr(socket, "socket", lambda *args: FakeSocket())
    monkeypatch.setattr("multi_robot_control.select.select", lambda sockets, *_: (sockets, [], []))
    assert discover_fixed((2, 3), 1, bind_ips=("192.168.33.112",)) == {
        2: ("192.168.33.102", 23), 3: ("192.168.33.103", 23)}
    assert len(created) == 1
    assert created[0].bound == ("192.168.33.112", 0)
    assert created[0].sent == [(b"BOID_CAR_DISCOVER", ("192.168.33.255", 4210))]
    assert created[0].closed


def test_no_yaya_address_reports_current_network_without_opening_sockets(monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *_: pytest.fail("unexpected socket"))
    with pytest.raises(RuntimeError, match=r"192\.168\.33\.0/24"):
        discover_fixed((2, 3), 1, bind_ips=("192.168.30.112",))


class FakeServer:
    def __init__(self):
        self.received = []
        self.closed = False
    def setsockopt(self, *args): pass
    def settimeout(self, *args): pass
    def sendall(self, data): self.received.append(data)
    def close(self): self.closed = True


def test_one_worker_failure_does_not_stop_another(monkeypatch):
    servers = {}
    def create(endpoint, timeout):
        car_id = int(endpoint[0].split(".")[-1])
        if car_id == 3:
            raise ConnectionRefusedError("offline")
        server = FakeServer()
        servers[car_id] = server
        return server
    monkeypatch.setattr(socket, "create_connection", create)
    good = RobotWorker(2, ("192.168.30.2", 23), "F", 100)
    bad = RobotWorker(3, ("192.168.30.3", 23), "F", 100)
    good.start(); bad.start()
    time.sleep(0.25)
    bad.stop(); good.stop()
    assert any(payload == b"F\n" for payload in servers[2].received)
    assert bad.snapshot().state == "STOPPED"
    assert good.snapshot().state == "STOPPED"
