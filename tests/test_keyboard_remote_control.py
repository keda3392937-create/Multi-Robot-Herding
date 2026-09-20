from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import keyboard_remote_control as remote

from keyboard_remote_control import (
    CarConnection,
    CommandRefresher,
    KeyboardControlState,
    motion_command_for_keys,
    parse_discovery_reply,
    parse_pong_id,
    validate_args,
)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("Q", "LF"),
        ("W", "F"),
        ("E", "RF"),
        ("A", "LB"),
        ("S", "B"),
        ("D", "RB"),
        ("UP", "F"),
        ("DOWN", "B"),
    ],
)
def test_single_direction_keys(key, expected):
    assert motion_command_for_keys({key}) == expected


def test_no_key_space_and_conflicting_directions_stop():
    assert motion_command_for_keys(set()) == "STOP"
    assert motion_command_for_keys({"SPACE", "W"}) == "STOP"
    assert motion_command_for_keys({"W", "S"}) == "STOP"
    assert motion_command_for_keys({"Q", "E"}) == "STOP"


def test_duplicate_keys_for_same_direction_are_unambiguous():
    assert motion_command_for_keys({"W", "UP"}) == "F"


def test_discovery_reply_uses_source_ip_and_advertised_port():
    reply = b"BOID_CAR id=27 name=ESP32_Car_27 ip=10.0.0.99 tcp=23\n"
    assert parse_discovery_reply(reply, "192.168.8.27") == (27, "192.168.8.27", 23)


@pytest.mark.parametrize(
    "reply",
    [
        b"BOID_CAR_DISCOVER",
        b"BOID_CAR name=missing_id tcp=23",
        b"BOID_CAR id=51 tcp=23",
        b"BOID_CAR id=1 tcp=99999",
    ],
)
def test_invalid_discovery_replies_are_ignored(reply):
    assert parse_discovery_reply(reply, "192.168.8.20") is None


def test_pong_parser_requires_an_id():
    assert parse_pong_id("PONG id=12\r") == 12
    assert parse_pong_id("PONG") is None
    assert parse_pong_id("other id=12") is None


class FakeConnection:
    def __init__(self):
        self.commands = []

    def send_line(self, command):
        self.commands.append(command)


def test_command_refresher_sends_release_stop_immediately_and_refreshes():
    connection = FakeConnection()
    refresher = CommandRefresher(connection, refresh_sec=0.12)

    assert refresher.update("F", 1.00)
    assert not refresher.update("F", 1.05)
    assert refresher.update("F", 1.12)
    assert refresher.update("STOP", 1.13)
    assert connection.commands == ["F", "F", "STOP"]


def test_control_state_stops_on_release_and_disarm():
    state = KeyboardControlState()
    assert state.arm()
    state.press_key("W")
    assert state.desired_command() == "F"
    state.release_key("W")
    assert state.desired_command() == "STOP"
    state.press_key("W")
    state.disarm()
    assert state.desired_command() == "STOP"
    assert not state.armed


class FakeSocket:
    def __init__(self, response):
        self.response = bytearray(response)
        self.sent = []
        self.closed = False

    def sendall(self, payload):
        self.sent.append(payload)

    def recv(self, size):
        data = bytes(self.response[:size])
        del self.response[:size]
        return data

    def close(self):
        self.closed = True


def test_identity_check_accepts_only_the_selected_car():
    good_socket = FakeSocket(b"PONG id=7\n")
    good = CarConnection(good_socket, car_id=7, ip="192.168.8.7", port=23)
    good.verify_identity()
    assert good_socket.sent == [b"PING\n"]

    wrong_socket = FakeSocket(b"PONG id=8\n")
    wrong = CarConnection(wrong_socket, car_id=7, ip="192.168.8.8", port=23)
    with pytest.raises(RuntimeError, match="selected car7"):
        wrong.verify_identity()


def test_connection_close_sends_final_stop():
    sock = FakeSocket(b"")
    connection = CarConnection(sock, car_id=4, ip="192.168.8.4", port=23)
    connection.close()
    assert sock.sent == [b"STOP\n"]
    assert sock.closed


@pytest.mark.parametrize("proof", [None, (2, "192.168.30.109", 23),
                                  (3, "192.168.30.110", 23), (3, "192.168.30.109", 24)])
def test_legacy_pong_rejected_without_matching_discovery(proof):
    connection = CarConnection(FakeSocket(b"PONG\n"), 3, "192.168.30.109", 23,
                               discovered_endpoint=proof)
    with pytest.raises(RuntimeError, match="Legacy PONG requires UDP discovery"):
        connection.verify_identity()


def test_legacy_pong_uses_discovery_and_accepts_followup_heartbeats():
    connection = CarConnection(FakeSocket(b"PONG\r\n"), 3, "192.168.30.109", 23,
                               discovered_endpoint=(3, "192.168.30.109", 23))
    connection.verify_identity()
    assert connection.legacy_pong
    assert connection.accepts_heartbeat("PONG\r")
    assert connection.accepts_heartbeat("PONG id=3")
    assert not connection.accepts_heartbeat("PONG id=4")
    assert not connection.accepts_heartbeat("random text")


def test_discovery_does_not_override_wrong_numbered_pong():
    connection = CarConnection(FakeSocket(b"PONG id=4\n"), 3, "192.168.30.109", 23,
                               discovered_endpoint=(3, "192.168.30.109", 23))
    with pytest.raises(RuntimeError, match="selected car3"):
        connection.verify_identity()


def test_numbered_firmware_cannot_downgrade_to_anonymous_heartbeat():
    connection = CarConnection(FakeSocket(b"PONG id=3\n"), 3, "192.168.30.109", 23,
                               discovered_endpoint=(3, "192.168.30.109", 23))
    connection.verify_identity()
    assert not connection.accepts_heartbeat("PONG")


def test_python_entry_defaults_to_the_current_network(monkeypatch):
    monkeypatch.setattr(remote.sys, "argv", ["keyboard_remote_control.py"])
    assert not remote.AUTO_CONNECT_WIFI
    assert not remote.parse_args().connect_wifi


def test_already_connected_wifi_is_not_reconfigured(monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=b"    SSID : YAYA\r\n", stderr=b""))
    add_profile = Mock(side_effect=AssertionError("Existing Wi-Fi must not be overwritten"))
    monkeypatch.setattr(remote.subprocess, "run", run)
    monkeypatch.setattr(remote, "add_windows_wifi_profile", add_profile)
    remote.connect_windows_wifi("YAYA", "kedayaya", 0)
    assert run.call_count == 1
    add_profile.assert_not_called()


def test_existing_wifi_profile_connect_failure_does_not_overwrite_it(monkeypatch):
    run = Mock(side_effect=[
        SimpleNamespace(returncode=0, stdout=b"SSID : OTHER\n", stderr=b""),
        SimpleNamespace(returncode=1, stdout=b"Connection failed", stderr=b""),
        SimpleNamespace(returncode=0, stdout=b"Existing policy profile", stderr=b""),
    ])
    add_profile = Mock(side_effect=AssertionError("Existing Wi-Fi must not be overwritten"))
    monkeypatch.setattr(remote.subprocess, "run", run)
    monkeypatch.setattr(remote, "add_windows_wifi_profile", add_profile)
    with pytest.raises(RuntimeError, match="existing YAYA profile"):
        remote.connect_windows_wifi("YAYA", "kedayaya", 0)
    add_profile.assert_not_called()


@pytest.mark.parametrize("car_id", [0, 51])
def test_settings_reject_out_of_range_car_ids(car_id):
    args = SimpleNamespace(
        car_id=car_id,
        speed=150,
        wifi_wait=0,
        discovery_timeout=1,
        car_ip="",
    )
    with pytest.raises(ValueError, match="1..50"):
        validate_args(args)
