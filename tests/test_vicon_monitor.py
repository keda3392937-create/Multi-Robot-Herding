import math
import socket
import time

import pytest

from keyboard_remote_control import CarConnection, CarHeartbeat, KeyboardControlState
from vicon_monitor import ViconMonitor, describe_snapshot, read_frame


class FakeVicon:
    def GetSubjectNames(self):
        return ["kedaya1", "kedaya2", "left-up"]

    def GetSubjectRootSegmentName(self, subject):
        return subject + "_root"

    def GetSegmentGlobalTranslation(self, subject, segment):
        assert segment == subject + "_root"
        return (10, 20, 30), subject == "kedaya2"

    def GetSegmentGlobalRotationEulerXYZ(self, subject, segment):
        return (0, 0, math.pi / 2), False

    def GetFrameNumber(self):
        return 100


def test_pose_uses_root_segment_mm_and_degrees():
    snapshot = read_frame(FakeVicon(), 10.0)
    assert snapshot["poses"] == {"kedaya1": (10, 20, 30, 90), "kedaya2": None}
    status, pose = describe_snapshot(snapshot, "kedaya1", 10.1)
    assert status.startswith("TRACKED")
    assert "90.0 deg" in pose


@pytest.mark.parametrize(("subject", "now", "expected"), [
    ("kedaya2", 10.1, "OCCLUDED"),
    ("kedaya3", 10.1, "SUBJECT NOT FOUND"),
    ("kedaya1", 11.0, "STALE"),
])
def test_invalid_or_old_poses_are_not_displayed_as_live(subject, now, expected):
    status, pose = describe_snapshot(read_frame(FakeVicon(), 10), subject, now)
    assert status.startswith(expected)
    assert pose == "--"


def test_sdk_error_does_not_require_a_pose():
    assert describe_snapshot({"status": "SDK ERROR: missing"}, "kedaya1", 10) == (
        "SDK ERROR: missing", "--")


def test_monitor_reports_sdk_failure_and_closes_its_process(tmp_path):
    (tmp_path / "vicon_dssdk.py").write_text(
        'raise ImportError("test-sdk-unavailable")\n', encoding="ascii")
    monitor = ViconMonitor("127.0.0.1", str(tmp_path))
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            status = monitor.poll()["status"]
            if "test-sdk-unavailable" in status:
                break
            time.sleep(0.05)
        assert "test-sdk-unavailable" in status
    finally:
        monitor.close()
    assert not monitor.process.is_alive()


def test_conflicting_keyboard_input_overrides_mouse_motion():
    state = KeyboardControlState()
    state.arm()
    state.press_key("W")
    state.press_key("S")
    state.press_mouse("F")
    assert state.desired_command() == "STOP"


def test_heartbeat_fragmented_reply_and_timeout():
    client, server = socket.socketpair()
    server.settimeout(1)
    try:
        connection = CarConnection(client, 1, "127.0.0.1", 23)
        heartbeat = CarHeartbeat(connection)
        heartbeat.poll(1)
        assert server.recv(64) == b"PING\n"
        server.sendall(b"PONG id=")
        heartbeat.poll(1.1)
        assert heartbeat.pending_at == 1
        server.sendall(b"1\n")
        heartbeat.poll(1.2)
        assert heartbeat.pending_at is None
        heartbeat.poll(2.3)
        assert server.recv(64) == b"PING\n"
        with pytest.raises(ConnectionError, match="timed out"):
            heartbeat.poll(3)
    finally:
        client.close()
        server.close()


def test_heartbeat_detects_wrong_car_and_disconnect():
    client, server = socket.socketpair()
    try:
        heartbeat = CarHeartbeat(CarConnection(client, 1, "127.0.0.1", 23))
        server.sendall(b"PONG id=2\n")
        with pytest.raises(RuntimeError, match="identity"):
            heartbeat.poll(1)
        server.close()
        with pytest.raises(ConnectionError, match="closed"):
            heartbeat.poll(2)
    finally:
        client.close()
        server.close()


def test_legacy_heartbeat_remains_connected_after_initial_handshake():
    client, server = socket.socketpair()
    server.settimeout(1)
    try:
        connection = CarConnection(client, 3, "127.0.0.1", 23,
                                   discovered_endpoint=(3, "127.0.0.1", 23))
        server.sendall(b"PONG\n")
        connection.verify_identity()
        assert server.recv(64) == b"PING\n"
        heartbeat = CarHeartbeat(connection)
        for now in (1.0, 2.2, 3.4):
            heartbeat.poll(now)
            assert server.recv(64) == b"PING\n"
            server.sendall(b"PONG\n")
            heartbeat.poll(now + 0.1)
            assert heartbeat.pending_at is None
        connection.close()
        assert server.recv(64) == b"STOP\n"
    finally:
        client.close()
        server.close()
