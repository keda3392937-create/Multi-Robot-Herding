from __future__ import annotations

import socket
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path

import fzmotion_client as client_module


def _description(message_type: int, remote_id: int, name: str) -> bytes:
    encoded = name.encode("utf-8") + b"\x00"
    payload = struct.pack("!I", len(encoded)) + encoded
    return client_module._pack_message(payload, message_type, remote_id, timestamp_s=100.25)


def _pose(remote_sender: int, remote_type: int, sensor: int = 0) -> bytes:
    payload = struct.pack("!ii7d", sensor, sensor, 1.25, -2.5, 3.75, 0.0, 0.0, 0.0, 1.0)
    return client_module._pack_message(payload, remote_type, remote_sender, timestamp_s=101.5)


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    data = bytearray()
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise RuntimeError("test peer disconnected")
        data.extend(chunk)
    return bytes(data)


class FakeVrpnServer:
    def __init__(self, transport: str = "tcp") -> None:
        self.transport = transport
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "FakeVrpnServer":
        self.thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.thread.join(timeout=2.0)
        self.listener.close()
        if self.thread.is_alive():
            raise RuntimeError("fake VRPN server did not finish")
        if self.error is not None:
            raise self.error

    def _run(self) -> None:
        peer: socket.socket | None = None
        try:
            peer, _address = self.listener.accept()
            peer.settimeout(2.0)
            cookie = _recv_exact(peer, client_module.VRPN_COOKIE_SIZE)
            if not client_module._valid_cookie(cookie):
                raise AssertionError("client cookie was invalid")
            peer.sendall(client_module._make_cookie())

            udp_target: tuple[str, int] | None = None
            if self.transport == "udp":
                header = _recv_exact(peer, client_module.VRPN_HEADER_SIZE)
                declared, _seconds, _micros, sender, message_type, _sequence = (
                    client_module._unpack_header(header)
                )
                payload_length = declared - client_module.VRPN_HEADER_SIZE
                padded = _recv_exact(peer, client_module._padded_size(payload_length))
                ip_address = padded[:payload_length].rstrip(b"\x00").decode("ascii")
                if message_type != client_module.UDP_DESCRIPTION:
                    raise AssertionError("client did not send a UDP description")
                udp_target = (ip_address, sender)

            peer.sendall(
                _description(client_module.SENDER_DESCRIPTION, 4, "Car")
                + _description(client_module.TYPE_DESCRIPTION, 7, client_module.TRACKER_POSE_TYPE)
            )
            time.sleep(0.05)
            if udp_target is None:
                peer.sendall(_pose(4, 7, sensor=2))
            else:
                udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    udp_sock.sendto(_pose(4, 7, sensor=2), udp_target)
                finally:
                    udp_sock.close()
            time.sleep(0.1)
        except BaseException as exc:
            self.error = exc
        finally:
            if peer is not None:
                peer.close()


class ProtocolTests(unittest.TestCase):
    def test_message_header_and_padding(self) -> None:
        packet = client_module._pack_message(b"abc", -3, 4567, sequence=9, timestamp_s=12.5)
        self.assertEqual(len(packet), 32)
        header = client_module._unpack_header(packet[:24])
        self.assertEqual(header, (27, 12, 500000, 4567, -3, 9))
        self.assertEqual(packet[24:27], b"abc")

    def test_pose_euler_identity(self) -> None:
        pose = client_module.Pose(
            "Car", 0, 0.0, 0.0, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0)
        )
        self.assertEqual(pose.rpy_deg, (0.0, 0.0, 0.0))
        self.assertEqual(pose.position("mm"), (1000.0, 2000.0, 3000.0))

    def test_tcp_tracker_pose(self) -> None:
        with FakeVrpnServer("tcp") as server:
            with client_module.VRPNClient("127.0.0.1", server.port, transport="tcp") as client:
                deadline = time.monotonic() + 1.5
                received = []
                while not received and time.monotonic() < deadline:
                    received.extend(client.poll_poses(0.25))
        self.assertEqual(len(received), 1)
        pose = received[0]
        self.assertEqual((pose.object_name, pose.sensor_id), ("Car", 2))
        self.assertEqual(pose.position_m, (1.25, -2.5, 3.75))
        self.assertEqual(pose.quaternion_xyzw, (0.0, 0.0, 0.0, 1.0))

    def test_udp_tracker_pose(self) -> None:
        with FakeVrpnServer("udp") as server:
            with client_module.VRPNClient("127.0.0.1", server.port, transport="udp") as client:
                self.assertIsNotNone(client.udp_port)
                deadline = time.monotonic() + 1.5
                received = []
                while not received and time.monotonic() < deadline:
                    received.extend(client.poll_poses(0.25))
        self.assertEqual(len(received), 1)
        self.assertEqual((received[0].object_name, received[0].sensor_id), ("Car", 2))

    def test_csv_writer_writes_header_and_scaled_pose(self) -> None:
        pose = client_module.Pose(
            "Car", 2, 101.5, 102.0, (1.25, -2.5, 3.75), (0.0, 0.0, 0.0, 1.0)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "pose.csv"
            with client_module.CsvPoseWriter(output_path, "mm") as writer:
                writer.write(pose)
            rows = output_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 2)
        self.assertIn("x_mm", rows[0])
        self.assertIn("1250.0", rows[1])

    def test_csv_writer_rejects_unit_mismatch(self) -> None:
        pose = client_module.Pose(
            "Car", 2, 101.5, 102.0, (1.25, -2.5, 3.75), (0.0, 0.0, 0.0, 1.0)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "pose.csv"
            with client_module.CsvPoseWriter(output_path, "m") as writer:
                writer.write(pose)
            with self.assertRaisesRegex(ValueError, "does not match"):
                with client_module.CsvPoseWriter(output_path, "mm"):
                    pass

    def test_null_config_value_uses_default(self) -> None:
        args = type("Args", (), {"port": None})()
        self.assertEqual(client_module._setting(args, {"port": None}, "port", 3883), 3883)

    def test_boolean_config_rejects_string(self) -> None:
        args = type("Args", (), {"reconnect": None})()
        with self.assertRaisesRegex(ValueError, "true or false"):
            client_module._boolean_setting(args, {"reconnect": "false"}, "reconnect", False)


if __name__ == "__main__":
    unittest.main()
