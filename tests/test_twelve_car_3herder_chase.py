import importlib
import socket
import sys
import threading
import time
from types import SimpleNamespace

import pytest


@pytest.fixture
def chase(monkeypatch):
    # The tests never construct a real SDK client or contact robots.
    try:
        return importlib.import_module("twelve_car_3herder_chase")
    except ModuleNotFoundError as exc:
        if exc.name != "vicon_dssdk":
            raise
        monkeypatch.setitem(sys.modules, "vicon_dssdk", SimpleNamespace(ViconDataStream=None))
    return importlib.import_module("twelve_car_3herder_chase")


def pose(chase, car_id, x, y):
    return chase.CarObservation(car_id, (x, y), (1.0, 0.0), 0.0, time.monotonic())


def calibrations(chase, *ids):
    calibration_type = importlib.import_module("six_car_repel").CommandCalibration
    return {car_id: {
        command: calibration_type(direction, 0.0, 50.0)
        for command, direction in {"F": (1.0, 0.0), "B": (-1.0, 0.0),
                                   "LF": (0.0, 1.0), "RB": (0.0, -1.0)}.items()
    } for car_id in ids}


def test_fixed_roles_and_discovery_skip_missing(chase, monkeypatch):
    args = chase.parse_args([])
    configs = chase.make_config_map(args)
    assert len(chase.HERDER_IDS) >= 1
    assert args.vicon_host == chase.VICON_HOST
    assert {i: c.subject_name for i, c in configs.items()} == {
        i: f"kedaya{i}" for i in chase.CAR_IDS}
    calls = []
    def discover(ids, timeout, bind_ips):
        calls.append(ids)
        return {chase.HERDER_IDS[0]: ("192.168.33.117", 23), chase.EVADER_IDS[0]: ("192.168.33.114", 23)}
    monkeypatch.setattr(chase, "discover_fixed", discover)
    assert set(chase.resolve_endpoints(configs, args)) == {chase.HERDER_IDS[0], chase.EVADER_IDS[0]}
    assert calls == [chase.CAR_IDS]


def test_manual_ips_bypass_discovery_and_reject_wrong_network(chase, monkeypatch):
    args = chase.parse_args([arg for i in chase.CAR_IDS
                             for arg in ("--car-ip", f"{i}=192.168.33.{100+i}")])
    monkeypatch.setattr(chase, "discover_fixed", lambda *_: pytest.fail("unexpected discovery"))
    assert len(chase.resolve_endpoints(chase.make_config_map(args), args)) == len(chase.CAR_IDS)
    with pytest.raises(ValueError):
        chase.make_config_map(chase.parse_args(["--car-ip", "1=192.168.1.101"]))
    with pytest.raises(ValueError):
        chase.make_config_map(chase.parse_args(["--subject", "50=kedaya50"]))
    config = chase.make_config_map(chase.parse_args([
        f"--subject", f"{chase.HERDER_IDS[0]}=custom17", "--car-ip", f"{chase.HERDER_IDS[0]}=192.168.33.117"]))[chase.HERDER_IDS[0]]
    assert config.subject_name == "custom17"
    assert config.ip == "192.168.33.117"


def test_missing_cars_do_not_block_chase_or_escape(chase):
    args = chase.parse_args([])
    herder, evader = chase.HERDER_IDS[0], chase.EVADER_IDS[0]
    observations = {herder: pose(chase, herder, 0, 0), evader: pose(chase, evader, 1000, 0)}
    _, commands, _, chased, _, _ = chase.compute_commands(
        observations, chase.make_config_map(args), calibrations(chase, herder, evader), args)
    assert chased[herder] == evader
    assert commands[herder] == "F"  # Chase toward +X.
    assert commands[evader] == "F"  # Escape farther toward +X.
    assert all(commands[i] == "STOP" for i in chase.CAR_IDS if i not in (herder, evader))
    # Losing one pose stops that car without requiring every configured pose.
    observations.pop(herder)
    _, commands, *_ = chase.compute_commands(
        observations, chase.make_config_map(args), calibrations(chase, herder, evader), args)
    assert commands[herder] == "STOP"


def test_herders_ignore_evaders_that_failed_calibration(chase):
    args = chase.parse_args([])
    herder = chase.HERDER_IDS[0]
    failed_evader = chase.EVADER_IDS[-1]
    active_evader = chase.EVADER_IDS[0]
    observations = {herder: pose(chase, herder, 0, 0), active_evader: pose(chase, active_evader, 100, 0),
                    failed_evader: pose(chase, failed_evader, 0, 1000)}
    commands, _, chased, _ = chase.compute_herder_commands(
        observations, calibrations(chase, herder, failed_evader), args)
    assert chased[herder] == failed_evader
    assert commands[herder] == "LF"


@pytest.mark.parametrize("stalled_id", [None, 19])
def test_shared_calibration_includes_new_car_ids_and_skips_stalled_car(chase, monkeypatch, stalled_id):
    base = importlib.import_module("six_car_repel")
    ids = tuple(range(1, 51))
    positions = {i: (float(i * 1000), 0.0) for i in ids}
    directions = {"F": (1, 0), "RF": (0.5, 0.866), "RB": (-0.5, 0.866),
                  "B": (-1, 0), "LB": (-0.5, -0.866), "LF": (0.5, -0.866)}
    class Controller:
        def __init__(self, car_id):
            self.car_id = car_id
            self.commands = []
        def send(self, command, force=False):
            self.commands.append(command)
            if command in directions and self.car_id != stalled_id:
                dx, dy = directions[command]
                x, y = positions[self.car_id]
                positions[self.car_id] = (x + dx * 50, y + dy * 50)
    class Tracker:
        def get_observations(self):
            return {i: pose(chase, i, *position) for i, position in positions.items()}
    def send_once(controllers, command, duration):
        for controller in controllers.values():
            controller.send(command, force=True)
    # Run real observation, displacement and six-direction completeness checks;
    # simulate only the movement duration and the vehicles' physical response.
    monkeypatch.setattr(base, "send_all_for_duration", send_once)
    monkeypatch.setattr(base.time, "sleep", lambda *_: None)
    controllers = {i: Controller(i) for i in ids}
    result = base.calibrate_all_cars(controllers, Tracker(), chase.parse_args([]))
    assert set(result) == set(ids) - {stalled_id}
    for car_id in result:
        assert set(result[car_id]) == set(base.MOTION_COMMANDS)
        assert all(c.displacement_mm >= 30 for c in result[car_id].values())
    assert all(c.commands[-1] == "STOP" for c in controllers.values())


def test_all_four_new_herders_receive_chase_commands(chase):
    args = chase.parse_args([])
    observations = {}
    test_evaders = tuple(car_id for car_id in chase.EVADER_IDS if car_id not in chase.HERDER_IDS)
    for index, (herder_id, evader_id) in enumerate(zip(chase.HERDER_IDS, test_evaders)):
        observations[herder_id] = pose(chase, herder_id, 0, index * 2000)
        observations[evader_id] = pose(chase, evader_id, 1000, index * 2000)
    _, commands, _, chased, _, _ = chase.compute_commands(
        observations, chase.make_config_map(args), calibrations(chase, *observations), args)
    active_herders = chase.HERDER_IDS[:len(test_evaders)]
    assert all(commands[i] in {"F", "RF", "LF"} for i in active_herders)
    assert set(chased) == set(chase.HERDER_IDS)
    assert all(chased[herder_id] == evader_id
               for herder_id, evader_id in zip(chase.HERDER_IDS, test_evaders))


def test_motion_request_expires_and_speed_is_preserved(chase):
    worker = chase.MotionWorker(1, ("192.168.30.111", 23), 200)
    worker.request("F")
    assert worker.pending_motion(worker.requested_at) == (200, "F")
    worker.request("SPD 180")
    assert worker.pending_motion(worker.requested_at + 0.5) == (180, "STOP")


def test_failed_worker_does_not_block_healthy_worker_and_expired_motion_stops(chase, monkeypatch):
    moved = threading.Event()
    expired = threading.Event()
    received = []
    class FakeSocket:
        def setsockopt(self, *args): pass
        def settimeout(self, *args): pass
        def close(self): pass
        def sendall(self, data):
            received.append(data)
            if b"F\n" in data:
                moved.set()
            elif moved.is_set() and data == b"STOP\n":
                expired.set()
    def connect(endpoint, timeout):
        if endpoint[0].endswith(".112"):
            raise ConnectionRefusedError("offline")
        return FakeSocket()
    monkeypatch.setattr(socket, "create_connection", connect)
    good = chase.MotionWorker(1, ("192.168.30.111", 23), 180)
    bad = chase.MotionWorker(2, ("192.168.30.112", 23), 180)
    good.request("F")
    good.start()
    bad.start()
    try:
        assert moved.wait(1.0)
        assert expired.wait(1.0)
        assert good.snapshot().state == "RUNNING"
        assert bad.snapshot().reconnects >= 1
        good.request("SPD 220")
        # A new speed also accompanies the next independent command update.
        good.request("B")
        deadline = time.monotonic() + 1.0
        while b"SPD 220\nB\n" not in received and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        assert b"SPD 220\nB\n" in received
    finally:
        good.stop()
        bad.stop()
    assert good.snapshot().state == bad.snapshot().state == "STOPPED"


def test_main_stops_every_car_if_calibration_raises(chase, monkeypatch):
    controllers = []
    disconnected = []
    class FakeController:
        def __init__(self, *args):
            self.commands = []
            self.closed = False
            controllers.append(self)
        def connect(self): pass
        def connected(self): return True
        def send(self, command, force=False): self.commands.append(command)
        def close(self): self.closed = True
    class FakeTracker:
        client = SimpleNamespace(Disconnect=lambda: disconnected.append(True))
        def __init__(self, *args): pass
        def connect(self): pass
    args = chase.parse_args([])
    monkeypatch.setattr(chase, "parse_args", lambda: args)
    monkeypatch.setattr(chase, "resolve_endpoints", lambda *_: {
        chase.HERDER_IDS[0]: ("192.168.33.117", 23), chase.EVADER_IDS[0]: ("192.168.33.114", 23)})
    monkeypatch.setattr(chase, "ChaseCarController", FakeController)
    monkeypatch.setattr(chase, "ViconTracker", FakeTracker)
    def fail(*args):
        raise RuntimeError("Vicon failed during calibration")
    monkeypatch.setattr(chase, "calibrate_all_cars", fail)
    with pytest.raises(RuntimeError, match="Vicon failed"):
        chase.main()
    assert len(controllers) == 2
    assert all(c.closed and c.commands[-1] == "STOP" for c in controllers)
    assert disconnected == [True]
