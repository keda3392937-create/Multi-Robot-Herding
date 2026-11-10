"""Offline checks that entry points can configure and drive car IDs through 50."""
import importlib
import socket
import sys
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def base(monkeypatch):
    try:
        return importlib.import_module("six_car_repel")
    except ModuleNotFoundError as exc:
        if exc.name != "vicon_dssdk":
            raise
        monkeypatch.setitem(sys.modules, "vicon_dssdk", SimpleNamespace(ViconDataStream=None))
    return importlib.import_module("six_car_repel")


def test_shared_configs_and_overrides_cover_all_50(base):
    assert base.CAR_IDS == tuple(range(1, 51))
    configs = base.make_config_map(SimpleNamespace(subject=["50=custom50"], car_ip=["50=192.168.33.150"]))
    assert set(configs) == set(range(1, 51))
    assert all(configs[i].subject_name == f"kedaya{i}" for i in range(1, 50))
    assert configs[50].subject_name == "custom50"
    assert configs[50].ip == "192.168.33.150"
    with pytest.raises(ValueError):
        base.parse_overrides(["51=invalid"], "subject")


@pytest.mark.parametrize("module_name", ["axis_move_calibration", "move_car_to_vicon_targets"])
def test_single_car_tools_accept_each_fleet_id(base, module_name):
    module = importlib.import_module(module_name)
    for car_id in range(1, 51):
        config = module.make_single_config(SimpleNamespace(
            car_id=car_id, car_ip=f"192.168.33.{100 + car_id}", subject="", skip_discovery=True))
        assert config.marker_id == car_id
        assert config.subject_name == f"kedaya{car_id}"


def test_chase_can_configure_all_50_and_target_evader50(base, monkeypatch):
    chase = importlib.import_module("twelve_car_3herder_chase")
    herders = (17, 18, 19, 20)
    evaders = tuple(i for i in range(1, 51) if i not in herders)
    monkeypatch.setattr(chase, "HERDER_IDS", herders)
    monkeypatch.setattr(chase, "EVADER_IDS", evaders)
    monkeypatch.setattr(chase, "CAR_IDS", herders + evaders)
    args = chase.parse_args([])
    configs = chase.make_config_map(args)
    assert len(configs) == 50
    poses = {
        17: base.CarObservation(17, (0, 0), (1, 0), 0, 0),
        50: base.CarObservation(50, (1000, 0), (1, 0), 0, 0),
    }
    calibrations = {i: {"F": base.CommandCalibration((1, 0), 0, 50)} for i in poses}
    _, commands, _, chased, _, _ = chase.compute_commands(poses, configs, calibrations, args)
    assert set(commands) == set(range(1, 51))
    assert commands[17] == commands[50] == "F"
    assert chased[17] == 50


def test_50_independent_workers_send_motion_and_stop(monkeypatch):
    control = importlib.import_module("multi_robot_control")
    assert control.DEFAULT_IDS == tuple(range(1, 51))
    all_moving = threading.Event()
    lock = threading.Lock()
    moving = set()
    stopped = set()
    class Socket:
        def __init__(self, car_id): self.car_id = car_id
        def setsockopt(self, *args): pass
        def settimeout(self, *args): pass
        def close(self): pass
        def sendall(self, payload):
            with lock:
                if payload == b"F\n":
                    moving.add(self.car_id)
                    if len(moving) == 50:
                        all_moving.set()
                elif payload == b"STOP\n":
                    stopped.add(self.car_id)
    monkeypatch.setattr(socket, "create_connection", lambda endpoint, timeout:
                        Socket(int(endpoint[0].rsplit(".", 1)[1]) - 100))
    workers = [control.RobotWorker(i, (f"192.168.33.{100+i}", 23), "F", 150) for i in range(1, 51)]
    for worker in workers:
        worker.start()
    try:
        assert all_moving.wait(3.0)
    finally:
        for worker in workers:
            worker.stop()
    assert moving == stopped == set(range(1, 51))
