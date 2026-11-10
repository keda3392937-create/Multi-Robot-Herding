import importlib
import itertools
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def remote(monkeypatch):
    try:
        return importlib.import_module("random_car1_vicon_test")
    except ModuleNotFoundError as exc:
        if exc.name != "vicon_dssdk":
            raise
        monkeypatch.setitem(sys.modules, "vicon_dssdk", SimpleNamespace(ViconDataStream=None))
    return importlib.import_module("random_car1_vicon_test")


@pytest.mark.parametrize("car_id", [1, 15, 16, 17, 20, 50])
def test_discovery_and_vicon_config_use_selected_id(remote, monkeypatch, car_id):
    calls = []
    def discover(ids, timeout, **kwargs):
        calls.append(ids)
        return {car_id: f"192.168.33.{100 + car_id}"}
    monkeypatch.setattr(remote, "discover_car_ips", discover)
    config = remote.make_car1_config(remote.parse_args(["--car-id", str(car_id)]))
    assert calls == [(car_id,)]
    assert config.marker_id == car_id
    assert config.name == f"car{car_id}_sta"
    assert config.subject_name == f"kedaya{car_id}"
    assert config.ip == f"192.168.33.{100 + car_id}"


def test_editing_car_id_still_sets_default(remote, monkeypatch):
    monkeypatch.setattr(remote, "CAR_ID", 19)
    assert remote.parse_args([]).car_id == 19


@pytest.mark.parametrize("car_id", [0, 51])
def test_invalid_id_is_rejected_before_discovery(remote, monkeypatch, car_id):
    monkeypatch.setattr(remote, "discover_car_ips", lambda *_: pytest.fail("unexpected discovery"))
    with pytest.raises(SystemExit):
        remote.parse_args(["--car-id", str(car_id)])


def test_explicit_ip_and_subject_for_car20_bypass_discovery(remote, monkeypatch):
    monkeypatch.setattr(remote, "discover_car_ips", lambda *_: pytest.fail("unexpected discovery"))
    args = remote.parse_args(["--car-id", "20", "--car-ip", "192.168.33.120",
                              "--subject", "custom20"])
    config = remote.make_car1_config(args)
    assert config.marker_id == 20
    assert config.ip == "192.168.33.120"
    assert config.subject_name == "custom20"


def test_discovery_failure_reports_selected_car(remote, monkeypatch):
    monkeypatch.setattr(remote, "discover_car_ips", lambda *_, **kwargs: {})
    with pytest.raises(RuntimeError, match="No IP for car20"):
        remote.make_car1_config(remote.parse_args(["--car-id", "20"]))


def test_main_tracks_and_moves_selected_car_instead_of_global_default(remote, monkeypatch, capsys):
    args = remote.parse_args(["--car-id", "50", "--car-ip", "192.168.33.150", "--skip-wifi",
                              "--duration", "0.3", "--move-sec", "0.03", "--settle-sec", "0.02"])
    selected = []
    commands = []
    class Controller:
        def __init__(self, config): selected.append(config.marker_id)
        def connect(self): return True
        def send(self, command, force=False): commands.append(command)
        def close(self): commands.append("closed")
    class Tracker:
        def __init__(self, host, configs):
            assert host == remote.DEFAULT_VICON_HOST
            assert set(configs) == {50}
        def connect(self): pass
        def get_observations(self):
            return {50: remote.CarObservation(50, (0.0, 0.0), (1.0, 0.0), 0.0, 0.0)}
    ticks = itertools.count(0.0, 0.01)
    monkeypatch.setattr(remote.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(remote.time, "sleep", lambda *_: None)
    monkeypatch.setattr(remote, "parse_args", lambda: args)
    monkeypatch.setattr(remote, "TcpCarController", Controller)
    monkeypatch.setattr(remote, "ViconTracker", Tracker)
    monkeypatch.setattr(remote, "connect_wifi", lambda *_: pytest.fail("unexpected Wi-Fi change"))
    remote.main()
    assert selected == [50]
    assert any(command in remote.MOTION_COMMANDS for command in commands)
    assert commands[-2:] == ["STOP", "closed"]
    output = capsys.readouterr().out
    assert "Vicon kedaya50:" in output
    assert "car50 stopped" in output
    assert "car1 " not in output
