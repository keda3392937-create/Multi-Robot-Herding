from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from keyboard_remote_control import RemoteControlWindow


class FakeConnection:
    car_id = 1
    ip = "127.0.0.1"
    port = 23

    def __init__(self):
        self.commands = []
        self.closed = False

    def send_line(self, line):
        self.commands.append(line)

    def set_speed(self, speed):
        self.send_line(f"SPD {speed}")

    def close(self):
        self.send_line("STOP")
        self.closed = True


@pytest.mark.parametrize(("focused", "toplevel", "should_stop"), [
    (".!frame.!combobox.popdown.f.l", ".!frame.!combobox.popdown", True),
    (".!frame.!combobox", ".", False),
    ("", "", True),
])
def test_tcl_popdown_focus_does_not_require_a_python_widget(focused, toplevel, should_stop):
    window = RemoteControlWindow.__new__(RemoteControlWindow)
    window.closed = False
    window.root = SimpleNamespace(
        _w=".", tk=SimpleNamespace(call=Mock(side_effect=[focused, toplevel])),
        focus_get=Mock(side_effect=KeyError("popdown")),
    )
    window._stop_and_disarm = Mock()
    window._disarm_if_unfocused()
    window.root.focus_get.assert_not_called()
    assert window._stop_and_disarm.called == should_stop


def test_window_selection_release_stop_and_disconnect(monkeypatch):
    import tkinter as tk

    real_tk = tk.Tk

    def hidden_root():
        root = real_tk()
        root.withdraw()
        return root

    monkeypatch.setattr(tk, "Tk", hidden_root)
    connection = FakeConnection()
    window = RemoteControlWindow(connection, 1, 100)
    try:
        window._arm()
        window._on_key_press(SimpleNamespace(keysym="w"))
        assert connection.commands[-1] == "F"
        window._on_key_release(SimpleNamespace(keysym="w"))
        assert connection.commands[-1] == "STOP"
        window._set_speed(120)
        assert connection.commands[-1] == "SPD 120"
        window.selected_car.set("kedaya2")
        window._selection_changed()
        assert connection.closed
        assert connection.commands[-1] == "STOP"
        assert not window.state.armed
        assert window.connection is None
        assert window.next_car_id == 2
        window._arm()
        assert not window.state.armed
    finally:
        window.close()


def test_connection_failure_keeps_window_available_for_reconnect(monkeypatch):
    import tkinter as tk

    real_tk = tk.Tk

    def hidden_root():
        root = real_tk()
        root.withdraw()
        return root

    monkeypatch.setattr(tk, "Tk", hidden_root)
    connection = FakeConnection()
    window = RemoteControlWindow(connection, 1, 100)
    try:
        window._network_failed(ConnectionError("test disconnect"))
        assert not window.closed
        assert connection.closed
        assert window.status.get() == "CONNECTION LOST"
        replacement = FakeConnection()
        window._connected(replacement)
        assert window.connection is replacement
        assert not window.state.armed
    finally:
        window.close()
