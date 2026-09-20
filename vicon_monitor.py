"""Read Vicon in a separate process so SDK calls cannot stall keyboard control."""

from __future__ import annotations

import math
import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path


DEFAULT_VICON_HOST = "192.168.30.100"
DEFAULT_SDK_PATH = r"D:\ViconDataStream\Win64\Python\vicon_dssdk"
MAX_FRAME_AGE_SEC = 0.5


def read_frame(client, now: float) -> dict:
    poses = {}
    for subject in client.GetSubjectNames():
        if not subject.startswith("kedaya"):
            continue
        segment = client.GetSubjectRootSegmentName(subject)
        position, hidden_position = client.GetSegmentGlobalTranslation(subject, segment)
        rotation, hidden_rotation = client.GetSegmentGlobalRotationEulerXYZ(subject, segment)
        values = (*position, *rotation)
        poses[subject] = None if (
            hidden_position or hidden_rotation or not all(math.isfinite(x) for x in values)
        ) else (*position, math.degrees(rotation[2]))
    return {"status": "STREAMING", "received_at": now,
            "frame": client.GetFrameNumber(), "poses": poses}


def describe_snapshot(snapshot: dict, subject: str, now: float) -> tuple[str, str]:
    if snapshot.get("status") != "STREAMING":
        return snapshot.get("status", "CONNECTING"), "--"
    age = now - snapshot["received_at"]
    if age > MAX_FRAME_AGE_SEC:
        return f"STALE ({age:.1f} s)", "--"
    poses = snapshot["poses"]
    if subject not in poses:
        return f"SUBJECT NOT FOUND: {subject}", "--"
    if poses[subject] is None:
        return f"OCCLUDED: {subject}", "--"
    x, y, z, yaw = poses[subject]
    return (f"TRACKED | frame {snapshot['frame']} | {age * 1000:.0f} ms",
            f"X {x:.1f} mm    Y {y:.1f} mm    Z {z:.1f} mm    Yaw {yaw:.1f} deg")


def _publish(output, snapshot: dict) -> None:
    try:
        output.put_nowait(snapshot)
    except queue.Full:
        try:
            output.get(timeout=0.05)
        except queue.Empty:
            pass
        try:
            output.put_nowait(snapshot)
        except queue.Full:
            pass


def _worker(host: str, sdk_path: str, output, stopping) -> None:
    try:
        path = Path(sdk_path)
        if path.exists():
            sys.path.insert(0, str(path))
        from vicon_dssdk import ViconDataStream
    except Exception as exc:
        _publish(output, {"status": f"SDK ERROR: {exc}"})
        return

    while not stopping.is_set():
        client = None
        try:
            _publish(output, {"status": f"CONNECTING: {host}"})
            client = ViconDataStream.Client()
            client.SetConnectionTimeout(1000)
            client.Connect(host)
            if not client.IsConnected():
                raise ConnectionError("Vicon connection failed")
            client.EnableSegmentData()
            client.SetStreamMode(ViconDataStream.Client.StreamMode.EClientPull)
            client.SetAxisMapping(
                ViconDataStream.Client.AxisMapping.EForward,
                ViconDataStream.Client.AxisMapping.ELeft,
                ViconDataStream.Client.AxisMapping.EUp,
            )
            _publish(output, {"status": "CONNECTED / WAITING FOR FRAMES"})
            last_frame = None
            while not stopping.is_set():
                if client.GetFrame():
                    frame = client.GetFrameNumber()
                    if frame != last_frame:
                        _publish(output, read_frame(client, time.monotonic()))
                        last_frame = frame
                stopping.wait(0.04)
        except Exception as exc:
            _publish(output, {"status": f"VICON ERROR: {exc}"})
            stopping.wait(1.0)
        finally:
            if client is not None:
                try:
                    client.Disconnect()
                except Exception:
                    pass


class ViconMonitor:
    def __init__(self, host: str, sdk_path: str):
        context = mp.get_context("spawn")
        self.output = context.Queue(maxsize=1)
        self.stopping = context.Event()
        self.process = context.Process(
            target=_worker, args=(host, sdk_path, self.output, self.stopping), daemon=True,
        )
        self.snapshot = {"status": "CONNECTING"}
        self.process.start()

    def poll(self) -> dict:
        try:
            while True:
                self.snapshot = self.output.get_nowait()
        except queue.Empty:
            if not self.process.is_alive() and not self.snapshot.get("status", "").startswith("SDK ERROR"):
                self.snapshot = {"status": "VICON WORKER STOPPED"}
            return self.snapshot

    def close(self) -> None:
        self.stopping.set()
        self.process.join(timeout=0.4)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        self.output.close()
