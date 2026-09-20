#!/usr/bin/env python3
"""Record car13 FZMOTION positions and build an estimated coverage heatmap."""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from tkinter import ttk
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for import_root in (SCRIPT_DIR, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from fzmotion_client import VRPNClient, VRPNError  # noqa: E402

from keyboard_remote_control import (  # noqa: E402
    CAR_ID,
    CAR_IP,
    COMMAND_REFRESH_SEC,
    CONTROL_PORT,
    DISCOVERY_TIMEOUT_SEC,
    MOTOR_SPEED,
    SPEED_STEP,
    TK_MOTION_KEYS,
    VALID_CAR_IDS,
    WIFI_PASSWORD,
    WIFI_SSID,
    WIFI_WAIT_SEC,
    CarConnection,
    CommandRefresher,
    KeyboardControlState,
    connect_windows_wifi,
    discover_car,
)


# Main settings. The command line can temporarily override each value.
FZMOTION_HOST = "192.168.2.3"
FZMOTION_PORT = 3883
TARGET_OBJECT_NAME = "car13"
TARGET_SENSOR_ID: int | None = None  # None locks to the first car13 sensor received.
GROUND_AXES = "xz"  # Change to "xy" if Z is vertical in your FZMOTION calibration.
TRANSPORT = "tcp"  # TCP avoids mistaking UDP packet loss for optical tracking loss.

CELL_SIZE_M = 0.10
FIELD_BOUNDS_M: tuple[float, float, float, float] | None = None
MIN_DROPOUT_GAP_S = 0.12
MAX_INFER_GAP_S = 5.0
MAX_PLAUSIBLE_SPEED_MPS = 2.0
MIN_SAMPLES_PER_CELL = 3
OUTPUT_ROOT = Path(__file__).with_name("coverage_runs")

LIVE_UI_REFRESH_MS = 20
LIVE_PLOT_REFRESH_MS = 100
LIVE_PLOT_MAX_POINTS = 4000

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class PositionSample:
    elapsed_s: float
    server_time_s: float
    receive_time_s: float
    sensor_id: int
    position_m: tuple[float, float, float]


@dataclass(frozen=True)
class TrackingGap:
    start: PositionSample
    end: PositionSample
    duration_s: float
    distance_m: float
    inferred_positions: tuple[tuple[float, float], ...]
    classification: str


def utc_text(timestamp_s: float | None = None) -> str:
    value = time.time() if timestamp_s is None else timestamp_s
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="milliseconds")


def parse_axes(value: str) -> tuple[int, int, str, str]:
    text = value.strip().lower()
    if len(text) != 2 or text[0] == text[1] or any(axis not in AXIS_INDEX for axis in text):
        raise ValueError("--axes must be one of xy, xz, yx, yz, zx, or zy")
    return AXIS_INDEX[text[0]], AXIS_INDEX[text[1]], text[0].upper(), text[1].upper()


def ground_position(
    sample: PositionSample, axis_indices: tuple[int, int]
) -> tuple[float, float]:
    return sample.position_m[axis_indices[0]], sample.position_m[axis_indices[1]]


def estimate_frame_period(samples: Sequence[PositionSample]) -> float:
    deltas = [
        right.server_time_s - left.server_time_s
        for left, right in zip(samples, samples[1:])
        if 0.0 < right.server_time_s - left.server_time_s <= 0.25
    ]
    return median(deltas) if deltas else 1.0 / 60.0


def find_tracking_gaps(
    samples: Sequence[PositionSample],
    axis_indices: tuple[int, int],
    *,
    min_gap_s: float,
    max_infer_gap_s: float,
    max_speed_mps: float,
) -> tuple[list[TrackingGap], float, float]:
    frame_period_s = estimate_frame_period(samples)
    gap_threshold_s = max(min_gap_s, 3.0 * frame_period_s)
    gaps: list[TrackingGap] = []

    for start, end in zip(samples, samples[1:]):
        duration_s = end.server_time_s - start.server_time_s
        if duration_s <= gap_threshold_s:
            continue

        start_xy = ground_position(start, axis_indices)
        end_xy = ground_position(end, axis_indices)
        distance_m = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
        average_speed = distance_m / duration_s if duration_s > 0 else math.inf
        plausible = (
            start.sensor_id == end.sensor_id
            and duration_s <= max_infer_gap_s
            and average_speed <= max_speed_mps
        )

        inferred: tuple[tuple[float, float], ...] = ()
        classification = "unknown_long_gap"
        if plausible:
            missing_count = max(1, min(10000, round(duration_s / frame_period_s) - 1))
            inferred = tuple(
                (
                    start_xy[0] + (end_xy[0] - start_xy[0]) * step / (missing_count + 1),
                    start_xy[1] + (end_xy[1] - start_xy[1]) * step / (missing_count + 1),
                )
                for step in range(1, missing_count + 1)
            )
            classification = "inferred_dropout"

        gaps.append(
            TrackingGap(
                start=start,
                end=end,
                duration_s=duration_s,
                distance_m=distance_m,
                inferred_positions=inferred,
                classification=classification,
            )
        )

    return gaps, frame_period_s, gap_threshold_s


def make_grid_edges(
    direct_positions: np.ndarray,
    inferred_positions: np.ndarray,
    bounds: tuple[float, float, float, float] | None,
    cell_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        points = direct_positions
        if inferred_positions.size:
            points = np.vstack((points, inferred_positions))
        min_u, min_v = np.min(points, axis=0)
        max_u, max_v = np.max(points, axis=0)
        padding = max(cell_size_m, 0.05 * max(max_u - min_u, max_v - min_v, cell_size_m))
        min_u -= padding
        max_u += padding
        min_v -= padding
        max_v += padding
    else:
        min_u, max_u, min_v, max_v = bounds
        if not min_u < max_u or not min_v < max_v:
            raise ValueError("field bounds must satisfy U_MIN < U_MAX and V_MIN < V_MAX")

    min_u = math.floor(min_u / cell_size_m) * cell_size_m
    max_u = math.ceil(max_u / cell_size_m) * cell_size_m
    min_v = math.floor(min_v / cell_size_m) * cell_size_m
    max_v = math.ceil(max_v / cell_size_m) * cell_size_m
    if max_u <= min_u:
        max_u = min_u + cell_size_m
    if max_v <= min_v:
        max_v = min_v + cell_size_m

    u_edges = np.arange(min_u, max_u + 0.5 * cell_size_m, cell_size_m)
    v_edges = np.arange(min_v, max_v + 0.5 * cell_size_m, cell_size_m)
    if (len(u_edges) - 1) * (len(v_edges) - 1) > 2_000_000:
        raise ValueError("coverage grid is too large; increase --cell-size-m or reduce --bounds")
    return u_edges, v_edges


def grid_status(detected: float, missing: float, min_samples: int) -> str:
    total = detected + missing
    if total < min_samples:
        return "unknown"
    availability = detected / total
    if missing == 0 or availability >= 0.95:
        return "detected"
    if availability >= 0.20:
        return "intermittent"
    return "dropout_candidate"


def write_analysis_outputs(
    session_dir: Path,
    samples: Sequence[PositionSample],
    gaps: Sequence[TrackingGap],
    axis_indices: tuple[int, int],
    axis_labels: tuple[str, str],
    *,
    bounds: tuple[float, float, float, float] | None,
    cell_size_m: float,
    min_samples: int,
    frame_period_s: float,
    gap_threshold_s: float,
    metadata: dict[str, object],
) -> Path:
    direct_positions = np.asarray(
        [ground_position(sample, axis_indices) for sample in samples], dtype=float
    )
    inferred_list = [
        position
        for gap in gaps
        if gap.classification == "inferred_dropout"
        for position in gap.inferred_positions
    ]
    inferred_positions = np.asarray(inferred_list, dtype=float).reshape((-1, 2))
    u_edges, v_edges = make_grid_edges(
        direct_positions, inferred_positions, bounds, cell_size_m
    )

    detected_counts, _, _ = np.histogram2d(
        direct_positions[:, 0], direct_positions[:, 1], bins=(u_edges, v_edges)
    )
    if inferred_positions.size:
        missing_counts, _, _ = np.histogram2d(
            inferred_positions[:, 0], inferred_positions[:, 1], bins=(u_edges, v_edges)
        )
    else:
        missing_counts = np.zeros_like(detected_counts)
    total_counts = detected_counts + missing_counts
    availability = np.divide(
        detected_counts,
        total_counts,
        out=np.full_like(detected_counts, np.nan, dtype=float),
        where=total_counts > 0,
    )
    availability[total_counts < min_samples] = np.nan

    grid_path = session_dir / "grid.csv"
    with grid_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                f"{axis_labels[0].lower()}_center_m",
                f"{axis_labels[1].lower()}_center_m",
                "detected_samples",
                "inferred_missing_samples",
                "estimated_availability",
                "status",
            ]
        )
        for u_index in range(len(u_edges) - 1):
            for v_index in range(len(v_edges) - 1):
                detected = float(detected_counts[u_index, v_index])
                missing = float(missing_counts[u_index, v_index])
                value = availability[u_index, v_index]
                writer.writerow(
                    [
                        0.5 * (u_edges[u_index] + u_edges[u_index + 1]),
                        0.5 * (v_edges[v_index] + v_edges[v_index + 1]),
                        int(detected),
                        int(missing),
                        "" if np.isnan(value) else float(value),
                        grid_status(detected, missing, min_samples),
                    ]
                )

    gaps_path = session_dir / "gaps.csv"
    with gaps_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "classification",
                "start_elapsed_s",
                "end_elapsed_s",
                "duration_s",
                "distance_m",
                f"start_{axis_labels[0].lower()}_m",
                f"start_{axis_labels[1].lower()}_m",
                f"end_{axis_labels[0].lower()}_m",
                f"end_{axis_labels[1].lower()}_m",
                "inferred_missing_samples",
            ]
        )
        for gap in gaps:
            start_uv = ground_position(gap.start, axis_indices)
            end_uv = ground_position(gap.end, axis_indices)
            writer.writerow(
                [
                    gap.classification,
                    gap.start.elapsed_s,
                    gap.end.elapsed_s,
                    gap.duration_s,
                    gap.distance_m,
                    start_uv[0],
                    start_uv[1],
                    end_uv[0],
                    end_uv[1],
                    len(gap.inferred_positions),
                ]
            )

    figure, axis = plt.subplots(figsize=(9, 7))
    color_map = plt.get_cmap("RdYlGn").copy()
    color_map.set_bad("#d8d8d8")
    image = axis.imshow(
        np.ma.masked_invalid(availability.T),
        origin="lower",
        extent=(u_edges[0], u_edges[-1], v_edges[0], v_edges[-1]),
        interpolation="nearest",
        cmap=color_map,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    color_bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    color_bar.set_label("Estimated detection availability")

    plot_points = direct_positions
    if len(plot_points) > 20000:
        plot_points = plot_points[:: math.ceil(len(plot_points) / 20000)]
    axis.scatter(
        plot_points[:, 0],
        plot_points[:, 1],
        s=2,
        color="black",
        alpha=0.22,
        linewidths=0,
        label="Direct FZMOTION positions",
    )

    inferred_label_used = False
    unknown_label_used = False
    for gap in gaps:
        start_uv = ground_position(gap.start, axis_indices)
        end_uv = ground_position(gap.end, axis_indices)
        if gap.classification == "inferred_dropout":
            axis.plot(
                [start_uv[0], end_uv[0]],
                [start_uv[1], end_uv[1]],
                color="#b2182b",
                linewidth=1.0,
                linestyle="--",
                alpha=0.75,
                label="Inferred dropout corridor" if not inferred_label_used else None,
            )
            inferred_label_used = True
        else:
            axis.plot(
                [start_uv[0], end_uv[0]],
                [start_uv[1], end_uv[1]],
                color="#ef8a62",
                linewidth=1.0,
                linestyle=":",
                alpha=0.75,
                label="Long gap, location unknown" if not unknown_label_used else None,
            )
            unknown_label_used = True

    axis.scatter(
        direct_positions[0, 0],
        direct_positions[0, 1],
        marker="o",
        s=28,
        color="#2166ac",
        label="Start",
        zorder=4,
    )
    axis.scatter(
        direct_positions[-1, 0],
        direct_positions[-1, 1],
        marker="s",
        s=28,
        color="#762a83",
        label="End",
        zorder=4,
    )
    axis.set_title(
        f"{metadata['object_name']} FZMOTION detection map\n"
        f"{len(samples)} direct samples, {len(gaps)} tracking gaps"
    )
    axis.set_xlabel(f"FZMOTION {axis_labels[0]} (m)")
    axis.set_ylabel(f"FZMOTION {axis_labels[1]} (m)")
    axis.set_xlim(u_edges[0], u_edges[-1])
    axis.set_ylim(v_edges[0], v_edges[-1])
    axis.grid(color="white", linewidth=0.25, alpha=0.35)
    axis.legend(loc="best", fontsize=8, framealpha=0.9)
    axis.text(
        0.01,
        0.01,
        "Gray = untested/unknown. Red dashed paths are inferred between reacquisition points.",
        transform=axis.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    figure.tight_layout()
    image_path = session_dir / "coverage.png"
    figure.savefig(image_path, dpi=180)
    plt.close(figure)

    metadata.update(
        {
            "sample_count": len(samples),
            "tracking_gap_count": len(gaps),
            "inferred_gap_count": sum(
                gap.classification == "inferred_dropout" for gap in gaps
            ),
            "estimated_frame_rate_hz": 1.0 / frame_period_s,
            "gap_threshold_s": gap_threshold_s,
            "cell_size_m": cell_size_m,
            "grid_bounds_m": [u_edges[0], u_edges[-1], v_edges[0], v_edges[-1]],
            "minimum_samples_per_cell": min_samples,
            "warning": (
                "Blank cells are untested/unknown. Dropout locations are straight-line "
                "estimates between the last visible and first reacquired positions."
            ),
        }
    )
    (session_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return image_path


class FZMotionCollector(threading.Thread):
    """Receive FZMOTION poses without blocking the Tk control window."""

    def __init__(
        self,
        args: argparse.Namespace,
        session_dir: Path,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="fzmotion-car13", daemon=False)
        self.args = args
        self.session_dir = session_dir
        self.stop_event = stop_event
        self.updates: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self.samples: list[PositionSample] = []
        self.selected_sensor: int | None = args.sensor
        self.capture_error: str | None = None
        self.started_wall = time.time()
        self.ended_wall = self.started_wall
        self.started_monotonic = time.monotonic()
        self.done_event = threading.Event()
        self.client: VRPNClient | None = None

    def request_stop(self) -> None:
        self.stop_event.set()
        client = self.client
        if client is not None:
            client.close()

    def run(self) -> None:
        pose_path = self.session_dir / "poses.csv"
        try:
            with pose_path.open("w", newline="", encoding="utf-8") as pose_file:
                writer = csv.writer(pose_file)
                writer.writerow(
                    [
                        "receive_time_utc",
                        "elapsed_s",
                        "server_time_s",
                        "object_name",
                        "sensor_id",
                        "x_m",
                        "y_m",
                        "z_m",
                    ]
                )

                self.client = VRPNClient(
                    self.args.host,
                    self.args.port,
                    transport=self.args.transport,
                    local_ip=self.args.local_ip or None,
                )
                with self.client as client:
                    self.updates.put(
                        (
                            "connected",
                            f"{self.args.host}:{self.args.port} via {self.args.transport}",
                        )
                    )
                    print(
                        f"FZMOTION connected: {self.args.host}:{self.args.port} "
                        f"via {self.args.transport}"
                    )
                    next_flush = time.monotonic() + 1.0
                    next_tracks_update = time.monotonic()

                    while not self.stop_event.is_set():
                        now = time.monotonic()
                        if (
                            self.args.duration_s is not None
                            and now - self.started_monotonic >= self.args.duration_s
                        ):
                            self.updates.put(("finished", "duration"))
                            break

                        for pose in client.poll_poses(0.02):
                            if pose.object_name != self.args.object_name:
                                continue
                            if self.selected_sensor is None:
                                self.selected_sensor = pose.sensor_id
                                self.updates.put(("sensor", pose.sensor_id))
                                print(
                                    f"Locked {self.args.object_name} to sensor "
                                    f"{self.selected_sensor}."
                                )
                            if pose.sensor_id != self.selected_sensor:
                                continue

                            received_monotonic = time.monotonic()
                            sample = PositionSample(
                                elapsed_s=received_monotonic - self.started_monotonic,
                                server_time_s=pose.server_time_s,
                                receive_time_s=pose.receive_time_s,
                                sensor_id=pose.sensor_id,
                                position_m=pose.position_m,
                            )
                            self.samples.append(sample)
                            writer.writerow(
                                [
                                    utc_text(pose.receive_time_s),
                                    sample.elapsed_s,
                                    pose.server_time_s,
                                    pose.object_name,
                                    pose.sensor_id,
                                    pose.position_m[0],
                                    pose.position_m[1],
                                    pose.position_m[2],
                                ]
                            )
                            self.updates.put(
                                ("pose", (sample, received_monotonic))
                            )

                        now = time.monotonic()
                        if now >= next_flush:
                            pose_file.flush()
                            next_flush = now + 1.0
                        if now >= next_tracks_update:
                            self.updates.put(
                                ("tracks", tuple(sorted(client.known_tracks)))
                            )
                            next_tracks_update = now + 1.0

                    pose_file.flush()
        except (VRPNError, OSError, RuntimeError) as exc:
            if not self.stop_event.is_set():
                self.capture_error = f"{type(exc).__name__}: {exc}"
                self.updates.put(("error", self.capture_error))
                print(f"FZMOTION capture error: {exc}", file=sys.stderr)
        except Exception as exc:  # Keep thread failures visible in the UI and metadata.
            if not self.stop_event.is_set():
                self.capture_error = f"{type(exc).__name__}: {exc}"
                self.updates.put(("error", self.capture_error))
                print(f"Unexpected capture error: {exc}", file=sys.stderr)
        finally:
            self.client = None
            self.ended_wall = time.time()
            self.done_event.set()


class IntegratedCoverageWindow:
    """Keyboard remote control plus live FZMOTION detection display."""

    def __init__(
        self,
        connection: CarConnection,
        collector: FZMotionCollector,
        stop_event: threading.Event,
        args: argparse.Namespace,
        axis_indices: tuple[int, int],
        axis_labels: tuple[str, str],
        session_dir: Path,
    ) -> None:
        self.connection = connection
        self.collector = collector
        self.stop_event = stop_event
        self.args = args
        self.axis_indices = axis_indices
        self.axis_labels = axis_labels
        self.session_dir = session_dir
        self.speed = args.speed
        self.control_state = KeyboardControlState()
        self.refresher = CommandRefresher(connection, COMMAND_REFRESH_SEC)
        self.closed = False
        self.car_operational = True
        self.capture_operational = True
        self.capture_connected = False
        self.capture_finished = False
        self.capture_error: str | None = None
        self.selected_sensor = args.sensor
        self.known_tracks: tuple[tuple[str, int], ...] = ()
        self.last_sample: PositionSample | None = None
        self.last_pose_monotonic: float | None = None
        self.live_points: list[tuple[float, float]] = []
        self.live_gap_segments: list[
            tuple[tuple[float, float], tuple[float, float]]
        ] = []
        self.last_plot_elapsed = float("-inf")
        self.tracking_detected = False
        self.plot_dirty = True
        self.next_plot_at = 0.0

        self.root = tk.Tk()
        self.root.title(f"car{args.car_id} FZMOTION coverage")
        self.root.geometry("980x720")
        self.root.minsize(820, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.finish)
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.bind("<FocusOut>", self._on_focus_out)
        self.root.bind("<Unmap>", self._on_unmap)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        top = ttk.Frame(self.root, padding=(16, 12, 16, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        ttk.Label(
            top,
            text=f"{args.object_name} coverage run",
            font=("Segoe UI", 16, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            top,
            text=f"car{args.car_id}  {connection.ip}:{connection.port}",
        ).grid(row=0, column=1, sticky="e")

        status_row = ttk.Frame(self.root, padding=(16, 0, 16, 8))
        status_row.grid(row=1, column=0, sticky="ew")
        status_row.columnconfigure(0, weight=2)
        status_row.columnconfigure(1, weight=1)
        self.tracking_status = tk.Label(
            status_row,
            text=f"WAITING FOR {args.object_name}",
            anchor="w",
            padx=12,
            pady=8,
            bg="#f0ad2c",
            fg="#1f2933",
            font=("Segoe UI", 15, "bold"),
        )
        self.tracking_status.grid(row=0, column=0, padx=(0, 8), sticky="ew")
        self.car_status = tk.Label(
            status_row,
            text=f"DISARMED  SPD {self.speed}",
            anchor="w",
            padx=12,
            pady=8,
            bg="#d9e2ec",
            fg="#1f2933",
            font=("Segoe UI", 11, "bold"),
        )
        self.car_status.grid(row=0, column=1, sticky="ew")

        self.details = tk.StringVar(value="Position --    Samples 0    Time 0.0 s")
        ttk.Label(
            self.root,
            textvariable=self.details,
            padding=(16, 0, 16, 8),
        ).grid(row=2, column=0, sticky="ew")

        self.canvas = tk.Canvas(
            self.root,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#bcccdc",
        )
        self.canvas.grid(row=3, column=0, padx=16, sticky="nsew")
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Button-1>", lambda _event: self.root.focus_set())

        controls = ttk.Frame(self.root, padding=(16, 10, 16, 14))
        controls.grid(row=4, column=0, sticky="ew")
        controls.columnconfigure(3, weight=1)
        self.arm_button = ttk.Button(
            controls, text="ARM", command=self._arm, takefocus=False
        )
        self.arm_button.grid(row=0, column=0, padx=(0, 8))
        ttk.Button(
            controls, text="STOP", command=self._stop_and_disarm, takefocus=False
        ).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(
            controls, text="FINISH", command=self.finish, takefocus=False
        ).grid(row=0, column=2)
        self.run_info = tk.StringVar(value=session_dir.name)
        ttk.Label(controls, textvariable=self.run_info).grid(
            row=0, column=3, sticky="e"
        )

        self.root.after(LIVE_UI_REFRESH_MS, self._tick)
        self.root.after(150, self.root.focus_force)

    @staticmethod
    def _motion_key(keysym: str) -> str | None:
        return TK_MOTION_KEYS.get(keysym, TK_MOTION_KEYS.get(keysym.lower()))

    def _arm(self) -> None:
        if not self.car_operational or not self.capture_operational:
            return
        if self.control_state.arm():
            self.car_status.configure(
                text=f"ARMED  STOP  SPD {self.speed}", bg="#c6f6d5"
            )
        else:
            self.car_status.configure(text="RELEASE DIRECTION KEYS", bg="#fbd38d")
        self.root.focus_set()

    def _stop_and_disarm(self) -> None:
        self.control_state.disarm()
        if self.car_operational:
            self.car_status.configure(
                text=f"DISARMED  SPD {self.speed}", bg="#d9e2ec"
            )
            self._transmit()

    def _on_key_press(self, event) -> str | None:
        if event.keysym == "Escape":
            self.finish()
            return "break"
        if event.keysym in {"Return", "KP_Enter"}:
            self._arm()
            return "break"
        if event.keysym == "space":
            self._stop_and_disarm()
            return "break"
        if event.keysym == "bracketleft":
            self._set_speed(max(0, self.speed - SPEED_STEP))
            return "break"
        if event.keysym == "bracketright":
            self._set_speed(min(255, self.speed + SPEED_STEP))
            return "break"

        key = self._motion_key(event.keysym)
        if key is not None:
            self.control_state.press_key(key)
            self._transmit()
            return "break"
        return None

    def _on_key_release(self, event) -> str | None:
        key = self._motion_key(event.keysym)
        if key is not None:
            self.control_state.release_key(key)
            self._transmit()
            return "break"
        return None

    def _on_focus_out(self, _event) -> None:
        self.root.after_idle(self._disarm_if_unfocused)

    def _disarm_if_unfocused(self) -> None:
        if not self.closed and self.root.focus_get() is None:
            self._stop_and_disarm()

    def _on_unmap(self, event) -> None:
        if event.widget is self.root:
            self._stop_and_disarm()

    def _on_canvas_resize(self, _event) -> None:
        self.plot_dirty = True

    def _set_speed(self, speed: int) -> None:
        if speed == self.speed or not self.car_operational:
            return
        try:
            self.connection.set_speed(speed)
            self.speed = speed
            self.refresher.force_next()
            state = "ARMED" if self.control_state.armed else "DISARMED"
            self.car_status.configure(
                text=f"{state}  SPD {self.speed}",
                bg="#c6f6d5" if self.control_state.armed else "#d9e2ec",
            )
        except (OSError, RuntimeError) as exc:
            self._car_network_failed(exc)

    def _transmit(self) -> None:
        if self.closed or not self.car_operational:
            return
        command = self.control_state.desired_command()
        try:
            if self.refresher.update(command, time.monotonic()):
                if self.control_state.armed:
                    self.car_status.configure(
                        text=f"ARMED  {command}  SPD {self.speed}", bg="#c6f6d5"
                    )
                else:
                    self.car_status.configure(
                        text=f"DISARMED  SPD {self.speed}", bg="#d9e2ec"
                    )
        except (OSError, RuntimeError) as exc:
            self._car_network_failed(exc)

    def _car_network_failed(self, exc: BaseException) -> None:
        if not self.car_operational:
            return
        print(f"Car connection lost: {exc}", file=sys.stderr)
        self.control_state.disarm()
        self.car_operational = False
        self.arm_button.state(["disabled"])
        self.car_status.configure(text="CAR CONNECTION LOST", bg="#e53e3e", fg="white")

    def _capture_failed(self, message: str) -> None:
        self.capture_error = message
        self.capture_operational = False
        self.arm_button.state(["disabled"])
        self._stop_and_disarm()
        self.tracking_status.configure(
            text="FZMOTION ERROR", bg="#c53030", fg="white"
        )
        self.run_info.set(message)

    def _accept_pose(self, sample: PositionSample, received_monotonic: float) -> None:
        point = ground_position(sample, self.axis_indices)
        if self.last_sample is not None:
            previous = ground_position(self.last_sample, self.axis_indices)
            if sample.server_time_s - self.last_sample.server_time_s > self.args.min_gap_s:
                self.live_gap_segments.append((previous, point))

        if (
            not self.live_points
            or sample.elapsed_s - self.last_plot_elapsed >= 0.05
            or math.hypot(
                point[0] - self.live_points[-1][0],
                point[1] - self.live_points[-1][1],
            )
            >= 0.01
        ):
            self.live_points.append(point)
            self.last_plot_elapsed = sample.elapsed_s
            self.plot_dirty = True

        self.last_sample = sample
        self.last_pose_monotonic = received_monotonic

    def _drain_capture_updates(self) -> None:
        while True:
            try:
                kind, payload = self.collector.updates.get_nowait()
            except queue.Empty:
                return

            if kind == "connected":
                self.capture_connected = True
                self.run_info.set(f"FZMOTION {payload}")
            elif kind == "sensor":
                self.selected_sensor = int(payload)
            elif kind == "tracks":
                self.known_tracks = payload  # type: ignore[assignment]
            elif kind == "pose":
                sample, received_monotonic = payload  # type: ignore[misc]
                self._accept_pose(sample, received_monotonic)
            elif kind == "error":
                self._capture_failed(str(payload))
            elif kind == "finished":
                self.capture_finished = True

    def _update_tracking_status(self, now: float) -> None:
        elapsed = now - self.collector.started_monotonic
        sample_count = len(self.collector.samples)
        if self.last_sample is None:
            position_text = "Position --"
        else:
            u, v = ground_position(self.last_sample, self.axis_indices)
            position_text = (
                f"{self.axis_labels[0]} {u:+.3f} m    "
                f"{self.axis_labels[1]} {v:+.3f} m"
            )
        self.details.set(
            f"{position_text}    Samples {sample_count}    Time {elapsed:.1f} s"
        )

        if self.capture_error is not None:
            self.tracking_detected = False
            return
        if self.last_pose_monotonic is None:
            self.tracking_detected = False
            suffix = "" if self.capture_connected else " - CONNECTING"
            self.tracking_status.configure(
                text=f"WAITING FOR {self.args.object_name}{suffix}",
                bg="#f0ad2c",
                fg="#1f2933",
            )
            return

        stale_s = now - self.last_pose_monotonic
        self.tracking_detected = stale_s < self.args.min_gap_s
        if self.tracking_detected:
            sensor_text = "" if self.selected_sensor is None else f" [{self.selected_sensor}]"
            self.tracking_status.configure(
                text=f"DETECTED  {self.args.object_name}{sensor_text}",
                bg="#2f855a",
                fg="white",
            )
        else:
            self.tracking_status.configure(
                text=f"LOST  {stale_s:.2f} s",
                bg="#c53030",
                fg="white",
            )
            self.plot_dirty = True

    def _current_plot_bounds(self) -> tuple[float, float, float, float]:
        if self.args.bounds is not None:
            return tuple(self.args.bounds)
        if not self.live_points:
            return -1.0, 1.0, -1.0, 1.0

        min_u = min(point[0] for point in self.live_points)
        max_u = max(point[0] for point in self.live_points)
        min_v = min(point[1] for point in self.live_points)
        max_v = max(point[1] for point in self.live_points)
        span = max(max_u - min_u, max_v - min_v, 0.5)
        padding = max(0.20, 0.08 * span)
        return min_u - padding, max_u + padding, min_v - padding, max_v + padding

    def _redraw_plot(self) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 400)
        height = max(self.canvas.winfo_height(), 300)
        left, top, right, bottom = 64.0, 24.0, 24.0, 48.0
        plot_width = max(1.0, width - left - right)
        plot_height = max(1.0, height - top - bottom)
        min_u, max_u, min_v, max_v = self._current_plot_bounds()

        range_u = max(max_u - min_u, 1e-6)
        range_v = max(max_v - min_v, 1e-6)
        scale = min(plot_width / range_u, plot_height / range_v)
        used_width = range_u * scale
        used_height = range_v * scale
        origin_x = left + 0.5 * (plot_width - used_width)
        origin_y = top + 0.5 * (plot_height - used_height)

        def to_canvas(point: tuple[float, float]) -> tuple[float, float]:
            return (
                origin_x + (point[0] - min_u) * scale,
                origin_y + used_height - (point[1] - min_v) * scale,
            )

        self.canvas.create_rectangle(
            origin_x,
            origin_y,
            origin_x + used_width,
            origin_y + used_height,
            outline="#829ab1",
        )
        for index in range(6):
            fraction = index / 5.0
            x = origin_x + used_width * fraction
            y = origin_y + used_height * fraction
            self.canvas.create_line(
                x, origin_y, x, origin_y + used_height, fill="#e4e7eb"
            )
            self.canvas.create_line(
                origin_x, y, origin_x + used_width, y, fill="#e4e7eb"
            )
            u_value = min_u + range_u * fraction
            v_value = max_v - range_v * fraction
            self.canvas.create_text(
                x,
                origin_y + used_height + 18,
                text=f"{u_value:.2f}",
                fill="#52606d",
                font=("Segoe UI", 8),
            )
            self.canvas.create_text(
                origin_x - 10,
                y,
                text=f"{v_value:.2f}",
                anchor="e",
                fill="#52606d",
                font=("Segoe UI", 8),
            )

        self.canvas.create_text(
            origin_x + used_width / 2,
            height - 10,
            text=f"{self.axis_labels[0]} (m)",
            fill="#334e68",
            font=("Segoe UI", 9, "bold"),
        )
        self.canvas.create_text(
            12,
            origin_y,
            text=f"{self.axis_labels[1]} (m)",
            anchor="nw",
            fill="#334e68",
            font=("Segoe UI", 9, "bold"),
        )

        if self.live_points:
            step = max(1, math.ceil(len(self.live_points) / LIVE_PLOT_MAX_POINTS))
            visible_points = self.live_points[::step]
            if visible_points[-1] != self.live_points[-1]:
                visible_points.append(self.live_points[-1])
            if len(visible_points) >= 2:
                coordinates = [
                    value
                    for point in visible_points
                    for value in to_canvas(point)
                ]
                self.canvas.create_line(
                    *coordinates,
                    fill="#2b6cb0",
                    width=2,
                    smooth=False,
                )

            for gap_start, gap_end in self.live_gap_segments:
                start_x, start_y = to_canvas(gap_start)
                end_x, end_y = to_canvas(gap_end)
                self.canvas.create_line(
                    start_x,
                    start_y,
                    end_x,
                    end_y,
                    fill="#c53030",
                    width=3,
                    dash=(6, 4),
                )

            start_x, start_y = to_canvas(self.live_points[0])
            self.canvas.create_oval(
                start_x - 4,
                start_y - 4,
                start_x + 4,
                start_y + 4,
                fill="#2b6cb0",
                outline="white",
            )
            current_x, current_y = to_canvas(self.live_points[-1])
            current_color = "#2f855a" if self.tracking_detected else "#c53030"
            self.canvas.create_oval(
                current_x - 7,
                current_y - 7,
                current_x + 7,
                current_y + 7,
                fill=current_color,
                outline="white",
                width=2,
            )
        else:
            self.canvas.create_text(
                width / 2,
                height / 2,
                text=f"Waiting for {self.args.object_name} position",
                fill="#7b8794",
                font=("Segoe UI", 12),
            )

        self.plot_dirty = False

    def _tick(self) -> None:
        if self.closed:
            return
        self._drain_capture_updates()
        now = time.monotonic()
        self._update_tracking_status(now)
        self._transmit()
        if self.plot_dirty and now >= self.next_plot_at:
            self._redraw_plot()
            self.next_plot_at = now + LIVE_PLOT_REFRESH_MS / 1000.0
        if self.capture_finished:
            self.finish()
            return
        self.root.after(LIVE_UI_REFRESH_MS, self._tick)

    def finish(self) -> None:
        if self.closed:
            return
        self.control_state.disarm()
        if self.car_operational:
            try:
                self.refresher.update("STOP", time.monotonic())
            except (OSError, RuntimeError):
                pass
        self.collector.request_stop()
        self.closed = True
        self.root.destroy()

    def run(self) -> None:
        print("Keyboard: Enter=arm, Q/W/E/A/S/D=move, Space=stop, Esc=finish")
        print("Up/Down also move forward/backward; [ and ] change speed.")
        self.root.mainloop()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Control one car while recording its FZMOTION positions, then create "
            "a detection heatmap."
        )
    )
    car_group = parser.add_argument_group("car remote control")
    car_group.add_argument("--car-id", type=int, default=CAR_ID)
    car_group.add_argument("--car-ip", default=CAR_IP)
    car_group.add_argument("--car-port", type=int, default=CONTROL_PORT)
    car_group.add_argument("--speed", type=int, default=MOTOR_SPEED)
    car_group.add_argument("--wifi-ssid", default=WIFI_SSID)
    car_group.add_argument("--wifi-password", default=WIFI_PASSWORD)
    car_group.add_argument("--skip-wifi", action="store_true")
    car_group.add_argument("--wifi-wait", type=float, default=WIFI_WAIT_SEC)
    car_group.add_argument(
        "--discovery-timeout", type=float, default=DISCOVERY_TIMEOUT_SEC
    )
    car_group.add_argument("--bind-ip", action="append", default=[])
    car_group.add_argument("--broadcast-ip", action="append", default=[])

    tracking_group = parser.add_argument_group("FZMOTION tracking")
    tracking_group.add_argument("--host", default=FZMOTION_HOST)
    tracking_group.add_argument("--port", type=int, default=FZMOTION_PORT)
    tracking_group.add_argument(
        "--transport", choices=("tcp", "udp"), default=TRANSPORT
    )
    tracking_group.add_argument("--local-ip", default="")
    tracking_group.add_argument(
        "--object", dest="object_name", default=TARGET_OBJECT_NAME
    )
    tracking_group.add_argument("--sensor", type=int, default=TARGET_SENSOR_ID)
    tracking_group.add_argument("--axes", default=GROUND_AXES)
    tracking_group.add_argument("--duration-s", type=float)

    output_group = parser.add_argument_group("heatmap output")
    output_group.add_argument("--cell-size-m", type=float, default=CELL_SIZE_M)
    output_group.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("U_MIN", "U_MAX", "V_MIN", "V_MAX"),
        default=FIELD_BOUNDS_M,
    )
    output_group.add_argument("--min-gap-s", type=float, default=MIN_DROPOUT_GAP_S)
    output_group.add_argument(
        "--max-infer-gap-s", type=float, default=MAX_INFER_GAP_S
    )
    output_group.add_argument(
        "--max-speed-mps", type=float, default=MAX_PLAUSIBLE_SPEED_MPS
    )
    output_group.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_CELL)
    output_group.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--print-rate-hz", type=float, default=5.0, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[int, int, str, str]:
    if args.car_id not in VALID_CAR_IDS:
        raise ValueError("--car-id must be between 1 and 50")
    if not 1 <= args.car_port <= 65535:
        raise ValueError("--car-port must be between 1 and 65535")
    if not 0 <= args.speed <= 255:
        raise ValueError("--speed must be between 0 and 255")
    if args.wifi_wait < 0 or args.discovery_timeout <= 0:
        raise ValueError("Wi-Fi wait cannot be negative and discovery timeout must be positive")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.cell_size_m <= 0:
        raise ValueError("--cell-size-m must be positive")
    if args.min_gap_s <= 0 or args.max_infer_gap_s <= args.min_gap_s:
        raise ValueError("gap settings must be positive and max infer gap must be larger")
    if args.max_speed_mps <= 0 or args.min_samples < 1:
        raise ValueError("--max-speed-mps and --min-samples must be positive")
    if args.duration_s is not None and args.duration_s <= 0:
        raise ValueError("--duration-s must be positive")
    if args.bounds is not None:
        min_u, max_u, min_v, max_v = args.bounds
        if not min_u < max_u or not min_v < max_v:
            raise ValueError("--bounds must satisfy U_MIN < U_MAX and V_MIN < V_MAX")
    return parse_axes(args.axes)


def prepare_car_connection(args: argparse.Namespace) -> CarConnection:
    if not args.skip_wifi:
        print(f"Connecting Windows Wi-Fi: {args.wifi_ssid}")
        connect_windows_wifi(args.wifi_ssid, args.wifi_password, args.wifi_wait)

    if args.car_ip:
        car_ip, car_port = args.car_ip, args.car_port
    else:
        print(f"Discovering car{args.car_id}...")
        car_ip, car_port = discover_car(
            args.car_id,
            args.discovery_timeout,
            bind_ips=args.bind_ip,
            broadcast_ips=args.broadcast_ip,
        )

    connection = CarConnection.connect(args.car_id, car_ip, car_port)
    try:
        connection.stop()
        connection.set_speed(args.speed)
    except BaseException:
        connection.close(send_stop=True)
        raise
    return connection


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        first_axis, second_axis, first_label, second_label = validate_args(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    connection: CarConnection | None = None
    try:
        connection = prepare_car_connection(args)
    except KeyboardInterrupt:
        print("Cancelled before capture started.")
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Could not prepare car{args.car_id}: {exc}", file=sys.stderr)
        return 1

    session_name = f"{args.object_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = args.output_root / session_name
    try:
        session_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        connection.close(send_stop=True)
        print(f"Could not create session directory: {exc}", file=sys.stderr)
        return 1
    print(f"Session directory: {session_dir}")

    axis_indices = (first_axis, second_axis)
    axis_labels = (first_label, second_label)
    stop_event = threading.Event()
    collector = FZMotionCollector(args, session_dir, stop_event)
    window: IntegratedCoverageWindow | None = None
    ui_error: str | None = None
    collector.start()
    try:
        window = IntegratedCoverageWindow(
            connection,
            collector,
            stop_event,
            args,
            axis_indices,
            axis_labels,
            session_dir,
        )
        window.run()
    except KeyboardInterrupt:
        print("Stopping capture...")
        if window is not None:
            window.finish()
        else:
            collector.request_stop()
    except (tk.TclError, RuntimeError) as exc:
        ui_error = f"{type(exc).__name__}: {exc}"
        print(f"Control window error: {exc}", file=sys.stderr)
        collector.request_stop()
    finally:
        collector.request_stop()
        collector.join()
        connection.close(send_stop=True)

    samples = list(collector.samples)
    capture_error = collector.capture_error or ui_error
    metadata: dict[str, object] = {
        "started_utc": utc_text(collector.started_wall),
        "ended_utc": utc_text(collector.ended_wall),
        "host": args.host,
        "port": args.port,
        "transport": args.transport,
        "object_name": args.object_name,
        "sensor_id": collector.selected_sensor,
        "ground_axes": args.axes.lower(),
        "capture_error": capture_error,
        "car_id": args.car_id,
        "car_ip": connection.ip,
        "car_port": connection.port,
        "car_speed": args.speed,
    }
    if not samples:
        metadata["sample_count"] = 0
        metadata["warning"] = "No matching pose was received; no heatmap was generated."
        (session_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"No {args.object_name} position was received. Data kept in {session_dir}.")
        return 1

    print("Generating coverage heatmap...")
    gaps, frame_period_s, gap_threshold_s = find_tracking_gaps(
        samples,
        axis_indices,
        min_gap_s=args.min_gap_s,
        max_infer_gap_s=args.max_infer_gap_s,
        max_speed_mps=args.max_speed_mps,
    )
    try:
        image_path = write_analysis_outputs(
            session_dir,
            samples,
            gaps,
            axis_indices,
            axis_labels,
            bounds=tuple(args.bounds) if args.bounds is not None else None,
            cell_size_m=args.cell_size_m,
            min_samples=args.min_samples,
            frame_period_s=frame_period_s,
            gap_threshold_s=gap_threshold_s,
            metadata=metadata,
        )
    except (OSError, ValueError) as exc:
        print(f"Error while generating heatmap: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {len(samples)} positions.")
    print(f"Heatmap: {image_path}")
    print(f"Raw positions: {session_dir / 'poses.csv'}")
    print("Gray cells are untested/unknown; dashed red corridors are estimates, not ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
