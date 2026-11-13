import csv
import json
import os

import pytest

pytest.importorskip("matplotlib")

from animate_herding_run import cage_segments, group_frames, latest_run, render_run


FIELDS = (
    "wall_time,monotonic_time,elapsed_sec,frame,car_id,role,active,tcp_active,visible,"
    "pose_age_sec,world_x_mm,world_y_mm,local_x_mm,local_y_mm,yaw_rad,accel_x,accel_y,"
    "model_vx,model_vy,measured_vx_mm_s,measured_vy_mm_s,target_x_mm,target_y_mm,"
    "command,direction_score,pwm,inside_cage,herder_hull,augmented_hull"
).split(",")

HEADER = {name: name for name in FIELDS}


def write_run(run_dir, frames=3):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps({
        "arena": {
            "origin_world": [0.0, 0.0],
            "x_axis_world": [1.0, 0.0],
            "y_axis_world": [0.0, 1.0],
            "width_mm": 4000.0,
            "height_mm": 4000.0,
            "cage_width_mm": 1600.0,
            "cage_height_mm": 1600.0,
        },
        "arguments": {"cage_opening_mm": 700.0},
    }), encoding="utf-8")
    with (run_dir / "states.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for step in range(frames):
            elapsed = step * 0.05
            for car_id, role, base_x, base_y in ((18, "herder", 1200.0, 1200.0),
                                                 (1, "evader", 2400.0, 1800.0)):
                row = {name: "" for name in FIELDS}
                row.update({
                    "elapsed_sec": f"{elapsed:.6f}",
                    "frame": 100 + step,
                    "car_id": str(car_id),
                    "role": role,
                    "active": "True",
                    "tcp_active": "True",
                    "visible": "True",
                    "pose_age_sec": "0.01",
                    "world_x_mm": f"{base_x + step * 10:.3f}",
                    "world_y_mm": f"{base_y + step * 5:.3f}",
                    "local_x_mm": f"{base_x + step * 10:.3f}",
                    "local_y_mm": f"{base_y + step * 5:.3f}",
                    "yaw_rad": "0.3",
                    "command": "F",
                    "pwm": "120",
                    "inside_cage": "False",
                    "herder_hull": str(role == "herder"),
                    "augmented_hull": "False",
                })
                writer.writerow(row)
    return run_dir


def test_render_run_writes_animation_and_overview(tmp_path):
    run_dir = write_run(tmp_path / "20260101_000000")
    paths = render_run(run_dir, fps=2, max_frames=3)
    assert paths["gif"].name == "trajectory.gif"
    assert paths["png"].name == "trajectory.png"
    assert paths["gif"].is_file() and paths["gif"].stat().st_size > 2000
    assert paths["png"].is_file() and paths["png"].stat().st_size > 2000


def test_latest_run_uses_the_most_recent_states_file(tmp_path):
    older = write_run(tmp_path / "20260101_000000")
    newer = write_run(tmp_path / "20260102_000000")
    os.utime(older / "states.csv", (1_000_000, 1_000_000))
    os.utime(newer / "states.csv", (2_000_000, 2_000_000))
    assert latest_run(tmp_path) == newer


def test_group_frames_keeps_control_cycles_together(tmp_path):
    run_dir = write_run(tmp_path / "20260101_000000", frames=4)
    import csv as csv_module
    with (run_dir / "states.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv_module.DictReader(handle))
    frames = group_frames(rows)
    assert len(frames) == 4
    assert all(len(rows_in_frame) == 2 for _, rows_in_frame in frames)


def test_cage_segments_leave_both_openings():
    segments = cage_segments(4000.0, 4000.0, 1600.0, 1600.0, 700.0)
    assert len(segments) == 6
    # 底边开口在 x=450..1150，右边开口在 y=2850..3550
    bottom = sorted(
        segment for segment in segments
        if segment[0][1] == 2400.0 and segment[1][1] == 2400.0
    )
    assert bottom[0][1][0] == pytest.approx(450.0)
    assert bottom[1][0][0] == pytest.approx(1150.0)
