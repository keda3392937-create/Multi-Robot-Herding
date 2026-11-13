"""把一次实验记录渲染成 GIF 动画和轨迹总览图。

数据直接来自实验日志：``relative_cage_runs/baseline/<时间>/states.csv`` 与
``metadata.json``，所以不需要重新跑一遍仿真，画的就是现场实际的运动。

命令行：

    python animate_herding_run.py                # 画最新一次
    python animate_herding_run.py --run relative_cage_runs/baseline/20261113_142036
    python animate_herding_run.py --latest --fps 8 --max-frames 200

输出（默认写在运行目录里）：

    trajectory.gif   随时间的动画，含轨迹拖尾、朝向、herder 凸包、笼子与开口
    trajectory.png   整段实验的轨迹总览（起点/终点标记）

坐标系使用算法内部的局部坐标：场地左下角为 (0,0)，笼子在左上角。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

DEFAULT_LOG_ROOT = Path("relative_cage_runs/baseline")

HARDER_COLOR = "#1f77b4"
EVADER_COLOR = "#ff7f0e"
INACTIVE_COLOR = "#9e9e9e"
CAGE_EDGE = "#444444"


def latest_run(log_root: Path = DEFAULT_LOG_ROOT) -> Path:
    root = Path(log_root)
    candidates = sorted(
        (path.parent for path in root.glob("*/states.csv")),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no run with states.csv under {root}")
    return candidates[-1]


def _float(text: str, default: float = float("nan")) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def load_run(run_dir: Path):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    rows: List[Dict[str, str]] = []
    with (run_dir / "states.csv").open(encoding="utf-8", newline="") as handle:
        rows.extend(csv.DictReader(handle))
    return meta, rows


def group_frames(rows: Sequence[Dict[str, str]]) -> List[Tuple[float, List[Dict[str, str]]]]:
    """按控制周期分组，保持时间顺序。"""
    buckets: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        buckets.setdefault(row.get("elapsed_sec", "0"), []).append(row)
    frames = []
    for key in sorted(buckets, key=lambda value: _float(value, 0.0)):
        frames.append((_float(key, 0.0), buckets[key]))
    return frames


def convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Andrew monotone chain；点数不足时原样返回。"""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return list(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: List[Tuple[float, float]] = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def cage_segments(width_mm: float, height_mm: float,
                  cage_width_mm: float, cage_height_mm: float,
                  opening_mm: float):
    """笼子墙段：opening<=0 时是完整的软边界，否则底边/右边各留一个开口。"""
    left, right = 0.0, cage_width_mm
    bottom, top = height_mm - cage_height_mm, height_mm
    if opening_mm <= 0.0:
        return (((left, bottom), (left, top)), ((left, top), (right, top)),
                ((right, top), (right, bottom)), ((right, bottom), (left, bottom)))
    centre_x = (left + right) / 2.0
    centre_y = (bottom + top) / 2.0
    half = opening_mm / 2.0
    return (
        ((left, bottom), (left, top)),
        ((left, top), (right, top)),
        ((left, bottom), (centre_x - half, bottom)),
        ((centre_x + half, bottom), (right, bottom)),
        ((right, bottom), (right, centre_y - half)),
        ((right, centre_y + half), (right, top)),
    )


def _heading_in_local(yaw: float, x_axis, y_axis) -> Tuple[float, float]:
    world = (math.cos(yaw), math.sin(yaw))
    return (
        world[0] * x_axis[0] + world[1] * x_axis[1],
        world[0] * y_axis[0] + world[1] * y_axis[1],
    )


def _frame_state(frame_rows):
    """把一个控制周期的行整理成 {car_id: {...}}，跳过没有坐标的车。"""
    state = {}
    for row in frame_rows:
        x = _float(row.get("local_x_mm", ""))
        y = _float(row.get("local_y_mm", ""))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        state[int(row["car_id"])] = {
            "x": x,
            "y": y,
            "yaw": _float(row.get("yaw_rad", ""), 0.0),
            "role": row.get("role", ""),
            "active": row.get("active") == "True",
            "inside": row.get("inside_cage") == "True",
            "hull": row.get("herder_hull") == "True",
            "command": row.get("command", ""),
        }
    return state


def render_run(run_dir: Path = None, *, fps: int = 10, max_frames: int = 240,
               labels: bool = True, log_root: Path = DEFAULT_LOG_ROOT) -> Dict[str, Path]:
    """渲染一次运行，返回 {'gif': 路径, 'png': 路径}。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    run_dir = Path(run_dir) if run_dir is not None else latest_run(log_root)
    meta, rows = load_run(run_dir)
    arena = meta.get("arena") or {}
    arguments = meta.get("arguments") or {}
    width_mm = float(arena.get("width_mm", 0.0))
    height_mm = float(arena.get("height_mm", 0.0))
    cage_width = float(arena.get("cage_width_mm", 0.0))
    cage_height = float(arena.get("cage_height_mm", 0.0))
    opening = float(arguments.get("cage_opening_mm") or 0.0)
    x_axis = tuple(arena.get("x_axis_world") or (1.0, 0.0))
    y_axis = tuple(arena.get("y_axis_world") or (0.0, 1.0))
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError(f"{run_dir.name} metadata.json has no field size")

    frames = group_frames(rows)
    if not frames:
        raise ValueError(f"{run_dir.name} states.csv has no control cycles")
    stride = max(1, math.ceil(len(frames) / max_frames))
    sampled = frames[::stride]
    if sampled[-1] is not frames[-1]:
        sampled.append(frames[-1])

    segments = cage_segments(width_mm, height_mm, cage_width, cage_height, opening)
    trails: Dict[int, List[Tuple[float, float]]] = {}

    fig, ax = plt.subplots(figsize=(8.0, 8.0 * height_mm / max(width_mm, 1.0)), dpi=110)

    def setup_axes():
        ax.set_xlim(-150, width_mm + 150)
        ax.set_ylim(-150, height_mm + 150)
        ax.set_aspect("equal")
        ax.set_xlabel("local x (mm)")
        ax.set_ylabel("local y (mm)")
        ax.add_patch(Rectangle((0, 0), width_mm, height_mm, fill=False,
                               edgecolor="#333333", linewidth=1.6))
        ax.add_patch(Rectangle((0, height_mm - cage_height), cage_width, cage_height,
                               facecolor="#fff3cd", edgecolor="none", zorder=0))
        for start, end in segments:
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color=CAGE_EDGE, linewidth=2.4, zorder=2)

    def draw_frame(index):
        ax.clear()
        setup_axes()
        elapsed, frame_rows = sampled[index]
        state = _frame_state(frame_rows)
        for car_id, item in state.items():
            trail = trails.setdefault(car_id, [])
            if not trail or math.dist(trail[-1], (item["x"], item["y"])) > 1.0:
                trail.append((item["x"], item["y"]))
        herders = {i: s for i, s in state.items() if s["role"] == "herder"}
        evaders = {i: s for i, s in state.items() if s["role"] == "evader"}
        # 轨迹拖尾
        for car_id, trail in trails.items():
            role = "herder" if car_id in herders else "evader"
            color = HARDER_COLOR if role == "herder" else EVADER_COLOR
            alpha = 0.45 if car_id in state and state[car_id]["active"] else 0.15
            if len(trail) > 1:
                ax.plot([p[0] for p in trail], [p[1] for p in trail],
                        color=color, alpha=alpha, linewidth=1.0, zorder=3)
        # herder 凸包
        hull_points = [(s["x"], s["y"]) for s in herders.values() if s["hull"]]
        if len(hull_points) >= 3:
            hull = convex_hull(hull_points)
            ax.fill([p[0] for p in hull], [p[1] for p in hull],
                    color=HARDER_COLOR, alpha=0.10, zorder=1)
        for car_id, item in herders.items():
            color = HARDER_COLOR if item["active"] else INACTIVE_COLOR
            ax.scatter([item["x"]], [item["y"]], s=95, color=color,
                       edgecolors="white", linewidths=0.8, zorder=6)
            dx, dy = _heading_in_local(item["yaw"], x_axis, y_axis)
            ax.arrow(item["x"], item["y"], dx * 220.0, dy * 220.0, width=14.0,
                     head_width=60.0, head_length=60.0, color=color,
                     alpha=0.9, length_includes_head=True, zorder=5)
            if labels:
                ax.annotate(f"H{car_id}", (item["x"], item["y"]), xytext=(7, 7),
                            textcoords="offset points", fontsize=7, color=color, zorder=7)
        for car_id, item in evaders.items():
            color = EVADER_COLOR if item["active"] else INACTIVE_COLOR
            edge = "#2e7d32" if item["inside"] else "white"
            ax.scatter([item["x"]], [item["y"]], s=80, color=color,
                       edgecolors=edge, linewidths=1.6, zorder=6)
            if labels:
                ax.annotate(f"E{car_id}", (item["x"], item["y"]), xytext=(7, -11),
                            textcoords="offset points", fontsize=7, color=color, zorder=7)
        inside = sum(1 for item in evaders.values() if item["inside"])
        active_h = sum(1 for item in herders.values() if item["active"])
        active_e = sum(1 for item in evaders.values() if item["active"])
        ax.set_title(
            f"t = {elapsed:5.1f}s   herders {active_h}/{len(herders)}   "
            f"evaders {active_e}/{len(evaders)}   in cage {inside}",
            fontsize=11,
        )
        fig.tight_layout()

    gif_path = run_dir / "trajectory.gif"
    png_path = run_dir / "trajectory.png"

    # 先写轨迹总览：即使动画渲染被 Ctrl+C 打断，也一定有一张图可看。
    # 轨迹总览：所有走过的路径 + 起点/终点
    for index in range(len(sampled)):
        elapsed, frame_rows = sampled[index]
        for car_id, item in _frame_state(frame_rows).items():
            trail = trails.setdefault(car_id, [])
            if not trail or math.dist(trail[-1], (item["x"], item["y"])) > 1.0:
                trail.append((item["x"], item["y"]))
    fig2, ax2 = plt.subplots(figsize=(8.0, 8.0 * height_mm / max(width_mm, 1.0)), dpi=110)
    ax2.set_xlim(-150, width_mm + 150)
    ax2.set_ylim(-150, height_mm + 150)
    ax2.set_aspect("equal")
    ax2.set_xlabel("local x (mm)")
    ax2.set_ylabel("local y (mm)")
    ax2.add_patch(Rectangle((0, 0), width_mm, height_mm, fill=False,
                            edgecolor="#333333", linewidth=1.6))
    ax2.add_patch(Rectangle((0, height_mm - cage_height), cage_width, cage_height,
                            facecolor="#fff3cd", edgecolor="none", zorder=0))
    for start, end in segments:
        ax2.plot([start[0], end[0]], [start[1], end[1]], color=CAGE_EDGE,
                 linewidth=2.4, zorder=2)
    last_frame_state = _frame_state(sampled[-1][1])
    for car_id, trail in trails.items():
        if len(trail) < 2:
            continue
        first, last = trail[0], trail[-1]
        role = last_frame_state.get(car_id, {}).get("role", "evader")
        color = HARDER_COLOR if role == "herder" else EVADER_COLOR
        ax2.plot([p[0] for p in trail], [p[1] for p in trail],
                 color=color, alpha=0.55, linewidth=0.9)
        ax2.scatter([first[0]], [first[1]], marker="o", s=26, color=color,
                    edgecolors="white", linewidths=0.6, zorder=6)
        ax2.scatter([last[0]], [last[1]], marker="X", s=46, color=color,
                    edgecolors="white", linewidths=0.6, zorder=6)
        ax2.annotate(str(car_id), (last[0], last[1]), xytext=(6, 5),
                     textcoords="offset points", fontsize=7, color=color)
    ax2.set_title(f"{run_dir.name}: start (o) and end (X) of every car", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(png_path)
    plt.close(fig2)

    # 再逐帧渲染动画。自己收集帧，Ctrl+C 时可以保存已经渲染的部分。
    from PIL import Image
    trails.clear()
    frames_rgb: List[Image.Image] = []
    palette_reference = None
    try:
        for index in range(len(sampled)):
            draw_frame(index)
            fig.canvas.draw()
            buffer = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            image = Image.fromarray(buffer)
            if palette_reference is None:
                palette_reference = image.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
                frames_rgb.append(palette_reference)
            else:
                frames_rgb.append(image.quantize(
                    palette=palette_reference, dither=Image.Dither.NONE))
            if (index + 1) % 25 == 0:
                print(f"  GIF frame {index + 1}/{len(sampled)}")
    except KeyboardInterrupt:
        print(f"  GIF rendering interrupted after {len(frames_rgb)}/{len(sampled)} frames; "
              "saving a shorter animation")
    finally:
        plt.close(fig)
    if frames_rgb:
        temporary = gif_path.with_suffix(".tmp.gif")
        try:
            frames_rgb[0].save(
                temporary, save_all=True, append_images=frames_rgb[1:],
                duration=int(1000 / max(fps, 1)), loop=0, optimize=False,
            )
            temporary.replace(gif_path)
        except KeyboardInterrupt:
            temporary.unlink(missing_ok=True)
            print("  GIF saving interrupted; the overview image is still available")
    return {"gif": gif_path, "png": png_path}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Render one recorded herding run into a GIF and a trajectory overview."
    )
    parser.add_argument("--run", type=Path, default=None,
                        help="run directory; defaults to the newest one")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=240,
                        help="cap the animation frame count; long runs are subsampled")
    parser.add_argument("--no-labels", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        paths = render_run(args.run, fps=args.fps, max_frames=args.max_frames,
                           labels=not args.no_labels, log_root=args.log_root)
    except (OSError, ValueError) as exc:
        print(f"Cannot render this run: {exc}")
        return 2
    print(f"GIF: {paths['gif']}")
    print(f"Trajectory overview: {paths['png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
