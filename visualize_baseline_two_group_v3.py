"""Live visualization launcher for the fixed 4+4 V3 baseline."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from baseline_two_group_v3_standalone import HerdingSimulation as V3Simulation


class VisualV3Simulation(V3Simulation):
    """V3 physics with a throttled, group-aware Matplotlib view."""

    def __init__(
        self,
        *,
        random_seed: int,
        dt: float,
        duration: float,
        frame_stride: int,
        pause_seconds: float,
        hold_at_end: bool,
    ) -> None:
        if isinstance(frame_stride, bool) or frame_stride < 1:
            raise ValueError("frame_stride must be a positive integer.")
        if not math.isfinite(pause_seconds) or pause_seconds <= 0.0:
            raise ValueError("pause_seconds must be finite and positive.")
        self._visualFrameStride = int(frame_stride)
        self._visualPauseSeconds = float(pause_seconds)
        self._holdAtEnd = bool(hold_at_end)
        self._visualCallCount = 0
        self._herderLabels = []
        super().__init__(random_seed=random_seed, render=True)
        self.dt = float(dt)
        self.totalTime = float(duration)
        self.validate_safety_timestep()

    def setup_visualization(self) -> None:
        super().setup_visualization()
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("V3 fixed 4+4 baseline")

        self.herder_scatter.set_edgecolors("black")
        self.herder_scatter.set_linewidths(1.0)
        self._herderLabels = [
            self.ax.text(
                self.positionsHerder[index, 0] + 0.8,
                self.positionsHerder[index, 1] + 0.8,
                f"H{index}",
                fontsize=8,
                color="black",
                zorder=8,
            )
            for index in range(self.numHerders)
        ]

        from matplotlib.lines import Line2D

        self.ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=self.group_colors[0],
                    markeredgecolor="black",
                    label="Herder group 0",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=self.group_colors[1],
                    markeredgecolor="black",
                    label="Herder group 1",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="yellow",
                    markeredgecolor="black",
                    label="Evader",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#16803a",
                    linestyle="--",
                    linewidth=2.0,
                    label="Target region",
                ),
            ],
            loc="lower right",
            fontsize=8,
            framealpha=0.9,
        )

    def update_visualization(self, current_time: float) -> bool:
        if not self.renderEnabled or self._plt is None:
            raise RuntimeError("update_visualization requires render=True.")
        complete = self.count_evaders_in_target() == self.numEvaders
        if not self._plt.fignum_exists(self.fig.number):
            return True

        self._visualCallCount += 1
        if (
            self._visualCallCount % self._visualFrameStride != 0
            and not complete
        ):
            return False

        group_colors = [
            self.group_colors[int(group_id)] for group_id in self.herderGroups
        ]
        self.evader_scatter.set_offsets(self.positionsEvader)
        self.herder_scatter.set_offsets(self.positionsHerder)
        self.herder_scatter.set_facecolors(group_colors)
        self.update_convex_hull_visualization()

        for index in range(self.numHerders):
            self.herder_fov_patches[index].remove()
            self.herder_fov_patches[index] = self.create_fov_patch(index)
            self.ax.add_patch(self.herder_fov_patches[index])
            position = self.positionsHerder[index]
            self.herder_repulsion_circles[index].center = tuple(position)
            self.herder_interaction_circles[index].center = tuple(position)
            self._herderLabels[index].set_position(
                (float(position[0] + 0.8), float(position[1] + 0.8))
            )

        count = self.count_evaders_in_target()
        display_time = min(float(current_time + self.dt), float(self.totalTime))
        horizon_reached = current_time + self.dt >= self.totalTime - 1e-9
        episode_finished = complete or horizon_reached
        group_zero = [
            f"H{index}" for index, group in enumerate(self.herderGroups) if group == 0
        ]
        group_one = [
            f"H{index}" for index, group in enumerate(self.herderGroups) if group == 1
        ]
        self.ax.set_title(
            f"V3 fixed 4+4 | t={display_time:.1f}s | "
            f"in target={count}/{self.numEvaders}\n"
            f"G0: {', '.join(group_zero)}    G1: {', '.join(group_one)}"
            + ("\nSimulation complete" if episode_finished else "")
        )
        self.fig.canvas.draw_idle()
        self._plt.pause(
            max(0.5, self._visualPauseSeconds)
            if episode_finished
            else self._visualPauseSeconds
        )
        if episode_finished and not self._holdAtEnd:
            self._plt.close(self.fig)
        return complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=930100)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--duration", type=float, default=1500.0)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Draw one frame per N physics steps; physics dt is unchanged.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.001,
        help="GUI event pause per rendered frame in seconds.",
    )
    parser.add_argument(
        "--hold-at-end",
        action="store_true",
        help="Keep the completed figure open until it is closed manually.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    simulation = VisualV3Simulation(
        random_seed=args.seed,
        dt=args.dt,
        duration=args.duration,
        frame_stride=args.frame_stride,
        pause_seconds=args.pause,
        hold_at_end=args.hold_at_end,
    )
    summary = simulation.run_simulation()
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
