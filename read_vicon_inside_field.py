import argparse
import math
import time
from typing import Dict, Iterable, Tuple

from six_car_repel import CarConfig, CarObservation, VICON_HOST, ViconTracker


# The fourth point in the note was repeated as (13600, 5800). For a rectangle,
# this script uses the implied opposite corner: (13600, 850).
FIELD_CORNERS = (
    (-500.0, 2900.0),
    (5600.0, 2900.0),
    (-500.0, -2100.0),
    (5600.0, -2100.0),
)
DEFAULT_FIRST_CAR_ID = 1
DEFAULT_LAST_CAR_ID = 50
DEFAULT_STATUS_INTERVAL_SEC = 0.10


def parse_args() -> argparse.Namespace:
    xmin, xmax, ymin, ymax = bounds_from_corners(FIELD_CORNERS)

    parser = argparse.ArgumentParser(
        description="Read Vicon positions for cars inside a virtual rectangular field."
    )
    parser.add_argument("--vicon-host", default=VICON_HOST, help="Vicon server IP address.")
    parser.add_argument("--first-id", type=int, default=DEFAULT_FIRST_CAR_ID)
    parser.add_argument("--last-id", type=int, default=DEFAULT_LAST_CAR_ID)
    parser.add_argument("--subject-prefix", default="kedaya")
    parser.add_argument("--xmin", type=float, default=xmin)
    parser.add_argument("--xmax", type=float, default=xmax)
    parser.add_argument("--ymin", type=float, default=ymin)
    parser.add_argument("--ymax", type=float, default=ymax)
    parser.add_argument(
        "--status-interval",
        type=float,
        default=DEFAULT_STATUS_INTERVAL_SEC,
        help="Seconds between coordinate prints.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one frame with at least one in-field car, then exit.",
    )
    parser.add_argument(
        "--show-empty",
        action="store_true",
        help="Print a short line even when no cars are inside the field.",
    )
    return parser.parse_args()


def bounds_from_corners(
    corners: Iterable[Tuple[float, float]]
) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return min(xs), max(xs), min(ys), max(ys)


def make_car_configs(first_id: int, last_id: int, subject_prefix: str) -> Dict[int, CarConfig]:
    if first_id > last_id:
        raise ValueError("--first-id must be less than or equal to --last-id")

    return {
        car_id: CarConfig(
            marker_id=car_id,
            name=f"car{car_id}_sta",
            ip="",
            subject_name=f"{subject_prefix}{car_id}",
        )
        for car_id in range(first_id, last_id + 1)
    }


def is_inside_field(
    observation: CarObservation,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> bool:
    x_mm, y_mm = observation.position
    return xmin <= x_mm <= xmax and ymin <= y_mm <= ymax


def format_observation(observation: CarObservation) -> str:
    x_mm, y_mm = observation.position
    yaw_deg = math.degrees(observation.yaw)
    return f"car{observation.marker_id}: x={x_mm:.1f} y={y_mm:.1f} yaw={yaw_deg:+.1f}deg"


def main() -> None:
    args = parse_args()
    configs = make_car_configs(args.first_id, args.last_id, args.subject_prefix)
    tracker = ViconTracker(args.vicon_host, configs)

    print(
        "Virtual field bounds: "
        f"x=[{args.xmin:.1f}, {args.xmax:.1f}], "
        f"y=[{args.ymin:.1f}, {args.ymax:.1f}]"
    )
    print(
        f"Reading Vicon subjects {args.subject_prefix}{args.first_id} "
        f"through {args.subject_prefix}{args.last_id}; outside-field cars are ignored."
    )

    tracker.connect()
    last_print = 0.0

    try:
        while True:
            observations = tracker.get_observations()
            now = time.monotonic()
            if now - last_print < args.status_interval:
                time.sleep(0.005)
                continue

            inside = {
                car_id: observation
                for car_id, observation in observations.items()
                if is_inside_field(observation, args.xmin, args.xmax, args.ymin, args.ymax)
            }

            if inside:
                line = " | ".join(
                    format_observation(inside[car_id]) for car_id in sorted(inside)
                )
                print(f"Inside={len(inside)} | {line}")
                if args.once:
                    break
            elif args.show_empty:
                print("Inside=0")

            last_print = now
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
