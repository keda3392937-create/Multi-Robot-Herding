"""baseline 围捕算法的实机入口（Vicon 定位 + 多车 TCP 控制）。

直接在 IDE 里运行本文件，不需要命令行启动器。启动流程与
twelve_car_3herder_chase.py 保持一致：

  1. 连接 HERDER_IDS / EVADER_IDS 指定的车辆；
  2. 六方向短脉冲标定：每轮全部车辆同时做一个方向（满 PWM、0.45 秒），
     Vicon 取脉冲前后位移，位移足够的方向才保留；
  3. 进入 baseline 围捕控制（baseline_physical + relative_cage_capture）。

场地和笼子现在使用本文件顶部写死的 Vicon 世界坐标（FIELD_CORNERS_WORLD /
CAGE_CORNERS_WORLD），不再根据初始车位自动推算。第一次使用请先加 --dry-run
核对坐标，再正式运行。
"""
import time
from pathlib import Path
from statistics import median

import relative_cage_capture as runtime
from baseline_physical import BaselineController, PhysicalCommandFilter, physical_parameters
from baseline_virtual_arena import FixedCornerArena, InitialCarArena
from relative_cage_core import (
    ArenaGeometryError, vec_norm, vec_normalize, vec_sub,
)
from six_car_repel import VICON_HOST


# ============================== 车辆编号 ==============================
HERDER_IDS = tuple(range(18, 26))        # 18~25 号作为 herder
EVADER_IDS = tuple(range(1, 18))         # 1~17 号作为 evader
IGNORE_FAILED_CARS = True                # 单车故障只剔除该车，其余继续
VICON_SERVER = VICON_HOST                # 默认 192.168.33.103

# ====================== 【需要改】场地与笼子坐标 ======================
# 单位 mm，直接填 Tracker 里读到的 Vicon 世界坐标（X 向右、Y 向上）。
# 四个点的顺序随意，脚本会自己判断矩形和笼子贴着场地的哪个角。
# 笼子必须紧贴场地的一个角，它会成为局部坐标的“左上角”，
# 两个开口（CAGE_OPENING_MM）自动开在朝向场地内部的两条边上。
FIELD_CORNERS_WORLD = (
    (4000.0, 1000.0),
    (8000.0, 1000.0),
    (8000.0, 5000.0),
    (4000.0, 5000.0),
)
CAGE_CORNERS_WORLD = (
    (4000.0, 3400.0),
    (5600.0, 3400.0),
    (5600.0, 5000.0),
    (4000.0, 5000.0),
)
# 想回到“由初始车位自动确定场地”的旧模式，把下面改成 False
USE_FIXED_ARENA = True
# ======================================================================

# 车体与笼子
ROBOT_DIAMETER_MM = 150.0                # 三角底盘最大外径
BODY_GAP_MM = 80.0                       # 额外车距，最小中心距 = 直径 + 该值
CAGE_OPENING_MM = 700.0                  # 笼子两条边中央的开口宽度
WALL_REPULSION_MARGIN_MM = 300.0         # 虚拟墙斥力作用范围
MIN_START_CLEARANCE_MM = 300.0           # 建议初始车位距场地边界的最小距离

# 自动模式（USE_FIXED_ARENA=False）才会用到
FIELD_WIDTH_MM = 4000.0
FIELD_HEIGHT_MM = 4000.0
INITIAL_FIELD_INSET_MM = 300.0
CAGE_WIDTH_RATIO = 0.40
CAGE_HEIGHT_RATIO = 0.40

# baseline 算法（见 docs/BASELINE_REAL_ROBOTS.md）
# 4 m 场地：herder 最大 160 mm/s，evader 最大 240 mm/s
SIMULATION_TIME_SCALE = 1.6
HERDER_MAX_ACCEL_MM_S2 = 300.0
EVADER_MAX_ACCEL_MM_S2 = 450.0
PREDICTION_HORIZON_SEC = 1.2
REACTION_TIME_SEC = 0.25
SPEED_UNCERTAINTY_FACTOR = 1.4
JAM_TIMEOUT_SEC = 30.0

# 运行 PWM。实测：170 PWM 约 200~230 mm/s，110 PWM 约 90~120 mm/s。
HERDER_MIN_PWM = 90
HERDER_MAX_PWM = 170
EVADER_MIN_PWM = 95
EVADER_MAX_PWM = 230
# Vicon 单帧跳变会放大瞬时速度，这里留出测量裕量；超过即为整组安全检查停车。
ACTUAL_SPEED_LIMIT_FACTOR = 2.5

# 六方向标定（与 twelve_car_3herder_chase.py 相同的脉冲参数）
CALIBRATION_SPEED = 255
CALIBRATION_MOVE_SEC = 0.45
CALIBRATION_SETTLE_SEC = 0.35
MIN_CALIBRATION_DISPLACEMENT_MM = 30.0
CALIBRATION_SAMPLES = 4                  # 旧串行标定使用
MIN_CALIBRATED_COMMANDS = 4              # 6 个方向里至少 4 个有效
MAX_CALIBRATION_ANGLE_GAP_DEG = 150.0    # 方向覆盖不能留下过大空档

# 启动检查
INITIAL_VISIBILITY_SEC = 1.0             # 短一点，避免车辆 TCP 空闲超时
INITIAL_VISIBILITY_RATIO = 0.5

# 记录与标定文件
LOG_ROOT = Path("relative_cage_runs/baseline")
CALIBRATION_FILE = Path("baseline_vicon_calibration.json")
CALIBRATION_TAG = "baseline-real-v3-fixed-4m-150mm"
# 每次运行都重新标定：既避免旧标定数据与当前 PWM 不匹配，也避免部分车辆在
# 别人标定的十几秒里空闲而被固件断开 TCP（空闲 2.5 秒即断开）。
RECALIBRATE_EVERY_RUN = True

SKIP_WIFI_SETUP = True                   # 电脑已经连好 YAYA
CAR_IP_OVERRIDES = ()                    # 可选：("3=192.168.33.115",)


def parse_args(argv=None):
    return runtime.parse_args(argv, defaults={
        "herders": ",".join(map(str, HERDER_IDS)),
        "evaders": ",".join(map(str, EVADER_IDS)),
        "arm": True, "vicon_host": VICON_SERVER,
        "robot_diameter_mm": ROBOT_DIAMETER_MM,
        "cage_width_ratio": CAGE_WIDTH_RATIO, "cage_height_ratio": CAGE_HEIGHT_RATIO,
        "skip_wifi": SKIP_WIFI_SETUP, "car_ip": list(CAR_IP_OVERRIDES),
        "log_root": LOG_ROOT,
        "calibration_file": CALIBRATION_FILE,
        "calibration_tag": CALIBRATION_TAG,
        "calibration_clearance_ratio": 0.01,
        "calibration_separation_ratio": 0.02,
        "allow_small_cage": False, "allow_partial_vicon": False,
        "isolate_car_failures": IGNORE_FAILED_CARS,
        "field_mode": "fixed_corners" if USE_FIXED_ARENA else "initial_car_positions",
        "field_width_mm": FIELD_WIDTH_MM, "field_height_mm": FIELD_HEIGHT_MM,
        "cage_opening_mm": CAGE_OPENING_MM,
        "recalibrate": RECALIBRATE_EVERY_RUN,
        "calibration_speed": CALIBRATION_SPEED,
        "calibration_move_sec": CALIBRATION_MOVE_SEC,
        "calibration_settle_sec": CALIBRATION_SETTLE_SEC,
        "calibration_samples": CALIBRATION_SAMPLES,
        "min_calibration_displacement_ratio": (
            MIN_CALIBRATION_DISPLACEMENT_MM / min(FIELD_WIDTH_MM, FIELD_HEIGHT_MM)
        ),
        "min_calibrated_commands": MIN_CALIBRATED_COMMANDS,
        "max_calibration_angle_gap_deg": MAX_CALIBRATION_ANGLE_GAP_DEG,
        "herder_min_pwm": HERDER_MIN_PWM, "herder_max_pwm": HERDER_MAX_PWM,
        "evader_min_pwm": EVADER_MIN_PWM, "evader_max_pwm": EVADER_MAX_PWM,
        "actual_speed_limit_factor": ACTUAL_SPEED_LIMIT_FACTOR,
        "initial_visibility_sec": INITIAL_VISIBILITY_SEC,
        "initial_visibility_ratio": INITIAL_VISIBILITY_RATIO,
    })


def calibrate_like_twelve_car(
    pending_ids,
    controllers,
    tracker,
    subjects_by_id,
    arena,
    args,
    logger,
):
    """六方向脉冲标定，逻辑与 twelve_car_3herder_chase.py 相同。

    每轮所有待标定车辆同时执行同一个方向；脉冲前后各取一次新鲜 Vicon 位姿，
    位移超过阈值的方向记为有效。与旧的 parallel 标定相比，这里不再检查车辆
    间距、虚拟场地边界和 yaw 变化，因此不会因为场地设置把正常车辆排除掉。
    返回 (每车方向标定, 被排除车辆及原因)。
    """
    active = [marker_id for marker_id in pending_ids if controllers[marker_id].active]
    results = {marker_id: {} for marker_id in pending_ids}
    excluded = {}
    sampling_missed = set()
    min_displacement = args.min_calibration_displacement_ratio * arena.scale_mm
    print(
        f"Six-direction calibration: {len(active)} cars, PWM {args.calibration_speed}, "
        f"{args.calibration_move_sec:.2f}s per direction, threshold {min_displacement:.0f}mm."
    )
    for marker_id in active:
        logger.event("calibration_started", "parallel_pulse", marker_id)

    def stop_cars(marker_ids):
        runtime.stop_all({
            marker_id: controllers[marker_id]
            for marker_id in marker_ids
            if marker_id in controllers and controllers[marker_id].active
        })

    def fresh_poses(marker_ids, timeout_sec=2.0):
        """每个车的第一帧新鲜位姿；等待期间继续发 STOP，防止 TCP 空闲断开。"""
        found = {}
        deadline = time.monotonic() + timeout_sec
        next_keepalive = time.monotonic() + 0.4
        while len(found) < len(marker_ids) and time.monotonic() < deadline:
            snapshot = tracker.get_snapshot()
            for marker_id in marker_ids:
                if marker_id in found:
                    continue
                subject_name = subjects_by_id[marker_id]
                if subject_name in snapshot.fresh_subjects and subject_name in snapshot.poses:
                    found[marker_id] = snapshot.poses[subject_name]
            now = time.monotonic()
            if now >= next_keepalive:
                stop_cars([marker_id for marker_id in marker_ids if marker_id not in found])
                next_keepalive = now + 0.4
            time.sleep(0.01)
        return found

    for command in runtime.MOTION_COMMANDS:
        active = [marker_id for marker_id in active if controllers[marker_id].active]
        if not active:
            break
        stop_cars(active)
        time.sleep(args.calibration_settle_sec)
        starts = fresh_poses(active)
        for marker_id in tuple(active):
            if marker_id not in starts:
                excluded.setdefault(marker_id, "calibration: no fresh Vicon start pose")
                controllers[marker_id].send_stop(force=True)
                controllers[marker_id].close(send_stop=False)
        active = [marker_id for marker_id in active if marker_id in starts]
        if not active:
            continue

        print(f"  {command}: moving {len(active)} cars")
        pulse_ends = dict(starts)
        failures = {}
        deadline = time.monotonic() + args.calibration_move_sec
        while time.monotonic() < deadline:
            now = time.monotonic()
            for marker_id in tuple(active):
                if not controllers[marker_id].set_motion(
                    command, args.calibration_speed, now, 0.10,
                ):
                    failures[marker_id] = (
                        controllers[marker_id].failure_reason or "TCP command failed"
                    )
                    active.remove(marker_id)
            snapshot = tracker.get_snapshot()
            for marker_id in active:
                subject_name = subjects_by_id[marker_id]
                pose = snapshot.poses.get(subject_name)
                if pose is not None and subject_name in snapshot.fresh_subjects:
                    pulse_ends[marker_id] = pose
            time.sleep(0.01)
        stop_cars(pending_ids)
        time.sleep(args.calibration_settle_sec)

        for marker_id, reason in failures.items():
            excluded.setdefault(marker_id, f"calibration pulse failed: {reason}")
            controllers[marker_id].close(send_stop=False)
        for marker_id in active:
            start = starts[marker_id]
            pulse_end = pulse_ends.get(marker_id)
            if pulse_end is None or pulse_end.seen_at <= start.seen_at:
                # One occluded frame should not drop a car that is otherwise fine;
                # it is excluded after the loop only if too few directions remain.
                sampling_missed.add(marker_id)
                continue
            offset = vec_sub(pulse_end.position, start.position)
            distance = vec_norm(offset)
            duration = max(pulse_end.seen_at - start.seen_at, 1e-6)
            if distance < min_displacement:
                print(
                    f"  car{marker_id} {command}: displacement {distance:.1f}mm "
                    f"< {min_displacement:.0f}mm"
                )
                logger.event(
                    "calibration_direction_rejected",
                    f"{command}: {distance:.1f}mm", marker_id,
                )
                continue
            results[marker_id][command] = runtime.MotionCalibration(
                direction_world=vec_normalize(offset),
                yaw=start.yaw,
                displacement_mm=distance,
                measured_speed_mm_s=distance / duration,
                calibration_pwm=args.calibration_speed,
                calibrated_at_unix=time.time(),
            )
            logger.event(
                "calibration_direction_accepted",
                f"{command}: {distance:.1f}mm", marker_id,
            )

    stop_cars(pending_ids)
    for marker_id in sampling_missed:
        if len(results[marker_id]) < args.min_calibrated_commands:
            excluded.setdefault(marker_id, "calibration: no Vicon pose during the pulse")
    speeds = [
        calibration.measured_speed_mm_s
        for commands in results.values()
        for calibration in commands.values()
    ]
    if speeds:
        print(
            f"Pulse speed at PWM {args.calibration_speed}: "
            f"median {median(speeds):.0f}mm/s "
            f"(range {min(speeds):.0f}..{max(speeds):.0f}mm/s); "
            "the collision guard scales this by the requested PWM."
        )
    return results, excluded


def build_arena_initializer():
    if USE_FIXED_ARENA:
        return FixedCornerArena(
            FIELD_CORNERS_WORLD, CAGE_CORNERS_WORLD, CAGE_OPENING_MM,
            ROBOT_DIAMETER_MM, BODY_GAP_MM, MIN_START_CLEARANCE_MM,
        )
    return InitialCarArena(
        FIELD_WIDTH_MM, FIELD_HEIGHT_MM, BODY_GAP_MM,
        CAGE_OPENING_MM, INITIAL_FIELD_INSET_MM,
    )


def main(argv=None):
    args = parse_args(argv)
    ids = runtime.parse_id_spec(args.herders) + runtime.parse_id_spec(args.evaders)
    if any(not 1 <= i <= 50 for i in ids):
        raise ValueError("This fleet uses robot IDs 1..50")
    if args.allow_partial_vicon or args.allow_small_cage:
        raise ValueError("Require complete active Vicon data and sufficient cage capacity")
    try:
        arena_initializer = build_arena_initializer()
    except (ArenaGeometryError, ValueError) as exc:
        print(f"Field/cage coordinates are invalid: {exc}")
        return 2
    if isinstance(arena_initializer, FixedCornerArena):
        # 标定阈值按真实场地尺寸换算，换成 3 m 或 5 m 场地不需要改别的地方。
        args.min_calibration_displacement_ratio = (
            MIN_CALIBRATION_DISPLACEMENT_MM / arena_initializer.arena.scale_mm
        )
        args.field_width_mm = arena_initializer.arena.width_mm
        args.field_height_mm = arena_initializer.arena.height_mm

    model = BaselineController(SIMULATION_TIME_SCALE, BODY_GAP_MM,
                               CAGE_OPENING_MM, WALL_REPULSION_MARGIN_MM)
    guard = PhysicalCommandFilter(
        BODY_GAP_MM, PREDICTION_HORIZON_SEC, REACTION_TIME_SEC,
        SPEED_UNCERTAINTY_FACTOR, JAM_TIMEOUT_SEC,
        herder_min_pwm=args.herder_min_pwm, evader_min_pwm=args.evader_min_pwm,
        isolate_failures=IGNORE_FAILED_CARS,
        cage_opening_mm=CAGE_OPENING_MM,
    )

    def parameters(arena, radius):
        args.calibration_separation_ratio = max(
            args.calibration_separation_ratio, (BODY_GAP_MM + 20) / arena.scale_mm)
        args.calibration_clearance_ratio = max(
            args.calibration_clearance_ratio, BODY_GAP_MM / (2 * arena.scale_mm))
        return physical_parameters(arena, radius, SIMULATION_TIME_SCALE,
                                   HERDER_MAX_ACCEL_MM_S2, EVADER_MAX_ACCEL_MM_S2)

    print("Baseline: acceleration-based variable speed; calibrated command collision checks.")
    if not args.parallel_calibration:
        print("Note: this entry always uses the twelve_car-style six-direction pulse calibration.")
    return runtime.main(args, control_step=model, decision_filter=guard,
                        params_factory=parameters, extra_obstacle_ids=tuple(range(1, 51)),
                        arena_initializer=arena_initializer,
                        calibrator=calibrate_like_twelve_car)


if __name__ == "__main__":
    raise SystemExit(main())
