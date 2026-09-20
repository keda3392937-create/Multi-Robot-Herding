# 项目结构与实现分析

## 系统链路

```text
Vicon DataStream (192.168.30.100)
        |
        | 位置、yaw、四角刚体
        v
笔记本 relative_cage_capture.py
        |
        | UDP 4210 发现 + TCP 23 指令/PING
        v
YAYA 路由器 Wi-Fi -> ESP32 小车固件 -> 六方向底盘运动
```

Vicon 世界坐标只作为输入。新脚本用 `left-up`、`left-down`、`right-up`、
`right-down` 拟合局部正交坐标系，后续场地、笼区、距离阈值和控制都在毫米制局部坐标中完成。

## 文件分组

| 文件 | 作用与现状 |
| --- | --- |
| `baseline.py` | 纯仿真基线；包含约 40% 场地边长的左上笼区、三笼角增广凸包和 herder/evader 动力学。 |
| `boid.py` | 较早的 boid/predator 图形仿真，与当前实机入口解耦。 |
| `six_car_repel.py` | 通用 Wi-Fi 配置、UDP 发现、早期 TCP/Vicon 控制和标定工具。新入口只复用其 Wi-Fi 与发现函数；默认 Vicon 地址已统一为 `192.168.30.100`。 |
| `read_vicon.py` | 单刚体 Vicon 读取脚本，当前地址为 `192.168.30.100`。 |
| `read_vicon_inside_field.py` | 用四角范围筛选场内车辆的读取工具，使用旧的轴对齐边界思路。 |
| `axis_move_calibration.py` | 单车六方向/坐标轴运动标定。 |
| `move_car_to_vicon_targets.py` | 单车移动到 Vicon 目标点的工具。 |
| `random_car1_vicon_test.py` | car1 随机运动与 Vicon 反馈的硬件冒烟测试。 |
| `four_herder_*.py`、`six_herder_*.py` | 多次实机围捕迭代，包含固定角色、绝对/轴对齐场地、凸包、多个 evader 和 roaming 等阶段性方案。保留作实验对照，不再作为本需求入口。 |
| `six_car_avoid_herder.py`、`six_car6_herder_chase.py`、`sixteen_car_4herder_chase.py` | 较早的追逐、排斥和固定角色实机脚本。 |
| `firmware/carN_sta_repel/carN_sta_repel.ino` | 每车独立 `CAR_ID` 的 ESP32 固件；连接 `YAYA`，UDP 4210 宣告身份，TCP 23 接收 `SPD`、六方向和 `STOP`，带运动/连接看门狗，并输出 Wi-Fi 事件、IP、MAC、RSSI 和断线 reason。心跳现返回 `PONG id=<CAR_ID>`。 |
| `multi_robot_control.py` | 固定编号多车网络测试；每车一个长 TCP 连接和独立重连线程，单车故障不会停止其他车辆。 |
| `relative_cage_core.py` | 新的纯算法核心；不连接 Vicon、网络或电机，可离线测试。 |
| `relative_cage_capture.py` | 新的唯一实机入口；负责相对几何、车辆筛选、安全标定、凸包控制、动力学、TCP 指令和日志。 |
| `tests/test_relative_cage_core.py` | 新实现的离线回归测试，包括几何、动力学、凸包、缓存、协议和多步围捕。 |
| `docs/RELATIVE_CAGE_CAPTURE.md` | 新脚本的运行与安全说明。 |

`legacy_external_cage_draft/` 保存了需求更正前生成的“场外笼区”草稿，只供源码对照，不能运行。

## 旧实现的主要风险

- 多个脚本把场地或笼区写成固定世界坐标，场地平移/旋转后失效。
- 不同脚本重复实现 Vicon、标定、TCP 和控制，阈值及角色定义不一致。
- 早期实现多使用固定 PWM 或无物理单位的内部速度，不能表达毫米/秒和毫米/秒²限制。
- 仅 TCP connect/send 不能证明连接的是正确车辆，也不能发现通信正常但电机不动。
- 固定车辆列表会让未联网车辆继续污染凸包和捕获分母。
- 部分旧控制只绘制或记录凸包，凸包状态没有进入控制闭环。

## 新入口的边界

新脚本不会修改 Vicon 系统配置，也不会自动判断物理场地是否已经清空。无 `--arm` 时只做
Wi-Fi/Vicon 和几何预检；带 `--arm` 后才连接车辆、执行低速标定并发送电机指令。实机首次运行仍需人工急停条件和场边监护。
