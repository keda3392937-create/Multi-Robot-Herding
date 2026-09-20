# Vicon 多小车定位与控制

## 当前配置

| 项目 | 配置 |
| --- | --- |
| 小车 Wi-Fi | `YAYA`，密码 `kedayaya`，使用 2.4 GHz |
| 实验网络 | `192.168.30.x`；小车通过 DHCP 获取地址 |
| Vicon Tracker 主机 | `192.168.30.100` |
| 小车与刚体对应 | 固件 `CAR_ID = N` 对应 Tracker 中的 `kedayaN`，支持 1–50 |
| 小车通信 | UDP `4210` 自动发现，TCP `23` 控制与身份心跳 |

电脑需要同时能访问小车和 Vicon 主机。通常连接 YAYA 后应获得 `192.168.30.x` 地址；
若使用常见的 `255.255.255.0` 掩码，广播地址为 `192.168.30.255`。
路由器需允许无线客户端互访；不要给电脑或小车使用已占用的 `.100`。

## 1. 烧录小车

固件统一放在 `firmware/carN_sta_repel/carN_sta_repel.ino`，每辆车一个 Arduino 工程。
例如 1 号车打开 `firmware/car1_sta_repel/car1_sta_repel.ino`。

1. USB 接入对应的小车，Arduino IDE 选择实际 ESP32 开发板和串口。
2. 使用支持 `ledcAttach(pin, frequency, resolution)` 的 Arduino-ESP32 3.x 开发板包；保留原底盘接线。
3. 上传对应编号的固件，避免给多辆车烧入相同 ID。
4. 串口监视器设为 `115200`，查看 `kedayaN IP: 192.168.30.xxx`。
5. Tracker 中相应车辆刚体命名为 `kedayaN`，大小写保持一致。

修改源码不会自动更新已经在小车上的程序，需要逐车重新烧录。

## 2. 键盘遥控与连接检查

先在 Windows 中连接 YAYA，再直接运行 `keyboard_remote_control.py`（也可在 IDE 中运行此文件）。
脚本默认沿用当前网络，不会添加、覆盖或切换 Wi-Fi 配置。在项目根目录运行：

```powershell
python keyboard_remote_control.py
```

窗口中选择 `kedaya1`、`kedaya2` 等，再点 `Connect`。程序会发现车辆并校验编号。
只有显式添加 `--connect-wifi` 时才尝试自动连接 YAYA；已有连接或配置会优先复用。
显示 `CAR: VERIFIED` 后，点击 `ARM` 或按回车启用控制。按住方向键移动，松开停车。

兼容两种固件回复：新版 `PONG id=N` 直接校验编号；旧版仅回复 `PONG` 时，
必须由本轮 UDP 发现确认该 IP、端口对应所选编号，连接后心跳也兼容旧格式。
仅手动指定 IP、未经过 UDP 发现的连接仍需新版带编号回复。

| 按键 | 底盘动作 |
| --- | --- |
| `W` / 上箭头 | 前进 |
| `S` / 下箭头 | 后退 |
| `Q` / `E` | 左前 / 右前 |
| `A` / `D` | 左后 / 右后 |
| `[` / `]` | PWM 减少 / 增加 10；也可拖动滑条 |
| 空格 / `STOP` | 停车并解除控制，需要再次 ARM |
| `Esc` | 停车并退出 |

这是三轮底盘原有的六方向平移指令，方向相对于车身，不是 Vicon 世界坐标。
切换车辆、窗口失焦或最小化均停车并解除控制；切换后重新点 `Connect` 和 `ARM`。
检测到心跳失联会停止发送运动并断开连接；固件自身保留约 700 ms 无运动指令停车机制。

Vicon 区域会显示所选 `kedayaN` 的 X/Y/Z（毫米）、Yaw（度）、帧号和数据年龄：

- `TRACKED`：收到该刚体的新鲜、无遮挡定位。移动该车时应看到相应坐标变化。
- `SUBJECT NOT FOUND`：有数据帧，但没有完全匹配的刚体名称。
- `OCCLUDED`：刚体被遮挡或位姿不可用。
- `STALE`：超过 0.5 秒没有新帧，旧坐标不继续冒充实时值。
- `VICON ERROR` / `SDK ERROR`：检查主机数据流、网络、防火墙或 SDK 安装。

遥控可独立于 Vicon 工作，因此 Vicon 异常不会禁止人工遥控。必须同时看到车辆响应和
`TRACKED` 坐标随该车变化，才能完成两条链路及刚体对应关系的人工检查。
车辆心跳只能证明固件通信，不能单独证明电机或 Tracker 标签安装正确。

```powershell
# 指定初始车辆及速度
python keyboard_remote_control.py --car-id 2 --speed 100
# 电脑已经连好网络，保持当前网卡连接
python keyboard_remote_control.py --skip-wifi
# 广播被网络配置阻挡时，手动指定车辆地址，仍会校验 CAR_ID
python keyboard_remote_control.py --skip-wifi --car-id 2 --car-ip 192.168.30.102
# 只检查小车通信，不启动 Vicon 读取
python keyboard_remote_control.py --no-vicon
```

Vicon 主机默认 `192.168.30.100`，可用 `--vicon-host` 覆盖。需要在 Tracker 开启
DataStream 输出。本机 SDK 默认目录为 `D:\ViconDataStream\Win64\Python\vicon_dssdk`；
换电脑后安装与 Python/系统位数兼容的 Vicon DataStream SDK，必要时通过
`--vicon-sdk-path` 指向包含 `vicon_dssdk` 包的目录。遥控本身使用 Python 标准库及 Tkinter。

## 3. 驱赶与围捕

运行实验前关闭遥控窗口，释放车辆 TCP 控制连接。

## 4. 固定编号多车通信测试

`multi_robot_control.py` 默认控制 `kedaya2,3,4,5,7,8,9,10,12,13`。它只向 YAYA 的
`192.168.30.255` 发现这些编号，建立每车一个 TCP 长连接，并为每辆车运行独立的发送/重连线程。
某辆车没有上线、TCP 连接失败或运行中断线时，只记录该车状态，其他车辆继续接收命令。
ESP32 的 700 ms 无命令看门狗仍会让掉线车辆停车。

```powershell
python multi_robot_control.py
# 降低速度，运行指定时间和方向
python multi_robot_control.py --speed 100 --duration 60 --command F
# 现场只测试指定的几辆车
python multi_robot_control.py --ids 2,3,7,10 --bind-ip 192.168.30.112
```

终端每秒显示每辆车的 `CONNECTED`、`RUNNING`、`CONNECT_FAILED` 或 `SEND_FAILED`，并附带异常类型、
最后发送时间和重连次数。脚本不会连接 Vicon；它用于先验证多车 Wi-Fi/TCP 与电机控制链路。Vicon 读取仍可
同时由 `read_vicon.py` 或其他实验入口运行。

固件串口监视器设为 `115200`。新版固件会输出 `[WiFi] connected`、IP、MAC、RSSI、
`[WiFi] disconnected reason=<code>`，并每 5 秒输出 RSSI；`[TCP]`/车辆控制日志用于区分无线断线和应用层连接问题。

- `six_car_repel.py`：驱赶及早期实机通信、标定工具。
- `relative_cage_capture.py`：相对场地坐标围捕入口，算法核心为 `relative_cage_core.py`。
- 参数与预检流程见 `docs/RELATIVE_CAGE_CAPTURE.md`，项目历史见 `docs/PROJECT_ANALYSIS.md`。
- 其他 `four_herder_*`、`six_herder_*`、`baseline*` 等保留原路径，维持已有导入和实验命令。

## 目录

| 目录 / 文件 | 用途 |
| --- | --- |
| `keyboard_remote_control.py` | 日常选车遥控入口，直接运行 Python 文件 |
| `multi_robot_control.py` | 固定编号多车长连接测试，单车故障隔离 |
| `vicon_monitor.py` | 独立进程中的 Vicon 监视 |
| `firmware/` | 50 辆车的独立烧录工程 |
| `tests/` | 固件配置、遥控、Vicon 与围捕回归测试 |
| `docs/` | 围捕操作与历史分析 |
| `relative_cage_runs/` | 已有实验记录 |
| `legacy_external_cage_draft/` | 已有历史草稿 |
| `FZMOTION小车/` | 另一套动捕系统的实验，不属于 Vicon DataStream 链路 |

离线测试：`python -m pytest`。测试使用模拟通信和窗口，不会遥控真实小车。
