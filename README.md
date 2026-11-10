# Vicon 多小车定位与控制

## 当前配置

| 项目 | 配置 |
| --- | --- |
| 小车 Wi-Fi | `YAYA`，密码 `kedayaya`；1～15 号中除 3、6、7 外使用 2.4 GHz；3、6、7 及 16～50 号 ESP32-C5 仅连接 5 GHz |
| 实验网络 | `192.168.33.x`；小车通过 DHCP 获取地址 |
| Vicon Tracker 主机 | `192.168.33.102` |
| 小车与刚体对应 | 固件 `CAR_ID = N` 对应 Tracker 中的 `kedayaN`，支持 1–50 |
| 小车通信 | UDP `4210` 自动发现，TCP `23` 控制与身份心跳 |

电脑需要同时能访问小车和 Vicon 主机。通常连接 YAYA 后应获得 `192.168.33.x` 地址；
若使用常见的 `255.255.255.0` 掩码，广播地址为 `192.168.33.255`。
路由器需允许两个频段的无线客户端互访；不要给电脑或小车使用 Vicon 已占用的 `.102`。

## 1. 烧录小车

固件统一放在 `firmware/carN_sta_repel/carN_sta_repel.ino`，每辆车一个 Arduino 工程。
例如 1 号车打开 `firmware/car1_sta_repel/car1_sta_repel.ino`。

1. USB 接入对应的小车，Arduino IDE 选择实际 ESP32 开发板和串口。
2. 1、2、4、5、8、9、10、11、12、13、14、15 号使用原 ESP32 和原接线；3、6、7 及 16～50 号使用 ESP32-C5-WIFI6-KIT-N16R8-M，按 [C5 接线与烧录说明](docs/ESP32_C5_WIRING.md) 接线。两种硬件不能混烧固件。
3. 上传对应编号的固件，避免给多辆车烧入相同 ID。
4. 串口监视器设为 `115200`，查看 `kedayaN IP: 192.168.33.xxx`。
5. Tracker 中相应车辆刚体命名为 `kedayaN`，大小写保持一致。

修改源码不会自动更新已经在小车上的程序，需要逐车重新烧录。

3、6、7 及 16～50 号固件在连接和重连前强制 `WIFI_BAND_MODE_5G_ONLY`，不会回落到同名的 2.4 GHz 网络。
串口连接日志会显示 `channel=... band=5GHz`；电脑和 Vicon 主机仍使用 `192.168.33.x`，
路由器需桥接两个频段并允许客户端互访。详细接线表及路由器自动信道说明见上面的 C5 文档。

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
python keyboard_remote_control.py --skip-wifi --car-id 2 --car-ip 192.168.33.120
# 只检查小车通信，不启动 Vicon 读取
python keyboard_remote_control.py --no-vicon
```

Vicon 主机默认 `192.168.33.102`，可用 `--vicon-host` 覆盖。需要在 Tracker 开启
DataStream 输出。本机 SDK 默认目录为 `D:\ViconDataStream\Win64\Python\vicon_dssdk`；
换电脑后安装与 Python/系统位数兼容的 Vicon DataStream SDK，必要时通过
`--vicon-sdk-path` 指向包含 `vicon_dssdk` 包的目录。遥控本身使用 Python 标准库及 Tkinter。

单车随机运动测试使用 `random_car1_vicon_test.py`。将文件顶部 `CAR_ID` 改成待测编号后
直接运行，支持 1～50 号，Vicon 刚体默认使用对应的 `kedayaN`。也可用
`python random_car1_vicon_test.py --car-id 20` 临时选车，文件名保留原名。
该测试的 Vicon 主机默认是 `192.168.33.102`。

## 3. 驱赶与围捕

运行实验前关闭遥控窗口，释放车辆 TCP 控制连接。

`baseline_vicon_herding.py` 是从桌面上的 `baseline.py` 迁移到实机的入口。
默认使用 **18～25 号作为 8 辆 herder、1～17 号作为 17 辆 evader**，共 25 辆；在文件顶部修改两组编号。
场地和笼子改为**写死的 Vicon 世界坐标**：在文件顶部 `FIELD_CORNERS_WORLD`、`CAGE_CORNERS_WORLD`
各填四个点（mm，顺序随意），脚本自动校验矩形、判断笼子贴的是哪个场地角，并把笼子作为局部坐标的
左上角。文件里的示例坐标是 **4 m × 4 m** 场地、**1.6 m × 1.6 m** 笼子；笼底边和右边各留
**700 mm 开口**，允许 evader 入笼。不需要四角标记，也不再用初始车位推算场地。
车体直径按 **150 mm** 配置，按实测尺寸修改 `ROBOT_DIAMETER_MM`。
连接 YAYA 后，在 IDE 中直接运行该 Python 文件即可；命令行等价操作为：

```powershell
python baseline_vicon_herding.py
# 只核对坐标：连接 Vicon，打印场地/笼子/每车局部坐标，不连接小车、不标定
python baseline_vicon_herding.py --dry-run
```

启动流程与 `twelve_car_3herder_chase.py` 相同：先做六方向并行短脉冲标定
（255 PWM、0.45 秒/方向，位移 ≥ 30 mm 才有效），6 个方向有效、方向覆盖合格的车辆才进入控制；
标定阶段不检查车距、场地边界和 yaw 变化，因此不会像旧版那样一开始就大量排除车辆。
每次运行都会重新标定，避免旧缓存与新 PWM 不匹配，也避免部分车辆在等待期间因固件 TCP 空闲
2.5 秒超时被断开。
若只想查看坐标而不连接小车，运行参数中加入 `--dry-run`。
原仿真保存在 `simulation/baseline.py`；凸包、视野、密度和追逐力公式提取至
`baseline_forces.py`，实机缩放和避碰位于 `baseline_physical.py`，场地/笼子几何和虚拟墙位于 `baseline_virtual_arena.py`。
速度通过加速度积分产生，并依六方向标定转换为各车 PWM；发送前检查实际方向、
车体尺寸、滑行和预测路径。受困边界的 evader 附近 herder 会尝试后退让位，
没有可行指令则停车。默认 `IGNORE_FAILED_CARS=True`：未发现、连接失败、标定不合格、
持续无运动响应或运行中断连的车单独退出，其余车辆继续，凸包和捕获数量按剩余车辆更新。
故障车仍按障碍处理；定位丢失时保留最后位置和额外避让范围。至少需要 3 辆有效 herder
和 1 辆有效 evader。算法检查不能替代实机验证。
具体设置、仿真差异和运行步骤见 [实机围捕说明](docs/BASELINE_REAL_ROBOTS.md)。

公共 `six_car_repel.py` 的 `CAR_IDS` 和 `CAR_CONFIGS` 已覆盖 1～50 号，
单车标定、移动到目标点及依赖它的旧实验脚本均可使用 50 号以内的编号。
各实验的角色分组、固定选车列表仍由该文件顶部或 `--herders`、`--evaders` 等参数指定；
支持 50 辆不代表所有特定实验都默认同时运行 50 辆。

`twelve_car_3herder_chase.py`：在顶部 `HERDER_IDS`、`EVADER_IDS` 设置两组编号，支持 1～50 号。
当前 herder 为 17、18、19、20 号，evader 为 1、2、4、5、8、9、10、11、12、13、14、15、16 号。
电脑连接 YAYA 后，直接在 IDE 中运行该 Python 文件即可；Vicon 主机默认
`192.168.33.102`，刚体名称为对应编号的 `kedayaN`，发现及手动 IP 校验共用
`multi_robot_control.py` 中的 `NETWORK`（当前为 `192.168.33.0/24`）。

脚本只发现这些编号，先同时进行六方向运动标定，再进入驱赶：herder 追逐最近的
可控制 evader，evader 远离附近 herder；两组各自保留同类排斥，减少聚集。
未发现、连接失败或标定不通过的车辆会跳过；运行时每辆车独立发送与重连，
该车丢失定位时停车，其他车辆继续。按 `Ctrl+C` 停止全部车辆。

```powershell
python twelve_car_3herder_chase.py
# 两组分别设置运行 PWM；不改变六方向标定的默认 PWM
python twelve_car_3herder_chase.py --herder-speed 220 --evader-speed 180
```

默认运行及标定 PWM 均为 255，沿用之前驱赶脚本的设置。evader 超出 herder 的
1300 mm 响应范围、且没有同类靠近时会停下；这是驱赶规则，并非断连。

## 4. 固定编号多车通信测试

`multi_robot_control.py` 默认测试全部 1～50 号，车辆编号由文件顶部 `DEFAULT_IDS` 或 `--ids` 指定。当前 YAYA 配置为
`192.168.33.0/24`，脚本只向 `192.168.33.255` 发现这些编号，建立每车一个 TCP 长连接，
并为每辆车运行独立的发送/重连线程。网卡筛选、广播和报错信息共用顶部 `NETWORK` 设置；
以后更换网段时只需修改这一处。当前 Vicon 主机是 `192.168.33.102`，此测试脚本不连接 Vicon。
某辆车没有上线、TCP 连接失败或运行中断线时，只记录该车状态，其他车辆继续接收命令。
ESP32 的 700 ms 无命令看门狗仍会让掉线车辆停车。

```powershell
python multi_robot_control.py
# 降低速度，运行指定时间和方向
python multi_robot_control.py --speed 100 --duration 60 --command F
# 现场只测试指定的几辆车
python multi_robot_control.py --ids 2,3,7,10 --bind-ip 192.168.33.112
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
| `twelve_car_3herder_chase.py` | 可配置 herder、evader 编号的 Vicon 驱赶实验，文件名保留原名 |
| `vicon_monitor.py` | 独立进程中的 Vicon 监视 |
| `firmware/` | 50 辆车的独立烧录工程 |
| `tests/` | 固件配置、遥控、Vicon 与围捕回归测试 |
| `docs/` | 围捕操作与历史分析 |
| `relative_cage_runs/` | 已有实验记录 |
| `legacy_external_cage_draft/` | 已有历史草稿 |
| `FZMOTION小车/` | 另一套动捕系统的实验，不属于 Vicon DataStream 链路 |

离线测试：`python -m pytest`。测试使用模拟通信和窗口，不会遥控真实小车。
