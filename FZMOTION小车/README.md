# FZMOTION 小车位姿读取

这个项目通过 FZMOTION 官方支持的标准 **VRPN** 数据流读取刚体位置和姿态。它直接使用 Windows 自带网络接口和本机 Python，不依赖 Vicon DataStream SDK，也不需要安装 C++ 编译器或第三方 Python 包。

脚本读取的数据包括：

- 刚体/目标名称和传感器编号
- 三维位置 `X/Y/Z`
- 四元数 `qx/qy/qz/qw`
- 由四元数换算的 `Roll/Pitch/Yaw`
- FZMOTION/VRPN 时间戳和本机接收时间
- 可选 CSV 或 JSON Lines 输出

## 已确认的本机环境

- Windows 64 位
- Python 3.13.3 x64 已安装，命令为 `python`
- 当前 WLAN 是 `192.168.2.102/24`
- 配置文件已将扫描网段设为 `192.168.2.0/24`
- 已找到并实测 FZMOTION VRPN 服务 `192.168.2.3:3883`（VRPN 07.35）
- TCP 与 UDP 均已收到实时位姿；当前目标为 `pi[0]` 和 `Rigid[0]`
- 本机虽已安装 Vicon DataStream SDK 1.13，但本项目不会使用它；FZMOTION 并不是 Vicon 服务端

## 1. 在 FZMOTION 主机上开启输出

先在 Moca/Motar/FZMOTION 软件中完成以下操作。不同版本的菜单名称可能略有区别：

1. 给小车贴好 Marker，创建刚体并给它一个简单名称，例如 `Car`。
2. 启动实时捕捉，确认软件界面里刚体的位置和姿态持续更新。
3. 打开“数据流 / 实时输出 / VRPN”设置。
4. 启用 **VRPN** 和 **刚体（Rigid Body）** 输出。
5. 监听地址选择 FZMOTION 主机位于 `192.168.2.x` 的网卡；端口优先使用 VRPN 标准端口 `3883`。
6. 在 FZMOTION 主机的 Windows 防火墙中允许该动捕软件通过专用网络。

如果软件里完全没有 VRPN 选项，需要向 FZMOTION/凌云光索取与你当前 Moca/Motar 版本匹配的 Windows x64 SDK 及示例。不要用 Vicon DataStream SDK 代替 FZMOTION SDK，两者的协议并不相同。

## 2. 扫描 FZMOTION 数据服务

在本机 PowerShell 中运行：

```powershell
Set-Location "D:\FZMOTION小车"
.\run.ps1 scan
```

成功时会看到类似：

```text
192.168.2.3:3883    vrpn: ver. 07.35
```

把这个 IP 填入 [config.json](./config.json) 的 `host`。也可以保持为空；脚本每次启动时会自动扫描，且只发现一台服务时会自动使用它。

本机已经找到并连接 `192.168.2.3:3883`，因此配置文件已直接填好这个地址。当前服务端公布了两个目标：`pi[0]` 和 `Rigid[0]`。需要在 FZMOTION 画面中移动小车，观察哪一组坐标同步变化，再把对应名称填入 `object_name`；不要只凭名称猜测。

## 3. 找出小车对应的名称/编号

```powershell
.\run.ps1 discover --host 192.168.2.3 --transport tcp --duration-s 10
```

先使用 `tcp` 排除 UDP 防火墙问题。输出示例：

```text
2026-08-28T05:20:30.123+00:00 pi[0] pos_m=(2.227000, 0.119000, 0.385000) ...
```

这里的目标名称是 `pi`，传感器编号是 `0`。当前还会发现 `Rigid[0]`；移动小车并观察哪组坐标同步变化后再选择。如果所有刚体共用一个名称，则用不同的传感器编号区分它们。

## 4. 持续读取和保存位置

先把 `$TargetName` 设为移动小车时同步变化的那个目标（`pi` 或 `Rigid`），再在控制台查看：

```powershell
$TargetName = "pi"
.\run.ps1 listen --host 192.168.2.3 --object $TargetName --sensor 0
```

以毫米显示，并把每一帧写入 CSV：

```powershell
.\run.ps1 listen --host 192.168.2.3 --object $TargetName --sensor 0 --units mm --csv data\car_pose.csv
```

输出 JSON Lines，供另一个程序读取：

```powershell
.\run.ps1 listen --host 192.168.2.3 --object $TargetName --sensor 0 --format jsonl --print-rate-hz 0
```

`config.json` 里可以永久保存这些参数。设置好后，双击或直接运行：

```powershell
.\run.ps1
```

## 5. 在小车程序中直接调用

```python
import time

from fzmotion_client import VRPNClient

with VRPNClient("192.168.2.3", transport="udp") as client:
    last_car_pose = time.monotonic()
    while True:
        for pose in client.poll_poses(timeout_s=0.02):
            if pose.object_name != "pi" or pose.sensor_id != 0:  # Confirm pi or Rigid first.
                continue
            x, y, z = pose.position_m
            qx, qy, qz, qw = pose.quaternion_xyzw
            last_car_pose = time.monotonic()
            # 在这里把位姿送入小车的定位/控制算法。

        if time.monotonic() - last_car_pose > 0.1:
            raise RuntimeError("FZMOTION pose is stale; stop the car")
```

## TCP 与 UDP

- `tcp`：先用它验证连接，最容易排查防火墙问题。
- `udp`：连接验证后用于低延迟闭环控制；运行时加 `--transport udp`，并允许 Python 的专用网络 UDP 入站通信。

VRPN 建连始终需要 TCP 3883；选择 `udp` 只是把实时位姿帧切换到协商出的 UDP 端口。脚本会从实际 TCP 路由自动选择本机网卡，避免 VMware 虚拟网卡干扰，也可用 `--local-ip 192.168.2.102` 强制指定 WLAN。

## 坐标、单位和控制安全

标准 VRPN Tracker 的位置单位是米，四元数顺序是 `(x, y, z, w)`。FZMOTION 的坐标轴方向取决于场地标定与软件坐标系设置。正式控制小车前必须做三个静态检查：

1. 把刚体放在原点附近，核对零点。
2. 分别沿场地三个正轴移动，核对 `X/Y/Z` 的方向和比例。
3. 旋转小车，核对偏航角正方向。

`--units mm` 只改变显示和 CSV 的数值比例，不会改变 VRPN 原始数据。命令行默认在目标连续 `0.5 s` 无更新时退出；真正的闭环控制应根据帧率把 stale 阈值收紧到约 2 至 3 帧，并同时检测跳变和遮挡。不要在没有急停与速度限制的情况下直接把动捕位置用于高速控制。

## 常见问题

**扫描不到服务器**

确认 FZMOTION 主机 IP、VRPN 开关、TCP 端口和 Windows 防火墙。先在 FZMOTION 主机上运行 `ipconfig`，不要把相机 PoE 网卡地址误当成对外数据网卡地址。

**能连接但没有位姿**

确认实时捕捉正在运行、刚体已经创建且“刚体输出”被勾选。若使用 `udp`，先改成 `--transport tcp`；TCP 有数据而 UDP 没数据通常是防火墙问题。

**提示没有匹配目标**

先运行 `discover`，然后把输出中的名称和方括号内编号原样写入 `--object`、`--sensor`。名称区分大小写。

**需要更低抖动**

最终控制建议使用有线千兆网、`udp` 传输，并关闭高频控制台打印（`--format none`）。CSV 会保存每一帧，而默认控制台只显示每个目标每秒 10 次。

## 6. 测试 car13 的场地检测覆盖

确认 FZMOTION 中刚体名称为小写 `car13`，然后运行：

```powershell
Set-Location "D:\Vicon定位小车\FZMOTION小车"
python car13_coverage_heatmap.py
```

这一个脚本现在同时负责遥控和检测，不再需要另外启动
`keyboard_remote_control.py`。它会连接 Windows Wi-Fi `YAYA`、发现并校验 13 号小车，
同时在后台通过 TCP 读取 `192.168.2.3:3883` 的 FZMOTION 数据。第一帧到达后会自动锁定
`car13` 的 sensor 编号。

可视化窗口中，绿色 `DETECTED` 表示正在收到位置，红色 `LOST` 表示已经超过检测
阈值没有收到位置；窗口会同时绘制实时轨迹、当前位置、样本数和运行时间。键盘操作为：

- `Enter`：解锁遥控；
- `Q/W/E/A/S/D`：左前、前进、右前、左后、后退、右后；
- `↑/↓`：前进/后退；
- `Space`：立即停车并锁定；
- `[` / `]`：降低/提高速度；
- `Esc`：停车、结束采集并生成热力图。

为防止按键卡住，窗口失去焦点或最小化时也会立即停车并重新锁定。测试结束时按 `Esc`
或点击窗口中的 `FINISH`，等待控制台打印保存路径后再关闭终端。结果保存在
`coverage_runs\car13_时间戳\`：

- `coverage.png`：检测可用率热力图和轨迹；
- `poses.csv`：每一帧原始 X/Y/Z 位置；
- `gaps.csv`：失联前后的位置与持续时间；
- `grid.csv`：每个栅格的检测、疑似丢帧和状态统计；
- `metadata.json`：本次连接、坐标轴和阈值配置。

当前默认地面坐标轴为 `X/Z`。如果你的 FZMOTION 标定以 `Z` 为高度轴，应改用：

```powershell
python car13_coverage_heatmap.py --axes xy
```

如果已知场地边界，建议显式提供，避免未经过区域被自动裁掉。例如：

```powershell
python car13_coverage_heatmap.py --axes xz --bounds -3 3 -2 2
```

默认控制 13 号小车，速度为 150。需要临时选择其他车或速度时可使用：

```powershell
python car13_coverage_heatmap.py --car-id 13 --speed 150
```

如果电脑已经连接到正确 Wi-Fi，可加 `--skip-wifi`；如果 UDP 自动发现受防火墙影响，
可用 `--car-ip 小车IP` 直接连接。

灰色栅格表示未测试或证据不足，不能直接认定为盲区；红色虚线是根据失联前后两个位置
直线插值得到的疑似失联走廊。若要严格确认盲区，应让小车按已知蛇形路线遍历全场，
并对可疑区域从不同方向重复测试。

## 测试

项目测试会启动本地模拟 VRPN 服务端，不需要 FZMOTION 主机：

```powershell
python -m unittest discover -v
```

协议依据：

- [FZMOTION 官方产品页：支持 VRPN/ZMQ/TCP/UDP 和 ROS/Matlab/LabView](https://www.lusterinc.com/fzmotion-baidu/)
- [VRPN 官方协议格式](https://github.com/vrpn/vrpn/blob/master/Format_Of_Protocol.txt)
- [VRPN 官方 Tracker 数据约定（米、`x/y/z/w` 四元数）](https://github.com/vrpn/vrpn/blob/master/vrpn_Tracker.h)
- [VRPN 官方 Tracker 编码实现](https://github.com/vrpn/vrpn/blob/master/vrpn_Tracker.C)
