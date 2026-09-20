# 相对坐标围捕脚本

主入口为 `relative_cage_capture.py`，纯算法模块为 `relative_cage_core.py`。

## 几何定义

脚本启动后，从 Vicon 连续读取以下四个刚体并取中位数：

- `left-up`
- `left-down`
- `right-up`
- `right-down`

四点用于拟合一个正交局部坐标系，单位仍为毫米：

```text
left-up=(0,H)  +------- cage -------+----------------+ right-up=(W,H)
                 |                  |
                 |                  + cage right-down
                 |                                      |
                 |             virtual field            |
                 |                                      |
left-down=(0,0) +--------------------------------------+ right-down=(W,0)
```

笼子位于场地内部左上角：

```text
x = [0, cage_width_ratio * W]
y = [H - cage_height_ratio * H, H]
```

默认宽、高比例均为 `0.40`，与 `baseline.py` 的笼区占比一致；仍会结合实际场地尺寸、车体直径和有效 evader 数量做容量检查。因此脚本不依赖 Vicon 世界坐标中的固定数值，场地平移或旋转后仍使用相同的相对几何。

四角名称、对边长度、夹角或矩形拟合误差不合理时，脚本拒绝启动。场地几何在启动时冻结；运行中任一角标丢失会暂停全部车辆，角标移动超过阈值会结束实验。

## 安全预检

不带 `--arm` 运行时，只连接 Wi-Fi/Vicon、读取四角并打印场地和笼区，不连接小车 TCP，也不会发送电机指令：

```powershell
python relative_cage_capture.py
```

先核对输出中的：

- 四个角标的世界坐标和局部坐标；
- 场地宽度、高度、夹角及拟合误差；
- 笼子左上角和右下角的世界坐标；
- 按小车直径估算的笼区容量。

## 正式运行

默认角色是 1-6 号 herder、7-16 号 evader：

```powershell
python relative_cage_capture.py --arm
```

指定其他角色：

```powershell
python relative_cage_capture.py --arm --herders 1-4 --evaders 5-6
```

如果 Windows 已保存 `YAYA` 配置，不需要在命令行提供密码。否则可在当前 PowerShell 会话设置：

```powershell
$env:VICON_CAR_WIFI_PASSWORD = "<Wi-Fi password>"
python relative_cage_capture.py --arm
```

Vicon 默认地址为 `192.168.30.100`，可以通过 `--vicon-host` 覆盖。小车 IP 默认通过 UDP 4210 自动发现，也可以使用多个 `--car-ip ID=IP` 指定。50 份固件源码的心跳均已升级为 `PONG id=<CAR_ID>`；覆盖 IP 必须由该响应或本轮 UDP 发现证明身份，避免把错误车辆当成目标 ID 控制。尚未重新烧录的旧固件仍可在自动发现成功时使用，但不能在跳过发现后仅靠裸 `PONG` 使用覆盖 IP。

## 车辆参与规则

一辆车只有依次满足以下条件，才会进入本次实验：

1. UDP 发现成功或提供了 IP；
2. 首次 TCP 连接成功，并通过 `PONG id=<CAR_ID>` 或已核验的旧固件 `PONG` 确认应用层存活和身份；
3. 启动采样期间 Vicon 可见率达标；
4. 需要控制时，至少四个运动方向标定有效且角度覆盖合格。

TCP 连接采用并行预检，车辆 Vicon subject 和 IP 都不允许重复；自动发现结果与 `--car-ip` 冲突时按身份错误剔除。失败车辆会记录原因并从角色、凸包和捕获分母中剔除。它若仍有 Vicon 坐标，会作为静态障碍物参与避障。少于 3 个有效 herder 或没有有效 evader 时，脚本拒绝运行。

运行中以公平错峰的 `PING/PONG` 检查连接；首次心跳或 TCP 写入失败后，该车在本次实验中永久退出，不进行阻塞式重连。非零 PWM 持续默认 2 秒但 Vicon 速度仍低于默认 15 mm/s，也视为电机/驱动无响应并剔除。角色集合变化时先让其余车辆停车、清空动力学状态，再按新集合计算凸包。herder 数量降到 3 以下时全部停车并结束。

## 标定

第一次运行或使用 `--recalibrate` 时，车辆逐台、低速执行六方向标定：

```powershell
python relative_cage_capture.py --arm --recalibrate
```

标定前和每个运动脉冲期间都会逐帧检查四角、目标车 Vicon、实际位移、场地边界和其他车辆距离；任一条件异常会立即先发送 `STOP`，再把 PWM 清零。位移过小、位移异常、yaw 变化过大或方向覆盖不足的车辆会被剔除。

结果保存在 `relative_cage_calibration.json`。缓存默认最多复用 24 小时，并绑定 `--calibration-tag`；Vicon 标记安装、电机、轮组或底盘发生变化后应更换 tag 或使用 `--recalibrate`。缓存中的异常方向会在使用前过滤。

## 控制与捕获

- herder 使用“笼子三个固定顶点 + 在线 herder”的增广凸包；凸包相邻角间隙直接参与控制，并优先围捕尚未进入增广凸包、且离笼中心最远的 evader；
- 同一轮的所有 herder 共享一个目标，在目标背离笼区的一侧形成弧形队列；该目标进入笼区后再处理下一个，避免多目标分散时队形互相冲突；
- evader 使用有界的分散、聚合、避开 herder、随机游走和场地边界力；
- herder 与 evader 的模型采用毫米、毫米/秒和毫米/秒²；每帧由 Vicon 位移估算实测速度，再经过最大加速度、最大速度和按秒计算的 PWM 变化率限制；
- 连续目标速度结合该车标定得到的实测速率映射到 PWM，不再始终使用一个固定速度；
- Vicon 原始帧间速度用于检测超速或标记跳变，低通速度只用于模型反馈；车体中心进入场地边界的底盘半径安全带时立即全车停车；
- 所有有效 evader 必须位于内缩后的笼区并持续默认 2 秒，才确认捕获并停车；
- 默认任一参与车辆 Vicon 丢失时全部暂停。只有明确使用 `--allow-partial-vicon` 才允许其余车辆继续。

`--passive-evaders` 可只跟踪 evader 而不给它们发送运动指令；此模式下 evader 不要求连接 Wi-Fi/TCP，只要 Vicon 可见就能参与围捕与捕获判定。正常方向切换会先沿原方向逐步把 PWM 降到零，再从零沿新方向升速；安全停车仍立即发送 `STOP`。

单次控制默认最多运行 300 秒，超时会停车并记入事件日志，可用 `--max-runtime-sec` 调整。任何暂停都会清空模型速度并把固件 PWM 置零，恢复后重新按 slew 限制起步。

## 日志和测试

每次正式实验默认在 `relative_cage_runs/<时间>/` 生成：

- `metadata.json`：参数、四角、相对变换、角色筛选和标定；
- `states.csv`：每个 Vicon 帧中每辆车的位置、Vicon 实测速度、模型状态、目标、指令和 PWM；
- `events.csv`：标定、剔除、暂停、恢复、掉线、超时和捕获事件。

离线测试命令：

```powershell
python -m pytest -q -p no:cacheprovider
```

测试不连接 Vicon、Wi-Fi 或小车，其中包含旋转场地、凸包、动力学、缓存、TCP 身份、故障筛选和代表性单/多 evader 多步闭环围捕场景。项目文件关系和旧实现风险见 `PROJECT_ANALYSIS.md`。`legacy_external_cage_draft/` 仅保留先前错误理解的场外笼区草稿，不是运行入口。
