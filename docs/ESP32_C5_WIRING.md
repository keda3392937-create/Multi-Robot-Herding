# ESP32-C5 小车接线与 5 GHz 固件

适用开发板：Waveshare **ESP32-C5-WIFI6-KIT-N16R8-M**，16 MB Flash、8 MB PSRAM、已焊排针。
适用编号：**3、6、7，以及 16～50**，共 38 辆。
其余 12 辆（1、2、4、5、8、9、10、11、12、13、14、15）使用原 ESP32 接线及 2.4 GHz 固件；
根目录 `base_move.ino` 也仍是旧 ESP32 接线。3、6、7 号改用 C5 后须按下表接线。

每辆车继续使用两个 TB6612FNG 驱动板、三个电机，原来的六方向运动算法不变。
以下编号是开发板丝印对应的 **GPIO 编号，不是排针从上往下的序号**。

## 1. 两个 TB6612 的控制线

把驱动两路电机的板叫“驱动板 A”，驱动单路电机的板叫“驱动板 B”。
M1、M2、M3 指原车固件对应的同一个电机，保持原电机位置和两根输出线的极性不变。

| 驱动板 | TB6612 引脚 | 新 C5 GPIO | 原 ESP32 GPIO | 用途 |
| --- | --- | --- | --- | --- |
| A | AIN1 | **0** | 27 | M1 方向 1 |
| A | AIN2 | **1** | 26 | M1 方向 2 |
| A | PWMA | **3** | 25 | M1 速度 PWM |
| A | BIN1 | **4** | 33 | M2 方向 1 |
| A | BIN2 | **5** | 32 | M2 方向 2 |
| A | PWMB | **8** | 23 | M2 速度 PWM |
| A | STBY | **23** | 14 | 驱动板 A 独立使能 |
| B | AIN1 | **9** | 18 | M3 方向 1 |
| B | AIN2 | **10** | 19 | M3 方向 2 |
| B | PWMA | **24** | 5 | M3 速度 PWM |
| B | STBY | **25** | 13 | 驱动板 B 独立使能 |

两个 STBY 分开接：**A 接 GPIO23，B 接 GPIO25**，不需要分叉线，也不要将两个 STBY 短接。
此前共用 GPIO23 的方案已替换；C5 基础自检和全部 38 份 C5 联网固件均使用本表。
**STBY 不要同时接到 3.3 V 或 5 V**；
若驱动板有将 STBY 固定拉高的跳线，先断开该连接，再接 GPIO23。
TB6612 的 STBY 输入自带下拉；若需外加下拉，可在每个 STBY 与 GND 之间各加一只 10 kΩ 电阻。

输出端：A 的 AO1/AO2 接 M1，A 的 BO1/BO2 接 M2，B 的 AO1/AO2 接 M3。
B 未使用的 BIN1、BIN2、PWMB 接 GND，BO1/BO2 留空。部分板标为 A01/A02、B01/B02，含义相同。
如果原车第三路接在另一块驱动板的 B 通道，应改接这里指定的 A 通道。

## 2. 电源与共地

| 端点 | 接到哪里 |
| --- | --- |
| 两块 TB6612 的 VCC（逻辑电源） | C5 的 **3V3** |
| 两块 TB6612 的 VM（电机电源） | 原有适合电机和驱动板额定电压的电源正极 |
| C5 的 5V | 稳定的 **5 V 稳压电源**，或使用 USB 供电 |
| C5 的 G/GND、两块驱动板的 GND、电机电源负极 | **全部共地** |

VM 与 VCC 是两个不同的电源接口。电机不要由 C5 的 3V3 引脚供电；多节电池电压也不要直接送入 C5 的 5V/3V3 引脚。
TB6612 的 VCC 使用 3.3 V，使方向、PWM、STBY 与 C5 的 3.3 V GPIO 电平匹配。
电机启动时若导致 ESP32 反复重启，先检查 5 V 稳压和共地；更换频段不能解决电源压降。

本接线特意避开 GPIO6（板载电池电压检测）、GPIO15（N16R8 的 PSRAM）、
GPIO11/12（UART）、GPIO13/14（原生 USB）、GPIO27（RGB）及 GPIO26/28（启动模式相关）。
GPIO3 和 GPIO25 在复位时兼有 SDIO 时钟边沿配置功能；本程序不使用 SDIO 下载/通信，
使用正常 SPI Flash 启动及 UART/USB 烧录，因此启动后将它们作为 PWM/数字输出使用。
GPIO0/1 在此用于普通输出；厂商原理图中的外置 32.768 kHz 晶振是未安装选件，本程序不启用它。

## 3. 对应程序

只检查新板、驱动板和电机时，打开 `firmware/biedong_c5/biedong_c5.ino`，按上表接线后烧录。
这是不使用 Wi-Fi/Vicon 的基础运动自检，所有 C5 车可共用，不需要车辆编号。
上电后自动循环：前进 → 后退 → 左前 → 右后 → 左后 → 右前，每个动作 2 秒；
前三个动作后停车 1 秒，后三个动作后停车 2 秒，PWM 保持原 `biedong` 的 150。
可修改 `speedVal` 调速。原来的 `firmware/biedong/biedong.ino` 继续供旧 ESP32 使用。
测试结束后，要恢复电脑遥控，需要重新烧录下面对应车辆编号的联网固件。

每辆车打开它自己的 Arduino 工程，例如：

- 3、6、7 号：各自的 `firmware/carN_sta_repel/carN_sta_repel.ino`
- 16 号：`firmware/car16_sta_repel/car16_sta_repel.ino`
- 17 号：`firmware/car17_sta_repel/car17_sta_repel.ino`
- 依此类推，现有工程一直到 50 号。

工程内的 `CAR_ID`、`CAR_NAME` 已分别设置为对应编号，不必逐次手动改编号。
以后新增 51 号及之后的车，从一个 C5 工程复制，修改工程目录/文件名及 `CAR_ID`、`CAR_NAME`，
并同步扩大电脑端当前 1～50 的编号范围。
不要从 1、2、4、5、8～15 号工程复制 C5 固件：这些工程使用旧 GPIO 和 2.4 GHz。

IDE 选择对应 ESP32-C5 目标（通用目标名为 `ESP32C5 Dev Module`）。
程序只允许 C5 编译，误选旧 ESP32 时会明确报错，防止把新接线程序烧入旧车。
串口波特率为 **115200**；用板上的 **UART USB-C** 接口查看串口日志最直接。

固件保留 TCP 23、UDP 4210、`PONG id=N`、`SPD` 和 F/B/LF/RF/LB/RB/停车指令，
因此现有电脑端 Python 脚本可继续使用。默认 PWM 为 150，电脑端仍可发送 `SPD` 调速。
700 ms 无运动指令停车机制继续保留；三个 PWM 初始化失败时，两个驱动板保持待机。

## 4. YAYA：只连接 5 GHz

- SSID：`YAYA`；密码：`kedayaya`。
- 首次连接及显式重连前设置并确认 `WIFI_BAND_MODE_5G_ONLY`；自动重连沿用此模式。
- 无法设置该模式时停车并停止连接；5 GHz 不可用时只等待，不改连 2.4 GHz。
- 固件不固定信道，会扫描允许的 5 GHz 信道，所以路由器目前的“自动”可以先保留。

连接成功时，串口应出现类似：

```text
[WiFi] 5GHz ONLY; 2.4GHz disabled.
[WiFi] connected SSID=YAYA IP=192.168.33.xxx MAC=... RSSI=-... channel=36 band=5GHz
```

若希望减少实验期间路由器换信道的影响，可将 **5 GHz** 固定为当地允许的非 DFS 信道，
例如路由器列表中可选的 36/40/44/48；不需要把固件同步改为该信道。
C5 在 STA 模式可连接扫描到的 DFS AP，但自动信道/DFS 引起的 AP 信道变化可能带来等待或短暂中断。

路由器的 2.4 GHz、5 GHz 都可以叫 YAYA，并使用同一密码。必须同时满足：

1. 两个频段桥接到同一 LAN，DHCP 地址仍是 `192.168.33.x`。
2. 关闭 AP/客户端隔离，避免使用隔离的访客网络；允许频段之间的 UDP 广播和 TCP 通信。
3. Vicon 主机仍是 `192.168.33.102`，不要把此地址分配给其他设备。

电脑可连任一频段，也可接路由器网线，只要能访问上述 LAN。
5 GHz 可以分担无线空口竞争，但不会自动增加路由器整机的最大客户端数量，也不能保证所有断连都消失。

## 5. 首辆 C5 车验证

1. 先断电，按表接好线，再将车轮架空，烧录 **16 号**固件。
2. 在串口确认编号是 `kedaya16`、地址为 `192.168.33.x`、实际显示 `band=5GHz`。
3. 电脑连接 YAYA，运行 `keyboard_remote_control.py`，选择 `kedaya16`，低速检查六个方向和停车。
4. 若运动方向与原车不符，检查 M1/M2/M3 是否对应原车同一位置以及输出线极性。
5. Vicon Tracker 中将该车刚体命名为 `kedaya16`，再检查坐标和现有运动脚本。

编译检查不能代替上述实车验证。确认 16 号正常后，其他 C5 车照同一接线表安装并烧录各自编号工程。

## 资料依据

- [Waveshare 本开发板原理图](https://files.waveshare.com/wiki/ESP32-C5-WIFI6-KIT-NXRX/ESP32-C5-WIFI6-KIT-NXRX-Schematic.pdf)
- [Espressif C5 GPIO 说明](https://docs.espressif.com/projects/esp-idf/en/stable/esp32c5/api-reference/peripherals/gpio.html)
- [Espressif C5 启动模式与 SDIO 配置](https://documentation.espressif.com/esp32-c5_datasheet_en.html)
- [Arduino-ESP32 频段 API 源码](https://github.com/espressif/arduino-esp32/blob/master/libraries/WiFi/src/WiFiGeneric.cpp)
- [Espressif C5 Wi-Fi 与 DFS 说明](https://docs.espressif.com/projects/esp-idf/en/stable/esp32c5/api-guides/wifi-driver/overview.html)
- [Toshiba TB6612FNG 数据手册](https://toshiba.semicon-storage.com/info/docget.jsp?did=10660)
