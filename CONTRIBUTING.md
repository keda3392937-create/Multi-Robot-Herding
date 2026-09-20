# 协作说明

这是一个私有的多小车 Vicon 控制项目。协作者在 GitHub 仓库中接受邀请后，可以按下面的流程更新代码。

## 开始工作

```powershell
git clone https://github.com/keda3392937-create/Multi-Robot-Herding.git
cd Multi-Robot-Herding
git switch -c feature/简短说明
```

每次开始工作前先同步主分支：

```powershell
git switch main
git pull --ff-only
git switch feature/简短说明
git rebase main
```

## 提交和合并

提交前运行离线测试：

```powershell
python -m pytest
```

提交时写清楚改动目的，不要提交 `__pycache__`、`.pytest_cache`、`.pytest_tmp_*` 或实验生成数据。
推送分支后，在 GitHub 创建 Pull Request，至少由另一名成员检查后再合并到 `main`。

## 现场安全

- 不要在没有急停准备的情况下直接运行运动脚本。
- 修改 ESP32 固件中的 `CAR_ID` 后，确认它与 Vicon 中的 `kedayaN` 对应。
- 上传固件前确认目标串口和小车编号，避免把相同编号烧录到多辆车。
- 任何通信故障都应让小车保持停车状态。

