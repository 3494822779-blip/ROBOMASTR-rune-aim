> ⚠️ **2026-09-01 重构说明**：项目已重构为新结构（src/{core,detect,track,fire,diag,debug} + config/ + tools/）。
> 本文中的旧路径 `rmcs/module/...` / `rmcs/utility/...` 已失效，对应新位置见 `README.md` 目录结构；
> 参数配置以 `config/*.yaml`（统一入口 `./build/rune_aim -c <yaml>`）为准。

# RMCS 火控 + 跟踪接入状态与使用指南

> 更新：2026-09-01。**火控与跟踪已经连接完成并验证通过**（无需再拷贝原项目 `rmcs_auto_aim_v2` 的 FireController —— 本仓库已有更贴合能量机关的 `RuneFireControl`）。
> 本文说明：已连接的模块、验证结果、4 个可执行测试程序、以及各部分的工作原理。

---

## 1. 结论：已经连接好了 ✅

完整链路 **`RuneDetector → RuneModel → RuneFireControl → RuneDiagnostics`** 已实现、已编译、已验证：

- 构建：`rmcs/build/` 下 4 个可执行文件全部链接成功（`rune_video_aim` / `rune_video_test` / `rune_virtual_test` / `rune_delay_calib_test`）；
- 全链路真值闭环测试（无 GPU，虚拟符）实测通过：
  - 小符：开火 9 次，预测相位误差 **mean 0.1° / max 0.2°**
  - 大符（正弦）：开火 9 次，预测相位误差 **mean 0.3° / max 0.4°**
- 火控状态机（COOLING → READY → FIRING → COOLING）正常流转，大/小符均能开火。

---

## 2. 已连接模块与文件清单

```
rmcs/
├── module/
│   ├── detector/rune.{hpp,cpp}               # RuneDetector：TensorRT 神经检测（本仓库替换版）
│   ├── gpu/rune_gpu.{hpp,cu}                 # CUDA 预处理 + 关键点细化
│   ├── tracker/model/
│   │   ├── rune.{hpp,cpp}                    # RuneModel：EKF 跟踪 + 正弦/线性转速拟合
│   │   ├── rune_energy_fitter.*              # 转速拟合器
│   │   └── virtual_rune.*                    # 虚拟符仿真（真值源，含像素噪声/漏检模拟）
│   ├── fire_control/
│   │   ├── rune_fire_control.{hpp,cpp}       # ★ 火控状态机（预瞄 + 弹道 + 开火判定）
│   │   └── trajectory_solution.{hpp,cpp}     # ★ 弹道解算（重力+空气阻力数值积分）
│   └── diagnostics/
│       ├── rune_diagnostics.{hpp,cpp}        # 预测误差诊断（量化"预测准不准"）
│       └── delay_calibrator.{hpp,cpp}        # 链路延迟互相关标定
├── app/
│   ├── video_test.cpp       → rune_video_aim      # 视频回放链路测试（检测→跟踪→火控，带可视化）
│   ├── virtual_test.cpp     → rune_virtual_test   # 虚拟符真值闭环测试（无 GPU，可带噪声/漏检）
│   └── delay_calib_test.cpp → rune_delay_calib_test
├── test/rune_video_test.cpp → rune_video_test     # 检测器性能测试
└── CMakeLists.txt                                # 已包含全部上述目标
```

> ⚠️ 说明：`module/fire_control/trajectory_solution.{hpp,cpp}` 由原项目 `rmcs_auto_aim_v2` 吸纳而来（与 `rmcs_auto_aim_v2/src/module/fire_control/trajectory_solution.*` 内容一致），是 `rune_fire_control.cpp` 的编译依赖（`#include "module/fire_control/trajectory_solution.hpp"`）。

---

## 3. 如何运行

### 3.1 构建（已构建，需要时重跑）

```bash
cd /home/nvidia/rune_aim/rmcs
./build.sh     # 或: cmake --build build -j2（增量）
```

### 3.2 全链路真值闭环测试（不需要 GPU/相机/显示）✅ 已通过

```bash
./build/rune_virtual_test [large=0|1] [seconds] [hz] [noise_px] [dropout]
# 小符 20 秒
./build/rune_virtual_test 0 20 200
# 大符（正弦）20 秒
./build/rune_virtual_test 1 20 200
# 带退化模拟（像素噪声 + 漏检），评估算法鲁棒性
./build/rune_virtual_test 0 20 200 2.0 0.1
```

每秒一行：火控状态 + 瞄准角 + 飞行时间 + 预测误差；结束输出汇总与 `/tmp/rune_diag.csv`。

### 3.3 视频回放链路测试（需要 GPU 引擎 + 图形会话）

```bash
./build/rune_video_aim \
  /home/nvidia/rune_aim/model/Rune-v8n-fp16-20260624_b1_gpu0_fp16.engine \
  /home/nvidia/rune_aim/test/rune_test_h264.mp4 \
  [max_frames] [fx] [fy] [cx] [cy]
```

弹出窗口显示：检测关键点/符叶、火控状态文字（含 `[FIRE]` 标记）、预测命中点、诊断误差；`q`/`Esc` 退出。
> 默认相机内参按 1440×1080 假设（fx=fy=1400, cx=720, cy=540），**实车请用标定值替换**。

### 3.4 检测器性能测试 / 延迟标定器自检

```bash
./build/rune_video_test <engine> <video> [max_frames] [score_thr] [keypoint_thr]   # 检测 FPS
./build/rune_delay_calib_test 120 2 30                                              # 延迟标定自检
```

---

## 4. 各部分工作原理

### 4.1 RuneFireControl（module/fire_control/rune_fire_control.cpp）

能量机关专用火控状态机，**直接吃 `RuneModel::State`**（无需 Trackable 包装）：

- **瞄准 + 弹道固定点迭代**（`aim_and_ballistic`）：
  1. `t_f = 距离 / bullet_speed`（上限 `max_fly_time`）；
  2. 拷贝状态并 `transition(algorithmic_delay + shoot_delay + t_f)` 外推到命中时刻；
  3. `get_aimpoints()` 取命中时刻的未激活符叶端点（无则瞄准符心、禁射）；
  4. `TrajectorySolution{v0, attack}` 数值积分弹道（重力 9.81 + 空气阻力 0.003）解出 `fly_time/yaw/pitch`，迭代 ≤5 次至收敛。
- **开火状态机**：`LOST → COOLING(初始冷却) → READY → FIRING(开火窗口) → COOLING(连发冷却)`，另有 `RECOVERING`（数据过期平滑回符心）。
- **切叶检测**：相位跳变连续 `switch_confirm` 帧 → 确认切叶，重建初始冷却（避免云台超调期开火）。
- **开火条件**：`READY/FIRING + 有未激活符叶 + |pitch| ≤ pitch_max + fly_time ≤ max_fly_time`。
- 输出 `Command{found, fire, yaw, pitch, fly_time, ff_v, ff_a, state, reason}`。

关键配置（`Config`，全部有默认值，实车必须按发射仓标定）：

| 字段 | 默认 | 含义 |
|---|---|---|
| `bullet_speed` | 22.5 m/s | **发射仓弹速（必标定）** |
| `shoot_delay` | 0.04 s | 扳机→出膛延迟 |
| `algorithmic_delay` | 0.05 s | 算法链路延迟（采集→解算→下发） |
| `max_fly_time` / `pitch_max` | 1.0 s / 0.61 rad | 飞行时间/俯仰上限 |
| `fire_cooldown_init` / `fire_cooldown` | 0.3 / 0.7 s | 初始冷却 / 连发冷却 |
| `fire_window` | 0.04 s | 连续开火窗口上限 |
| `data_life` / `recover_time` | 0.2 / 0.2 s | 数据寿命 / 回符心时长 |
| `switch_angle` / `switch_confirm` | 0.30 rad / 5 帧 | 切叶检测 |
| `offset_yaw` / `offset_pitch` | 0 | 弹道机械偏置（rad） |

### 4.2 TrajectorySolution（module/fire_control/trajectory_solution.cpp）

含重力与空气阻力的数值积分弹道：5ms 步长积分 `(vx,vy)`（`vx -= c·v·vx·dt; vy -= (g + c·v·vy)·dt`），迭代求 pitch 使落点高度匹配目标；输出 `fly_time / yaw / pitch`。空气阻力系数 `kAirResistanceCoefficient = 0.003`（代码内常量）。

### 4.3 RuneDiagnostics（module/diagnostics/rune_diagnostics.cpp）

预测误差诊断：火控求解成功时记录"预测命中时刻 + 预测相位（外推）"，每帧记录实测相位（EKF 校正后），容差内配对，误差 = `wrap(实测 − 预测)`。**任何预测器改动都应先看这里的误差**。输出 `/tmp/rune_diag.csv`。

### 4.4 delay_calibrator（module/diagnostics/delay_calibrator.cpp）

链路延迟互相关标定（吸纳 Climber serial_delay 核心算法），用于标定 `algorithmic_delay`。

---

## 5. 已知注意点

1. **大符收敛门控**：`RuneModel::State::get_aimpoints()` 在收敛期内返回空（大符正弦 6s、小符 3s，见 `rune.cpp`），期间火控瞄准符心且禁射——测试时长要大于收敛期（大符跑 ≥8s 才能看到开火）。
2. **`bullet_speed`/`shoot_delay` 决定预瞄量**：不准则系统性偏前/偏后；`shoot_delay` 偏小→打在目标后面，偏大→打超。
3. **相机参数**：`rune_video_aim` 默认演示内参，实车必须换标定值；外参（相机→Odom）真机来自 TF（当前测试用单位外参）。
4. **GPU 权限**：`rune_video_aim`/`rune_video_test` 需要可用的 CUDA 引擎加载（本机无 GPU 权限的 shell 会报 `NvRmMemInitNvmap failed`，属正常；`rune_virtual_test` 不依赖 GPU）。
5. **与老文档的关系**：`docs/END_TO_END_ANALYSIS.md` 第 8 层描述的"原项目 FireController + Trackable 接入"方案在本仓库已被 `RuneFireControl`（直接吃 `RuneModel::State`）取代，接入方式以本文为准。

---

## 6. 下一步（如需上真机）

- 把 `rune_video_aim` 的相机内参/外参换成标定值与 TF；
- 用 `delay_calibrator` 标定链路延迟 → 填 `algorithmic_delay`；
- 按发射仓标定 `bullet_speed`/`shoot_delay`；
- 将 `RuneFireControl::Command` 接到云台/发射执行层（原项目在 `component.cpp` 的 ROS topics 层完成，本仓库不含）。
