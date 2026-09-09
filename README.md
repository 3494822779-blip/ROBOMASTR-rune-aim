# rune_aim —— RoboMaster 能量机关检测 / 跟踪 / 预测 / 火控

面向 RoboMaster 能量机关（符）的完整工程：TensorRT 神经检测 → PnP 初始化 → EKF 跟踪 →
转速拟合 → 弹道解算 → 火控状态机 → 预测误差诊断。**纯 C++/CUDA，不依赖 ROS2。**

## 快速开始

```bash
./build.sh                                    # 一键构建（输出 build/ 与 lib/librune_core.a）

./build/rune_aim -c config/rune_small_virtual.yaml   # 小符虚拟符闭环（无 GPU，调参首选）
./build/rune_aim -c config/rune_large_virtual.yaml   # 大符（正弦）虚拟符闭环
./build/rune_aim -c config/rune_gimbal_virtual.yaml  # 云台运动仿真（验证外参时变链路，无 GPU）
./build/rune_aim -c config/rune_video.yaml           # 视频回放（需 GPU 引擎 + 图形会话）
./build/rune_aim -c config/camera_sentry.yaml        # 真机相机（替换标定值后使用）
```

按键：`q`/`ESC` 退出 · `空格` 暂停 · `s` 截图 · `v` 开关可视化。

## 目录结构

```
rune_aim/
├── config/          ★ 所有参数集中于此：一个场景 = 一个 yaml
├── src/             源码（单层功能目录）
│   ├── core/        基础：类型/相机/PnP/弹道支撑/参数加载（config_loader）
│   ├── detect/      RuneDetector（TensorRT）+ CUDA 预处理/细化
│   ├── track/       RuneModel（EKF）+ EnergyFitter + VirtualRune
│   ├── fire/        RuneFireControl（预瞄+开火状态机）+ Trajectory（弹道）
│   ├── diag/        RuneDiagnostics（预测误差）+ DelayCalibrator（链路延迟标定）
│   └── debug/       统一可视化绘制层
├── tools/           命令行工具：rune_aim（统一入口）/ rune_bench / delay_calib
├── data/            测试视频（rune_test_h264.mp4）
├── model/           ONNX 与 TensorRT FP16 engine
├── docs/            文档（PIPELINE.md 为全链路详解，推荐先读）
└── build.sh / CMakeLists.txt
```

## 参数（config/*.yaml）

**template.yaml 是唯一参数源**：所有可调参数先从这里读（含默认值），场景 yaml
（rune_*.yaml）只写本场景差异并覆盖。改公共参数（相机标定/阈值/冷却等）只改
template.yaml 一处即可全局生效；启动日志打印 base/scene 两层与实际生效值。各段对应：

| 段 | 对应 | 关键项 |
|---|---|---|
| `input` | 数据源 | `mode: video\|virtual\|camera`、`source`、`max_frames`、`hz` |
| `camera` | 标定 | `matrix`(9) `distortion`(5) `transform`(t+q，相机→Odom) |
| `detect` | RuneDetector::Config | `engine`、`score_threshold`、`keypoint_threshold`、`center_distance`、细化参数 |
| `track` | RuneModel::Config | `noise_observation`、`gate_threshold`、初始化门限 |
| `fire` | RuneFireControl::Config | ★ `bullet_speed` `shoot_delay`（发射仓标定）、`algorithmic_delay`、开火窗口/冷却/切叶 |
| `virtual_rune` | VirtualRuneModel | `large`、符位置、像素噪声/漏检模拟、`gimbal`（云台运动仿真） |
| `diag` | RuneDiagnostics | `match_tolerance_ms` |
| `display` | 可视化 | `enabled`、各层开关 |

## 命令行工具

| 工具 | 用途 | 示例 |
|---|---|---|
| `rune_aim` | 统一入口（video/virtual/camera 三模式 + 可视化调试） | `./build/rune_aim -c config/rune_small_virtual.yaml --no-display` |
| `rune_bench` | 检测器性能基准 | `./build/rune_bench <engine> <video> [max_frames] [score_thr] [keypoint_thr]` |
| `delay_calib` | 链路延迟互相关标定 | `./build/delay_calib 120 2 30` |

## 真值闭环测试结果（虚拟符）

误差口径：预测命中相位 vs 实测相位在配对容差内的 wrap 误差（`RuneDiagnostics`），
与 RP-26Rune PowerRuneDiagnostics 一致。CSV 输出 `/tmp/rune_diag.csv`。

| 场景 | 模式 | 平均预测误差 | 最大预测误差 |
| --- | --- | --- | --- |
| 无退化（纯算法误差） | 大符（正弦） | 0.0021 rad (0.1°) | 0.0087 rad (0.5°) |
| 无退化（纯算法误差） | 小符（恒速） | 0.0024 rad (0.1°) | 0.0048 rad (0.3°) |
| 轻度退化（1.5px 噪声 + 10% 漏检） | 大符 | 0.0080 rad (0.5°) | 0.0202 rad (1.2°) |
| 重度退化（3px 噪声 + 20% 漏检） | 大符 | 0.0204 rad (1.2°) | 0.0635 rad (3.6°) |

> ⚠️ 上表数字产生于**画幅裁剪修复之前**。此前 VirtualRune 的投影只判断"点在相机前方"，
> 不判断是否落在画幅内，而 template.yaml 的演示几何（单位外参 + 符心仰角 18°）会让符叶
> 有相当一部分投影到画幅上方之外——那些真实检测器根本给不出的观测也被喂进了 EKF。
> 修复后同一条 `rune_small_virtual` 命令的 `aim_ok` 从 1999/2000 掉到 1227/2000。
> 上表需重新测量；若要沿用旧口径，请把相机外参改为对准符心（见下节场景）。
>
> ⚠️ 另：大符一行（0.1°/0.5°）目前**复现不出来**。在画幅裁剪修复**之前**的代码上实测
> `rune_large_virtual` 为 1000 帧 6.57°、6000 帧 2.18°（收敛需要约 1.5 个正弦周期以上），
> 与本表相差一个数量级。即该偏差早于本次改动存在，需单独定位是回归还是当初口径不同。

## 云台运动仿真结果（`config/rune_gimbal_virtual.yaml`）

真机上 `camera.transform` 由 IMU 回读驱动、每帧变化，但工具里一直是常量，这条链路在
装相机之前从未被执行过。本场景让外参按 yaw ±10°@0.30Hz / pitch ±3°@0.17Hz 摆动
（峰值角速度约 19°/s），并分别注入 IMU 回读滞后与外参姿态噪声。
小符、6000 帧 @200Hz、相机基准外参对准符心。

**外参滞后（`transform_delay`，噪声=0）—— 线性、温和：**

| 滞后 | init_ok | aim_ok | 平均预测误差 | 最大预测误差 |
| --- | --- | --- | --- | --- |
| 0（云台静止参照） | 1 | 5999 | 0.03° | 0.06° |
| 0 ms（摆动，外参精确） | 1 | 5999 | 0.03° | 0.06° |
| 5 ms | 1 | 5999 | 0.03° | 0.09° |
| 10 ms | 1 | 5999 | 0.05° | 0.12° |
| 20 ms | 1 | 5999 | 0.10° | 0.25° |
| 30 ms | 1 | 5999 | 0.17° | 0.37° |
| 50 ms | 1 | 5999 | 0.27° | 0.50° |

**外参姿态噪声（`transform_noise`，滞后=0）—— 超线性、会打断跟踪：**

| 噪声 σ | init_ok | aim_ok | 平均预测误差 | 最大预测误差 |
| --- | --- | --- | --- | --- |
| 0° | 1 | 5999 | 0.04° | 0.06° |
| 0.1° | 2 | 5997 | 1.80° | 19.94° |
| 0.2° | 2 | 5997 | 3.37° | 11.51° |
| 0.5° | 35 | 5912 | 28.97° | 104.03° |
| 1.0° | 291 | 4674 | 87.64° | 174.44° |

结论：

1. **外参滞后不可怕**，误差约 `0.5 × 云台角速度 × 滞后`，线性可预测。50 ms 滞后仍只有
   0.27°，且从不打断跟踪。IMU 回读延迟做到 20 ms 以内即可忽略。
2. **外参角度噪声才是杀手**。0.1° σ 就已经超过上表"重度退化"档；0.5° 时 30 秒内重初始化
   35 次，1.0° 时 291 次，EKF 基本锁不住。原因是外参误差是**共模**的——整帧 6 个特征点
   一起偏——而 EKF 的观测噪声 R 是对角阵，结构上无法表达这种相关性。
3. 放大 `noise_observation` 只能救回稳定性，救不回精度。固定 0.2° 噪声下实测：

   | `noise_observation` | init_ok | 平均 | 最大 |
   | --- | --- | --- | --- |
   | 2（默认） | 2 | 3.59° | 27.36° |
   | 10 | 2 | 3.29° | 14.97° |
   | 40 | 1 | 3.18° | 23.74° |
   | 100 | 1 | 1.45° | 4.92° |

   即：R 调大后不再重初始化、最大误差从 27° 收到 4.9°，但平均误差仍停在 1.45°，
   是无噪声基线（0.04°）的约 36 倍。**这个下限由 IMU 决定，调参消不掉。**

> 口径与局限：`n` 约 30 个配对样本（诊断仅在开火帧记录），个位数百分比差异无意义，
> 看趋势即可。噪声模型是逐帧独立高斯，未建模**缓变偏置**——真机上标定误差更接近
> 常值偏置，且常值偏置不可观测，预计比白噪声更难处理，这是下一步要补的仿真项。

## 链路

```
数据源(video/virtual/camera)
  → RuneDetector(TensorRT，5 关键点 + 3 类)    [video/camera]
  → RuneModel(PnP 初始化 + 6 维 EKF + 正弦/线性能量拟合)
  → RuneFireControl(弹道固定点迭代 + 开火状态机)  → yaw/pitch/fire + ff_v/ff_a 前馈
  → RuneDiagnostics(预测误差评估闭环)
```

## 实车集成要点

- 火控输出 Odom 系射线角 yaw/pitch 与开火标志；坐标系约定与 rmcs_auto_aim_v2 一致；
- 接入实车：`RuneFireControl::Command` → 云台目标角 + 开火指令；回读 IMU 更新相机外参
  （`RuneModel::update_transform`，config 的 `camera.transform`）；
- 必改参数：`camera.*`（标定）、`fire.bullet_speed` / `fire.shoot_delay`（发射仓标定）、
  `fire.algorithmic_delay`（可用 `delay_calib` 标定）。
- 接工业相机（海康 MVS / 大恒 Galaxy 等 `cv::VideoCapture` 打不开的设备）：实现
  `src/core/frame_source.hpp` 的 `FrameSource`（`grab()` 返回图像 + **采集时刻**），
  主循环不需要改。有硬件采集时间戳时务必填进 `Frame::stamp`，那是全链路时间对齐的基准。

## 升级记录

| 项 | 内容 |
| --- | --- |
| P0-1 | RuneEnergyFitter ω 两阶段精化：粗扫 + 黄金分割，分辨率 ~1e-4 rad/s |
| P0-2 | RuneEnergyFitter Cauchy IRLS 鲁棒加权，切叶/遮挡离群点自动降权 |
| P1-3 | 恢复 RuneModel::converge() 协方差收敛判据 |
| P1-4 | RuneBullseye 保留大/小符激活类别（Activation 枚举） |
| P2-6 | virtual_rune 观测退化模拟（像素噪声 + 漏检） |
| P2-7 | 剥离 rclcpp：库纯 C++，构建/运行不依赖 ROS2 |
| P2-8 | RuneFireControl::Command 透出 ff_v/ff_a 云台前馈 |
| **R1** | **全面重构：src/{core,detect,track,fire,diag,debug} 单层结构 + config/ 参数集中 + 统一 CLI（rune_aim）+ 删除 deepstream/kernel/refiner/无用 utility** |
| R2 | 轨迹积分 Euler → RK4，提高远距/高速场景精度 |
| **G1** | **云台运动仿真：`virtual_rune.gimbal` 让相机外参随时间摆动，可注入 IMU 回读滞后与外参标定噪声；配套修复 virtual 模式缺失的画幅裁剪与 init 前外参未刷新** |
| **G2** | **数据源抽象 `FrameSource`：帧自带采集时间戳，消除双缓冲流水线的一帧时间戳偏差；检测改为常驻线程（原每帧新建 `std::async` 线程）；每帧 3 次全图 clone 降为 1 次；修复 camera 模式非数字 source 触发 `std::stoi` 未捕获异常导致的崩溃** |
