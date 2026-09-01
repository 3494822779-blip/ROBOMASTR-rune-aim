# rune_aim —— RoboMaster 能量机关检测 / 跟踪 / 预测 / 火控

面向 RoboMaster 能量机关（符）的完整工程：TensorRT 神经检测 → PnP 初始化 → EKF 跟踪 →
转速拟合 → 弹道解算 → 火控状态机 → 预测误差诊断。**纯 C++/CUDA，不依赖 ROS2。**

## 快速开始

```bash
./build.sh                                    # 一键构建（输出 build/ 与 lib/librune_core.a）

./build/rune_aim -c config/rune_small_virtual.yaml   # 小符虚拟符闭环（无 GPU，调参首选）
./build/rune_aim -c config/rune_large_virtual.yaml   # 大符（正弦）虚拟符闭环
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

所有可调参数集中在一个 yaml，启动即打印实际生效值。各段对应：

| 段 | 对应 | 关键项 |
|---|---|---|
| `input` | 数据源 | `mode: video\|virtual\|camera`、`source`、`max_frames`、`hz` |
| `camera` | 标定 | `matrix`(9) `distortion`(5) `transform`(t+q，相机→Odom) |
| `detect` | RuneDetector::Config | `engine`、`score_threshold`、`keypoint_threshold`、`center_distance`、细化参数 |
| `track` | RuneModel::Config | `noise_observation`、`gate_threshold`、初始化门限 |
| `fire` | RuneFireControl::Config | ★ `bullet_speed` `shoot_delay`（发射仓标定）、`algorithmic_delay`、开火窗口/冷却/切叶 |
| `virtual_rune` | VirtualRuneModel | `large`、符位置、像素噪声/漏检模拟 |
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
