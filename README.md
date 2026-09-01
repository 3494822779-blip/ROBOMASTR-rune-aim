# Rune DeepStream / RMCS

面向 RoboMaster 能量机关的检测、跟踪、预测与火控工程。`deepstream/` 是可视化推理程序，`rmcs/` 是不依赖 ROS 2 的 C++/CUDA 算法库与闭环测试。

## 快速开始

```bash
cd rmcs
./build.sh                         # 配置、编译并运行 CTest
./build/rune_virtual_test 1 7 100 # 大符虚拟闭环
```

工具链可通过环境变量覆盖：

```bash
CUDACXX=/usr/local/cuda/bin/nvcc CC=gcc CXX=g++ JOBS=2 ./build.sh
```

## Layout

- `model/`: ONNX and TensorRT FP16 engine
- `deepstream/`: runnable visualization application, config and custom parser
- `rmcs/`: 从观测到火控的完整算法库（检测 → PnP → EKF 跟踪 → 拟合 → 弹道 → 火控 → 诊断）
- `test/`: H.264 test video and original AVI link
- `docs/`: integration notes（[`docs/PIPELINE.md`](docs/PIPELINE.md) 为从观测到击打全链路详解，推荐先读）

## DeepStream test

```bash
cd /home/nvidia/rune_deepstream/deepstream
make CUDA_VER=12.6
./bin/deepstream \
  -s file:///home/nvidia/rune_deepstream/test/rune_test_h264.mp4 \
  -c config/config_infer_primary_rune_pose.txt
```

## Build RMCS library

```bash
cd /home/nvidia/rune_deepstream/rmcs
./build.sh
```

Output: `rmcs/lib/librune_full.a`.

详细构建、测试和故障处理见 `rmcs/README.md` 与 `docs/BUILD_TROUBLESHOOTING.md`。

## rmcs/ 库：从观测到火控

`rmcs/` 是完整能量机关算法库（产物 `lib/librune_full.a`），按方案 D（单体主链路 +
DeepStream 验证路径 + 库/应用分离）组织。模块与链路：

```
RuneDetector(TensorRT) ──> RuneModel(PnP 初始化 + 6 维 EKF + 正弦/线性能量拟合)
        ──> RuneFireControl(弹道 + 开火状态机) ──> yaw/pitch/fire
        ──> RuneDiagnostics(预测误差评估闭环)
```

| 模块 | 文件 | 来源 |
| --- | --- | --- |
| 神经检测 | `module/detector/rune.cpp` + `module/gpu/rune_gpu.cu` | 本仓库（YOLOv8-pose 5 关键点 + 3 类） |
| EKF 跟踪 / PnP 初始化 | `module/tracker/model/rune.cpp` | rmcs_auto_aim_v2 |
| 能量拟合 | `module/tracker/model/rune_energy_fitter.cpp` | rmcs_auto_aim_v2 |
| 虚拟符真值 | `module/tracker/model/virtual_rune.cpp` | rmcs_auto_aim_v2 |
| 弹道解算 | `module/fire_control/trajectory_solution.cpp` | rmcs_auto_aim_v2（吸纳） |
| 火控状态机 | `module/fire_control/rune_fire_control.cpp` | 新增（RP-26Rune RuneDecisionModule 设计） |
| 预测误差诊断 | `module/diagnostics/rune_diagnostics.cpp` | 新增（RP-26Rune PowerRuneDiagnostics 设计） |
| PnP/重投影支撑 | `utility/math/solve_pnp/pnp_solution.cpp` 等 | rmcs_auto_aim_v2（此前漏编译，已补） |

### 可执行目标

| 目标 | 用途 | 运行 |
| --- | --- | --- |
| `rune_video_test` | 检测器性能测试（原有） | `./build/rune_video_test <engine> <video>` |
| `rune_virtual_test` | **全链路真值闭环**：虚拟符 → EKF → 火控 → 误差诊断 | `./build/rune_virtual_test [large=0/1] [seconds] [hz]` |
| `rune_video_aim` | 视频回放链路：检测 → 跟踪 → 火控（可视化） | `./build/rune_video_aim <engine> <video>` |

### 真值闭环测试结果（虚拟符）

误差口径与 RP-26Rune PowerRuneDiagnostics 一致：「预测命中相位 vs 实测相位」在配对容差内的
wrap 误差。CSV 输出 `/tmp/rune_diag.csv`。

| 场景 | 模式 | 平均预测误差 | 最大预测误差 |
| --- | --- | --- | --- |
| 无退化（纯算法误差） | 大符（正弦） | 0.0021 rad (0.1°) | 0.0087 rad (0.5°) |
| 无退化（纯算法误差） | 小符（恒速） | 0.0024 rad (0.1°) | 0.0048 rad (0.3°) |
| 轻度退化（1.5px 噪声 + 10% 漏检） | 大符 | 0.0080 rad (0.5°) | 0.0202 rad (1.2°) |
| 重度退化（3px 噪声 + 20% 漏检） | 大符 | 0.0204 rad (1.2°) | 0.0635 rad (3.6°) |

重度退化场景的 0.020 rad 与 RP-26Rune 宣称的 0.02 rad（真实噪声环境）持平。

### 升级记录（已有模块）

| 项 | 内容 | 验证 |
| --- | --- | --- |
| **P0-1** | RuneEnergyFitter ω 两阶段精化：41 步粗扫 + 黄金分割，分辨率 0.01 → ~1e-4 rad/s | 大符 mean 0.0041 → 0.0019 rad（↓54%） |
| **P0-2** | RuneEnergyFitter Cauchy IRLS 鲁棒加权（w=1/(1+(r/2.5·RMS)²)），切叶/遮挡离群点自动降权 | 退化场景误差可控 |
| **P1-3** | 恢复 RuneModel::converge() 协方差收敛判据（此前被注释恒 true） | virtual_test 显示 conv 状态 |
| **P1-4** | RuneBullseye 保留大/小符激活类别（Activation 枚举），不再合并为 bool | video_test 显示 SMALL/BIG 标签 |
| **P2-6** | virtual_rune 观测退化模拟：像素高斯噪声 + 每片符叶独立漏检/遮挡（seed 可复现） | 见上表退化场景 |
| **P2-7** | 剥离 rclcpp：库纯 C++（日志改 stderr、serializable 去耦合），构建/运行不再依赖 ROS2 | ./build.sh 无需 source ROS2 |
| **P2-8** | RuneFireControl::Command 透出 ff_v/ff_a 云台前馈 | — |

### 实车集成

火控输出为车体系（Odom）射线角 yaw/pitch 与开火标志，坐标系约定与 rmcs_auto_aim_v2 一致。
接入实车时：将 `RuneFireControl::Command` 转换为云台目标角 + 开火指令下发，
并回读 IMU 姿态更新 `RuneModel::update_transform`（相机外参）。参数集中在
`RuneFireControl::Config` 与 `RuneModel::Config`。
