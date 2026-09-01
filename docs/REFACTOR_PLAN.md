# 项目重构方案（rune_aim → rune_aim）

> 状态：✅ 已执行（2026-09-01）。实施结果见 commit 记录与 README；小符/大符闭环回归误差 0.03°/0.02°，优于基线。
> 目标：易改参数、可视化调试方便、结构明了、终端命令行启动、参数集中一个文件夹。

---

## 1. 现状问题诊断（为什么"乱"）

当前代码由三拨来源拼成，各留各的痕迹：

| 来源 | 痕迹 | 问题 |
|---|---|---|
| 原项目 rmcs_auto_aim_v2 | `kernel/`、`module/refiner/`、`utility/` 大量未用文件、`rclcpp/` 整套 | `kernel/detector.cpp` 引用不存在的头文件（死代码）；`utility/` 84 个文件只有 ~12 个被真正用到 |
| 并行新增（RuneFireControl 等） | `module/fire_control/`、`module/diagnostics/`、`app/` | 命名与旧结构混在一起（module 里既有旧 detector 又有新 fire_control） |
| NVIDIA DeepStream 样例 | `deepstream/`（GStreamer 管线，另一套技术栈） | 与 rmcs 功能重叠（都是"检测+可视化"），参数/启动方式完全不同 |

**具体痛点**：
- 参数散落：`rune_video_aim` 内参硬编码在代码里（fx=1400...）、`virtual_test` 的 K/D 硬编码、火控参数在 `Config` 默认值里、`kernel/detector.cpp` 里还硬编码 engine 绝对路径——**没有一个统一的参数入口**；
- 4 个可执行文件启动方式各不相同、参数靠手敲位置参数（`<engine> <video> [max_frames] [fx] [fy]...`），记不住；
- 目录嵌套两层（`module/tracker/model/`）、同名文件多个（`rune.cpp` ×2）、死代码与活代码混排；
- 可视化逻辑散落在各测试程序里（video_test 有 imshow，virtual_test 只打印），不统一、不可开关。

---

## 2. 重构目标（四个诉求如何落地）

| 诉求 | 落地方式 |
|---|---|
| **易改参数** | 所有参数集中到顶层 `config/` 目录（YAML）；程序启动只认 `--config`，场景 = 一个 yaml 文件 |
| **可视化调试方便** | 统一调试绘制层 `src/debug/`（关键点/符叶/预瞄点/状态文字/误差），`display:` 段一键开关；video / virtual / camera 三种数据源共用 |
| **结构明了** | 按功能分层 `src/{core,detect,track,fire,diag,debug}`，单层目录、无重复命名、死代码全部删除 |
| **终端命令行启动** | 唯一主程序 `rune_aim`：`./rune_aim -c config/rune_small.yaml`，`--mode` 选数据源 |

---

## 3. 新目录结构

```
rune_aim/                       # 项目根（重构后 rmcs/、app/、test/ 均消失）
├── README.md                          # 唯一使用说明（构建/运行/参数说明）
├── build.sh                           # 一键构建
├── CMakeLists.txt
│
├── config/              ★★★ 所有参数集中在这里，一个场景 = 一个 yaml
│   ├── common.yaml                   # 公共：engine 路径、GPU、日志级别
│   ├── rune_small_video.yaml         # 场景：小符 + 视频回放
│   ├── rune_large_video.yaml         # 场景：大符 + 视频回放
│   ├── rune_small_virtual.yaml       # 场景：小符 + 虚拟符（无 GPU 调试）
│   ├── rune_large_virtual.yaml       # 场景：大符 + 虚拟符
│   └── camera_sentry.yaml            # 场景：真机相机（待补标定值）
│
├── src/                # 源码（单层功能目录，全部新命名）
│   ├── core/                        # 通用基础（原 utility 精简到 ~12 个文件）
│   │   ├── types.{hpp}              # Point2d/Point3d/Transform（原 math/linear.hpp）
│   │   ├── clock.hpp
│   │   ├── angle.hpp
│   │   ├── camera.{hpp,cpp}         # CameraFeature
│   │   ├── conversion.hpp           # ROS↔OpenCV 坐标
│   │   ├── pnp.{hpp,cpp}            # SingleRunePnpSolution（原 solve_pnp/pnp_solution）
│   │   ├── reprojection.{hpp,cpp}
│   │   ├── corners.{hpp,cpp}        # 角点优化（原 corners_optimizor）
│   │   ├── hungarian.hpp            # 匈牙利匹配
│   │   ├── mahalanobis.hpp
│   │   ├── pimpl.hpp
│   │   └── rune_types.hpp           # RuneBullseye/RuneIcon/物点常数（原 robot/rune.hpp）
│   ├── detect/                      # 检测
│   │   ├── detector.{hpp,cpp}       # ← module/detector/rune（类 RuneDetector）
│   │   └── gpu_{preprocess,refine}.{hpp,cu}  # ← module/gpu/rune_gpu
│   ├── track/                       # 跟踪
│   │   ├── rune_model.{hpp,cpp}     # ← module/tracker/model/rune（类 RuneModel）
│   │   ├── energy_fitter.{hpp,cpp}  # ← rune_energy_fitter
│   │   └── virtual_rune.{hpp,cpp}   # ← virtual_rune（虚拟符，无 GPU 调试用）
│   ├── fire/                        # 火控
│   │   ├── fire_control.{hpp,cpp}   # ← rune_fire_control（类 RuneFireControl）
│   │   └── trajectory.{hpp,cpp}     # ← trajectory_solution（弹道）
│   ├── diag/                        # 诊断
│   │   ├── diagnostics.{hpp,cpp}    # ← rune_diagnostics（预测误差）
│   │   └── delay_calibrator.{hpp,cpp}
│   └── debug/                       # ★ 新：统一可视化调试层
│       ├── draw.{hpp,cpp}           # 画关键点/符叶/R标/预瞄点/状态/误差，全部可开关
│       └── panel.{hpp,cpp}          # 调试面板（按键：q退出/空格暂停/s截图/v切开关）
│
├── tools/             # 命令行工具
│   ├── rune_aim.cpp                # ★ 唯一主程序（--mode video|virtual|camera）
│   ├── rune_bench.cpp              # 检测性能基准（原 test/rune_video_test）
│   └── delay_calib.cpp             # 链路延迟标定（原 app/delay_calib_test）
│
├── model/            # 模型文件（现状保留）
├── data/             # 测试视频软链 → ../test/rune_test_h264.mp4
├── demos/
│   └── deepstream/   # 原 deepstream/ 原样移入（另一技术栈演示，不参与重构）
└── docs/
```

---

## 4. 文件改名映射表（关键）

| 现在 | 新位置/新名 |
|---|---|
| `rmcs/module/detector/rune.{hpp,cpp}` | `src/detect/detector.{hpp,cpp}` |
| `rmcs/module/gpu/rune_gpu.{hpp,cu}` | `src/detect/gpu_preprocess.{hpp,cu}` + `gpu_refine.{hpp,cu}` |
| `rmcs/module/tracker/model/rune.{hpp,cpp}` | `src/track/rune_model.{hpp,cpp}` |
| `rmcs/module/tracker/model/rune_energy_fitter.*` | `src/track/energy_fitter.*` |
| `rmcs/module/tracker/model/virtual_rune.*` | `src/track/virtual_rune.*` |
| `rmcs/module/fire_control/rune_fire_control.*` | `src/fire/fire_control.*` |
| `rmcs/module/fire_control/trajectory_solution.*` | `src/fire/trajectory.*` |
| `rmcs/module/diagnostics/rune_diagnostics.*` | `src/diag/diagnostics.*` |
| `rmcs/module/diagnostics/delay_calibrator.*` | `src/diag/delay_calibrator.*` |
| `rmcs/utility/math/linear.hpp` 等 12 个在用头文件 | `src/core/*` |
| `rmcs/app/video_test.cpp` | `tools/rune_aim.cpp`（--mode video） |
| `rmcs/app/virtual_test.cpp` | 并入 `tools/rune_aim.cpp`（--mode virtual） |
| `rmcs/app/delay_calib_test.cpp` | `tools/delay_calib.cpp` |
| `rmcs/test/rune_video_test.cpp` | `tools/rune_bench.cpp` |
| `rmcs/kernel/`、`rmcs/module/refiner/` | **删除**（死代码，无引用） |
| `rmcs/utility/` 未用文件（rclcpp/image/coroutine/csv/thread/singleton/shared/tf 等） | **删除**（约 70 个文件） |
| `deepstream/` | 移入 `demos/deepstream/`（不动内部） |

> 类名保留（`RuneDetector`/`RuneModel`/`RuneFireControl` 本身清晰），只改文件/目录/命名空间归属。
> 命名空间：建议保留 `rmcs`（机械替换 `rmcs::` → `rune::` 收益低、风险高；也可选做，见待确认项）。

---

## 5. 参数集中方案（config/）

**原则：一个场景一个 yaml，程序零硬编码参数。** 所有程序用统一加载器读取：

```yaml
# config/rune_small_video.yaml —— 示例
model:
  engine: model/Rune-v8n-fp16-20260624_b1_gpu0_fp16.engine   # 相对项目根

input:
  mode: video              # video | virtual | camera
  source: data/rune_test_h264.mp4
  max_frames: 0            # 0 = 不限

camera:                    # 标定值；virtual 模式可省略（用内置演示内参）
  matrix: [1400, 0, 720, 0, 1400, 540, 0, 0, 1]
  distortion: [0, 0, 0, 0, 0]
  transform: [0, 0, 0, 1, 0, 0, 0]        # t_xyz + q_xyzw（相机→Odom）

detect:
  score_threshold: 0.8
  keypoint_threshold: 0.8
  center_distance: 30.0
  refine_radius: 10
  icon_refine_radius: 14
  max_refine_shift: 7.0
  min_refine_gradient: 12.0
  min_refine_support: 6

track:                     # EKF 参数（对应 RuneModel::Config）
  noise_observation: 20.0
  gate_threshold: 13.816
  init_seed_mean_error: 10.0
  init_seed_max_error: 20.0
  init_center_gate: 30.0
  init_pitch_bound: 20.0
  diverge_face_angle: 45.0

fire:                      # 火控参数（对应 RuneFireControl::Config）
  bullet_speed: 22.5       # ★ 发射仓标定
  shoot_delay: 0.04        # ★ 发射仓标定
  algorithmic_delay: 0.05
  max_fly_time: 1.0
  pitch_max: 0.61
  fire_cooldown_init: 0.3
  fire_cooldown: 0.7
  fire_window: 0.04
  data_life: 0.2
  recover_time: 0.2
  switch_angle: 0.30
  switch_confirm: 5
  offset_yaw: 0.0
  offset_pitch: 0.0

display:                   # 可视化调试开关
  enabled: true
  show_keypoints: true
  show_blades: true
  show_aimpoint: true
  show_state_text: true
  show_error_text: true
  save_screenshot_dir: ""  # 非空则 s 键存图
```

加载机制：`src/core/config_loader.{hpp,cpp}`（yaml-cpp，`-c` 指定，缺省字段用代码内默认值，未知字段告警）。

---

## 6. 命令行启动设计

```bash
./build/rune_aim -c config/rune_small_video.yaml            # 最常用
./build/rune_aim -c config/rune_large_virtual.yaml          # 无 GPU 调参（虚拟符）
./build/rune_aim -c config/rune_small_video.yaml --max-frames 300   # 覆盖单参数
./build/rune_aim -c config/camera_sentry.yaml --mode camera # 切真机相机
```

- 只暴露少数高频覆盖项（`--mode`/`--source`/`--max-frames`/`--no-display`），其余一律进 yaml；
- 启动即打印"实际生效参数"（读到的 yaml 值），避免"我以为改的是这个参数"；
- 日志带时间戳到终端 + 可选文件。

---

## 7. 可视化调试方案（src/debug/）

统一绘制层，三种数据源共用同一套：

- **检测层**：5 关键点（按分数着色）、符叶中心/角点、R 标、类别文字（inactive/SMALL/BIG）；
- **跟踪层**：EKF 预测的 6 个特征点投影、符叶多边形（R 标 + 5 叶片）、正弦拟合参数文字 `v+a·sin(ωt+φ)`；
- **火控层**：预瞄点（外推到命中时刻的位置）、当前 yaw/pitch、状态机（COOLING/READY/FIRING…，开火红色闪烁）、飞行时间；
- **诊断层**：预测误差曲线（mean/max/latest，可选 imshow 小窗或终端行）；
- **交互**：`q` 退出、`空格` 暂停、`s` 截图、`v` 切换可视化、`1/2/3` 开关各层；
- 全部开关在 yaml `display:` 段，`enabled: false` 时纯终端运行（适合无人值守/性能测试）。

---

## 8. 实施阶段（每阶段可编译、可回归）

| 阶段 | 内容 | 验证 |
|---|---|---|
| 0 | `git init` + 提交现状（留退路）；`cp -r` 备份到 `backup/` | — |
| 1 | 新建 `config/` + `src/core/config_loader`；写 `tools/rune_aim.cpp` 骨架（先调用现有模块，不改内部） | 能读 yaml 并启动 virtual 模式 |
| 2 | 文件搬移 + 改名（按映射表）+ CMake 重写 + include 修正 | **编译通过** + `rune_virtual_test` 等价回归（误差 <0.5°） |
| 3 | 删除死代码（kernel/、refiner/、无用 utility/） | 编译通过，功能不变 |
| 4 | `src/debug/` 绘制层 + 三模式统一 + 交互按键 | video/virtual 两种模式可视化一致 |
| 5 | 文档（README 重写、config 模板注释）、`build.sh` 收尾 | 全新克隆即可构建运行 |

---

## 9. 待确认决策点（确认后再开工）

1. **`deepstream/` 怎么处理**？
   - A. 移入 `demos/deepstream/` 原样保留（推荐：另一技术栈演示，`rune_aim --mode video` 已覆盖其功能，但保留无成本）
   - B. 直接删除（彻底告别"别人的代码"）
   - C. 保留在根目录不动
2. **命名空间** `rmcs::` 是否改成 `rune::`？（推荐保留，改动大且无功能收益；除非你想要 100% 属于自己）
3. **类名**：`RuneDetector/RuneModel/RuneFireControl` 保留？（推荐保留，已清晰）
4. **库形态**：继续产出 `librune_core.a` 静态库 + 独立可执行？（推荐保留，方便外部集成）
5. **测试视频**：`test/rune_test_h264.mp4` 移入 `data/`？（推荐，与结构统一）

---

## 10. 风险与对策

| 风险 | 对策 |
|---|---|
| 搬移后 include 路径/CMake 出错 | 阶段 2 一次搬完并立即编译；`git` 可回退 |
| 并行工作再次修改 | 开工前先 `git add -A && git commit` 锁定基线；若发现新改动先沟通 |
| 行为回归 | 每个阶段跑 `rune_virtual_test` 等价闭环（当前误差基线：小符 0.1°、大符 0.3°） |
| 命名空间/头文件循环依赖 | `src/core` 只放无依赖基础类型；`draw` 依赖其余层但不被依赖 |
