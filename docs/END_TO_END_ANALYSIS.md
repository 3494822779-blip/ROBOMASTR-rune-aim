# 端到端剖析：从物理世界到弹丸命中

> 本文按**数据流主线**从头到尾剖析整个项目（含原项目 `rmcs_auto_aim_v2` 的完整系统）。
> 主线：物理世界 → 成像 → 解码 → 推理 → 后处理 → 2D 观测 → PnP 3D → EKF 跟踪 → 运动预测 → 弹道解算 → 云台/发射。
> 每一层都会指出：输入、输出、代码位置、以及"路径 A（DeepStream 可视化）"与"路径 B（RMCS 库）"各自的实现。

---

## 第 0 层：物理世界（能量机关）—— 一切参数的源头

比赛里的能量机关是一块立着的圆面：

- **几何**（规则书物理常数，`rmcs/utility/robot/rune.hpp`）：
  - `kRuneGlobalRadius = 0.7` m（符心→符叶端点）
  - `kRuneBullseyeRadius = 0.15` m（十字端点相对符叶端点的偏移）
  - `kRuneIconProminentDistance = 0.1` m（R 标从符面突出）
  - 由此定义 5 个 3D 物点 `RunePagePoints::kPoints`（局部系：原点符心、X 朝背面）
- **运动学**（`virtual_rune.cpp`，规则）：
  - 小符：恒速 `π/3` rad/s
  - 大符：正弦 `spd = 1·sin(1.884t) + 1`（ω≈1.884 rad/s，幅度 1）
  - 5 片符叶间隔 72°，绕符面法线旋转，符心静止
- **规则**：符叶只有"未激活"时可击打；激活符叶轮转（代码里每 100s 轮换一次，仅用于仿真）；每片符叶有 idle（等待）/ shoot（击打）时序（`FireController` 的 `rune_idle_duration` / `rune_shoot_duration`）

> 这套"物理参数"就是第 5 层 PnP 的物点来源、第 6 层 EKF 的观测模型来源、第 7 层转速拟合的模型来源、第 8 层预瞄外推的模型来源——**全链路共用同一套常数**。

---

## 第 1 层：成像（相机 + 标定）

- **输入**：真实相机（或测试视频 `test/rune_test_h264.mp4`，1440×1080 @ 50fps，800 帧）
- **相机参数**（`CameraFeature`，`rmcs/utility/math/camera.hpp`）：
  - 内参 `camera_matrix`（3×3 行优先）、畸变 `distort_coeff`（k1 k2 p1 p2 k3）
  - 外参 `orientation`（四元数）+ `translation`（相机在 odom 系）
- **来源**：内参/畸变来自相机标定；外参来自 TF 树（`transform_tree.hpp`：`world → odom → imu → pitch → {camera, muzzle}`，即相机挂在云台 pitch 轴上，枪口 `muzzle_link` 同级）
- **坐标系约定**：ROS（x前/y左/z上）↔ OpenCV（x右/y下/z前），用 `conversion.hpp` 的 `ros2opencv_*` / `opencv2ros_*` 互相转换

**相机参数注入的完整链路（原项目 Tracker::execute，第 294-299 行）**：

```
上层(component/auto_aim) 
  → Tracker::update_camera(内参9, 畸变5, Transform外参)     ← tracker.cpp:529-538
  → Tracker::Impl::camera (CameraFeature)
  → execute() 每帧:
      rune->update_camera(bit_cast<array<9>>(camera_matrix), distort_coeff)
      rune->update_transform({camera.translation, camera.orientation})
  → RuneModel::Impl::configure_camera / update_transform
  → 写入 camera_feature + observable.feature（两处，供 PnP 与 EKF 观测共用）
```

> ⚠️ 本仓库（rune_deepstream）没有这个调用方——它被抽取成了独立库；相机参数注入发生在原项目 `Tracker` 里。

---

## 第 2 层：解码与预处理（两条路径的分叉点）

| | 路径 A（deepstream/） | 路径 B（rmcs/） |
|---|---|---|
| 解码 | GStreamer `uridecodebin` → `nvstreammux`（1920×1080 RGBA） | `cv::VideoCapture` 或相机直接给 `cv::Mat`（BGR） |
| 缩放/归一化 | `nvinfer` 内部（net-scale-factor=1/255，RGB） | CUDA kernel `preprocess_kernel`（letterbox 640×480，灰边114，双线性，RGB，÷255） |
| 入口文件 | `deepstream/src/deepstream.c` | `rmcs/module/gpu/rune_gpu.cu` |

---

## 第 3 层：神经网络推理

**模型**：YOLOv8-pose 风格，输入 `[1,3,480,640]`，输出 `[1,18,6300]`（`model/Rune-v8n-fp16-20260624.*`，TensorRT 10.3 FP16）。

- 18 通道 = 3 类别（未激活/小符激活/大符激活）+ 5 关键点 × (x,y,score)
- 6300 = stride 8/16/32 三尺度锚点（4800+1200+300）
- 内存列主序：`output[c*6300+n]`

**两条推理路径**：

| | 路径 A | 路径 B |
|---|---|---|
| 引擎载体 | `nvinfer` GStreamer 元素（config 指向 engine） | `RuneDetector` 直接 TensorRT C++ API（deserializeCudaEngine + enqueueV3） |
| 输出解析 | 自定义插件 `NvDsInferParseRunePose`（`plugin/nvdsparsepose_Rune.cpp`） | `RuneDetector::detect` 内联后处理（`rmcs/module/detector/rune.cpp`） |

**后处理（两路逻辑一致，细节略有差异）**：

| 步骤 | 路径 A | 路径 B |
|---|---|---|
| 类别 | 3 类 argmax | 同 |
| 阈值 | pre-cluster-threshold=0.8 | `score_threshold`=0.8 |
| 关键点 | score≥0.8 有效，≥3 个 | 同（`keypoint_threshold`=0.8） |
| 检测框 | 有效点包围盒 + 15% padding → bbox | 不生成 bbox，直接输出关键点 |
| NMS | 关键点中心距离 < 30px 抑制 | 同（`center_distance`=30，可配） |
| 坐标 | 网络坐标（640×480），主程序反 letterbox 到 1920×1080 | 直接还原到原图坐标（去 pad ÷ scale） |
| 细化 | 无 | CUDA `refine_kernel`：端点沿径向找梯度最强点、R 标梯度加权空间矩，位移限 7px |

---

## 第 4 层：2D 观测结构

路径 B 的输出（`utility/robot/rune.hpp`）：

```cpp
struct RuneBullseye { Point2d center; std::array<Point2d,4> corners; bool active; double score; };
struct RuneIcon     { Point2d center; double score; };
```

- `points[2]`(R) → `RuneIcon.center`
- `points[0,1,3,4]` → `corners`（top,left,bottom,right）；四点均值 → `center`
- class 0 → `active=false`；class 1/2 → `active=true`（激活信息合并为布尔）

路径 A 的等价物：`NvDsObjectMeta.mask_params` 里的 5×(x,y,score)，主程序画圈/画线/画框。

---

## 第 5 层：3D 重建（PnP）

位置：`rmcs/utility/math/solve_pnp/pnp_solution.cpp`（`SingleRunePnpSolution`），被 `RuneModel::init` 调用（`tracker/model/rune.cpp` 的 `make_init_candidate`）。

- **输入**：相机内参/畸变/外参 + 单帧的 icon 像素 + 一片未激活符叶的 4 角点像素 + 物理物点 `RunePagePoints::kPoints`（5 点）
- **过程**：按"到 R 标的距离"把 4 角点排序为 t/l/b/r → `cv::solvePnP(EPNP)` 粗解 → `cv::solvePnP(ITERATIVE)` 精化 → 坐标系转回 ROS
- **输出**：相机系下的符位姿（平移 + 四元数）
- **验证与评分**（`make_init_candidate` / `evaluate_seed_reprojection`）：
  - 重投影误差：均值 ≤10px、最大 ≤20px
  - 符面俯仰角 ≤20°
  - 多候选按"inactive 内点数 > 中心误差 > icon 误差 > …"排序取最优
- **转世界系**：`center_odom = q_odom_camera·t_camera_rune + t_odom_camera`；符面法线 → `face_yaw_odom`；由法线解出该符叶的当前旋转角 `seed_angle`

> 尺度信息完全由第 0 层的物理常数（0.7m 等）提供——没有它们，单目 PnP 只能给出无尺度解。

---

## 第 6 层：EKF 跟踪（RuneModel）

位置：`rmcs/module/tracker/model/rune.cpp`（1053 行，核心）。

- **状态（6 维）**：`[x,y,z, ω, θ, ψ]`（符心位置、旋转角速度、旋转角、符面朝向）
- **观测（6 个特征点）**：R 标 + 5 片符叶端点，用物点几何投影到像素（`Observable::update`）
- **生命周期**（在原项目 `Tracker::execute` 驱动，`tracker.cpp` 第 290-327 行）：
  ```
  rune == nullptr 且有观测 → update_camera/update_transform → init(icons, bullseyes)
  否则 → update_transform（云台动了要刷新外参）
        → predict(dt)            （θ += ω·dt，协方差传播）
        → correct(icons, bullseyes)
        → diverged()? → 销毁重建
  ```
- **correct 细节**：6 个预测特征 vs N 个观测做**匈牙利匹配**（马氏距离，gate 13.816，dummy 拒绝）→ 逐点 EKF 更新（数值稳定形式）→ inactive 符叶标记（100ms 超时清除）
- **converge/diverged**：发散检测（协方差过大、位置越界、朝向夹角>45°、NaN）

---

## 第 7 层：运动预测（转速拟合）

位置：`rmcs/module/tracker/model/rune_energy_fitter.cpp`，在 `correct()` 内驱动。

- **数据**：6s 滑动窗口的 (t, θ)，指数权重（半衰期 3s，老数据权重低）
- **线性拟合**：`θ = C + v·t`（小符近似）
- **正弦拟合**：`θ = C + v·t − (a/ω)·cos(ωt+φ)`，ω 在 [1.80, 2.20] rad/s 扫 41 步、每步最小二乘取最优
- **选择**：正弦更优（或处于强制正弦期）→ `sine_valid`，用正弦模型；否则用线性速度
- **外推能力**：`State::transition(seconds)` 用当前模型把 `θ/ω` 外推到任意未来时刻——这是第 8 层预瞄的基础

---

## 第 8 层：火控 / 预瞄（原项目 FireController）

位置：`/home/nvidia/RM/rmcs_auto_aim_v2-main/src/kernel/fire_control.cpp`（+ `module/fire_control/trajectory_solution.cpp` 弹道、`shoot_evaluator.cpp` 射击评估）。本仓库未包含，但与本仓库代码同源（utility、rune 模型逐文件 diff 相同）。

**输入**：`Trackable`（把 `RuneModel::State` 用 `make_trackable(stamp, state, DeviceId::RUNE)` 包一层即可）+ 云台反馈（当前 yaw/pitch、最大角速度/加速度）。

**aim() 主流程**：

```
1. fly_time = distance / config.bullet_speed        // 发射仓弹速（YAML 配置）
   increment = shoot_delay + 链路延迟               // 射击延迟 + 数据延迟
2. future = trackable.clone(); future->jump_into(fly_time + increment)
                                                   // 用正弦/线性模型外推到命中时刻
3. evaluate_blade(*future)：选 inactive 符叶 + idle/shoot 时序
   （rune_idle_duration / rune_shoot_duration 轮转）
4. 迭代 ≤5 次：attack = future 下的击打点
   → TrajectorySolution{v0=bullet_speed, point=attack}
   → 数值积分弹道（重力 9.81 + 空气阻力 0.003，5ms 步长，迭代求 pitch）
   → 新 fly_time → 收敛
5. 输出 Aimed { aim_yaw, pitch, shoot(是否开火), pre_aim, target(方向+ff_v/ff_a), center, attack }
```

**弹道模型**（`trajectory_solution.cpp`）：`estimate()` 对 `(vx,vy)` 做 5ms 步长数值积分（`vx -= c·v·vx·dt; vy -= (g + c·v·vy)·dt`），迭代求 pitch 使落点高度匹配目标；输出 `fly_time / yaw / pitch`。

---

## 第 9 层：输出与执行

- `Aimed.aim_yaw / pitch` → 云台角度指令（原项目由 `component.cpp` 的 gimbal 部分执行，本仓库不含）
- `Aimed.shoot` → 发射触发（结合 `ShootEvaluator` 的容差评估 yaw_tolerance=0.07/pitch_tolerance=0.04）
- `Aimed.target`（方向向量 + ff_v/ff_a）→ 云台运动前馈
- 调试信息（`Tracker::Addition`）：`rune_polygon`（R 标 + 5 符叶预测投影）、正弦参数文本 `spd(t)=v+a·sin(ωt+φ), e=...`、`/tmp/rune_sine_dump.csv`（EKF 角 vs 模型角 vs 拟合参数 dump）

---

## 纵向对照：两条实现路径每层对应关系

| 层 | 路径 A（deepstream 可视化） | 路径 B（rmcs 库 → 完整系统） |
|---|---|---|
| 解码 | GStreamer uridecodebin | cv::VideoCapture / 相机 |
| 预处理 | nvinfer 内部 | CUDA letterbox kernel |
| 推理 | nvinfer + engine | TensorRT C++ API + engine |
| 后处理 | 自定义解析插件 | RuneDetector 内联 |
| 关键点细化 | 无 | CUDA refine_kernel |
| 输出 | 屏幕画框/画点（无 3D） | RuneIcon/RuneBullseye |
| 3D | — | PnP（SingleRunePnpSolution） |
| 跟踪 | — | RuneModel EKF |
| 预测 | — | RuneEnergyFitter 正弦/线性 |
| 火控 | — | FireController + TrajectorySolution（原项目） |
| 用途 | 模型部署验证 / demo | 真机感知闭环 |

---

## 完整调用链（原项目视角，含本仓库部分）

```
component.cpp (ROS 节点)
 ├─ 相机采集 (hikcamera) ──> Detector ──> 2D 结果
 ├─ Tracker (tracker.cpp)
 │    ├─ store(RuneIcon/RuneBullseye/...)
 │    ├─ execute(timestamp)
 │    │    ├─ RuneModel::update_camera / update_transform   ← 相机参数注入点
 │    │    ├─ RuneModel::init / predict / correct / diverged
 │    │    └─ make_trackable(state, RUNE) ──> Trackable
 ├─ FireController (fire_control.cpp)
 │    ├─ update(云台反馈)
 │    └─ aim(trackable) ──> Aimed{yaw, pitch, shoot, ...}
 │         └─ TrajectorySolution{v0, attack} ──> fly_time/yaw/pitch
 └─ 云台 / 发射仓执行
```

## 数据流一句话版

> 符(物理几何+运动规则) → 相机(内参/畸变/外参标定) → 图像 → [解码 → letterbox → TRT FP16 推理 → 3类+5点后处理 → NMS → 细化] → RuneIcon/RuneBullseye → [PnP 物点匹配 → 6维状态初始化] → [EKF 预测/校正(匈牙利匹配)] → [线性/正弦转速拟合] → [外推 fly_time → 弹道解算(重力+阻力)] → 云台 yaw/pitch + 开火信号 → 弹丸命中 R 标
