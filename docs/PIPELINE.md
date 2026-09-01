# 从观测到击打：能量机关全链路详解

> 适用对象：接手 `rune_aim/rmcs` 的队员。
> 平台：Jetson (aarch64) · TensorRT 10.3 · CUDA 12.x · C++23。
> 本文描述**当前代码实际实现**的完整链路（检测 → 位姿恢复 → EKF 跟踪 → 运动建模 → 外推 → 弹道 → 火控 → 诊断），所有参数均与源码/配置一致。

---

## 1. 链路总览

```
┌──────────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ 相机/视频  │ → │  RuneDetector │ → │    RuneModel      │ → │  RuneFireControl │
│ (BGR帧)   │   │  神经检测+细化  │   │ PnP初始化+EKF跟踪  │   │  弹道+开火状态机   │
└──────────┘   └──────────────┘   └──────────────────┘   └────────┬─────────┘
                                                                 │ yaw/pitch/fire
                                   ┌──────────────────┐          │ ff_v/ff_a
                                   │ RuneEnergyFitter  │          ▼
                                   │  ω精化+Cauchy拟合  │   ┌──────────────┐
                                   │   （RuneModel内）  │   │ 下位机/云台/电控 │
                                   └──────────────────┘   └──────────────┘
                                   ┌──────────────────┐
                                   │ RuneDiagnostics   │  ← 预测 vs 实测误差闭环
                                   │  (1ms 配对评估)    │
                                   └──────────────────┘
```

**核心设计原则**（继承自 rmcs_auto_aim_v2）：检测器只输出"谁在哪儿"，**位姿与运动由几何模型（PnP/EKF）与物理模型（正弦/弹道）决定**；`RuneModel` 在初始化后不再依赖每帧 PnP，只用像素观测做 EKF 校正——这是链路低延迟、高稳定的关键。

数据流一句话：`图像 → 5 关键点+3 类 → 多候选 PnP 恢复 6 维初值 → 6 维 EKF 持续跟踪 → 正弦/线性能量拟合 → 命中时刻外推 → 弹道解算 → 状态机开火 → 误差诊断回流`。

---

## 2. 环节一：观测与神经检测（RuneDetector）

**文件**：`module/detector/rune.cpp` + `module/gpu/rune_gpu.cu`（CUDA 预处理/细化）

### 2.1 模型与输出

- 网络：YOLOv8-pose 风格，输入 `[1,3,480,640]`（RGB，0~1 归一化），输出 `[1,18,6300]`
- 18 通道 = 3 类置信度（`0=未激活 / 1=小符已激活 / 2=大符已激活`）+ 5 关键点 × 3（x,y,score）
- 关键点顺序：`0=上 1=左 2=R标 3=右 4=下`；6300 = 3 个尺度锚点（stride 8/16/32）
- 引擎：`model/Rune-v8n-fp16-*.engine`（FP16，TensorRT 10.3）

### 2.2 处理流程（每帧）

```
BGR 帧 ──(CUDA)──▶ letterbox 等比缩放+居中填充到 640×480 ──▶ TensorRT 推理
  ──▶ 候选解析：每个锚点取 3 类最大得分，≥ score_threshold(0.8) 保留
  ──▶ 关键点校验：有效点(score≥0.8) ≥ 3 且坐标在图内才保留
  ──▶ 质量分 = 类别分 × 有效点得分均值，降序排序
  ──▶ NMS：按关键点中心距离（center_distance=30px）抑制重复框
  ──▶ (CUDA) 局部细化：每个关键点在邻域内按梯度精修（refine_radius=10, icon=14）
  ──▶ 输出映射
```

### 2.3 输出映射（与跟踪器的契约）

| 网络输出 | 映射为 |
| --- | --- |
| 点 2（R 标） | `RuneIcon.center`（带 score） |
| 点 0,1,4,3 | `RuneBullseye.corners`（top,left,bottom,right） |
| 四点均值 | `RuneBullseye.center` |
| 类 0 / 1 / 2 | `Activation = Inactive / SmallActive / BigActive`（P1-4 升级：不再合并为 bool） |

### 2.4 关键参数（`RuneDetector::Config`）

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `score_threshold` | 0.8 | 类别置信度阈值 |
| `keypoint_threshold` | 0.8 | 关键点置信度阈值 |
| `center_distance` | 30.0 | NMS 关键点中心距离（px） |
| `refine_radius` / `icon_refine_radius` | 10 / 14 | 普通点/R 标的细化邻域半径（px） |
| `max_refine_shift` | 7.0 | 细化最大偏移（px），超过则保留网络原始点 |
| `min_refine_gradient` | 12.0 | 细化所需最小梯度 |
| `min_refine_support` | 6 | 细化所需最小支撑像素 |

---

## 3. 环节二：初始化位姿恢复（PnP 多候选投票）

**文件**：`module/tracker/model/rune.cpp`（`init()`）+ `utility/math/solve_pnp/pnp_solution.cpp`

### 3.1 时机

首次检测到 **1~2 个未激活靶心 + 至少 1 个 R 标** 时触发（之后不再重复，除非发散重锚）。

### 3.2 流程

```
对每个 (R标 × 未激活靶心) 候选：
  1. 5 点 PnP：EPNP 初解 → ITERATIVE 精化
     3D 点：kIcon(-0.1,0,0), kT(0,0,0.85), kL(0,0.15,0.7), kB(0,0,0.55), kR(0,-0.15,0.7)
     2D 点：R 标中心 + 靶心 4 角点（按"距 R 标最近=底、最远=顶"排序）
  2. 重投影校验：5 点 RMS ≤ 10px 且最大误差 ≤ 20px，否则丢弃
  3. 几何校验：符面法线俯仰 ≤ 20° 且法线朝向相机
  4. 布局投票：用候选位姿重投影全部观测——
     · R 标投影误差计分
     · 种子符叶中心误差 ≤ 30px
     · 其余未激活靶心与 5 片符叶最近匹配，30px 内算内点
  5. 多候选择优：内点数多 → seed 中心误差小 → R 标误差小 → 最大误差小 → SSE 小
```

**准度意义**：初始化认错符叶（72° 级错误）是能量机关最致命的误差来源，布局投票用"整片符叶的相对位置"做一致性校验，从源头杜绝。

### 3.3 初始化后的首次校正

`init()` 成功后用 R 标像素与种子符叶像素各做一次 EKF 更新；状态初值协方差 `diag(64,64,64,100,25,10)`（位置/速度/角度/朝向均不确定），由后续观测快速收敛。

---

## 4. 环节三：EKF 跟踪（RuneModel）

**文件**：`module/tracker/model/rune.cpp`（1053 行核心）

### 4.1 状态与模型

状态向量 6 维（Odom/世界系）：

```
x = [ x, y, z,          ω(旋转角速度), θ(旋转角), ψ(符面朝向yaw) ]
     符心位置            转速          相位        符面法线
```

- 预测：`θ += ω·dt`，`ψ` 归一化；雅可比含 `∂θ/∂ω = dt`
- 观测模型：把 6 维状态投影为 6 个特征点的像素坐标（R 标 + 5 片符叶中心，符叶位置由 `θ` 与 72° 叶角决定），针孔投影 + 相机外参，解析雅可比

### 4.2 数据关联（马氏门控 + 匈牙利）

每帧 `correct()`：

```
观测集合：R 标中心（若干）+ 靶心中心（若干，含激活/未激活）
1. 计算每个观测到每个预测特征的马氏距离²（S = HPHᵀ + R）
2. 约束：R 标只能匹配特征 0，靶心只能匹配符叶 1~5
3. 匈牙利算法最优分配，门限 gate_threshold = 13.816（χ²₂ 的 99.9% 分位）
4. 分配后按最新状态重算新息协方差二次门控
5. 顺序 EKF 更新（Joseph 形式协方差），ψ 归一化
```

**准度意义**：遮挡/误检产生多余观测时，门控保证**只吸收统计上可信的观测**，离群观测不会污染状态。

### 4.3 收敛与发散

- **收敛**（P1-3 升级后生效）：`converge()` = 更新次数 ≥ 10 且 位置协方差 < 1.0 m² 且 朝向协方差 < 0.002 rad² 且初始化后 ≥ 0.1s
- **发散**（`diverged()`）：协方差 > 150、位置超界（15/15/5 m）、转速 > 10π、符面法线与车心连线夹角 > 45°、NaN——任一触发即判发散，调用方应重新 `init()`（自动重锚）

### 4.4 关键参数（`RuneModel::Config`）

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `noise_x/y/z` | 1e-5 | 位置过程噪声 |
| `noise_rotation_speed` | 1e0 | 转速过程噪声 |
| `noise_rotation_angle` | 1e-3 | 相位过程噪声 |
| `noise_face_yaw` | 1e-5 | 朝向过程噪声 |
| `noise_observation` | 20.0 | 像素观测噪声方差（px²） |
| `gate_threshold` | 13.816 | 马氏距离²门限（χ²₂ 99.9%） |
| `init_seed_mean_error` / `max_error` | 10 / 20 px | PnP 初始化重投影门限 |
| `init_center_gate` | 30 px | 布局投票门限 |
| `init_pitch_bound` | 20° | 初始化俯仰上限 |
| `diverge_face_angle` | 45° | 发散判据：法线夹角 |

---

## 5. 环节四：旋转运动建模（RuneEnergyFitter）

**文件**：`module/tracker/model/rune_energy_fitter.cpp`

### 5.1 模型形式

- **线性**（小符/回退）：`θ(t) = C + v·t`
- **正弦**（大符，规则模型）：`θ(t) = C + v·t − (a/ω)·cos(ωt+φ)`，即 `ω(t) = v + a·sin(ωt+φ)`

规则物理先验：`ω ∈ [1.80, 2.20] rad/s`、`a ∈ [0.6, 1.2]`（实际大符范围 0.78~1.045）。

### 5.2 拟合方法（P0 升级后）

```
数据：EKF 平滑后的展开角 θ（窗口 6s，最短 1.5s 才拟合，指数时间权重半衰期 3s）

对每个候选 ω：
  内层 IRLS（3 次迭代）：
    X = [1, t, cos(ωt), sin(ωt)]
    权重 = 时间权重 × Cauchy 残差权重（w = 1/(1+(r/2.5·RMS)²)）
    加权最小二乘（列主元 QR）→ 残差 → 更新权重
  → 加权 MSE 作为该 ω 的代价

阶段 1：粗扫 41 步（ω ∈ [1.80, 2.20]）
阶段 2：最优邻域黄金分割精化（分辨率 0.01 → ~1e-4 rad/s）

还原：a = ω·√(A²+B²)，φ = atan2(B, −A)
```

### 5.3 模型选择（在 rune.cpp 的 `correct()` 内）

```
每帧同时拟合线性与正弦：
  选用正弦 ⇔ 存在正弦解 且（无线性解 ∨ 强制正弦期 ∨（正弦 cost < 线性 cost 且 a ≥ 0.6））
  · a ≥ 0.6：振幅显著，避免把匀速段误报为正弦
  · 强制正弦：一帧内纠正 >1 个未激活靶心（判为符叶轮转）后 3 秒内强制用正弦——
    切换瞬间线性外推必然失真，正弦的周期性可跨符叶延续
回退：仅线性可用时用线性斜率 v 作为预测转速
```

**准度意义**：Cauchy 权重让切叶/遮挡产生的离群相位自动降权（P0-2）；ω 精化消除 0.01 rad/s 的网格量化误差——0.5s 外推的相位误差因此降低约一半（实测 0.0041 → 0.0019 rad）。

---

## 6. 环节五：命中预测外推（State::transition）

**文件**：`module/tracker/model/rune.cpp`（`State::transition`）

把状态推进 `dt` 秒（火控在"当前时刻 + 延迟 + 飞行时间"处求瞄准点）：

```
正弦有效：θ += v·dt + (a/ω)·(cos(φ₀) − cos(φ₀+ω·dt))   ← 正弦速度的解析积分（精确）
          ω(t) = v + a·sin(φ₀+ω·dt)
否则    ：θ += ω·dt
```

外推后的瞄准点由 `get_aimpoints()` 给出：取**第一个未激活符叶**的世界系位置（符心 + 0.7m 半径 × 相位角，符面法线 ψ 旋转），同时解析给出该点的**射线角速度/角加速度前馈** `ff_v/ff_a`（供云台动态跟踪）。

> 收敛门控：初始化后 3s（恒速/线性）或 6s（正弦）内 `get_aimpoints()` 返回空——期间火控只瞄符心、禁射，避免模型未收敛就开火。

---

## 7. 环节六：弹道解算（TrajectorySolution）

**文件**：`module/fire_control/trajectory_solution.cpp`（吸纳自 rmcs_auto_aim_v2）

- 物理模型：重力 g=9.81 + **平方空气阻力**（系数 0.003），`dt=5ms` 数值积分
- 求解：俯仰角迭代（最多 10 次，高度误差 < 1mm 收敛，俯仰上限 80°）
- 输出：`[fly_time, yaw, pitch]`（Odom 系射线角）

```
输入：弹速 v0 + 攻击点世界坐标（外推后的符叶位置）
输出：飞行时间 t_f、云台 yaw、云台 pitch
```

---

## 8. 环节七：火控决策（RuneFireControl）

**文件**：`module/fire_control/rune_fire_control.cpp`（新增，RP RuneDecisionModule 状态机设计）

### 8.1 每帧流程

```
1. 时间步进：dt = now − last（钳制 ≤ 0.5s 防暂停跳变）
2. 数据新鲜度：RuneModel 的 update_count 变化 → 新鲜（数据年龄清零）；否则年龄累加
3. 切叶检测：|Δθ| > 0.30 rad 计一次跳变；连续 ≥ 5 帧确认切叶 → 重置初始冷却
4. 瞄准+弹道（固定点迭代，最多 5 轮）：
   t_f₀ = 距离/弹速
   每轮：克隆状态 → transition(algorithmic_delay + shoot_delay + t_f)
        → get_aimpoints() 取第一个未激活符叶（无则瞄符心、禁射）
        → TrajectorySolution 解出新 t_f/yaw/pitch
   |Δ t_f| < 1ms 收敛
5. 状态机推进（见 8.2）
6. 输出：yaw/pitch/fire/fly_time/ff_v/ff_a + 状态与原因
```

### 8.2 开火状态机

```
                ┌────────────────────────────────────────────────────┐
                ▼                                                    │
 LOST ──新目标──▶ COOLING(初始冷却0.3s) ──冷却结束──▶ READY            │
                ▲                                    │              │
                │                             命中窗口条件满足         │
                │                                    ▼              │
       数据恢复：重新初始冷却              FIRING(连续开火窗口0.04s)     │
                ▲                                    │              │
                │                              窗口用尽             │
         数据过期>0.2s：                            ▼              │
       平滑回符心(recover_time 0.2s) ◀── COOLING(冷却0.7s) ──────────┘
```

**开火条件**（READY/FIRING 且）：目标新鲜 ∧ 有未激活符叶瞄准点 ∧ |pitch| ≤ 0.61 rad ∧ t_f ≤ 1.0s ∧ 弹道求解成功。

**数据过期保护**：无新观测 > 0.2s → 禁射，并在 0.2s 内从当前指向**线性插值回符心**（防云台突跳）；恢复观测 → 重新初始冷却。

### 8.3 关键参数（`RuneFireControl::Config`）

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `bullet_speed` | 22.5 m/s | 弹丸初速（实车按电控实测改） |
| `shoot_delay` | 0.04 s | 发弹延迟（扳机→出膛） |
| `algorithmic_delay` | 0.05 s | 算法链路延迟（采集→解算→下发） |
| `max_fly_time` | 1.0 s | 飞行时间上限 |
| `pitch_max` | 0.61 rad | 云台俯仰上限（≈35°） |
| `fire_cooldown_init` | 0.3 s | 新目标/切叶初始冷却 |
| `fire_cooldown` | 0.7 s | 连发窗口后冷却 |
| `fire_window` | 0.04 s | 连续开火窗口上限 |
| `data_life` | 0.2 s | 目标数据寿命 |
| `recover_time` | 0.2 s | 平滑回符心时长 |
| `switch_angle` | 0.30 rad | 切叶跳变阈值 |
| `switch_confirm` | 5 帧 | 切叶确认帧数 |
| `max_iterate` / `iterate_epsilon` | 5 / 0.001 s | 固定点迭代 |
| `offset_yaw/pitch` | 0 | 机械偏置 |

---

## 9. 环节八：误差诊断（RuneDiagnostics）

**文件**：`module/diagnostics/rune_diagnostics.cpp`（新增，RP PowerRuneDiagnostics 设计）

**核心思想**：火控求解成功时记录「预测命中时刻 + 预测相位」；每帧记录实测相位（EKF 校正后的 `rotation_angle`）；当真实观测时刻到达命中时刻（±1ms 容差）时配对，误差 = wrap(实测 − 预测)。

```
push_predict(t_hit, θ_pred)      ← 火控每次 fire 时（虚拟符模式）
push_observation(t, θ_obs)       ← 每帧 EKF 校正后
→ stats(): latest/mean/max 误差 + 样本数
→ dump_csv(): /tmp/rune_diag.csv（配对样本全量落盘）
```

**这是全链路准度的"仪表盘"**：任何环节改动（模型、拟合、延迟参数）都应在真值闭环下观察误差曲线的变化。

---

## 9.5 链路延迟标定（DelayCalibrator）

**文件**：`module/diagnostics/delay_calibrator.{hpp,cpp}`（吸纳自 Climber serial_delay 核心算法）

**用途**：实测 `RuneFireControl::Config.algorithmic_delay`（算法链路延迟）——这是弹道/火控里唯一不能靠打靶测、只能靠信号相关性测的参数。

**原理**：云台 IMU 反馈与视觉目标是同一运动的两种观测，前者先于后者（曝光+检测+解算+下发+执行延迟）。两路信号等间隔重采样（100Hz）后做**滑动互相关**，相关系数最大的位移即链路总延迟：

```
每帧 push({ 检测结束时刻, 目标像素位移(相对图像中心), IMU yaw/pitch(度) })
→ 门槛：像素跨度 ≥40/30px、云台角跨度 ≥2.0/1.5°、|corr| > 0.3、搜索 ±3s
→ estimate() → { yaw_ms, pitch_ms, corr, hit_edge }
→ 稳定后的延迟（两轴均值）填入 algorithmic_delay
```

**离线自检**（已内置 CTest）：

```bash
./build/rune_delay_calib_test [delay_ms=120] [noise_px=2] [seconds=30]
# 合成已知延迟信号对，验证互相关恢复精度（实测四组场景均 0.0ms 误差）
```

**实车标定步骤**：
1. 相机对准装甲板/符，运行检测，记录最大目标中心相对图像中心的像素位移
2. 同一时刻读云台 IMU 姿态（yaw/pitch）
3. 手动摆动云台让目标在画面中大幅移动（跨度达标），观察 `estimate()`
4. 稳定值即链路总延迟 → 填入 `algorithmic_delay`（该值含曝光+检测+解算+下发+执行+IMU 反馈全程）

> 注意：与 `algorithmic_delay` 不同，`shoot_delay`（扳机→出膛）无法用互相关测，需高速相机或电控时间戳。

---

## 10. 坐标系与时间基准

| 项 | 约定 |
| --- | --- |
| 相机系 | ROS 约定：前 x、左 y、上 z（`ros2opencv_position` 转换投影） |
| 世界系（Odom） | `RuneModel` 状态所在系；`update_transform(Transform)` 提供相机→世界外参 |
| 瞄准输出 | Odom 系射线角 yaw/pitch（与 rmcs_auto_aim_v2 一致） |
| 时间 | `steady_clock`；`Timestamp` 贯穿检测/跟踪/火控/诊断 |
| 相位 θ | **展开角**（不归一化），天然连续，供拟合器直接拟合；ψ 单独归一化 |

**虚拟符测试**：符心 (6.67, 0, 2.172) m、法线 +x、单位外参（相机=世界）——视线与法线夹角 ≈18°，近似正视角。

---

## 11. 一帧处理时序（主程序侧）

```
now = Clock::now()
│
├─ 1. elements = detector.detect(frame)          // 检测（TRT + CUDA 细化）
├─ 2. 若未初始化：model.init(icons, bullseyes, now)
├─ 3. model.predict(dt, now)                     // EKF 预测
├─ 4. model.correct(icons, bullseyes)            // EKF 校正（含拟合器更新）
├─ 5. state = model.state()
├─ 6. diag.push_observation(now, state.rotation_angle)
├─ 7. cmd = fire_control.update(state, now)      // 弹道 + 状态机
├─ 8. 若 cmd.fire：diag.push_predict(t_hit, θ_pred)
└─ 9. 下发 cmd.yaw/pitch/fire/ff_v/ff_a（实车：串口/ROS2）
```

> 完整代码骨架见 `app/virtual_test.cpp`（真值闭环）与 `app/video_test.cpp`（视频回放）。

---

## 12. 验证结果（真值闭环）

`./build/rune_virtual_test [large] [seconds] [hz] [noise_px] [dropout]`

| 场景 | 模式 | mean | max |
| --- | --- | --- | --- |
| 无退化（纯算法误差） | 大符 | 0.0021 rad (0.1°) | 0.0087 rad (0.5°) |
| 无退化 | 小符 | 0.0024 rad (0.1°) | 0.0048 rad (0.3°) |
| 轻度退化（1.5px + 10% 漏检） | 大符 | 0.0080 rad (0.5°) | 0.0202 rad (1.2°) |
| 重度退化（3px + 20% 漏检） | 大符 | 0.0204 rad (1.2°) | 0.0635 rad (3.6°) |

重度退化（模拟真实赛场噪声强度）下 0.020 rad ≈ RP-26Rune 宣称的 0.02 rad（真实环境）——链路精度已达一线水平。

---

## 13. 已知限制与后续工作

1. **实车外参**：`update_transform` 目前为单位变换（测试）；实车需 IMU 姿态 + 手眼标定结果接入。
2. **大小符自动判别**：检测器已输出 `Activation` 类别（P1-4），但火控侧仍按配置决定大小符模型——后续可让 `RuneModel` 根据观测类别自动切换。
3. **弹道模型**：TrajectorySolution 为平方阻力 + 迭代解，若追求极限（远距离/高速弹）可升级为 RK4+Ceres 联合求解（RP 方案），依赖 Ceres。
4. **串口/ROS2 下发**：`Command` 输出已就绪，下位机协议（C 板/云台串口）未接入。
5. **前馈闭环**：`ff_v/ff_a` 已透出，实车云台控制若支持前馈应直接使用（rmcs 控制栈已有对应接口）。
6. **拟合器参数**：窗口 6s/半衰期 3s/Cauchy 尺度 2.5 为虚拟符与规则参数下的经验值，实车噪声特性不同时应回归测试调整。
