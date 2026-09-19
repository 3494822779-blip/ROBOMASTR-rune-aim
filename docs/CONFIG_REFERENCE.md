# config 参数参考（CONFIG_REFERENCE）

> 所有可调参数集中在 `config/*.yaml`。
> **加载层次**：`config/template.yaml`（唯一参数源，公共默认）→ 场景 yaml（只写差异，覆盖）。
> 改公共参数只改 template.yaml 一处；启动日志打印 base/scene 与实际生效值。
> 加载器：`src/core/config_loader.cpp`；修改 yaml 后**无需重新编译**，重启 `rune_aim` 即可生效。

---

## 1. input —— 数据源

| 参数 | 默认 | 含义 |
|---|---|---|
| `mode` | `virtual` | 数据源：`video`（普通视频）/ `fixed_video`（固定外参视频，不使用 IMU）/ `virtual`（虚拟符，无 GPU）/ `camera`（真机相机） |
| `source` | 空 | video: 视频路径（如 `data/rune_test_h264.mp4`）；camera: 设备索引字符串（`"0"`） |
| `max_frames` | 0 | 最多处理帧数，0 = 不限 |
| `hz` | 200 | virtual 模式仿真帧率（Hz），调高减小诊断配对离散误差 |
| `video_fps` | 0 | 视频 PTS 缺失、重复或回退时的兜底帧率；正常文件按逐帧 PTS 支持可变帧率 |

## 2. camera —— 相机标定

| 参数 | 默认 | 含义 |
|---|---|---|
| `matrix` | 1400 演示内参 | 内参 3×3 行优先：`[fx, 0, cx, 0, fy, cy, 0, 0, 1]`。**实车必换标定值** |
| `distortion` | 全 0 | 畸变系数 `[k1, k2, p1, p2, k3]` |
| `transform` | 单位 | 相机→Odom 外参：`[t_x, t_y, t_z, q_x, q_y, q_z, q_w]`（平移 + 四元数）。真机来自标定/TF |
| `image_width` | 1440 | 画幅宽（px），须与 `cx` 对应 |
| `image_height` | 1080 | 画幅高（px），须与 `cy` 对应 |

> 影响：PnP 3D 解算与预瞄点投影。内参错 → 距离错；外参错 → 世界坐标整体偏。
> `image_width/height` 仅 virtual 模式使用：虚拟符的投影只判断点在相机前方，不判画幅，
> 据此裁掉落在画幅外的观测，对齐真实检测器"5 个关键点全在画幅内"的判据。

## 3. detect —— 检测（RuneDetector::Config）

| 参数 | 默认 | 含义 |
|---|---|---|
| `engine` | 空 | TensorRT 引擎路径（相对项目根或绝对） |
| `score_threshold` | 0.8 | 类别置信度阈值。调低 → 召回↑误检↑ |
| `keypoint_threshold` | 0.8 | 关键点置信度阈值（≥3 个有效点才保留目标） |
| `center_distance` | 30.0 | NMS 用关键点中心距离（px），重叠目标合并 |
| `refine_radius` | 10 | 符叶端点 GPU 细化的径向搜索半径（px） |
| `icon_refine_radius` | 14 | R 标细化的搜索半径（px） |
| `max_refine_shift` | 7.0 | 细化最大位移（px），超限回退网络原始点 |
| `min_refine_gradient` | 12.0 | 细化要求的最小梯度响应（过低视为证据不足） |
| `min_refine_support` | 6 | 细化要求的最小支撑点数 |

## 4. track —— EKF 跟踪（RuneModel::Config）

| 参数 | 默认 | 含义 |
|---|---|---|
| `noise_x/y/z` | 1e-5 | 符心位置的过程噪声（EKF 预测协方差增长） |
| `noise_rotation_angle` | 1e-3 | 旋转角过程噪声 |
| `noise_rotation_speed` | 1e-0 | 角速度过程噪声（角速度变化快的场景调大） |
| `noise_face_yaw` | 1e-5 | 符面朝向过程噪声 |
| `noise_observation` | 20.0 | 观测噪声（px²）。**虚拟模式收紧到 2**；真机按检测点抖动调：抖动大调大，跟踪迟钝调小 |
| `gate_threshold` | 13.816 | 观测关联门限（马氏距离²，6 自由度 99% 卡方值）。调小 → 关联更严格、易丢观测 |
| `init_seed_mean_error` | 10.0 | 初始化 PnP 重投影 RMS 误差上限（px），超过拒绝初始化 |
| `init_seed_max_error` | 20.0 | 初始化重投影最大误差上限（px） |
| `init_center_gate` | 30.0 | 初始化时符叶中心与预测的匹配门限（px） |
| `init_pitch_bound` | 20.0 | 初始化允许的符面俯仰角上限（degree） |
| `diverge_face_angle` | 45.0 | 发散判定：符面朝向与"相机→符心"视线夹角超过即认为发散重建 |

## 5. fire —— 火控（RuneFireControl::Config）

| 参数 | 默认 | 含义 |
|---|---|---|
| `bullet_speed` | 22.5 | ★ **弹速 m/s（发射仓标定）**。飞行时间 = 距离/弹速，直接决定预瞄提前量 |
| `shoot_delay` | 0.04 | ★ 扳机→出膛延迟（s），与 `bullet_speed` 一起决定预瞄量 |
| `algorithmic_delay` | 0.05 | 算法链路延迟（采集→解算→下发），用 `delay_calib` 标定 |
| `max_fly_time` | 1.0 | 飞行时间上限（s），超出视为弹道无解 |
| `pitch_max` | 0.61 | 云台俯仰上限（rad ≈ 35°），超出禁止开火 |
| `fire_cooldown_init` | 0.3 | 新目标 / 切叶确认后的初始冷却（s） |
| `fire_cooldown` | 0.7 | 连续开火窗口结束后的冷却（s） |
| `fire_window` | 0.04 | 连续开火窗口时长上限（s），窗口内可持续发射 |
| `data_life` | 0.2 | 目标数据寿命（s），无新观测超时后禁射并平滑回符心 |
| `recover_time` | 0.2 | 数据过期后从上次瞄准点平滑回到符心的时间（s） |
| `switch_angle` | 0.30 | 切叶检测：相位跳变阈值（rad ≈ 17°） |
| `switch_confirm` | 5 | 连续跳变帧数达到后确认切叶（重建初始冷却） |
| `offset_yaw` | 0.0 | 弹道机械偏置（rad），往左增（校准用） |
| `offset_pitch` | 0.0 | 弹道机械偏置（rad），往下增（校准用） |
| `max_iterate` | 5 | 弹道固定点迭代次数（外推→解弹道→收敛） |
| `iterate_epsilon` | 0.001 | 飞行时间收敛判据（s） |

## 6. virtual_rune —— 虚拟符（VirtualRuneModel::Config）

| 参数 | 默认 | 含义 |
|---|---|---|
| `large` | false | true=大符（正弦运动）/ false=小符（恒速） |
| `x/y/z` | 6.67/0/2.17 | 符心位置（Odom 系，米） |
| `face_yaw` | 0.0 | 符面朝向（rad） |
| `pixel_noise_px` | 0.0 | 关键点/角点高斯噪声 σ（px），模拟真实检测抖动 |
| `dropout_prob` | 0.0 | 每片符叶独立漏检/遮挡概率 0~1，模拟遮挡 |

> 用途：无 GPU 离线调参。加噪声/漏检评估算法鲁棒性（见 README 退化测试表）。

### 6.1 virtual_rune.gimbal —— 云台运动仿真

| 参数 | 默认 | 含义 |
|---|---|---|
| `enable` | false | true = 相机外参随时间摆动，而非固定为 `camera.transform` |
| `yaw_amp` / `yaw_freq` | 10.0 deg / 0.30 Hz | yaw 摆幅与频率（绕 Odom +z） |
| `pitch_amp` / `pitch_freq` | 3.0 deg / 0.17 Hz | pitch 摆幅与频率（绕 Odom +y） |
| `transform_delay` | 0.0 | 喂给 EKF 的外参滞后（s），模拟 IMU 回读延迟 |
| `transform_noise` | 0.0 | 喂给 EKF 的外参姿态噪声 σ（deg），模拟标定/量测误差 |
| `seed` | 7 | 噪声随机种子 |

> 用途：真机上 `camera.transform` 由 IMU 回读驱动、每帧变化，但工具里一直是常量，
> 这条链路在装相机之前无法验证。开启后 VirtualRune 用**真实**外参生成观测，
> RuneModel 收到的是**滞后 + 带噪**的那一份，两者之差即真机上的外参误差来源。
> 场景文件：`config/rune_gimbal_virtual.yaml`。

## 7. diag —— 诊断（RuneDiagnostics::Config）

| 参数 | 默认 | 含义 |
|---|---|---|
| `match_tolerance_ms` | 1.0 | 预测命中时刻与实测时刻的配对容差（ms） |
| `max_queue` | 1000 | 预测记录缓冲上限 |
| `max_history` | 10000 | 配对样本历史上限（CSV 导出 `/tmp/rune_diag.csv`） |

## 8. display —— 可视化调试

| 参数 | 默认 | 含义 |
|---|---|---|
| `enabled` | true | false = 纯终端运行（无窗口，适合性能测试/无人值守） |
| `keypoints` | true | 画检测关键点/符叶/R 标 + 激活类别文字 |
| `aimpoint` | true | 画外推到命中时刻的预瞄点（黄色圆） |
| `hitpoint` | false | 开火窗口上升沿登记一次，于预计命中时间最接近的视频帧画解算落点（红色 HIT） |
| `state_text` | true | 画火控状态机文字（COOLING/READY/FIRING…，开火红色闪烁） |
| `error_text` | true | 画诊断误差（mean/max/样本数） |

---

## 调参速查（常见目标 → 改哪里）

| 想解决 | 改 |
|---|---|
| 检测不到符 / 框乱跳 | `detect.score_threshold` ↓ / `detect.keypoint_threshold` ↑ |
| 跟踪跟不上/发散 | `track.noise_rotation_speed` ↑、`track.noise_observation` ↑ |
| 初始化总失败（init_ok=0） | `track.init_seed_mean_error` ↑、`track.init_center_gate` ↑、检查 `camera.matrix` |
| 弹丸打在目标后面 | `fire.shoot_delay` ↑ 或 `fire.bullet_speed` ↓（预瞄提前量增大） |
| 弹丸打超目标 | `fire.shoot_delay` ↓ 或 `fire.bullet_speed` ↑ |
| 开火太频繁/太稀 | `fire.fire_window` / `fire.fire_cooldown` 调 |
| 弹道系统性偏移 | `fire.offset_yaw` / `fire.offset_pitch` 校准 |
