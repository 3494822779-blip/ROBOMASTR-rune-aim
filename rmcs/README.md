# Rune DeepStream / RMCS 能量机关工程说明

## 1. 工程简介

本工程是一个面向 RoboMaster 能量机关的独立 C++/CUDA 解算库，提供从视频图像到云台瞄准和开火决策的完整链路：

```text
相机 BGR 图像
    │
    ▼
TensorRT 关键点检测 ── CUDA 预处理与局部精修
    │                  输出 R 标、符叶角点和激活类别
    ▼
RuneModel
    │  PnP 初始化、EKF 跟踪、转速估计、大符正弦拟合
    ▼
RuneFireControl
    │  命中时刻外推、弹道迭代、开火状态机
    ▼
yaw / pitch / fire / 前馈量
    │
    └── RuneDiagnostics：预测相位与后续观测闭环配对
```

工程主体已经剥离 ROS 2，可作为静态库 `librune_full.a` 集成到其他程序。当前顶层 CMake 构建的是能量机关独立链路，不包含 `kernel/detector.cpp` 中依赖旧装甲识别模块的统一检测器。

## 2. 功能组成

### 2.1 神经网络检测

`RuneDetector` 使用 TensorRT engine 完成三分类、五关键点检测：

- 类别 0：未激活符叶；
- 类别 1：小符激活符叶；
- 类别 2：大符激活符叶；
- 五个关键点：四个符叶角点和一个 R 标点；
- 输入张量固定为 `1 × 3 × 480 × 640`；
- 输出张量固定为 `1 × 18 × 6300`。

输入图像必须是非空的 `CV_8UC3` BGR 图像。图像在 GPU 上进行 letterbox 缩放、BGR→RGB、归一化及 CHW 排布。

检测输出经过：

1. 类别置信度筛选；
2. 关键点置信度及图像边界筛选；
3. 基于关键点中心距离的去重；
4. CUDA 局部梯度精修；
5. 转换为跟踪器使用的 `RuneIcon` 和 `RuneBullseye`。

### 2.2 CUDA 关键点精修

`RuneGpuPipeline` 负责：

- 将原始 BGR 图像异步上传到 GPU；
- TensorRT 输入预处理；
- 沿符叶径向搜索局部边缘；
- 对 R 标邻域进行梯度加权定位；
- 检查精修位移，证据不足时保留网络原始关键点。

精修接口中的数量单位是“目标数”，每个目标必须提供五个连续关键点。

### 2.3 跟踪与运动拟合

`RuneModel` 维护能量机关的三维位置、符面朝向、旋转角和旋转速度。

- 使用相机内参、畸变参数和相机到世界坐标变换完成初始化；
- 通过 EKF 的 `predict()` / `correct()` 完成逐帧跟踪；
- 小符采用恒速运动描述；
- 大符可使用正弦速度模型；
- `State::transition(seconds)` 可将状态外推到未来命中时刻；
- `State::get_aimpoints()` 返回当前可攻击符叶的三维瞄准点及角速度、角加速度前馈。

跟踪器具有收敛等待、观测门控和发散检测。应用层发现 `diverged()` 为真时，应停止射击并重新调用 `init()`。

### 2.4 火控

`RuneFireControl` 输入跟踪状态，输出：

- `found`：目标是否可用于控制；
- `fire`：当前是否允许开火；
- `yaw`、`pitch`：世界坐标系下的瞄准角；
- `fly_time`：弹丸预计飞行时间；
- `ff_v`、`ff_a`：射线角速度和角加速度前馈；
- `state`、`reason`：火控状态及调试原因。

状态机包含 `LOST`、`COOLING`、`READY`、`FIRING` 和 `RECOVERING`。它会处理初始冷却、连续开火窗口、切叶确认、数据过期和回符心过程。

### 2.5 预测诊断

`RuneDiagnostics` 将“预测命中时刻的相位”与该时刻附近的真实观测相位配对，统计：

- 最近一次相位误差；
- 平均绝对相位误差；
- 最大绝对相位误差；
- 已配对样本数。

它还可将完整样本写入 CSV，用于比较模型、参数或代码改动前后的效果。

## 3. 目录结构

```text
rmcs/
├── CMakeLists.txt                   构建定义与 CTest 用例
├── build.sh                         Release 构建脚本
├── app/
│   ├── virtual_test.cpp             虚拟小符/大符闭环测试
│   └── video_test.cpp               视频检测→跟踪→火控演示
├── test/
│   └── rune_video_test.cpp          TensorRT 检测性能测试
├── module/
│   ├── detector/rune.*              TensorRT 检测与后处理
│   ├── gpu/rune_gpu.*               CUDA 预处理和关键点精修
│   ├── tracker/model/               EKF、正弦拟合、虚拟能量机关
│   ├── fire_control/                弹道和火控状态机
│   ├── diagnostics/                 预测误差闭环诊断
│   └── refiner/                     CPU 关键点精修参考实现
├── utility/                         数学、坐标、PnP 和基础数据结构
├── kernel/                          更大 RMCS 工程的统一检测器接口
└── lib/                             build.sh 导出的静态库
```

## 4. 环境与依赖

最低构建要求：

- Linux，目标环境目前为 NVIDIA Jetson；
- CMake 3.22 或更高；
- 支持 C++23 的编译器，构建脚本默认 GCC/G++ 14；
- CUDA Toolkit 和 `nvcc`；
- TensorRT，需提供 `nvinfer`；
- OpenCV：`core`、`imgproc`、`calib3d`、`ximgproc`、`videoio`、`highgui`；
- Eigen3。

TensorRT engine 与 GPU 架构、TensorRT/CUDA 版本和构建参数通常相关。不要默认把其他设备生成的 engine 直接复制到当前设备使用。

## 5. 编译

### 5.1 使用构建脚本

```bash
cd /home/nvidia/rune_deepstream/rmcs
./build.sh
```

脚本会：

1. 自动查找常见路径下的 `nvcc`；
2. 使用 GCC/G++ 14 配置 Release 构建；
3. 并行编译所有目标；
4. 将 `build/librune_full.a` 复制到 `lib/librune_full.a`。

### 5.2 手动构建

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build -j2
```

主要产物：

- `build/librune_full.a`：能量机关静态库；
- `build/rune_video_test`：检测器性能测试；
- `build/rune_virtual_test`：虚拟闭环测试；
- `build/rune_video_aim`：视频完整链路演示。

静态库启用了 `POSITION_INDEPENDENT_CODE`，可继续链接进共享库。

## 6. 测试和运行

### 6.1 自动回归测试

```bash
ctest --test-dir build --output-on-failure
```

当前包含：

- `rune_virtual_small`：3 秒、100 Hz 小符闭环；
- `rune_virtual_large`：7 秒、100 Hz 大符闭环。

这两个测试验证跟踪、拟合、状态外推、弹道、火控和诊断链路，不需要 TensorRT engine 或正常工作的 GPU 推理环境。测试通过代表算法链路可运行，但不代表真实检测精度已经达标。

### 6.2 虚拟闭环测试

```bash
./build/rune_virtual_test [large] [seconds] [hz] [noise_px] [dropout]
```

参数：

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `large` | `0` 为小符，`1` 为大符 | `0` |
| `seconds` | 模拟持续时间 | `20` |
| `hz` | 模拟频率 | `200` |
| `noise_px` | 关键点高斯噪声标准差，像素 | `0` |
| `dropout` | 每片符叶独立漏检概率 | `0` |

示例：

```bash
./build/rune_virtual_test 1 20 200 1.5 0.05
```

程序每秒输出火控状态和误差统计，结束后把配对样本写入 `/tmp/rune_diag.csv`。

### 6.3 检测器性能测试

```bash
./build/rune_video_test \
  <engine> <video> [max_frames] [score_threshold] [keypoint_threshold]
```

示例：

```bash
./build/rune_video_test \
  /home/nvidia/rune_deepstream/model/Rune-v8n-fp16-20260624_b1_gpu0_fp16.engine \
  /home/nvidia/rune_deepstream/test/rune_test_h264.mp4 \
  300 0.8 0.8
```

输出帧数、检出帧数、目标数、平均单帧耗时和流水线 FPS。该耗时包括上传、预处理、TensorRT 推理、后处理和 CUDA 精修。

### 6.4 视频完整链路演示

```bash
./build/rune_video_aim \
  <engine> <video> [max_frames] [fx] [fy] [cx] [cy]
```

程序会显示检测点、激活类别、跟踪状态和预测瞄准点。按 `q` 或 `Esc` 退出。

默认相机参数仅用于演示：

```text
fx = 1400, fy = 1400, cx = 720, cy = 540
畸变参数 = 0
相机到世界变换 = 单位变换
```

实车使用时必须替换为实际标定结果。错误的内参或外参会直接影响 PnP、距离、弹道和最终命中精度。

## 7. 检测器配置

直接使用 `RuneDetector` 时，可设置以下参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `engine_path` | 空 | TensorRT engine 路径，必须设置 |
| `score_threshold` | `0.8` | 类别最低置信度 |
| `keypoint_threshold` | `0.8` | 单关键点有效阈值 |
| `center_distance` | `30` | 中心距离去重阈值，像素 |
| `refine_radius` | `10` | 符叶角点径向搜索半径 |
| `icon_refine_radius` | `14` | R 标精修邻域半径 |
| `max_refine_shift` | `7` | 允许的最大精修位移，像素 |
| `min_refine_gradient` | `12` | 精修最低梯度证据 |

通过 `kernel::Detector` 接入时，当前支持以下 YAML 项：

```yaml
rune_network:
  engine: /absolute/path/to/rune.engine
  score_threshold: 0.8
  keypoint_threshold: 0.8
  center_distance: 30.0
```

`engine` 不再提供机器相关的默认绝对路径，缺失时初始化会返回明确错误。注意：`kernel::Detector` 不是当前独立 CMake 目标的一部分；若在完整 RMCS 工程中启用它，还需提供 YAML-CPP 和旧装甲检测相关源文件及依赖。

## 8. 推荐集成流程

核心调用顺序如下：

```cpp
RuneDetector detector;
detector.config.engine_path = engine_path;
if (!detector.initialize()) {
    // 禁止进入推理循环，记录 engine/CUDA/TensorRT 错误
}

RuneModel model(RuneModel::Config{});
model.update_camera(camera_matrix, distortion);
model.update_transform(camera_to_odom);

RuneFireControl fire_control(RuneFireControl::Config{});
bool initialized = false;

// 每帧：
const auto now = Clock::now();
const auto elements = detector.detect(frame);

if (!initialized) {
    initialized = model.init(elements.icons, elements.bullseyes, now);
} else {
    model.predict(frame_dt, now);
    const bool corrected = model.correct(elements.icons, elements.bullseyes);
    (void)corrected;
}

if (initialized && !model.diverged()) {
    const auto command = fire_control.update(model.state(), now);
    // 将 command.yaw/pitch/ff_v/ff_a 下发给云台。
    // 只有 command.found && command.fire 时才允许触发发射机构。
} else if (model.diverged()) {
    fire_control.reset();
    initialized = false;
}
```

关键约束：

- 时间戳必须来自同一个单调时钟；
- `frame_dt` 应使用真实采集时间差，不能长期写死；
- 每帧先 `predict()`，再 `correct()`，最后调用火控；
- 相机内参、外参、弹速和链路延迟必须经过实测标定；
- 初始化失败或状态发散时必须禁射；
- `fire` 只是算法许可，实车层仍应保留设备状态和安全联锁。

## 9. 重要坐标和数据约定

- 跟踪状态位于 Odom/世界坐标系；
- 工程内部三维方向采用前 `x`、左 `y`、上 `z` 的约定；
- OpenCV 图像坐标为右 `u`、下 `v`；
- `RuneBullseye::corners` 顺序为上、左、下、右；
- R 标作为 `RuneIcon` 单独输出；
- 角度单位均为弧度；
- 长度单位为米；
- 时间单位为秒，诊断配对容差除外，其配置单位为毫秒。

## 10. 常见故障

### 10.1 TensorRT engine 初始化失败

检查：

1. engine 文件是否存在且可读；
2. engine 是否由当前 TensorRT/CUDA 和 GPU 环境生成；
3. engine 输入输出维度是否严格符合本工程固定约定；
4. CUDA 驱动是否正常；
5. 当前进程是否具有访问 GPU 的权限。

### 10.2 `CUDA initialization failure with error: 999`

如果同时出现：

```text
NvRmMemInitNvmap failed
NvRmMemMgrInit failed
```

通常表示 Jetson 驱动、`nvmap` 设备节点、容器设备映射或系统 CUDA 状态异常，而不是模型阈值或跟踪算法问题。建议依次检查：

```bash
ls -l /dev/nvmap /dev/nvhost* /dev/nvidia* 2>/dev/null
lsmod | rg 'nvidia|nvmap'
/usr/src/tensorrt/bin/trtexec --loadEngine=/path/to/rune.engine
```

若 `trtexec` 也无法创建 runtime，应先恢复系统 GPU 环境，再运行本工程的 GPU 回归测试。重启设备可用于确认是否是驱动临时失效，但不应代替设备节点、驱动版本和容器映射检查。

### 10.3 能检测但无法初始化跟踪

重点检查：

- 四角点和 R 标语义是否与模型导出一致；
- 相机标定是否对应当前分辨率；
- 外参方向是否正确；
- 检测结果是否包含足够的未激活符叶；
- 初始化像素误差门限是否过严；
- 输入画面是否存在镜像、旋转或裁切。

### 10.4 跟踪正常但命中偏差大

按以下顺序排查：

1. 相机内参和畸变；
2. 相机到云台/世界坐标外参；
3. 弹丸实测速度；
4. `algorithmic_delay` 和 `shoot_delay`；
5. 云台机械零偏；
6. 大符速度拟合收敛时间；
7. 关键点系统性偏差。

不要首先通过机械偏置掩盖相机外参或时间延迟错误。

## 11. 当前验证状态

截至 2026-08-28：

- Release 构建通过；
- CTest 小符闭环通过；
- CTest 大符闭环通过；
- 已修复 GPU 精修目标数/关键点数混淆导致的越界读取；
- 已加固 GPU 缓冲区扩容失败路径；
- 已缓存 TensorRT 张量名并复用逐帧后处理容器；
- 已删除统一检测器中的本机 engine 硬编码路径；
- 当前设备的真实视频 GPU 回归受 CUDA error 999 / `NvRmMem` 初始化故障阻断。

因此，当前可以确认 CPU 侧闭环算法与构建系统正常；真实视频的检测精度、GPU 性能和完整链路指标需要在 CUDA 环境恢复后重新测量。

## 12. 后续优化建议

建议按以下优先级继续：

1. 恢复 CUDA/TensorRT 环境，完成固定视频的稳定性和延迟基线；
2. 将推理输出主机缓冲改为页锁定内存，测量 D2H 收益；
3. 使用 CUDA Event 分离上传、预处理、推理、复制和精修耗时；
4. 建立标注视频集，记录召回率、关键点像素误差和类别混淆；
5. 用诊断 CSV 联合标定弹速、算法延迟和发弹延迟；
6. 增加噪声、遮挡、切叶和长时间运行回归测试；
7. 为 CUDA/TensorRT 错误增加结构化状态和限频日志，避免静默返回空检测；
8. 将尚未接入的精修参数统一到 YAML 或应用层配置。

任何算法参数调整都应同时比较检测指标、闭环相位误差和真实命中结果，避免仅针对单段视频过拟合。
