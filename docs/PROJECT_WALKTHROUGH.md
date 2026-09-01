> ⚠️ **2026-09-01 重构说明**：项目已重构为新结构（src/{core,detect,track,fire,diag,debug} + config/ + tools/）。
> 本文中的旧路径 `rmcs/module/...` / `rmcs/utility/...` 已失效，对应新位置见 `README.md` 目录结构；
> 参数配置以 `config/*.yaml`（统一入口 `./build/rune_aim -c <yaml>`）为准。

# rune_deepstream 项目全解析（梳理文档）

> 适用对象：对项目完全不了解的新成员。
> 平台：Jetson（aarch64）· Ubuntu 22.04 · DeepStream 7.x · TensorRT 10.3 · CUDA 12.x · ROS 2 Humble

---

## 0. 一句话总结

这是一个 **RoboMaster 能量机关（符）神经检测 + 跟踪**项目。仓库里其实是**两条互相独立的实现路径**：

| 路径 | 位置 | 内容 | 用途 |
|---|---|---|---|
| A. DeepStream 可视化应用 | `deepstream/` | GStreamer 管线 + nvinfer + 自定义输出解析插件，把检测结果画在画面上 | 模型部署验证 / 演示 |
| B. RMCS C++ 静态库 | `rmcs/` | 直接调 TensorRT 的检测器 + EKF 跟踪器 + 转速拟合，产物 `lib/librune_full.a` | 给机器人主程序集成 |

两者**互不调用**：A 用 DeepStream 的 nvinfer 元素做推理，B 用 TensorRT C++ API 自己做推理；它们各自都有完整的"推理 → 后处理"逻辑。README 里的两条命令对应这两条路径。

---

## 1. 背景：RoboMaster 能量机关（符）是什么

- 能量机关是一块立着的圆形"符面"，上面有 5 片符叶（间隔 72°），符面中央偏上有一个突起的 **R 标**。
- 符叶会绕符面中心**旋转**；大符转速是正弦规律，小符近似恒速。
- 打击目标：在符叶旋转到"待击打位"（inactive）时，把弹丸打到 **R 标中心**。
- 因此感知需要：
  1. 检测出符的 **5 个关键点**：上、左、R、右、下（十字形端点 + R 标）；
  2. 分类：**未激活 / 小符已激活 / 大符已激活**（激活与否决定能不能打）；
  3. 跟踪预测：估计符心 3D 位置、符面朝向、旋转角度/角速度，从而预测击打点。

本项目的检测部分改用神经网络（YOLOv8-pose 风格），跟踪/预测部分保留了原来 RMCS 的 EKF + 曲线拟合方案。

---

## 2. 目录结构总览

```
rune_deepstream/
├── README.md            # 顶层说明（路径正确，以它为准）
├── model/               # 模型文件
│   ├── Rune-v8n-fp16-20260624.onnx          # 11.7 MB，YOLOv8 风格
│   └── Rune-v8n-fp16-20260624_b1_gpu0_fp16.engine  # 8.3 MB，TensorRT 10.3 FP16 引擎
├── deepstream/          # 路径 A：可视化应用
│   ├── bin/deepstream               # 已编译好的可执行程序（aarch64）
│   ├── config/
│   │   ├── config_infer_primary_rune_pose.txt  # nvinfer 推理配置
│   │   └── labels_rune.txt                    # 类别标签（中文）
│   ├── plugin/
│   │   ├── nvdsparsepose_Rune.cpp             # 自定义输出解析插件源码
│   │   ├── libnvdsinfer_custom_impl_Rune_pose.so  # 编译好的插件
│   │   └── Makefile
│   ├── src/deepstream.c            # 主程序（GStreamer 管线）
│   ├── src/modules/                # 中断处理 + 性能统计
│   └── Makefile                    # ⚠️ 里面的 plugin 目标路径是坏的（见 §4.6）
├── rmcs/                # 路径 B：C++ 检测/跟踪库
│   ├── build.sh                    # 一键构建脚本
│   ├── CMakeLists.txt              # 只编译 rune 相关 6 个源文件
│   ├── kernel/                     # ⚠️ 遗留代码，不参与构建
│   ├── module/
│   │   ├── detector/rune.{hpp,cpp}     # RuneDetector：TensorRT 检测器
│   │   ├── gpu/rune_gpu.{hpp,cu}       # CUDA：预处理 + 关键点细化
│   │   ├── refiner/                    # ⚠️ CPU 版细化，已不参与构建
│   │   └── tracker/model/
│   │       ├── rune.{hpp,cpp}          # RuneModel：EKF 跟踪器（核心，1053 行）
│   │       ├── rune_energy_fitter.*    # 线性/正弦转速拟合
│   │       └── virtual_rune.*          # 虚拟符仿真器（按规则生成观测）
│   ├── test/rune_video_test.cpp  # 检测器性能测试程序
│   ├── lib/librune_full.a        # 构建产物（228 KB）
│   └── utility/                  # 头文件工具库（数学/图像/日志/ROS 适配等）
├── test/
│   ├── rune_test_h264.mp4        # 测试视频（1440×1080, 50fps, 800 帧）
│   └── rune_test.avi             # 符号链接 → 外部目录（可能失效，不用管）
└── docs/
    ├── PROJECT_README.md         # 旧笔记（路径是 /home/nvidia/project/...，已过时）
    ├── NEURAL_RUNE_INTEGRATION.md# 神经检测器集成说明（输出映射、配置项）
    └── PROJECT_WALKTHROUGH.md    # 本文档
```

---

## 3. 模型（model/）

### 3.1 网络结构（从 ONNX 元数据验证）

- 输入：`images`，形状 `[1, 3, 480, 640]`（NCHW，RGB，0~1 归一化）。
- 输出：`output0`，形状 `[1, 18, 6300]`。
- 层名含 `model.22 / cv3 / cv4` 等，是标准 YOLOv8 检测头；
  6300 = 4800（stride 8）+ 1200（stride 16）+ 300（stride 32），即 3 个尺度的锚点总数。

### 3.2 输出通道含义（18 = 3 + 5×3）

```
通道 0~2    ：3 个类别的置信度（未激活 / 小符已激活 / 大符已激活）
通道 3~5    ：关键点 0（上）  的 x, y, score
通道 6~8    ：关键点 1（左）  的 x, y, score
通道 9~11   ：关键点 2（R 标）的 x, y, score
通道 12~14  ：关键点 3（右）  的 x, y, score
通道 15~17  ：关键点 4（下）  的 x, y, score
```

内存布局按**列主序（column-major）**：`output[c * 6300 + n]` 是第 c 通道第 n 个锚点的值。
坐标是相对 640×480 输入图的像素坐标。

### 3.3 引擎

- `.engine` 由上述 ONNX 用 TensorRT 10.3、FP16、batch=1 导出（文件名 `b1_gpu0_fp16`，引擎文件头 `TR~` + 版本 10.3，与系统安装的 `libnvinfer 10.3.0` 匹配，可直接加载）。
- 引擎里确实包含张量名 `images` / `output0` 和 6300 维输出，确认就是本 ONNX 转出来的。

---

## 4. 路径 A：DeepStream 可视化应用

### 4.1 运行命令（README 原文）

```bash
cd /home/nvidia/rune_deepstream/deepstream/config
../bin/deepstream \
  -s file:///home/nvidia/rune_deepstream/test/rune_test_h264.mp4 \
  -c config_infer_primary_rune_pose.txt
```

程序弹出窗口播放测试视频，画上检测框、5 个关键点圆圈和骨架连线。需要图形会话（DISPLAY）。

### 4.2 GStreamer 管线（src/deepstream.c 的 main）

```
uridecodebin ──> nvstreammux ──> nvinfer ──> nvvideoconvert ──> capsfilter(RGBA) ──> nvdsosd ──> sink
   (解码视频)      (批处理/缩放)   (推理)       (转 RGBA)          (格式约束)        (画框画字)   (nv3dsink Jetson / nveglglessink)
```

- `uridecodebin`：自动解码；Jetson 上强制用 `nvv4l2decoder`（硬件解码）。
- `nvstreammux`：batch-size=1，输出 1920×1080；`live-source` 对文件源关掉。
- `nvinfer`：加载 `config_infer_primary_rune_pose.txt` 描述的模型做推理。
- `nvdsosd`：OSD 绘制，`process-mode` GPU。
- 主循环前的 `gst_element_seek(pipeline, 0.75, ...)`：**以 0.75 倍速播放**，保证每帧都参与推理。
- 命令行参数：`-s` 源、`-c` 推理配置、`-b/-w/-e/-g` 批大小/宽/高/GPU 号。

### 4.3 推理配置 config_infer_primary_rune_pose.txt 关键项

| 配置项 | 值 | 含义 |
|---|---|---|
| `onnx-file` / `model-engine-file` | `../../model/...` | 相对 config 目录的路径，引擎优先 |
| `batch-size` | 1 | 单帧推理 |
| `network-mode` | 2 | FP16 |
| `num-detected-classes` | 3 | 3 个类别 |
| `net-scale-factor` | 0.00392156979… | 即 1/255，像素归一化 |
| `model-color-format` | 0 | RGB 输入 |
| `network-type` | 3 | YOLO 型输出 |
| `parse-bbox-instance-mask-func-name` | `NvDsInferParseRunePose` | 自定义解析函数 |
| `custom-lib-path` | `../plugin/libnvdsinfer_custom_impl_Rune_pose.so` | 插件动态库 |
| `output-instance-mask` | 1 | 把网络原始输出作为 instance mask 挂到 obj_meta |
| `pre-cluster-threshold` | 0.8 | 类内预聚类阈值 |

### 4.4 自定义解析插件 plugin/nvdsparsepose_Rune.cpp

nvinfer 推理完后不会自动理解 YOLO 输出，需要这个插件把 `[18, N]` 原始张量解析成目标框：

1. 遍历每个锚点 n：取 3 个类别分数最大值 → `class_id` + `confidence`；低于阈值（默认 0.8，来自 pre-cluster-threshold）丢弃。
2. 读 5 个关键点 (x, y, score)，score ≥ 0.8 的算"有效点"；有效点 < 3 丢弃。
3. 用有效点包围盒外扩 15% 作为检测框（`left/top/width/height`）。
4. 最终置信度 = 类别分数 × 关键点分数均值。
5. 把关键点存进 `object.mask`（每点 3 个 float：x, y, score），mask 宽高记为网络输入 640×480。
6. NMS：按关键点**中心距离** 30 px 抑制重复框。
7. 导出符号 `NvDsInferParseRunePose`（CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE 宏）。

### 4.5 主程序如何画图（src/deepstream.c）

- 在 `nvdsosd` 的 sink pad 上挂 probe（`nvosd_sink_pad_buffer_probe`），对每个目标的 `obj_meta->mask_params` 处理：
  - `parse_pose_from_meta`：把 mask 里的 640×480 坐标**反 letterbox** 回 1920×1080 显示帧（gain + pad 换算），置信度 ≥ 0.5 的点画蓝色圆圈；按骨架表 `{1,2},{2,5},{5,4},{4,1}`（top↔left↔bottom↔right 的十字连线）画线。画完释放 mask 内存。
  - `set_custom_bbox`：蓝色边框 + 白色文字（类别标签"未激活/小符已激活/大符已激活"来自 labels_rune.txt）。
- `modules/perf.c`：每 5 秒打印 FPS 等性能。
- `modules/interrupt.c`：Ctrl+C 优雅退出。

### 4.6 构建与输出兼容性

1. 顶层 Makefile 会调用 `deepstream/plugin/Makefile`，执行 `make CUDA_VER=12.6` 即可依次构建插件和主程序。

2. 解析器兼容 DeepStream 可能上报的 `[18,N]` 与 TensorRT 常见的 `[1,18,N]`，并拒绝空缓冲区或不匹配的输出。

---

## 5. 路径 B：RMCS 检测/跟踪库（rmcs/）

### 5.1 构建

```bash
cd /home/nvidia/rune_deepstream/rmcs
./build.sh
```

build.sh 做的事：检查工具链 → 运行 CMake（Release，C++23 + CUDA 17）→ 编译 → 执行 CTest → 拷贝 `build/librune_full.a` 到 `lib/`。RMCS 主链路不依赖 ROS 2。

CMakeLists 只编译 6 个源文件（其余都是头文件或遗留代码）：

```
module/detector/rune.cpp                  # RuneDetector（TRT 推理）
module/gpu/rune_gpu.cu                    # CUDA 预处理 + 关键点细化
module/tracker/model/rune.cpp             # RuneModel（EKF 跟踪）
module/tracker/model/rune_energy_fitter.cpp  # 转速拟合
module/tracker/model/virtual_rune.cpp     # 虚拟符仿真
utility/math/camera.cpp                   # 相机参数工具
```

依赖：OpenCV（core/imgproc/calib3d/ximgproc/videoio）、Eigen3、CUDA、TensorRT（nvinfer）、rclcpp（ROS 2）。

同时生成测试程序 `rune_video_test`。

### 5.2 数据流（核心链路）

```
cv::Mat (BGR 帧)
   │
   ▼
RuneDetector::detect()            module/detector/rune.cpp
   │  ① CUDA 预处理（letterbox 640×480 → RGB float 张量）   module/gpu/rune_gpu.cu
   │  ② TensorRT enqueueV3 推理（FP16 engine）
   │  ③ 后处理：类别 argmax / 阈值 / 坐标去 padding / NMS
   │  ④ CUDA 关键点细化（沿梯度把点挪到真实边缘）
   ▼
RuneIcon{R 标像素} + RuneBullseye{符叶中心/四角点/active/score}
   │
   ▼
RuneModel                          module/tracker/model/rune.cpp
   │  init:   单帧 PnP 初始化 6 维状态（符心 xyz + 角速度 + 转角 + 符面朝向）
   │  predict: 状态转移 + 协方差传播
   │  correct: 匈牙利匹配（马氏距离）+ 逐特征点 EKF 更新 + 转速拟合
   ▼
RuneModel::State / get_aimpoints()   → 击打点 3D 坐标 + 前馈角速度/角加速度
```

### 5.3 RuneDetector（module/detector/rune.cpp）

- **初始化**：读 engine 文件 → createInferRuntime → deserializeCudaEngine → createExecutionContext；校验输入 `[1,3,480,640]`、输出 `[1,18,6300]`；分配 GPU 显存和 CUDA stream。
- **detect() 步骤**：
  1. `scale = min(640/w, 480/h)` 等比例缩放 + 居中 padding（灰边 114），与训练一致；
  2. `gpu_pipeline.upload_bgr()` 把 BGR 帧拷上 GPU；`preprocess()` CUDA kernel 双线性缩放 + 转 RGB + 归一化，写入 `input_device`；
  3. `enqueueV3` 推理，`cudaMemcpyAsync` 拉回 `output_host`（18×6300）；
  4. 后处理：每列取类别 argmax（阈值 `score_threshold=0.8`），关键点坐标减去 padding 再除以 scale 还原到原图；有效点（score ≥ 0.8）≥ 3 且都在图内才保留；质量分 = 类别分 × 有效点分数均值；按关键点中心距离（`center_distance=30`）做 NMS；
  5. **GPU 关键点细化**（`refine_kernel`）：4 个十字端点沿"端点→符心"方向 ±radius 内找梯度最强处（符叶边缘更精确）；R 标用周围梯度加权的高斯空间矩（更接近 R 标中心）；位移超限（>7px）或证据不足则回退用网络原始点；
  6. 输出映射（与 docs/NEURAL_RUNE_INTEGRATION.md 一致）：
     - `points[2]`（R）→ `RuneIcon.center`；
     - `points[0,1,3,4]` → `RuneBullseye.corners`（top,left,bottom,right）；
     - 四点均值 → `RuneBullseye.center`；
     - class 0 → `active=false`；class 1/2 → `active=true`（小符/大符激活信息合并，因为跟踪器接口只有布尔）；
     - 类别分数 → `score`。

### 5.4 RuneModel：EKF 跟踪器（module/tracker/model/rune.cpp，核心）

**状态向量（6 维）**：`[x, y, z, ω(角速度), θ(旋转角), ψ(符面朝向 yaw)]`，坐标系是 odom（世界）。

**可观测特征（6 个）**：R 标 + 5 片符叶端点，按符面局部系几何（半径 0.7m、R 突出 0.1m）投影到像素。

- **init()**：用单帧的 icon + 未激活符叶做 `SingleRunePnpSolution` 解算位姿；校验重投影误差（均值 ≤10px、最大 ≤20px）、符面俯仰角 ≤20°、符面朝向合理性；若有多片未激活符叶，把所有候选中"内点最多、误差最小"的作为初始状态；随后用 icon 和 seed 两个观测做两次 EKF update 精化。
- **predict(dt)**：`θ += ω·dt`，协方差用雅可比传播 + 过程噪声。
- **correct()**：把这一帧所有 icon/bullseye 观测与 6 个预测特征点做**匈牙利匹配**（马氏距离，gate 13.816，不可匹配用 dummy 拒绝）；匹配上的逐点做标准 EKF 更新（数值稳定形式，协方差对称化）；标记 inactive 符叶（超时 100ms 自动清除）；记录 `update_count`。
- **转速模型（RuneEnergyFitter）**：观测累计 1s 后开始拟合——
  - **线性拟合**：`θ = C + v·t`（小符近似）；
  - **正弦拟合**：`θ = C + v·t − (a/ω)·cos(ωt+φ)`，ω 在 1.80~2.20 rad/s 扫 41 步，每步最小二乘，取误差最小（大符规则）；
  - 都是 6s 滑动窗口 + 指数权重（半衰期 3s，老数据权重低）；
  - 正弦拟合更优（或处于强制正弦期）→ 用正弦模型外推；否则用线性速度。
- **get_aimpoints()**：对当前 inactive 符叶计算其 3D 位置（世界系），并给出方向向量的角速度 `ff_v` / 角加速度 `ff_a`（供云台前馈），收敛 3~6 秒后才输出。
- **converge()/diverged()**：发散检测（协方差过大、位置越界、朝向与视线夹角 >45°、NaN 等）会打警告日志。

### 5.5 VirtualRuneModel：虚拟符仿真器（module/tracker/model/virtual_rune.cpp）

不依赖相机/网络，**按规则在虚拟场景里生成观测**：
- 小符：恒速 π/3 rad/s；大符：正弦 `v = 1·sin(1.884t) + 1`；
- 每 100 秒轮转一个 active 符叶；只生成"待击打"符叶的 bullseye（大符 2 片、小符 1 片）+ 恒定的 R 标 icon；
- 用相机内参/外参把 3D 点投影成像素，输出与真实检测器**完全相同的 RuneIcon/RuneBullseye 结构**。

用途：脱离真实相机测试/调参跟踪器（原 RMCS 项目里用它做仿真验证）。本仓库没有调用它的示例，但它编译进库里供外部使用。

### 5.6 测试程序 test/rune_video_test.cpp

```bash
cd /home/nvidia/rune_deepstream/rmcs
./build/rune_video_test \
  /home/nvidia/rune_deepstream/model/Rune-v8n-fp16-20260624_b1_gpu0_fp16.engine \
  /home/nvidia/rune_deepstream/test/rune_test_h264.mp4 [max_frames] [score_thr] [keypoint_thr]
```

逐帧跑 `RuneDetector::detect`，打印：帧数、检出帧数、目标总数、平均推理耗时、pipeline FPS。
（注意：这个程序**只测检测器**，不跑跟踪器。）

### 5.7 ⚠️ 不参与构建的遗留文件

- `kernel/detector.cpp`：原 RMCS 的装甲板检测入口（颜色分割、灯条、绿灯逻辑），include 了 `module/detector/armor_detection.hpp`、`green_light.hpp`、`lightbar.hpp` 等**本仓库不存在**的文件，且不在 CMakeLists 里——已废弃，别被它迷惑。
- `module/refiner/rune_keypoint_refiner.cpp`：CPU 版关键点细化，被 GPU 版（rune_gpu.cu）取代。
- `utility/` 里大量头文件（image/logging/rclcpp/robot/csv/coroutine…）是为原 RMCS 项目准备的，只有少数被 rune 部分用到。

---

## 6. 两条路径怎么选 / 什么关系

- **想快速看效果/验证模型部署** → 路径 A（deepstream 应用），开窗口看检测框。
- **要做机器人上的感知闭环**（检测 + 跟踪 + 预测击打点）→ 路径 B（`librune_full.a`），把它链接进主程序（原 RMCS 项目里由 Tracker 调用 `RuneModel`，本仓库只抽出了 rune 相关部分）。
- 两者共享同一个模型（model/ 下的 ONNX/engine）和同一套输出语义（5 关键点 + 3 类别），但推理代码互不共用。

---

## 7. 运行/构建环境清单

| 依赖 | 版本/位置 |
|---|---|
| 硬件 | Jetson（aarch64），需要图形会话（DISPLAY=:0） |
| DeepStream SDK | `/opt/nvidia/deepstream/deepstream`（libnvdsgst_meta 等已装） |
| TensorRT | 10.3.0（libnvinfer 10.3，与 engine 匹配） |
| CUDA | /usr/local/cuda-12.6（含 cudart） |
| GStreamer | 1.20.3 |
| ROS 2 | Humble（/opt/ros/humble，rclcpp） |
| 编译器 | gcc-14 / g++-14（rmcs 构建要求） |
| OpenCV / Eigen3 | 系统安装 |

---

## 8. 常见疑问速查

- **为什么跑起来没框？** ① 检查 4.6-2 的解析器维度校验是否报错；② 检查 DISPLAY/GPU 权限（本机 `nvidia-smi` 在无 GPU 权限的 shell 里会失败，属正常）；③ 确认 config 里 `../../model/` 相对路径正确（在 config 目录下运行）。
- **engine 和 onnx 用哪个？** nvinfer 优先用 `model-engine-file`；engine 与系统 TRT 10.3 匹配，可直接用。
- **改阈值在哪？** 路径 A：`config_infer_primary_rune_pose.txt` 的 `pre-cluster-threshold`（0.8）+ 插件里硬编码的 `kKeypointThreshold`(0.8)/`kNmsDistance`(30)/`kBoxPadding`(0.15)；路径 B：`RuneDetector::Config`（score/keypoint/center_distance 等，可被 YAML `rune_network` 节点覆盖，见 NEURAL_RUNE_INTEGRATION.md）。
- **测试视频**：`test/rune_test_h264.mp4`（1440×1080、50fps、800 帧）；`rune_test.avi` 是指向外部目录的软链，若目标不存在则失效，不影响。
- **路径不一致**：docs/ 两篇旧笔记写的 `/home/nvidia/project/rune_deepstream/...` 是旧位置，README.md 的 `/home/nvidia/rune_deepstream/...` 才是当前实际路径。
