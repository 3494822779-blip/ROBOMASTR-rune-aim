#pragma once
// config_loader —— 全项目统一参数加载器
// 所有可调参数集中到 config/*.yaml，程序启动只读这一个文件。
// 缺省字段自动用代码内默认值；加载完成后打印实际生效参数，避免"改错文件"。
#include "core/types.hpp"
#include "track/rune_model.hpp"
#include "fire/fire_control.hpp"
#include "diag/diagnostics.hpp"

#include <array>
#include <string>

namespace rmcs::cfg {

struct InputConfig {
    std::string mode = "virtual";  // video | virtual | camera
    std::string source;            // video: 视频路径 / camera: 设备索引 "0"
    int max_frames = 0;            // 0 = 不限
    double hz = 200.0;             // virtual 模式仿真帧率
    double video_fps = 0.0;        // video 模式帧率（0 = 默认 30）
};

struct CameraConfig {
    int capture_width = 0;
    int capture_height = 0;
    double capture_fps = 0.0;
    int buffer_size = 1;
    // 标定值（行优先 3×3）；virtual 模式用内置演示内参
    std::array<double, 9> matrix { 1400, 0, 720, 0, 1400, 540, 0, 0, 1 };
    std::array<double, 5> distortion { 0, 0, 0, 0, 0 };
    // 相机→Odom 外参：平移(x,y,z) + 四元数(x,y,z,w)
    Translation translation { 0, 0, 0 };
    Orientation orientation { 0, 0, 0, 1 };
    // 画幅尺寸（默认与上面的演示内参 cx/cy 对应）。virtual 模式据此裁掉落在
    // 画幅外的观测——真实检测器不会输出这些点，不裁会让仿真结果偏乐观。
    int image_width  = 1440;
    int image_height = 1080;
    auto transform() const -> Transform { return { translation, orientation }; }
};

struct DetectConfig {
    std::string engine;                    // TensorRT engine 路径（相对项目根或绝对）
    float score_threshold = 0.8F;
    float keypoint_threshold = 0.8F;
    float center_distance = 30.0F;
    int refine_radius = 10;
    int icon_refine_radius = 14;
    float max_refine_shift = 7.0F;
    float min_refine_gradient = 12.0F;
    int min_refine_support = 6;
};

using TrackConfig = RuneModel::Config;
using FireConfig = RuneFireControl::Config;
using DiagConfig = RuneDiagnostics::Config;

// 云台运动仿真（virtual 模式）：让相机外参随时间摆动，用于在没有真机的情况下
// 验证"外参随云台运动"这条链路。真机上这个外参由 IMU 回读驱动，目前工具里是常量，
// 也就是说该链路至今没有被执行过——这里补上离线验证手段。
struct GimbalConfig {
    bool enable = false;
    double yaw_amp    = 10.0;  // 摆幅 deg（绕 Odom +z）
    double yaw_freq   = 0.30;  // 频率 Hz
    double pitch_amp  = 3.0;   // 摆幅 deg（绕 Odom +y）
    double pitch_freq = 0.17;  // 频率 Hz
    // 喂给 RuneModel 的外参 = 真实外参延迟 transform_delay 秒 + transform_noise 噪声。
    // 二者分别对应真机的 IMU 回读滞后与外参标定/量测误差。
    double transform_delay = 0.0;  // s
    double transform_noise = 0.0;  // σ deg
    unsigned seed = 7;
};

struct VirtualConfig {
    bool large = false;
    double x = 6.670, y = 0.0, z = 2.172;
    double face_yaw = 0.0;
    double pixel_noise_px = 0.0;   // 像素高斯噪声 σ
    double dropout_prob = 0.0;     // 每片符叶漏检概率
    GimbalConfig gimbal;
};

struct DisplayConfig {
    bool enabled = true;
    bool keypoints = true;   // 检测关键点/符叶/R标
    bool aimpoint = true;    // 命中时刻预瞄点
    bool state_text = true;  // 火控状态文字
    bool error_text = true;  // 诊断误差文字
};

struct AppConfig {
    InputConfig input;
    CameraConfig camera;
    DetectConfig detect;
    TrackConfig track;
    FireConfig fire;
    VirtualConfig virtual_rune;
    DiagConfig diag;
    DisplayConfig display;
};

// 从 yaml 加载全部配置（相对路径基于项目根解析）。
AppConfig load_config(const std::string& yaml_path);

// 在 yaml 与命令行覆盖均应用后打印最终生效参数。
void print_config(const AppConfig& config);

}  // namespace rmcs::cfg
