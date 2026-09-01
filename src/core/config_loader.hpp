#pragma once
// config_loader —— 全项目统一参数加载器
// 所有可调参数集中到 config/*.yaml，程序启动只读这一个文件。
// 缺省字段自动用代码内默认值；加载完成后打印实际生效参数，避免"改错文件"。
#include "core/types.hpp"

#include <array>
#include <string>

namespace rmcs::cfg {

struct InputConfig {
    std::string mode = "virtual";  // video | virtual | camera
    std::string source;            // video: 视频路径 / camera: 设备索引 "0"
    int max_frames = 0;            // 0 = 不限
    double hz = 200.0;             // virtual 模式仿真帧率
};

struct CameraConfig {
    // 标定值（行优先 3×3）；virtual 模式用内置演示内参
    std::array<double, 9> matrix { 1400, 0, 720, 0, 1400, 540, 0, 0, 1 };
    std::array<double, 5> distortion { 0, 0, 0, 0, 0 };
    // 相机→Odom 外参：平移(x,y,z) + 四元数(x,y,z,w)
    Translation translation { 0, 0, 0 };
    Orientation orientation { 0, 0, 0, 1 };
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

struct TrackConfig {  // 对应 RuneModel::Config
    double noise_observation = 20.0;
    double gate_threshold = 13.816;
    double init_seed_mean_error = 10.0;
    double init_seed_max_error = 20.0;
    double init_center_gate = 30.0;
    double init_pitch_bound = 20.0;
    double diverge_face_angle = 45.0;
};

struct FireConfig {  // 对应 RuneFireControl::Config
    double bullet_speed = 22.5;         // ★ 发射仓标定
    double shoot_delay = 0.04;          // ★ 发射仓标定
    double algorithmic_delay = 0.05;
    double max_fly_time = 1.0;
    double pitch_max = 0.61;            // rad ≈ 35°
    double fire_cooldown_init = 0.3;
    double fire_cooldown = 0.7;
    double fire_window = 0.04;
    double data_life = 0.2;
    double recover_time = 0.2;
    double switch_angle = 0.30;
    int switch_confirm = 5;
    double offset_yaw = 0.0;            // rad
    double offset_pitch = 0.0;          // rad
};

struct VirtualConfig {  // 对应 VirtualRuneModel::Config
    bool large = false;
    double x = 6.670, y = 0.0, z = 2.172;
    double face_yaw = 0.0;
    double pixel_noise_px = 0.0;   // 像素高斯噪声 σ
    double dropout_prob = 0.0;     // 每片符叶漏检概率
};

struct DiagConfig {
    double match_tolerance_ms = 1.0;
};

struct DisplayConfig {
    bool enabled = true;
    bool keypoints = true;
    bool blades = true;
    bool aimpoint = true;
    bool state_text = true;
    bool error_text = true;
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

// 从 yaml 加载全部配置（相对路径基于项目根解析）；打印实际生效参数。
AppConfig load_config(const std::string& yaml_path);

}  // namespace rmcs::cfg
