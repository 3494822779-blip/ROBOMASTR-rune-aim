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

struct TrackConfig {  // 对应 RuneModel::Config（EKF 过程噪声 + 观测噪声 + 门限）
    double noise_x = 1e-5;               // 位置过程噪声
    double noise_y = 1e-5;
    double noise_z = 1e-5;
    double noise_rotation_angle = 1e-3;  // 转角过程噪声
    double noise_rotation_speed = 1e-0;  // 角速度过程噪声
    double noise_face_yaw = 1e-5;        // 符面朝向过程噪声
    double noise_observation = 20.0;     // 观测噪声（像素²），虚拟模式可收紧到 2
    double gate_threshold = 13.816;      // 观测关联门限（卡方 6 自由度 99%）
    double init_seed_mean_error = 10.0;  // 初始化重投影均方根误差上限 px
    double init_seed_max_error = 20.0;   // 初始化重投影最大误差上限 px
    double init_center_gate = 30.0;      // 初始化符叶中心匹配门限 px
    double init_pitch_bound = 20.0;      // 初始化符面俯仰角上限 degree
    double diverge_face_angle = 45.0;    // 符面朝向与视线夹角发散阈值 degree
};

struct FireConfig {  // 对应 RuneFireControl::Config
    double bullet_speed = 22.5;         // ★ 发射仓弹速标定 m/s
    double shoot_delay = 0.04;          // ★ 扳机→出膛延迟标定 s
    double algorithmic_delay = 0.05;    // 算法链路延迟估计 s（可用 delay_calib 标定）
    double max_fly_time = 1.0;          // 飞行时间上限 s（超出视为无解）
    double pitch_max = 0.61;            // 云台俯仰上限 rad ≈ 35°
    double fire_cooldown_init = 0.3;    // 新目标/切叶确认后初始冷却 s
    double fire_cooldown = 0.7;         // 连续开火窗口结束后的冷却 s
    double fire_window = 0.04;          // 连续开火窗口时长上限 s
    double data_life = 0.2;             // 目标数据寿命 s，超时禁射并回符心
    double recover_time = 0.2;          // 数据过期后平滑回符心时长 s
    double switch_angle = 0.30;         // 切叶检测相位跳变阈值 rad ≈ 17°
    int switch_confirm = 5;             // 连续跳变帧数达到后确认切叶
    double offset_yaw = 0.0;            // 弹道机械偏置 rad（往左增）
    double offset_pitch = 0.0;          // 弹道机械偏置 rad（往下增）
    int max_iterate = 5;                // 弹道固定点迭代次数
    double iterate_epsilon = 0.001;     // 飞行时间收敛判据 s
};

struct VirtualConfig {  // 对应 VirtualRuneModel::Config
    bool large = false;
    double x = 6.670, y = 0.0, z = 2.172;
    double face_yaw = 0.0;
    double pixel_noise_px = 0.0;   // 像素高斯噪声 σ
    double dropout_prob = 0.0;     // 每片符叶漏检概率
};

struct DiagConfig {
    double match_tolerance_ms = 1.0;  // 预测/实测时刻配对容差 ms
    std::size_t max_queue = 1000;     // 预测记录缓冲上限
    std::size_t max_history = 10000;  // 配对样本历史上限（CSV 导出）
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

// 从 yaml 加载全部配置（相对路径基于项目根解析）；打印实际生效参数。
AppConfig load_config(const std::string& yaml_path);

}  // namespace rmcs::cfg
