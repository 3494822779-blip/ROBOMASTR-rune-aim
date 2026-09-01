#pragma once

// DelayCalibrator —— 链路延迟互相关标定器
//
// 吸纳自：Climber_Vision_26 (同济 SuperPower) calibration/serial_delay.cpp
//
// 原理：视觉目标在图像中的像素位移与云台实际姿态（IMU 反馈）是同一运动的两种观测，
// 二者之间的时间偏移就是「曝光 → 检测 → 解算 → 串口下发 → 下位机执行 → IMU 反馈」
// 的完整链路延迟。把两路信号等间隔重采样后做滑动互相关，相关系数最大的位移即延迟。
//
// 用法（实车）：
//   每帧调用 push({ 检测结束时刻, 目标像素位移(相对图像中心), IMU yaw/pitch(度) })
//   窗口积累足够（像素跨度 ≥40px、云台角跨度 ≥2°）后 estimate() 返回延迟估计，
//   将结果填入 RuneFireControl::Config.algorithmic_delay（算法链路延迟）。
//
// 有效性门槛（与 Climber 一致）：像素跨度 40/30px、云台角跨度 2.0/1.5°、相关系数 >0.3；
// 搜索范围 ±3s（覆盖常见链路延迟），100Hz 重采样时间轴。

#include <cstddef>
#include <deque>
#include <limits>

namespace rmcs {

class DelayCalibrator {
public:
    struct Config {
        double resample_step_s   = 0.01;  // 重采样步长（100Hz 时间轴）
        double max_delay_search_s = 3.0;  // 互相关最大搜索时长
        double min_vision_span_px = 40.0; // 水平像素跨度门槛
        double min_vision_span_py = 30.0; // 垂直像素跨度门槛
        double min_yaw_span_deg   = 2.0;  // yaw 跨度门槛
        double min_pitch_span_deg = 1.5;  // pitch 跨度门槛
        double min_corr           = 0.3;  // 相关系数门槛（|r| 大于此值才采信）
        std::size_t max_buffer    = 1200; // 样本缓冲上限（约 40s@30fps）
    };

    struct Sample {
        double t;         // s，统一零点后的时刻（建议用检测结束时刻）
        double vision_x;  // px，目标中心相对图像中心的水平位移
        double vision_y;  // px，垂直位移
        double imu_yaw_deg;   // °，云台 IMU yaw
        double imu_pitch_deg; // °，云台 IMU pitch
    };

    struct Estimate {
        double yaw_ms   = std::numeric_limits<double>::quiet_NaN(); // 视觉x vs yaw 延迟
        double pitch_ms = std::numeric_limits<double>::quiet_NaN(); // 视觉y vs pitch 延迟
        double yaw_corr   = std::numeric_limits<double>::quiet_NaN();
        double pitch_corr = std::numeric_limits<double>::quiet_NaN();
        bool   hit_edge   = false; // 最佳位移撞到搜索边界（结果不可信，应增大窗口）
        bool   valid      = false; // 至少一个轴得到可靠估计
    };

    DelayCalibrator() noexcept;
    explicit DelayCalibrator(const Config& config) noexcept;

    void push(const Sample& sample); // 每帧喂样本
    auto estimate() const -> Estimate; // 窗口互相关延迟估计
    void reset() noexcept;

    auto size() const noexcept -> std::size_t { return buffer_.size(); }
    auto config() const noexcept -> const Config& { return config_; }

private:
    Config config_;
    std::deque<Sample> buffer_;
};

} // namespace rmcs
