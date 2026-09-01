#pragma once

// RuneDiagnostics —— 能量机关预测误差诊断（评估闭环）
//
// 设计来源：RP-26Rune (深圳大学 RobotPilots) PowerRuneDiagnostics
//   - 火控求解成功时记录「预测命中时刻 + 预测相位」；
//   - 每帧记录实测相位（EKF 校正后的 rotation_angle）；
//   - 实测时刻与预测命中时刻在容差内配对，误差 = wrap(实测 − 预测)。
// 该闭环是量化「预测准不准」的核心工具：任何预测器改动都应先看这里。

#include "core/clock.hpp"

#include <cstddef>
#include <deque>
#include <string>
#include <utility>

namespace rmcs {

class RuneDiagnostics {
public:
    struct Config {
        double match_tolerance_ms = 1.0;  // 预测/实测时刻配对容差（RP 同款 1ms）
        std::size_t max_queue     = 1000; // 预测记录缓冲上限
        std::size_t max_history   = 10000; // 配对样本历史上限（CSV 导出用）
    };

    struct Sample {
        double t_hit_s;   // s，预测命中时刻（相对 steady_clock epoch）
        double t_obs_s;   // s，实测时刻
        double phase_pred; // rad
        double phase_obs;  // rad
        double error_rad;  // rad，wrap 后误差
    };

    struct Stats {
        double latest_error = 0.0; // rad，最近一次配对误差
        double mean_error   = 0.0; // rad，平均绝对误差
        double max_error    = 0.0; // rad，最大绝对误差
        std::size_t samples = 0;   // 配对样本数
    };

    RuneDiagnostics() noexcept;
    explicit RuneDiagnostics(const Config& config) noexcept;

    // 预测侧：火控/预测器求解命中时刻时调用（phase 为同一时刻的预测相位，rad）
    void push_predict(Timestamp hit_time, double phase);

    // 观测侧：每帧 EKF 校正后调用（phase 为实测 rotation_angle，rad）
    void push_observation(Timestamp time, double phase);

    auto stats() const -> Stats;
    void reset() noexcept;

    // 导出全部配对样本（Sample 字段顺序），返回写入行数
    auto dump_csv(const std::string& path) const -> std::size_t;

private:
    Config config_;
    std::deque<std::pair<Timestamp, double>> predicts_;
    // 最近一次实测
    Timestamp last_obs_time_ { };
    double    last_obs_phase_ = 0.0;
    bool      has_obs_        = false;

    // 配对样本历史（CSV 导出）
    std::deque<Sample> history_;

    // 统计
    double    sum_abs_error_ = 0.0;
    double    max_abs_error_ = 0.0;
    double    latest_error_  = 0.0;
    std::size_t samples_     = 0;
};

} // namespace rmcs
