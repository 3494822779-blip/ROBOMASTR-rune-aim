#pragma once

// RuneFireControl —— 能量机关专用火控状态机
//
// 设计来源：
//   - 瞄准/弹道固定点迭代：rmcs_auto_aim_v2 (南京理工大学 Alliance) kernel/fire_control.cpp
//   - 开火时机状态机：RP-26Rune (深圳大学 RobotPilots) RuneDecisionModule
//     （初始冷却 → 连续开火窗口 → 冷却；数据过期平滑回符心；切叶确认后重置）
//
// 坐标系约定（与 rmcs_auto_aim_v2 一致）：
//   输入为 RuneModel::State 的世界系（Odom）状态；TrajectorySolution 输入 Odom 系攻击点，
//   输出 Odom 系 yaw/pitch 射线角。实车集成时若云台参考系不同，请在调用方做变换。

#include "track/rune_model.hpp"
#include "core/clock.hpp"

#include <memory>
#include <string>

namespace rmcs {

class RuneFireControl {
public:
    struct Config {
        // ---- 弹道与时间 ----
        double bullet_speed        = 22.5;  // m/s，弹丸初速
        double shoot_delay         = 0.04;  // s，发弹延迟（扳机到出膛）
        double algorithmic_delay   = 0.05;  // s，算法链路延迟估计（采集→解算→下发）
        double max_fly_time        = 1.0;   // s，飞行时间上限
        double pitch_max           = 0.61;  // rad，云台俯仰上限（约 35°）

        // ---- 开火状态机（RP RuneDecisionModule 参数） ----
        double fire_cooldown_init  = 0.3;   // s，新目标/切叶确认后的初始冷却
        double fire_cooldown       = 0.7;   // s，连续开火窗口结束后的冷却
        double fire_window         = 0.04;  // s，连续开火窗口上限
        double data_life           = 0.2;   // s，目标数据寿命，超时禁射
        double recover_time        = 0.2;   // s，过期后平滑回符心时长

        // ---- 切叶确认（RP） ----
        double switch_angle        = 0.30;  // rad，相位跳变阈值（≈17°）
        int    switch_confirm      = 5;     // 连续跳变帧数达到后确认切叶

        // ---- 固定点迭代（rmcs） ----
        int    max_iterate         = 5;
        double iterate_epsilon     = 0.001; // s，飞行时间收敛判据

        // ---- 机械偏置 ----
        double offset_yaw          = 0.0;   // rad
        double offset_pitch        = 0.0;   // rad
    };

    enum class State {
        LOST,        // 无目标
        COOLING,     // 冷却中（初始冷却 / 连发冷却）
        READY,       // 冷却结束，等待开火
        FIRING,      // 连续开火窗口内
        RECOVERING,  // 数据过期，平滑回符心
    };

    struct Command {
        bool   found    = false;  // 目标是否可用
        bool   fire     = false;  // 是否允许开火
        bool   shot_started = false; // 开火窗口上升沿（无拨弹反馈时作为发射事件）
        double yaw      = 0.0;    // rad，Odom 系
        double pitch    = 0.0;    // rad，Odom 系
        double fly_time = 0.0;    // s
        Point3d attack_point = Point3d::kZero(); // 固定点迭代最终使用的目标点
        bool has_attack_point = false;
        // P2-8：云台角速度/角加速度前馈（命中时刻符叶的射线运动，实车动态跟踪用）
        Vector3d ff_v = Vector3d::kZero();
        Vector3d ff_a = Vector3d::kZero();
        State  state    = State::LOST;
        std::string reason;       // 调试用：当前状态原因

        auto state_name() const -> const char* {
            switch (state) {
            case State::LOST:       return "LOST";
            case State::COOLING:    return "COOLING";
            case State::READY:      return "READY";
            case State::FIRING:     return "FIRING";
            case State::RECOVERING: return "RECOVERING";
            }
            return "?";
        }
    };

    RuneFireControl() noexcept;
    explicit RuneFireControl(const Config& config) noexcept;
    ~RuneFireControl() noexcept;

    // 每帧调用：输入当前 RuneModel 状态（应在 correct() 之后立即调用），输出瞄准指令。
    auto update(const RuneModel::State& state, Timestamp now) -> Command;

    void reset() noexcept;

    auto config() const noexcept -> const Config& { return config_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Config config_;
};

} // namespace rmcs
