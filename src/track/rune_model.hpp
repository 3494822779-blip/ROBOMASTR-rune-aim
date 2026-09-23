#pragma once

#include "core/clock.hpp"
#include "core/pimpl.hpp"
#include "core/aimpoint.hpp"
#include "core/rune_types.hpp"

#include <array>
#include <span>
#include <vector>

namespace rmcs {

class RuneModel {
    RMCS_PIMPL_DEFINITION(RuneModel)

public:
    struct State {
        double x;
        double y;
        double z;

        Timestamp start_timestamp;
        Timestamp current_timestamp;  // 当前模型时刻（视频模式为合成时间戳）

        double rotation_speed;
        double filter_rotation_speed = 0.0; // EKF 最近观测速度，用于检查拟合起点偏差
        double sine_speed_correction = 0.0; // 正弦拟合速度与最近 EKF 速度的起点差
        double rotation_angle;

        double face_yaw = 0;

        std::array<bool, 5> inactive;
        // 最近一次确认未激活的符叶（0..4），-1 表示无记忆。
        // inactive[] 的有效期（kInactiveTimeout）短于火控 data_life，两者之间的夹缝会让
        // get_aimpoints() 空返、瞄准点回退到符心；该记忆用 EKF 相位把这段夹缝补上。
        int  last_inactive_blade = -1;
        bool last_inactive_valid = false; // 记忆是否仍在 kInactiveMemory 有效期内
        bool converged = false;

        bool use_prediction_speed = false;
        double prediction_cost    = 0.0;

        double sine_C     = 0.0;
        double sine_v     = 0.0;
        double sine_a     = 0.0;
        double sine_omega = 0.0;
        double sine_phase = 0.0;
        double sine_t     = 0.0;
        bool sine_valid   = false;

        std::size_t update_count = 0;

        auto transition(double seconds) -> void;

        auto get_direction() const -> Point3d;
        auto get_aimpoints() const -> AimPoints;
        auto get_rotation_speed() const -> double;
    };

    struct Config {
        double noise_x = 1e-5;
        double noise_y = 1e-5;
        double noise_z = 1e-5;

        double noise_rotation_angle = 1e-3;
        double noise_rotation_speed = 1e-0;
        double noise_face_yaw       = 1e-5;

        double noise_observation = 20.0;

        double gate_threshold = 13.816;

        double init_seed_mean_error = 10.0;
        double init_seed_max_error  = 20.0;
        double init_center_gate     = 30.0;
        double init_pitch_bound     = 20.0;

        double diverge_face_angle = 45.0;

        double diverge_cov_max = 150.0;
        double diverge_pos_xy_max = 15.0;
        double diverge_pos_z_max = 5.0;
        double diverge_speed_factor = 10.0;
    };

    struct Addition {
        struct Tracked {
            int feature_id;
            Point2d point;
        };
        std::vector<Tracked> tracked;
        std::vector<Tracked> predicted;
    };

    explicit RuneModel(const Config&) noexcept;

    auto update_camera(std::array<double, 9>, std::array<double, 5>) noexcept -> void;
    auto update_transform(const Transform&) noexcept -> void;

    auto init(std::span<const RuneIcon>, std::span<const RuneBullseye>, Timestamp) noexcept -> bool;
    auto predict(double dt, Timestamp now) noexcept -> void;
    [[nodiscard]] auto correct(std::span<const RuneIcon>, std::span<const RuneBullseye>) noexcept
        -> bool;

    auto converge() const -> bool;
    auto diverged() const -> bool;

    auto state() const noexcept -> State;
    auto addition() const -> const Addition&;
};

}
