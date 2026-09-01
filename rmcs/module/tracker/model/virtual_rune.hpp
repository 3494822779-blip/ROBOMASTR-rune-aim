#pragma once

#include "utility/clock.hpp"
#include "utility/math/linear.hpp"
#include "utility/pimpl.hpp"
#include "utility/robot/rune.hpp"

#include <array>
#include <span>

namespace rmcs {

class VirtualRuneModel {
    RMCS_PIMPL_DEFINITION(VirtualRuneModel)

public:
    // P2-7：剥离 rclcpp —— 原继承 util::Serializable（ROS2 参数适配）已移除，
    // 库不再依赖 ROS2；如需参数加载请在应用层自行适配。
    struct Config {
        bool enable = false;
        bool large  = false; // false=小符（恒速 π/3），true=大符（正弦）

        // 符中心，固定 OdomLink 坐标
        double x = 6.670;
        double y = 0.0;
        double z = 2.172;

        double face_yaw = 0.0; // 符面法线（局部 +x 背面）远离相机，R 标突出朝向相机

        // P2-6：观测退化模拟（默认关闭，用于评估算法在真实噪声下的表现）
        double pixel_noise_px = 0.0; // 关键点/角点高斯噪声 σ（像素）
        double dropout_prob   = 0.0; // 每片符叶（含 R 标）独立整片漏检/遮挡概率 0~1
        unsigned seed         = 42;  // 随机种子（固定便于复现对比）

        // 相机内参：不参与序列化，由 Tracker 拷贝填充
        std::array<double, 9> camera_matrix = { };
        std::array<double, 5> distort_coeff = { };

        static constexpr std::tuple metas {
            // clang-format off
            &Config::enable,   "enable",
            &Config::large,    "large",
            &Config::x,        "x",
            &Config::y,        "y",
            &Config::z,        "z",
            &Config::face_yaw, "face_yaw",
            // clang-format on
        };
    };

    explicit VirtualRuneModel(const Config&) noexcept;

    auto update_camera(const std::array<double, 9>&) noexcept -> void;
    auto update_camera(const std::array<double, 5>&) noexcept -> void;
    auto update_transform(const Transform&) noexcept -> void;

    auto update(Timestamp) noexcept -> void;

    auto icons() const noexcept -> std::span<const RuneIcon>;
    auto bullseyes() const noexcept -> std::span<const RuneBullseye>;
};

}
