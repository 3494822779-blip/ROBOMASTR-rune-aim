#include "trajectory.hpp"

#include <cmath>
#include <tuple>

using namespace rmcs;

namespace details {

constexpr auto kMaxIterateCount          = int { 10 };
constexpr auto kMaxPitchThreold          = double { 80.0 / 57.3 };
constexpr auto kEstimateDeltaTime        = double { 0.005 };
constexpr auto kHeightErrorThreold       = double { 0.001 };
constexpr auto kEstimateTimeOutThreold   = double { 4.0 };
constexpr auto kMinVelocityX             = double { 0.1 };
constexpr auto kGravity                  = double { 9.81 };
constexpr auto kAirResistanceCoefficient = double { 0.003 };
constexpr auto kPitchUpdateGain           = double { 0.65 };

constexpr auto estimate(double v0, double pitch, double d, double air_resistance)
    -> std::tuple<double, double> {

    double x = 0, y = 0, t = 0;
    double vx = v0 * std::cos(pitch);
    double vy = v0 * std::sin(pitch);

    double prev_x = 0, prev_y = 0, prev_t = 0;

    while (x < d) {
        prev_x = x;
        prev_y = y;
        prev_t = t;

        const auto accel = [&](double vx_, double vy_) {
            const double v = std::sqrt(vx_ * vx_ + vy_ * vy_);
            return std::pair {
                -air_resistance * v * vx_,
                -(kGravity + air_resistance * v * vy_)
            };
        };

        const auto [ax1, ay1] = accel(vx, vy);
        const double vx2 = vx + ax1 * 0.5 * kEstimateDeltaTime;
        const double vy2 = vy + ay1 * 0.5 * kEstimateDeltaTime;
        const auto [ax2, ay2] = accel(vx2, vy2);
        const double vx3 = vx + ax2 * 0.5 * kEstimateDeltaTime;
        const double vy3 = vy + ay2 * 0.5 * kEstimateDeltaTime;
        const auto [ax3, ay3] = accel(vx3, vy3);
        const double vx4 = vx + ax3 * kEstimateDeltaTime;
        const double vy4 = vy + ay3 * kEstimateDeltaTime;
        const auto [ax4, ay4] = accel(vx4, vy4);

        // Save pre-step velocities for trapezoidal position integration.
        const double vx_old = vx;
        const double vy_old = vy;

        vx += (ax1 + 2.0 * ax2 + 2.0 * ax3 + ax4) * (kEstimateDeltaTime / 6.0);
        vy += (ay1 + 2.0 * ay2 + 2.0 * ay3 + ay4) * (kEstimateDeltaTime / 6.0);

        // Trapezoidal position integration: average pre- and post-step velocity.
        // Noticeably more accurate than post-step-only (especially near the target).
        x += 0.5 * (vx_old + vx) * kEstimateDeltaTime;
        y += 0.5 * (vy_old + vy) * kEstimateDeltaTime;
        t += kEstimateDeltaTime;

        if (t > kEstimateTimeOutThreold || vx <= kMinVelocityX) [[unlikely]]
            break;
    }

    if (x >= d && x > prev_x) {
        double ratio = (d - prev_x) / (x - prev_x);
        return { std::lerp(prev_y, y, ratio), std::lerp(prev_t, t, ratio) };
    }
    return { y, t };
}

}

auto TrajectorySolution::solve() const -> std::optional<Output> {

    const auto target_d = std::hypot(input.point.x, input.point.y);
    const auto target_h = input.point.z;

    if (!std::isfinite(input.v0) || !std::isfinite(input.point.x) ||
        !std::isfinite(input.point.y) || !std::isfinite(input.point.z) ||
        input.v0 <= 0 || target_d <= 0) return std::nullopt;

    const auto yaw = std::atan2(input.point.y, input.point.x);

    double pitch = std::atan2(target_h, target_d);
    for (int i = 0; i < details::kMaxIterateCount; ++i) {
        auto [actual_h, t] =
            details::estimate(input.v0, pitch, target_d, details::kAirResistanceCoefficient);

        if (!std::isfinite(actual_h) || !std::isfinite(t)) return std::nullopt;
        auto h_error = target_h - actual_h;
        if (std::abs(h_error) < details::kHeightErrorThreold) {
            auto result     = Output { };
            result.fly_time = t;
            result.yaw      = yaw;
            result.pitch    = -pitch;
            return result;
        }

        // Damped Newton-like update prevents oscillation for close targets or
        // when drag makes the height response strongly nonlinear.
        pitch += details::kPitchUpdateGain * std::atan2(h_error, target_d);

        if (std::abs(pitch) > details::kMaxPitchThreold) break;
    }

    return std::nullopt;
}
