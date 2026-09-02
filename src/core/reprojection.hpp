#pragma once

#include "core/camera.hpp"

#include <array>
#include <limits>
#include <ranges>
#include <span>

#include <opencv2/core/types.hpp>

namespace rmcs {

namespace details {

    struct project_points {
        bool success = false;
        explicit project_points(std::span<const cv::Point3f> object_points,
            const cv::Mat& intrinsic, const cv::Mat& distortion,
            std::span<cv::Point2f> projected_points)
            : success { impl(object_points, intrinsic, distortion, projected_points) } { };
        explicit operator bool() const { return success; }

    private:
        static auto impl(std::span<const cv::Point3f> object_points, const cv::Mat& intrinsic,
            const cv::Mat& distortion, std::span<cv::Point2f> projected_points) -> bool;
    };

}

template <std::size_t N>
struct ReprojectionSolution {
    static_assert(N > 0, "ReprojectionSolution requires at least one point");

    struct Input {
        util::CameraFeature camera;
        std::array<cv::Point3f, N> object_points;
        std::array<cv::Point2f, N> image_points;
    } input;

    struct Result {
        std::array<cv::Point2f, N> projected_points;
        double error;
        double mean_error;
    } result;

    auto solve() -> bool {
        result.error      = std::numeric_limits<double>::max();
        result.mean_error = std::numeric_limits<double>::max();

        if (!details::project_points { input.object_points, input.camera.intrinsic(),
                input.camera.distortion(), result.projected_points })
            return false;

        result.error = 0.0;
        for (const auto& [projected, detected] :
            std::views::zip(result.projected_points, input.image_points)) {
            result.error += cv::norm(projected - detected);
        }

        result.mean_error = result.error / static_cast<double>(N);
        return true;
    }
};

namespace util {
    // 注意，point_camera 为相机坐标系，而非 ROS 系
    auto reproject_point(const Point3d& point_camera, const util::CameraFeature& camera)
        -> std::optional<Point2d>;

    // 快速单点投影（含 5 参数畸变模型），避免 cv::projectPoints 的 vector/Mat 分配开销。
    // point_camera 同为 OpenCV 相机坐标系，外参为恒等（已在调用方变换到相机系）。
    inline auto reproject_point_fast(const Point3d& p,
        const std::array<std::array<double, 3>, 3>& K,
        const std::array<double, 5>& D) -> std::optional<Point2d> {
        if (p.z <= 0.0) return std::nullopt;
        const double xn = p.x / p.z;
        const double yn = p.y / p.z;
        const double r2 = xn * xn + yn * yn;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double radial = 1.0 + D[0] * r2 + D[1] * r4 + D[4] * r6;
        const double xd = xn * radial + 2.0 * D[2] * xn * yn + D[3] * (r2 + 2.0 * xn * xn);
        const double yd = yn * radial + D[2] * (r2 + 2.0 * yn * yn) + 2.0 * D[3] * xn * yn;
        return Point2d { K[0][0] * xd + K[0][2], K[1][1] * yd + K[1][2] };
    }
}
}
