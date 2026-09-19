#include "debug/draw.hpp"

#include "core/conversion.hpp"
#include "core/reprojection.hpp"

#include <eigen3/Eigen/Geometry>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace rmcs::debug {
namespace {

auto to_cv(const Point2d& p) -> cv::Point2f {
    return cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y));
}

auto project_to_image(const Point3d& world, const std::array<double, 9>& K,
    const std::array<double, 5>& distortion, const Transform& camera_transform)
    -> std::optional<cv::Point2f> {
    const auto q_odom_from_cam = camera_transform.orientation.make<Eigen::Quaterniond>();
    const auto cam_position = camera_transform.translation.make<Eigen::Vector3d>();
    const auto point_world = world.make<Eigen::Vector3d>();
    const auto point_camera_ros = q_odom_from_cam.conjugate() * (point_world - cam_position);
    const auto point_camera_cv = util::ros2opencv_position(point_camera_ros);

    const std::array<std::array<double, 3>, 3> matrix {{
        { K[0], K[1], K[2] },
        { K[3], K[4], K[5] },
        { K[6], K[7], K[8] },
    }};
    const auto pixel = util::reproject_point_fast(Point3d { point_camera_cv }, matrix, distortion);
    if (!pixel) return std::nullopt;
    return to_cv(*pixel);
}

auto draw_marker(cv::Mat& img, const cv::Point2f& pixel, const cv::Scalar& color,
    const char* label) -> bool {
    if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y)
        || pixel.x < 0.0F || pixel.x >= static_cast<float>(img.cols)
        || pixel.y < 0.0F || pixel.y >= static_cast<float>(img.rows))
        return false;
    cv::drawMarker(img, pixel, color, cv::MARKER_CROSS, 24, 2);
    cv::circle(img, pixel, 12, color, 2);
    constexpr double font_scale = 0.55;
    constexpr int thickness = 2;
    int baseline = 0;
    const auto label_size = cv::getTextSize(
        label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    const auto label_x = std::clamp(static_cast<int>(pixel.x) + 16, 0,
        std::max(0, img.cols - label_size.width));
    const auto label_y = std::clamp(static_cast<int>(pixel.y) - 12, label_size.height,
        std::max(label_size.height, img.rows - baseline));
    cv::putText(img, label, cv::Point(label_x, label_y),
        cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness);
    return true;
}

}  // namespace

void draw_detection(cv::Mat& img, const std::vector<RuneIcon>& icons,
    const std::vector<RuneBullseye>& bullseyes) {
    for (const auto& bs : bullseyes) {
        cv::circle(img, to_cv(bs.center), 4,
            bs.active ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), -1);
        for (const auto& c : bs.corners) cv::circle(img, to_cv(c), 3, cv::Scalar(255, 255, 0), -1);
        const char* tag = bs.activation == RuneBullseye::Activation::Inactive   ? "inactive"
                        : bs.activation == RuneBullseye::Activation::SmallActive ? "SMALL" : "BIG";
        cv::putText(img, tag, to_cv(bs.center) + cv::Point2f(6, -6), cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(255, 255, 255), 1);
    }
    for (const auto& ic : icons) cv::circle(img, to_cv(ic.center), 6, cv::Scalar(255, 0, 255), -1);
}

void draw_aimpoint(cv::Mat& img, const RuneModel::State& state, double lead_time,
    const std::array<double, 9>& K, const std::array<double, 5>& distortion,
    const Transform& camera_transform) {
    auto clone = state;
    clone.transition(lead_time);
    const auto aimpoints = clone.get_aimpoints();
    for (const auto& ap : aimpoints) {
        const auto pixel = project_to_image(ap, K, distortion, camera_transform);
        if (!pixel) continue;
        draw_marker(img, *pixel, cv::Scalar(0, 255, 255), "AIM");
        break;  // 只画第一个未激活符叶
    }
}

auto draw_hitpoint(cv::Mat& img, const Point3d& hitpoint,
    const std::array<double, 9>& K, const std::array<double, 5>& distortion,
    const Transform& camera_transform) -> bool {
    const auto pixel = project_to_image(hitpoint, K, distortion, camera_transform);
    return pixel && draw_marker(img, *pixel, cv::Scalar(0, 0, 255), "HIT");
}

void draw_status(cv::Mat& img, const RuneFireControl::Command& cmd,
    const RuneDiagnostics::Stats& stats, const DrawOptions& opt) {
    if (opt.state_text) {
        std::string status = cmd.state_name();
        if (cmd.fire) status += " [FIRE]";
        cv::putText(img, status, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9,
            cmd.fire ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255), 2);
    }
    if (opt.error_text) {
        cv::putText(img,
            cv::format("err: mean %.3f max %.3f n %zu", stats.mean_error, stats.max_error,
                stats.samples),
            cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1);
    }
}

}  // namespace rmcs::debug
