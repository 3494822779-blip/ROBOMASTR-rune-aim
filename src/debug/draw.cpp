#include "debug/draw.hpp"

#include <opencv2/imgproc.hpp>

namespace rmcs::debug {
namespace {

auto to_cv(const Point2d& p) -> cv::Point2f {
    return cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y));
}

// ROS 前=x 左=y 上=z → OpenCV 右=-y 下=-z 前=x（单位外参演示）
auto project_to_image(const Point3d& world, const std::array<double, 9>& K) -> cv::Point2f {
    return {
        static_cast<float>(K[0] * (-world.y) / world.x + K[2]),
        static_cast<float>(K[4] * (-world.z) / world.x + K[5]),
    };
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
    const std::array<double, 9>& K) {
    auto clone = state;
    clone.transition(lead_time);
    const auto aimpoints = clone.get_aimpoints();
    for (const auto& ap : aimpoints) {
        cv::circle(img, project_to_image(ap, K), 10, cv::Scalar(0, 255, 255), 2);
        break;  // 只画第一个未激活符叶
    }
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
