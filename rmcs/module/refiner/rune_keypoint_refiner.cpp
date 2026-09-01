#include "rune_keypoint_refiner.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace rmcs {
namespace {
auto inside(const cv::Mat& image, const cv::Point2f& p) -> bool {
    return p.x >= 1.0F && p.y >= 1.0F && p.x < image.cols - 1 && p.y < image.rows - 1;
}

auto bilinear(const cv::Mat& image, cv::Point2f p) -> float {
    const int x = static_cast<int>(std::floor(p.x));
    const int y = static_cast<int>(std::floor(p.y));
    const float ax = p.x - x, ay = p.y - y;
    return (1 - ax) * (1 - ay) * image.at<float>(y, x) +
           ax * (1 - ay) * image.at<float>(y, x + 1) +
           (1 - ax) * ay * image.at<float>(y + 1, x) +
           ax * ay * image.at<float>(y + 1, x + 1);
}

auto refine_blade(const cv::Mat& gray, cv::Point2f seed, cv::Point2f center,
                  int radius, float min_gradient, int min_support,
                  cv::Point2f& result) -> bool {
    cv::Point2f normal = seed - center;
    const float length = cv::norm(normal);
    if (length < 2.0F) return false;
    normal *= 1.0F / length;
    const cv::Point2f tangent{-normal.y, normal.x};
    const int half_width = std::clamp(radius / 3, 2, 5);
    const int margin = radius + half_width + 3;
    cv::Rect rect{static_cast<int>(std::floor(seed.x)) - margin,
                  static_cast<int>(std::floor(seed.y)) - margin,
                  2 * margin + 1, 2 * margin + 1};
    rect &= cv::Rect{0, 0, gray.cols, gray.rows};
    if (rect.width < 7 || rect.height < 7) return false;
    cv::Mat gx, gy, gradient;
    cv::Scharr(gray(rect), gx, CV_32F, 1, 0, 1.0 / 32.0);
    cv::Scharr(gray(rect), gy, CV_32F, 0, 1, 1.0 / 32.0);
    cv::magnitude(gx, gy, gradient);
    const cv::Point2f local_seed = seed - cv::Point2f(rect.x, rect.y);
    // Configured radii are small on Jetson. Fixed storage avoids heap work for
    // every keypoint/candidate.
    constexpr int kMaxRadius = 24;
    if (radius < 1 || radius > kMaxRadius) return false;
    std::array<float, 2 * kMaxRadius + 1> response{};
    std::array<int, 2 * kMaxRadius + 1> support{};
    for (int offset = -radius; offset <= radius; ++offset) {
        for (int lateral = -half_width; lateral <= half_width; ++lateral) {
            const cv::Point2f p = local_seed + normal * static_cast<float>(offset) +
                                  tangent * static_cast<float>(lateral);
            if (!inside(gradient, p)) continue;
            response[offset + radius] += bilinear(gradient, p);
            ++support[offset + radius];
        }
        if (support[offset + radius] > 0)
            response[offset + radius] /= support[offset + radius];
    }
    const auto response_end = response.begin() + 2 * radius + 1;
    const auto best_it = std::max_element(response.begin(), response_end);
    const int best = static_cast<int>(best_it - response.begin());
    if (*best_it < min_gradient || support[best] < min_support || best == 0 ||
        best + 1 == 2 * radius + 1) return false;
    const float left = response[best - 1], middle = response[best], right = response[best + 1];
    const float denominator = left - 2.0F * middle + right;
    const float fraction = std::abs(denominator) > 1e-5F
        ? std::clamp(0.5F * (left - right) / denominator, -0.5F, 0.5F) : 0.0F;
    result = seed + normal * (static_cast<float>(best - radius) + fraction);
    return true;
}

auto refine_icon(const cv::Mat& gray, cv::Point2f seed, int radius,
                 float min_gradient, int min_support, cv::Point2f& result) -> bool {
    cv::Rect rect{static_cast<int>(std::floor(seed.x)) - radius,
                  static_cast<int>(std::floor(seed.y)) - radius,
                  2 * radius + 1, 2 * radius + 1};
    rect &= cv::Rect{0, 0, gray.cols, gray.rows};
    if (rect.width < 7 || rect.height < 7) return false;
    // A fixed-size gradient moment is substantially cheaper than allocating a
    // binary image, contours and an ellipse fit. Gaussian spatial weighting
    // keeps the moment attached to the neural seed when other edges enter ROI.
    const cv::Mat roi = gray(rect);
    const int local_seed_x = static_cast<int>(std::round(seed.x)) - rect.x;
    const int local_seed_y = static_cast<int>(std::round(seed.y)) - rect.y;
    const float sigma2 = std::max(4.0F, 0.35F * radius * radius);
    double weight_sum = 0.0, x_sum = 0.0, y_sum = 0.0;
    int support = 0;
    for (int y = 1; y < roi.rows - 1; ++y) {
        const auto* previous = roi.ptr<unsigned char>(y - 1);
        const auto* current = roi.ptr<unsigned char>(y);
        const auto* next = roi.ptr<unsigned char>(y + 1);
        for (int x = 1; x < roi.cols - 1; ++x) {
            const float gx = static_cast<float>(current[x + 1]) - current[x - 1];
            const float gy = static_cast<float>(next[x]) - previous[x];
            const float magnitude = std::abs(gx) + std::abs(gy);
            if (magnitude < 2.0F * min_gradient) continue;
            const float dx = static_cast<float>(x - local_seed_x);
            const float dy = static_cast<float>(y - local_seed_y);
            const float spatial = std::exp(-(dx * dx + dy * dy) / (2.0F * sigma2));
            const double weight = magnitude * spatial;
            weight_sum += weight;
            x_sum += weight * (x + rect.x);
            y_sum += weight * (y + rect.y);
            ++support;
        }
    }
    if (support < min_support || weight_sum <= 0.0) return false;
    result = {static_cast<float>(x_sum / weight_sum),
              static_cast<float>(y_sum / weight_sum)};
    return true;
}
}

RuneKeypointRefiner::RuneKeypointRefiner(Config config) : config_(config) {}

auto RuneKeypointRefiner::refine(const cv::Mat& image,
                                 const std::array<cv::Point2f, 5>& initial,
                                 std::array<cv::Point2f, 5>& refined) const -> bool {
    if (image.empty()) return false;
    cv::Mat gray;
    if (image.channels() == 1) gray = image;
    else cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    const cv::Point2f blade_center = (initial[0] + initial[1] + initial[3] + initial[4]) * 0.25F;
    for (const int i : std::array{0, 1, 3, 4}) {
        if (!inside(image, initial[i]) ||
            !refine_blade(gray, initial[i], blade_center, config_.blade_radius,
                          config_.min_gradient, config_.min_support, refined[i])) return false;
    }
    if (!inside(image, initial[2]) ||
        !refine_icon(gray, initial[2], config_.icon_radius, config_.min_gradient,
                     config_.min_support, refined[2]))
        return false;
    for (int i = 0; i < 5; ++i) {
        const cv::Point2f shift = refined[i] - initial[i];
        if (!inside(image, refined[i]) || shift.dot(shift) > config_.max_shift * config_.max_shift)
            return false;
    }
    if (!(refined[0].y < refined[4].y && refined[1].x < refined[3].x)) return false;
    const float area = std::abs(cv::contourArea(std::vector<cv::Point2f>{
        refined[0], refined[3], refined[4], refined[1]}));
    return area >= config_.min_cross_area;
}

}
