#pragma once

#include <array>
#include <opencv2/core.hpp>

namespace rmcs {

class RuneKeypointRefiner {
public:
    struct Config {
        int blade_radius = 10;
        int icon_radius = 14;
        float max_shift = 7.0F;
        float min_gradient = 12.0F;
        int min_support = 6;
        float min_cross_area = 20.0F;
    };

    explicit RuneKeypointRefiner(Config config);
    [[nodiscard]] auto refine(const cv::Mat& image,
                              const std::array<cv::Point2f, 5>& initial,
                              std::array<cv::Point2f, 5>& refined) const -> bool;

private:
    Config config_;
};

}
