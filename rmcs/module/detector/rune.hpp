#pragma once

#include "utility/robot/rune.hpp"
#include <opencv2/core/mat.hpp>
#include <memory>
#include <string>

namespace rmcs {

class RuneDetector {
public:
    struct Config {
        std::string engine_path;
        float score_threshold = 0.8F;
        float keypoint_threshold = 0.8F;
        float center_distance = 30.0F;
        int refine_radius = 10;
        int icon_refine_radius = 14;
        float max_refine_shift = 7.0F;
        float min_refine_gradient = 12.0F;
        int min_refine_support = 6;
    } config;

    struct Elements {
        std::vector<RuneBullseye> bullseyes;
        std::vector<RuneIcon> icons;
    };

    RuneDetector() noexcept;
    ~RuneDetector() noexcept;
    RuneDetector(RuneDetector&&) noexcept;
    auto operator=(RuneDetector&&) noexcept -> RuneDetector&;

    auto initialize() noexcept -> bool;
    auto detect(const cv::Mat&) noexcept -> Elements;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
