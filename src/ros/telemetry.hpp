#pragma once
#include "core/config_loader.hpp"
#include <opencv2/core.hpp>
#include <memory>

namespace rmcs {
// Optional observation-only ROS adapter. Core library remains ROS independent.
class RosTelemetry {
public:
    explicit RosTelemetry(const cfg::AppConfig& config, double image_fps = 15.0);
    ~RosTelemetry();
    bool ok() const;
    bool image_due() const;
    void publish(Timestamp stamp, const cv::Mat& image,
                 const std::vector<RuneIcon>& icons,
                 const std::vector<RuneBullseye>& bullseyes,
                 const RuneModel::State* state, const RuneFireControl::Command& command,
                 const RuneDiagnostics::Stats& stats, bool corrected);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
