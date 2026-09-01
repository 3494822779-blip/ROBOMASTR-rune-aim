#pragma once
// debug/draw —— 统一可视化调试层
// 所有模式（video/virtual/camera）共用同一套绘制：检测结果、预瞄点、火控状态、诊断误差。
// 开关由调用方（config 的 display 段）决定。
#include "core/rune_types.hpp"
#include "core/types.hpp"
#include "diag/diagnostics.hpp"
#include "fire/fire_control.hpp"
#include "track/rune_model.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <vector>

namespace rmcs::debug {

struct DrawOptions {
    bool keypoints = true;
    bool aimpoint = true;
    bool state_text = true;
    bool error_text = true;
};

// 检测层：符叶中心/角点（激活红/未激活绿）、类别文字（inactive/SMALL/BIG）、R 标
void draw_detection(cv::Mat& img, const std::vector<RuneIcon>& icons,
    const std::vector<RuneBullseye>& bullseyes);

// 火控层：把命中时刻的未激活符叶端点投影回图像（单位外参演示；真机用 TF 外参）
void draw_aimpoint(cv::Mat& img, const RuneModel::State& state, double lead_time,
    const std::array<double, 9>& K);

// 状态层：火控状态机文字（开火红色）+ 诊断误差
void draw_status(cv::Mat& img, const RuneFireControl::Command& cmd,
    const RuneDiagnostics::Stats& stats, const DrawOptions& opt);

}  // namespace rmcs::debug
