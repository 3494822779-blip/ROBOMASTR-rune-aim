#pragma once
// FrameSource —— 数据源统一抽象（视频文件 / 相机）
//
// 存在的理由是时间戳：此前主循环在循环顶部取 Clock::now()，而双缓冲流水线当轮
// 处理的是**上一轮**采集的帧，观测因此被贴上晚一个帧周期的时间戳，EKF 状态系统性
// 滞后。时间戳跟着帧走之后，这个偏差从结构上消失。
//
// 另一个作用是把"怎么拿到一帧"和主循环解耦：真机相机（工业相机 SDK / GStreamer）
// 到位时只需实现一个新的 FrameSource，主循环不动。
#include "core/clock.hpp"

#include <opencv2/core/mat.hpp>

#include <memory>
#include <string>

namespace rmcs {

// 一帧图像 + 它的采集时刻。图像用 shared_ptr<const> 传递，流水线各级只共享不拷贝。
struct Frame {
    std::shared_ptr<const cv::Mat> image;
    Timestamp stamp {};

    auto valid() const noexcept -> bool { return image && !image->empty(); }
};

class FrameSource {
public:
    FrameSource() = default;
    FrameSource(const FrameSource&) = delete;
    auto operator=(const FrameSource&) -> FrameSource& = delete;
    virtual ~FrameSource() = default;

    // 取下一帧；返回无效帧表示流结束或取帧失败。
    virtual auto grab() -> Frame = 0;
    virtual auto name() const -> std::string = 0;
};

// 视频文件回放。时间戳优先使用视频 PTS；PTS 无效时按 fps 外推（fps <= 0 时取 30），
// 使解码/推理耗时的抖动不进入 EKF 的 dt，并支持可变帧率视频。打开失败返回 nullptr。
auto make_video_source(const std::string& path, double fps) -> std::unique_ptr<FrameSource>;

// 相机。source 为纯十进制整数时按设备索引打开，否则按字符串打开
// （/dev/videoN、GStreamer pipeline、RTSP URL 等）。打开失败返回 nullptr。
auto make_camera_source(const std::string& source, int width = 0, int height = 0,
                       double fps = 0.0, int buffer_size = 1) -> std::unique_ptr<FrameSource>;

}  // namespace rmcs
