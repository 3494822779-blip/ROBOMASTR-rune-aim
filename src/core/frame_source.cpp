#include "core/frame_source.hpp"

#include <opencv2/videoio.hpp>

#include <charconv>
#include <chrono>
#include <system_error>
#include <utility>

namespace rmcs {
namespace {

// VideoCapture::read() 给出的 Mat 在下一次 read() 后可能失效（部分后端直接指向
// 解码器内部缓冲），跨帧持有必须先复制到独享缓冲。整条流水线只在这里拷贝一次。
auto detach(const cv::Mat& source) -> std::shared_ptr<const cv::Mat> {
    auto owned = std::make_shared<cv::Mat>();
    source.copyTo(*owned);
    return owned;
}

// 整串必须是十进制整数才算设备索引。此前用的是无保护的 std::stoi：
// source 写成 /dev/video0 或 GStreamer pipeline 时会抛 invalid_argument
// 而无人捕获，直接 std::terminate。from_chars 不抛异常。
auto parse_device_index(const std::string& text, int& index) -> bool {
    if (text.empty()) return false;
    const auto* const first = text.data();
    const auto* const last = first + text.size();
    const auto result = std::from_chars(first, last, index);
    return result.ec == std::errc {} && result.ptr == last;
}

class VideoSource final : public FrameSource {
public:
    VideoSource(std::string path, double fps)
        : path_ { std::move(path) }
        , fps_ { fps > 0.0 ? fps : 30.0 } {
        opened_ = capture_.open(path_);
    }

    auto opened() const -> bool { return opened_; }
    auto name() const -> std::string override { return "video:" + path_; }

    auto grab() -> Frame override {
        cv::Mat raw;
        if (!capture_.read(raw) || raw.empty()) return {};
        if (index_ == 0) start_ = Clock::now();
        const auto offset = std::chrono::duration<double> { static_cast<double>(index_) / fps_ };
        ++index_;
        return Frame { detach(raw), start_ + std::chrono::duration_cast<Duration>(offset) };
    }

private:
    std::string path_;
    double fps_;
    cv::VideoCapture capture_;
    bool opened_ = false;
    std::size_t index_ = 0;
    Timestamp start_ {};
};

class CameraSource final : public FrameSource {
public:
    explicit CameraSource(std::string source)
        : source_ { std::move(source) } {
        if (int index = 0; parse_device_index(source_, index))
            opened_ = capture_.open(index);
        else
            opened_ = capture_.open(source_);
    }

    auto opened() const -> bool { return opened_; }
    auto name() const -> std::string override { return "camera:" + source_; }

    auto grab() -> Frame override {
        cv::Mat raw;
        if (!capture_.read(raw) || raw.empty()) return {};
        // 真实曝光时刻早于此处：曝光 + 传输 + 驱动缓冲。没有硬件采集时间戳时
        // 这段固定偏移只能由 fire.algorithmic_delay 统一吸收。
        return Frame { detach(raw), Clock::now() };
    }

private:
    std::string source_;
    cv::VideoCapture capture_;
    bool opened_ = false;
};

}  // namespace

auto make_video_source(const std::string& path, double fps) -> std::unique_ptr<FrameSource> {
    auto source = std::make_unique<VideoSource>(path, fps);
    if (!source->opened()) return nullptr;
    return source;
}

auto make_camera_source(const std::string& source) -> std::unique_ptr<FrameSource> {
    auto camera = std::make_unique<CameraSource>(source);
    if (!camera->opened()) return nullptr;
    return camera;
}

}  // namespace rmcs
