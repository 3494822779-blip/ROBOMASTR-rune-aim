#include "detect/detect_worker.hpp"

#include <utility>

namespace rmcs {

DetectWorker::DetectWorker(RuneDetector& detector)
    : detector_ { detector }
    , worker_ { [this] { run(); } } { }

DetectWorker::~DetectWorker() {
    {
        const std::lock_guard lock { mutex_ };
        stop_ = true;
    }
    job_ready_.notify_all();
    if (worker_.joinable()) worker_.join();
}

auto DetectWorker::submit(Frame frame) -> void {
    {
        const std::lock_guard lock { mutex_ };
        job_ = std::move(frame);
        has_job_ = true;
    }
    job_ready_.notify_one();
}

auto DetectWorker::take(Frame& frame, RuneDetector::Elements& elements) -> void {
    std::unique_lock lock { mutex_ };
    result_ready_.wait(lock, [this] { return has_result_ || stop_; });
    if (!has_result_) return;
    frame = std::move(result_frame_);
    elements = std::move(result_);
    result_frame_ = Frame {};
    result_ = RuneDetector::Elements {};
    has_result_ = false;
}

auto DetectWorker::run() -> void {
    while (true) {
        Frame frame;
        {
            std::unique_lock lock { mutex_ };
            job_ready_.wait(lock, [this] { return has_job_ || stop_; });
            if (stop_) return;
            frame = std::move(job_);
            job_ = Frame {};
            has_job_ = false;
        }

        // 检测在锁外执行，主线程可同时跑上一帧的 EKF/火控。
        auto elements = frame.valid() ? detector_.detect(*frame.image) : RuneDetector::Elements {};

        {
            const std::lock_guard lock { mutex_ };
            result_frame_ = std::move(frame);
            result_ = std::move(elements);
            has_result_ = true;
        }
        result_ready_.notify_one();
    }
}

}  // namespace rmcs
