#pragma once
// DetectWorker —— 常驻检测线程
//
// 替代此前每帧一次的 std::async(std::launch::async)：libstdc++ 下它不走线程池，
// 60~120fps 就是每秒创建销毁上百个线程，每个新线程首次 CUDA 调用还要重新绑定
// context；资源紧张时 std::async 还会抛 system_error 而无人接。
//
// 单槽设计：调用方按 submit → take → submit → take 交替使用，同一时刻最多一个
// 在途任务，与原来的双缓冲流水线语义一致（检测第 N+1 帧的同时跑第 N 帧的
// EKF/火控）。连续两次 submit 会覆盖前一个未取走的任务。
#include "core/frame_source.hpp"
#include "detect/detector.hpp"

#include <condition_variable>
#include <mutex>
#include <thread>

namespace rmcs {

class DetectWorker {
public:
    // detector 必须在 DetectWorker 之后析构，且不得再被其他线程使用。
    explicit DetectWorker(RuneDetector& detector);
    DetectWorker(const DetectWorker&) = delete;
    auto operator=(const DetectWorker&) -> DetectWorker& = delete;
    ~DetectWorker();

    // 提交一帧待检测，不阻塞。
    auto submit(Frame frame) -> void;

    // 取回检测结果，阻塞至就绪。返回的 frame 即当初 submit 的那一帧，
    // 其采集时间戳随之带回——这正是时间对齐的关键。
    auto take(Frame& frame, RuneDetector::Elements& elements) -> void;

private:
    auto run() -> void;

    RuneDetector& detector_;
    std::mutex mutex_;
    std::condition_variable job_ready_;
    std::condition_variable result_ready_;

    Frame job_;
    bool has_job_ = false;

    Frame result_frame_;
    RuneDetector::Elements result_;
    bool has_result_ = false;

    bool stop_ = false;
    std::thread worker_;
};

}  // namespace rmcs
