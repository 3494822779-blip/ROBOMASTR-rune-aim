// video_test —— 能量机关视频回放链路测试（RuneDetector → RuneModel → RuneFireControl）
//
// 用法：
//   ./rune_video_aim <engine> <video> [max_frames] [fx] [fy] [cx] [cy]
//
// 说明：
//   - 演示「检测 → PnP 初始化 → EKF 跟踪 → 拟合 → 弹道 → 火控」完整链路；
//   - 无 IMU/外参，使用单位外参（相机系 = 车体系），只验证链路与可视化，不做准度结论；
//   - 相机内参默认按 1440×1080 视频假设（fx=fy=1400, cx=720, cy=540），实车请替换标定值。

#include "module/detector/rune.hpp"
#include "module/diagnostics/rune_diagnostics.hpp"
#include "module/fire_control/rune_fire_control.hpp"
#include "module/tracker/model/rune.hpp"
#include "utility/math/linear.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace rmcs;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <engine> <video> [max_frames] [fx] [fy] [cx] [cy]\n",
            argv[0]);
        return 2;
    }
    const std::string engine_path = argv[1];
    const std::string video_path  = argv[2];
    const int max_frames = argc > 3 ? std::atoi(argv[3]) : 0;

    const double fx = argc > 4 ? std::atof(argv[4]) : 1400.0;
    const double fy = argc > 5 ? std::atof(argv[5]) : 1400.0;
    const double cx = argc > 6 ? std::atof(argv[6]) : 720.0;
    const double cy = argc > 7 ? std::atof(argv[7]) : 540.0;

    const std::array<double, 9> K { fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0 };
    const std::array<double, 5> D { 0.0, 0.0, 0.0, 0.0, 0.0 };
    const auto identity = Transform::kIdentity();

    // ---- 检测器 ----
    RuneDetector detector;
    detector.config.engine_path = engine_path;
    if (!detector.initialize()) {
        std::fprintf(stderr, "failed to load TensorRT engine: %s\n", engine_path.c_str());
        return 3;
    }

    // ---- 跟踪 ----
    RuneModel model(RuneModel::Config { });
    model.update_camera(K, D);
    model.update_transform(identity);

    // ---- 火控 ----
    RuneFireControl fire_control(RuneFireControl::Config { });

    // ---- 诊断 ----
    RuneDiagnostics diag;

    cv::VideoCapture capture(video_path);
    if (!capture.isOpened()) {
        std::fprintf(stderr, "failed to open video: %s\n", video_path.c_str());
        return 4;
    }

    cv::Mat frame;
    int frames = 0;
    int inited = 0;
    int fire_frames = 0;
    double pipeline_ms = 0.0;

    while ((max_frames <= 0 || frames < max_frames) && capture.read(frame)) {
        const auto begin = std::chrono::steady_clock::now();

        const auto now = Clock::now();
        const auto elements = detector.detect(frame);

        const auto icons     = elements.icons;
        const auto bullseyes = elements.bullseyes;

        bool have_track = (inited > 0);
        const double video_fps = capture.get(cv::CAP_PROP_FPS);
        const double frame_dt  = (video_fps > 0.0 && video_fps < 500.0) ? 1.0 / video_fps : 1.0 / 50.0;
        if (!have_track) {
            if (model.init(icons, bullseyes, now)) {
                inited = 1;
                have_track = true;
            }
        } else {
            model.predict(frame_dt, now);
            (void)model.correct(icons, bullseyes);
        }

        RuneFireControl::Command cmd;
        if (have_track) {
            const auto state = model.state();
            cmd = fire_control.update(state, now);
            if (cmd.fire) {
                fire_frames++;
                const auto hit_dt = fire_control.config().algorithmic_delay
                    + fire_control.config().shoot_delay + cmd.fly_time;
                auto clone = state;
                clone.transition(hit_dt);
                diag.push_predict(
                    now + std::chrono::duration_cast<Timestamp::duration>(
                              std::chrono::duration<double>(hit_dt)),
                    clone.rotation_angle);
            }
            diag.push_observation(now, state.rotation_angle);
        }

        const auto end = std::chrono::steady_clock::now();
        pipeline_ms += std::chrono::duration<double, std::milli>(end - begin).count();

        // ---- 可视化 ----
        const auto to_cv = [](const Point2d& p) {
            return cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y));
        };
        cv::Mat display = frame.clone();
        for (const auto& bs : bullseyes) {
            cv::circle(display, to_cv(bs.center), 4, bs.active ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0),
                -1);
            for (const auto& c : bs.corners) {
                cv::circle(display, to_cv(c), 3, cv::Scalar(255, 255, 0), -1);
            }
            // P1-4：显示激活类别（未激活/小符/大符）
            const char* tag =
                bs.activation == RuneBullseye::Activation::Inactive   ? "inactive"
                : bs.activation == RuneBullseye::Activation::SmallActive ? "SMALL"
                                                                         : "BIG";
            cv::putText(display, tag, to_cv(bs.center) + cv::Point2f(6, -6),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
        for (const auto& ic : icons) {
            cv::circle(display, to_cv(ic.center), 6, cv::Scalar(255, 0, 255), -1);
        }
        if (have_track) {
            const auto state = model.state();
            // 瞄准点（外推到命中时刻的第一个未激活符叶）
            auto clone = state;
            clone.transition(fire_control.config().algorithmic_delay
                + fire_control.config().shoot_delay + cmd.fly_time);
            const auto aimpoints = clone.get_aimpoints();
            for (const auto& ap : aimpoints) {
                // 世界系 → 图像（单位外参，相机系=车体系；ROS 前=x 左=y 上=z → OpenCV 右=-y 下=-z 前=x）
                const auto p_cam = ap;
                const auto u = fx * (-p_cam.y) / p_cam.x + cx;
                const auto v = fy * (-p_cam.z) / p_cam.x + cy;
                cv::circle(display, cv::Point2f(static_cast<float>(u), static_cast<float>(v)), 10,
                    cv::Scalar(0, 255, 255), 2);
                break;
            }
        }
        std::string status = have_track ? cmd.state_name() : "NO TRACK";
        if (cmd.fire) status += " [FIRE]";
        cv::putText(display, status, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9,
            cmd.fire ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255), 2);
        const auto s = diag.stats();
        cv::putText(display,
            cv::format("err: mean %.3f max %.3f n %zu", s.mean_error, s.max_error, s.samples),
            cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1);

        cv::imshow("rune aim", display);
        const auto key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;

        frames++;
    }

    const double avg = frames > 0 ? pipeline_ms / frames : 0.0;
    std::printf("frames=%d inited=%d fire_frames=%d avg_pipeline=%.2fms\n", frames, inited,
        fire_frames, avg);
    const auto s = diag.stats();
    std::printf("predict err: mean=%.4f rad max=%.4f rad n=%zu\n", s.mean_error, s.max_error,
        s.samples);
    return 0;
}
