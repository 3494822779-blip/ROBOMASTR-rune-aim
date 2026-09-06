// rune_aim —— 统一命令行入口（video / virtual / camera 三模式）
//
// 用法：
//   ./build/rune_aim -c config/rune_small_video.yaml
//   ./build/rune_aim -c config/rune_large_virtual.yaml --no-display
//   ./build/rune_aim -c config/camera_sentry.yaml --mode camera --source 0
//
// 覆盖项（其余参数一律进 yaml）：--mode --source --max-frames --no-display
// 按键：q/ESC 退出，空格 暂停，s 截图，v 切换可视化
//
// 链路：数据源(视频/虚拟符/相机) → RuneDetector(视频/相机) 或 VirtualRune
//       → RuneModel(EKF 跟踪) → RuneFireControl(预瞄+开火) → RuneDiagnostics(误差)
#include "core/config_loader.hpp"
#include "debug/draw.hpp"
#include "detect/detector.hpp"
#include "diag/diagnostics.hpp"
#include "fire/fire_control.hpp"
#include "track/rune_model.hpp"
#include "track/virtual_rune.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <numbers>
#include <optional>
#include <string>
#include <thread>

using namespace rmcs;

namespace {

struct Cli {
    std::string config = "config/rune_small_virtual.yaml";
    std::string mode;            // 空 = 用 yaml
    std::string source;          // 空 = 用 yaml
    int max_frames = -1;         // -1 = 用 yaml
    bool no_display = false;
};

auto parse_cli(int argc, char** argv) -> Cli {
    Cli cli;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 < argc) return argv[++i];
            std::fprintf(stderr, "缺少参数: %s\n", name);
            std::exit(2);
        };
        if (a == "-c" || a == "--config") cli.config = next("--config");
        else if (a == "--mode") cli.mode = next("--mode");
        else if (a == "--source") cli.source = next("--source");
        else if (a == "--max-frames") cli.max_frames = std::atoi(next("--max-frames"));
        else if (a == "--no-display") cli.no_display = true;
        else if (a == "-h" || a == "--help") {
            std::printf("用法: rune_aim -c <config.yaml> [--mode video|virtual|camera]"
                        " [--source <path|dev>] [--max-frames N] [--no-display]\n");
            std::exit(0);
        } else {
            std::fprintf(stderr, "未知参数: %s\n", a.c_str());
            std::exit(2);
        }
    }
    return cli;
}

}  // namespace

int main(int argc, char** argv) {
    const auto cli = parse_cli(argc, argv);
    auto cfg = cfg::load_config(cli.config);
    if (!cli.mode.empty()) cfg.input.mode = cli.mode;
    if (!cli.source.empty()) cfg.input.source = cli.source;
    if (cli.max_frames >= 0) cfg.input.max_frames = cli.max_frames;
    if (cli.no_display) cfg.display.enabled = false;

    const bool is_virtual = cfg.input.mode == "virtual";
    const bool is_video   = cfg.input.mode == "video";
    const bool is_camera  = cfg.input.mode == "camera";
    if (!is_virtual && !is_video && !is_camera) {
        std::fprintf(stderr, "未知 mode: %s（可选 video|virtual|camera）\n", cfg.input.mode.c_str());
        return 2;
    }

    // ---- 检测器（video/camera 需要；virtual 不需要 GPU）----
    RuneDetector detector;
    if (!is_virtual) {
        detector.config.engine_path = cfg.detect.engine;
        detector.config.score_threshold = cfg.detect.score_threshold;
        detector.config.keypoint_threshold = cfg.detect.keypoint_threshold;
        detector.config.center_distance = cfg.detect.center_distance;
        detector.config.refine_radius = cfg.detect.refine_radius;
        detector.config.icon_refine_radius = cfg.detect.icon_refine_radius;
        detector.config.max_refine_shift = cfg.detect.max_refine_shift;
        detector.config.min_refine_gradient = cfg.detect.min_refine_gradient;
        detector.config.min_refine_support = cfg.detect.min_refine_support;
        if (!detector.initialize()) {
            std::fprintf(stderr, "TensorRT 引擎加载失败: %s\n", cfg.detect.engine.c_str());
            return 3;
        }
    }

    // ---- 跟踪 ----
    RuneModel model(cfg.track);
    model.update_camera(cfg.camera.matrix, cfg.camera.distortion);
    model.update_transform(cfg.camera.transform());

    // ---- 火控 ----
    RuneFireControl fire(cfg.fire);

    // ---- 诊断 ----
    RuneDiagnostics diag(cfg.diag);

    // ---- 数据源 ----
    std::optional<VirtualRuneModel> virtual_rune;
    cv::VideoCapture capture;
    if (is_virtual) {
        VirtualRuneModel::Config vcfg;
        vcfg.enable = true;
        vcfg.large = cfg.virtual_rune.large;
        vcfg.x = cfg.virtual_rune.x;
        vcfg.y = cfg.virtual_rune.y;
        vcfg.z = cfg.virtual_rune.z;
        vcfg.face_yaw = cfg.virtual_rune.face_yaw;
        vcfg.camera_matrix = cfg.camera.matrix;
        vcfg.distort_coeff = cfg.camera.distortion;
        vcfg.pixel_noise_px = cfg.virtual_rune.pixel_noise_px;
        vcfg.dropout_prob = cfg.virtual_rune.dropout_prob;
        virtual_rune.emplace(vcfg);
        virtual_rune->update_camera(cfg.camera.matrix);
        virtual_rune->update_camera(cfg.camera.distortion);
        virtual_rune->update_transform(cfg.camera.transform());
        std::printf("[aim] virtual rune @ (%.2f, %.2f, %.2f) %s\n", vcfg.x, vcfg.y, vcfg.z,
            vcfg.large ? "LARGE" : "SMALL");
    } else {
        const std::string src = cfg.input.source.empty() ? "data/rune_test_h264.mp4" : cfg.input.source;
        const bool opened = is_camera ? capture.open(std::stoi(src)) : capture.open(src);
        if (!opened) {
            std::fprintf(stderr, "无法打开数据源: %s\n", src.c_str());
            return 4;
        }
        std::printf("[aim] %s source: %s\n", is_camera ? "camera" : "video", src.c_str());
    }

    // 视频模式用固定帧率驱动 EKF，避免 GPU 推理延迟波动导致 dt 抖动掉帧。
    // 帧率从 yaml 的 input.video_fps 读取（避免 Jetson GStreamer 的 CAP_PROP_FPS 查询干扰解码器）。
    double video_fps = cfg.input.video_fps;
    if (is_video) {
        if (video_fps <= 0.0 || video_fps > 1000.0) video_fps = 30.0;
        std::printf("[aim] video fps: %.2f\n", video_fps);
    }

    // ---- 主循环 ----
    const auto window = "rune_aim";
    if (cfg.display.enabled) cv::namedWindow(window, cv::WINDOW_AUTOSIZE);

    bool rune_inited = false;
    Timestamp rune_stamp{};
    Timestamp rune_corrected_stamp{};
    Timestamp last_now{};
    double dt = 1.0 / cfg.input.hz;
    int frame_id = 0, init_count = 0, aim_count = 0, fire_count = 0;
    bool paused = false;
    bool tracking_corrected = false;
    // 实时显示帧率（按实际完成绘制的帧数统计，适用于 video/camera/virtual）。
    double display_fps = 0.0;
    int fps_frames = 0;
    auto fps_stamp = std::chrono::steady_clock::now();
    RuneFireControl::Command last_cmd{};  // 主循环计算，可视化只读，避免重复推进火控状态机
    // Reuse per-frame containers to avoid allocator churn in the real-time loop.
    std::vector<RuneIcon> icons;
    std::vector<RuneBullseye> bullseyes;
    icons.reserve(32);
    bullseyes.reserve(32);

    std::future<RuneDetector::Elements> pending_detect;
    cv::Mat prev_frame;
    double last_read_ms = 0.0;
    double last_detect_wait_ms = 0.0;
    double last_display_ms = 0.0;
    double last_track_ms = 0.0;
    auto loop_stamp = std::chrono::steady_clock::now();

    const auto run_frame = [&](const cv::Mat& frame, std::vector<RuneIcon>& icons,
                                std::vector<RuneBullseye>& bullseyes, Timestamp now) {
        tracking_corrected = false;
        // ---- 跟踪生命周期 ----
        if (!rune_inited) {
            if ((!icons.empty() || !bullseyes.empty()) && model.init(icons, bullseyes, now)) {
                rune_inited = true;
                rune_stamp = rune_corrected_stamp = now;
                std::printf("[aim] frame %d: RuneModel init OK\n", frame_id);
                ++init_count;
            }
            return;
        }
        if (std::chrono::duration<double>(now - rune_corrected_stamp).count() > 1.5) {
            rune_inited = false;
            return;
        }
        model.update_transform(cfg.camera.transform());
        model.predict(dt, now);
        rune_stamp = now;
        const bool corrected = model.correct(icons, bullseyes);
        if (model.diverged()) { rune_inited = false; return; }
        if (corrected) rune_corrected_stamp = now;
        tracking_corrected = corrected;

        // ---- 火控 + 诊断 ----
        const auto state = model.state();
        last_cmd = fire.update(state, now);
        const auto& cmd = last_cmd;
        if (cmd.found) {
            ++aim_count;
            if (cmd.fire) ++fire_count;
        }
        diag.push_observation(now, state.rotation_angle);
        if (cmd.found && cmd.fire) {
            const auto hit_dt = cfg.fire.algorithmic_delay + cfg.fire.shoot_delay + cmd.fly_time;
            auto clone = state;
            clone.transition(hit_dt);
            diag.push_predict(now + std::chrono::duration_cast<Timestamp::duration>(
                                      std::chrono::duration<double>(hit_dt)),
                clone.rotation_angle);
        }

        // ---- 终端输出 ----
        if (cmd.found && (frame_id % 30 == 0)) {
            std::printf("frame %d | %s | yaw=%.2f pitch=%.2f fly=%.3fs | fire=%d | %s\n", frame_id,
                cmd.state_name(), cmd.yaw, cmd.pitch, cmd.fly_time, static_cast<int>(cmd.fire),
                cmd.reason.c_str());
        }
    };

    while (true) {
        if (cfg.input.max_frames > 0 && frame_id >= cfg.input.max_frames) break;

        // 视频模式：固定 dt = 1/fps，合成时间戳按视频帧率匀速推进，
        //           消除 TensorRT 推理延迟波动对 EKF 的影响。
        // 相机/虚拟模式：仍用挂钟时间。
        Timestamp now;
        if (is_video) {
            dt = 1.0 / video_fps;
            if (last_now == Timestamp{}) {
                now = Clock::now();
            } else {
                now = last_now + std::chrono::duration_cast<Duration>(
                    std::chrono::duration<double>(dt));
            }
        } else {
            now = Clock::now();
            if (!paused && last_now != Timestamp{}) {
                dt = std::chrono::duration<double>(now - last_now).count();
                dt = std::clamp(dt, 0.0, 0.5);
            }
        }
        last_now = now;

        cv::Mat frame;
        icons.clear();
        bullseyes.clear();

        if (is_virtual) {
            if (paused) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); continue; }
            virtual_rune->update(now);
            const auto virtual_icons = virtual_rune->icons();
            icons.insert(icons.end(), virtual_icons.begin(), virtual_icons.end());
            const auto virtual_bullseyes = virtual_rune->bullseyes();
            bullseyes.insert(bullseyes.end(), virtual_bullseyes.begin(), virtual_bullseyes.end());
            frame = cv::Mat::zeros(540, 960, CV_8UC3);
            if (!paused) run_frame(frame, icons, bullseyes, now);
        } else {
            // Double-buffered pipeline: overlap TensorRT inference with EKF+fire control.
            // First iteration: no pending detection, just kick off async detect.
            if (!pending_detect.valid()) {
                const auto read_begin = std::chrono::steady_clock::now();
                if (!capture.read(frame)) break;
                last_read_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - read_begin).count();
                prev_frame = frame.clone();
                pending_detect = std::async(std::launch::async, [&detector](cv::Mat f) {
                    return detector.detect(f);
                }, frame.clone());
                // No tracking data yet for this first frame — just grab and display.
            } else {
                // Retrieve detection results from previous frame's async detect.
                const auto detect_begin = std::chrono::steady_clock::now();
                auto elements = pending_detect.get();
                last_detect_wait_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - detect_begin).count();
                icons = std::move(elements.icons);
                bullseyes = std::move(elements.bullseyes);

                // Kick off next frame's detection in parallel with this frame's tracking.
                cv::Mat next_frame;
                const auto read_begin = std::chrono::steady_clock::now();
                bool has_next = capture.read(next_frame);
                last_read_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - read_begin).count();
                if (has_next) {
                    pending_detect = std::async(std::launch::async, [&detector](cv::Mat f) {
                        return detector.detect(f);
                    }, next_frame.clone());
                }
                frame = prev_frame;
                if (has_next) prev_frame = next_frame;

                if (!paused) {
                    const auto track_begin = std::chrono::steady_clock::now();
                    run_frame(frame, icons, bullseyes, now);
                    last_track_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - track_begin).count();
                }

                if (!has_next) {
                    // Process display for last frame then exit.
                    if (cfg.display.enabled) {
                        cv::Mat display = frame.clone();
                        debug::DrawOptions opt{
                            .keypoints = cfg.display.keypoints,
                            .aimpoint = cfg.display.aimpoint,
                            .state_text = cfg.display.state_text,
                            .error_text = cfg.display.error_text,
                        };
                        if (opt.keypoints) debug::draw_detection(display, icons, bullseyes);
                        cv::imshow(window, display);
                        cv::waitKey(1);
                    }
                    ++frame_id;
                    break;
                }
            }
        }

        // ---- 可视化（统一 draw 层）----
        if (cfg.display.enabled) {
            const auto display_begin = std::chrono::steady_clock::now();
            cv::Mat display = frame.clone();
            debug::DrawOptions opt{
                .keypoints = cfg.display.keypoints,
                .aimpoint = cfg.display.aimpoint,
                .state_text = cfg.display.state_text,
                .error_text = cfg.display.error_text,
            };
            if (opt.keypoints) debug::draw_detection(display, icons, bullseyes);
            if (rune_inited) {
                const auto& cmd = last_cmd;  // 复用主循环结果，不重复调用 fire.update
                if (opt.aimpoint && tracking_corrected) {
                    debug::draw_aimpoint(display, model.state(),
                        cfg.fire.algorithmic_delay + cfg.fire.shoot_delay + cmd.fly_time, cfg.camera.matrix);
                }
                if (opt.state_text || opt.error_text) debug::draw_status(display, cmd, diag.stats(), opt);
            }
            // 每约半秒更新一次，避免瞬时值抖动；文字始终显示在左上角。
            ++fps_frames;
            const auto fps_now = std::chrono::steady_clock::now();
            const double fps_elapsed = std::chrono::duration<double>(fps_now - fps_stamp).count();
            if (fps_elapsed >= 0.5) {
                display_fps = static_cast<double>(fps_frames) / fps_elapsed;
                fps_frames = 0;
                fps_stamp = fps_now;
            }
            cv::putText(display, cv::format("FPS: %.1f", display_fps), cv::Point(20, 115),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::imshow(window, display);
            const auto key = cv::waitKey(1);
            last_display_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - display_begin).count();
            if (key == 'q' || key == 27) break;
            if (key == ' ') paused = !paused;
            if (key == 's') cv::imwrite(cv::format("shot_%04d.png", frame_id), display);
        }
        if (is_video && frame_id > 0 && frame_id % 30 == 0) {
            const auto loop_now = std::chrono::steady_clock::now();
            const double loop_ms = std::chrono::duration<double, std::milli>(loop_now - loop_stamp).count() / 30.0;
            loop_stamp = loop_now;
            std::printf("[timing] frame %d read=%.1fms detect_wait=%.1fms display=%.1fms\n",
                frame_id, last_read_ms, last_detect_wait_ms, last_display_ms);
            std::printf("[timing] track=%.1fms\n", last_track_ms);
            std::printf("[timing] avg_loop=%.1fms (%.1f FPS)\n", loop_ms, loop_ms > 0.0 ? 1000.0 / loop_ms : 0.0);
        }
        ++frame_id;
        if (is_virtual && !paused) {
            const auto next = now + std::chrono::duration_cast<Timestamp::duration>(
                                        std::chrono::duration<double>(1.0 / cfg.input.hz));
            std::this_thread::sleep_until(next);
        }
    }

    // ---- 汇总 ----
    const auto s = diag.stats();
    std::printf("\n=== summary (mode=%s) ===\n", cfg.input.mode.c_str());
    std::printf("  frames      : %d\n", frame_id);
    std::printf("  init_ok     : %d\n", init_count);
    std::printf("  aim_ok      : %d\n", aim_count);
    std::printf("  fire_frames : %d\n", fire_count);
    std::printf("  predict err : mean=%.4f rad (%.2f deg), max=%.4f rad (%.2f deg), n=%zu\n",
        s.mean_error, s.mean_error * 180.0 / std::numbers::pi, s.max_error,
        s.max_error * 180.0 / std::numbers::pi, s.samples);
    return 0;
}
