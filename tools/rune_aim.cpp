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
#include "core/angle.hpp"
#include "core/config_loader.hpp"
#include "core/frame_source.hpp"
#include "debug/draw.hpp"
#include "detect/detect_worker.hpp"
#include "detect/detector.hpp"
#include "diag/diagnostics.hpp"
#include "fire/fire_control.hpp"
#include "track/rune_model.hpp"
#include "track/virtual_rune.hpp"

#include <eigen3/Eigen/Geometry>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
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

// 云台运动仿真：在标定的基准外参上叠加 yaw(绕 Odom +z) / pitch(绕 Odom +y) 正弦摆动。
// 相机绕自身位置转动，平移不变；返回值与 camera.transform 同义（相机→Odom）。
auto gimbal_pose(const cfg::AppConfig& cfg, double t) -> Transform {
    const auto& g = cfg.virtual_rune.gimbal;
    const auto yaw = util::deg2rad(g.yaw_amp)
                   * std::sin(2.0 * std::numbers::pi * g.yaw_freq * t);
    const auto pitch = util::deg2rad(g.pitch_amp)
                     * std::sin(2.0 * std::numbers::pi * g.pitch_freq * t);
    const auto q_base = cfg.camera.orientation.make<Eigen::Quaterniond>();
    const Eigen::Quaterniond q = Eigen::AngleAxisd { yaw, Eigen::Vector3d::UnitZ() }
                               * Eigen::AngleAxisd { pitch, Eigen::Vector3d::UnitY() } * q_base;
    return Transform { cfg.camera.translation, Orientation { q } };
}

// 给外参叠加高斯姿态噪声，模拟真机的 IMU 量测 / 外参标定误差。
auto perturb_pose(const Transform& pose, double sigma_deg, std::mt19937& gen) -> Transform {
    if (sigma_deg <= 0.0) return pose;
    std::normal_distribution<double> noise(0.0, util::deg2rad(sigma_deg));
    const auto q_base = pose.orientation.make<Eigen::Quaterniond>();
    const Eigen::Quaterniond q = Eigen::AngleAxisd { noise(gen), Eigen::Vector3d::UnitZ() }
                               * Eigen::AngleAxisd { noise(gen), Eigen::Vector3d::UnitY() }
                               * q_base;
    return Transform { pose.translation, Orientation { q } };
}

}  // namespace

int main(int argc, char** argv) {
    const auto cli = parse_cli(argc, argv);
    auto cfg = cfg::load_config(cli.config);
    if (!cli.mode.empty()) cfg.input.mode = cli.mode;
    if (!cli.source.empty()) cfg.input.source = cli.source;
    if (cli.max_frames >= 0) cfg.input.max_frames = cli.max_frames;
    if (cli.no_display) cfg.display.enabled = false;
    cfg::print_config(cfg);

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
    std::unique_ptr<FrameSource> source;
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
        // 视频模式的合成帧率从 yaml 的 input.video_fps 读取（避免 Jetson GStreamer 的
        // CAP_PROP_FPS 查询干扰解码器）；0 或越界时由 FrameSource 回退到 30。
        double video_fps = cfg.input.video_fps;
        if (video_fps <= 0.0 || video_fps > 1000.0) video_fps = 30.0;
        source = is_camera ? make_camera_source(src) : make_video_source(src, video_fps);
        if (!source) {
            std::fprintf(stderr, "无法打开数据源: %s\n", src.c_str());
            return 4;
        }
        std::printf("[aim] source: %s%s\n", source->name().c_str(),
            is_video ? cv::format(" (fps %.2f)", video_fps).c_str() : "");
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
    bool last_fire = false;
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

    std::optional<DetectWorker> detect_worker;
    if (!is_virtual) detect_worker.emplace(detector);
    bool pipeline_primed = false;

    // 喂给 RuneModel 的相机外参。静止相机下恒等于标定值；云台运动仿真下每帧刷新，
    // 且可带上回读滞后与量测噪声（真机上这个值来自 IMU 回读）。
    const auto& gimbal = cfg.virtual_rune.gimbal;
    Transform model_pose = cfg.camera.transform();
    std::mt19937 gimbal_gen { gimbal.seed };
    Timestamp sim_start{};

    const auto run_frame = [&](std::vector<RuneIcon>& icons,
                                std::vector<RuneBullseye>& bullseyes, Timestamp now) {
        tracking_corrected = false;
        // 外参每帧刷新。注意必须放在 init 之前：云台运动下 init() 也要用当前帧的外参，
        // 此前只在 rune_inited 之后更新，静止相机看不出问题，一旦外参时变就会用错。
        model.update_transform(model_pose);
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
        if (cmd.found && (frame_id % 30 == 0 || cmd.fire != last_fire)) {
            std::printf("frame %d | %s | yaw=%.2f pitch=%.2f fly=%.3fs | fire=%d | %s\n", frame_id,
                cmd.state_name(), cmd.yaw, cmd.pitch, cmd.fly_time, static_cast<int>(cmd.fire),
                cmd.reason.c_str());
        }
        last_fire = cmd.fire;
    };

    while (true) {
        if (cfg.input.max_frames > 0 && frame_id >= cfg.input.max_frames) break;

        // 每帧的时间戳由各分支确定：虚拟模式用挂钟，video/camera 用帧**自带的采集时刻**。
        // 后者是本次改动的要点——此前在循环顶部取 Clock::now()，而流水线当轮处理的是
        // 上一轮采集的帧，观测被贴上晚一个帧周期的时间戳，EKF 状态因此系统性滞后。
        Timestamp now {};
        cv::Mat frame;  // 仅可视化用；与源共享像素，不拷贝
        icons.clear();
        bullseyes.clear();
        bool last_frame = false;

        if (is_virtual) {
            if (paused) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); continue; }
            now = Clock::now();
            if (sim_start == Timestamp{}) sim_start = now;
            const auto t_sec = std::chrono::duration<double>(now - sim_start).count();
            if (gimbal.enable) {
                // 真实外参驱动观测生成；喂给 EKF 的那份滞后 transform_delay 秒并叠加噪声。
                // 负的 t 直接代入正弦即可，正是"纯延迟"该有的外推行为。
                virtual_rune->update_transform(gimbal_pose(cfg, t_sec));
                model_pose = perturb_pose(gimbal_pose(cfg, t_sec - gimbal.transform_delay),
                    gimbal.transform_noise, gimbal_gen);
            }
            virtual_rune->update(now);
            const auto virtual_icons = virtual_rune->icons();
            icons.insert(icons.end(), virtual_icons.begin(), virtual_icons.end());
            const auto virtual_bullseyes = virtual_rune->bullseyes();
            bullseyes.insert(bullseyes.end(), virtual_bullseyes.begin(), virtual_bullseyes.end());

            // 画幅裁剪：VirtualRune 的投影只判断 z>0，不判画幅，目标转出视野后仍会
            // 产生观测。真实检测器要求 5 个关键点全部在画幅内（detector.cpp 的
            // in_bounds），这里对齐该判据，否则云台摆动实验会得到偏乐观的结果。
            const auto inside = [w = cfg.camera.image_width, h = cfg.camera.image_height](
                                    const Point2d& p) {
                return p.x >= 0.0 && p.x < w && p.y >= 0.0 && p.y < h;
            };
            std::erase_if(icons, [&](const RuneIcon& i) { return !inside(i.center); });
            std::erase_if(bullseyes, [&](const RuneBullseye& b) {
                return !inside(b.center)
                    || std::ranges::any_of(b.corners,
                        [&](const Point2d& c) { return !inside(c); });
            });
            // 画布尺寸跟随标定画幅：此前写死 960×540 与内参 (cx=720,cy=540) 不符，
            // 关键点会画到画布外看不见。
            if (cfg.display.enabled)
                frame = cv::Mat::zeros(cfg.camera.image_height, cfg.camera.image_width, CV_8UC3);
        } else {
            // 单深度流水线：检测第 N+1 帧的同时跑第 N 帧的 EKF/火控。
            // 预热轮只提交，没有可处理的结果。
            if (!pipeline_primed) {
                auto first = source->grab();
                if (!first.valid()) break;
                detect_worker->submit(std::move(first));
                pipeline_primed = true;
                continue;
            }

            Frame done;
            RuneDetector::Elements elements;
            detect_worker->take(done, elements);

            // 先把下一帧丢进检测线程，再处理本帧，两者重叠。
            auto next = source->grab();
            last_frame = !next.valid();
            if (!last_frame) detect_worker->submit(std::move(next));

            icons = std::move(elements.icons);
            bullseyes = std::move(elements.bullseyes);
            now = done.stamp;
            // 浅拷贝，与 Frame 共享像素；真正的拷贝只在开了可视化时的 clone() 发生。
            if (cfg.display.enabled && done.valid()) frame = *done.image;
        }

        // ---- 时间步进 ----
        if (!paused && last_now != Timestamp{}) {
            dt = std::clamp(std::chrono::duration<double>(now - last_now).count(), 0.0, 0.5);
        }
        last_now = now;
        if (!paused) run_frame(icons, bullseyes, now);

        // ---- 可视化（统一 draw 层）----
        if (cfg.display.enabled && !frame.empty()) {
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
            if (key == 'q' || key == 27) break;
            if (key == ' ') paused = !paused;
            if (key == 's') cv::imwrite(cv::format("shot_%04d.png", frame_id), display);
        }
        ++frame_id;
        if (last_frame) break;  // 数据源已耗尽：本帧处理并显示完再退出
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
    if (is_virtual && gimbal.enable) {
        std::printf("  gimbal      : yaw=%.1fdeg@%.2fHz pitch=%.1fdeg@%.2fHz "
                    "delay=%.3fs noise=%.3fdeg\n",
            gimbal.yaw_amp, gimbal.yaw_freq, gimbal.pitch_amp, gimbal.pitch_freq,
            gimbal.transform_delay, gimbal.transform_noise);
    }
    std::printf("  predict err : mean=%.4f rad (%.2f deg), max=%.4f rad (%.2f deg), n=%zu\n",
        s.mean_error, s.mean_error * 180.0 / std::numbers::pi, s.max_error,
        s.max_error * 180.0 / std::numbers::pi, s.samples);
    return 0;
}
