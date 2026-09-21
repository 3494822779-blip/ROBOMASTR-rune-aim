// rune_aim —— 统一命令行入口（video / fixed_video / virtual / camera）
//
// 用法：
//   ./build/rune_aim -c config/rune_small_video.yaml
//   ./build/rune_aim -c config/rune_large_virtual.yaml --no-display
//   ./build/rune_aim -c config/camera_sentry.yaml --mode camera --source 0
//
// 覆盖项（其余参数一律进 yaml）：--mode --source --max-frames --output --no-display
// 按键：q/ESC 退出，空格 暂停，s 截图，v 切换可视化
//
// 链路：数据源(视频/虚拟符/相机) → RuneDetector(视频/相机) 或 VirtualRune
//       → RuneModel(EKF 跟踪) → RuneFireControl(预瞄+开火) → RuneDiagnostics(误差)
#ifdef RUNE_ENABLE_ROS2
#include "ros/telemetry.hpp"
#endif
#include "core/angle.hpp"
#include "core/config_loader.hpp"
#include "core/frame_source.hpp"
#include "core/simulator.hpp"
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
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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
    std::string output;          // 空 = 不保存标注视频
    int max_frames = -1;         // -1 = 用 yaml
    bool no_display = false;
    bool display = false;
    bool sim_fire = false;
    bool ros = false;
    double ros_image_fps = 15.0;
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
        else if (a == "--output") cli.output = next("--output");
        else if (a == "--max-frames") cli.max_frames = std::atoi(next("--max-frames"));
        else if (a == "--ros-image-fps") {
            char* end = nullptr;
            const char* value = next("--ros-image-fps");
            cli.ros_image_fps = std::strtod(value, &end);
            if (*end != '\0' || !std::isfinite(cli.ros_image_fps) ||
                cli.ros_image_fps <= 0 || cli.ros_image_fps > 240) {
                std::fprintf(stderr, "--ros-image-fps must be in (0, 240]\n");
                std::exit(2);
            }
        }
        else if (a == "--sim-fire") cli.sim_fire = true;
        else if (a == "--ros") cli.ros = true;
        else if (a == "--display") cli.display = true;
        else if (a == "--no-display") cli.no_display = true;
        else if (a == "-h" || a == "--help") {
            std::printf("用法: rune_aim -c <config.yaml> [--mode video|fixed_video|virtual|camera|simulator]"
                        " [--source <path|dev>] [--output <video.mp4>] [--max-frames N]"
                        " [--display|--no-display] [--sim-fire] [--ros] [--ros-image-fps 15]\n");
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

int main(int argc, char** argv) try {
    const auto cli = parse_cli(argc, argv);
    auto cfg = cfg::load_config(cli.config);
    if (!cli.mode.empty()) cfg.input.mode = cli.mode;
    if (!cli.source.empty()) cfg.input.source = cli.source;
    if (cli.max_frames >= 0) cfg.input.max_frames = cli.max_frames;
    if (cli.display) cfg.display.enabled = true;
    if (cli.no_display) cfg.display.enabled = false;
    cfg::print_config(cfg);
#ifdef RUNE_ENABLE_ROS2
    std::unique_ptr<RosTelemetry> ros;
    if (cli.ros) ros = std::make_unique<RosTelemetry>(cfg, cli.ros_image_fps);
#else
    if (cli.ros) {
        std::fprintf(stderr, "ROS support requires cmake -DENABLE_ROS2=ON\n");
        return 2;
    }
#endif

    const bool is_virtual = cfg.input.mode == "virtual";
    const bool is_video   = cfg.input.mode == "video" || cfg.input.mode == "fixed_video";
    const bool is_fixed_video = cfg.input.mode == "fixed_video";
    const bool is_simulator = cfg.input.mode == "simulator";
    const bool is_camera  = cfg.input.mode == "camera";
    if (!is_virtual && !is_video && !is_camera && !is_simulator) {
        std::fprintf(stderr, "未知 mode: %s（可选 video|fixed_video|virtual|camera|simulator）\n",
            cfg.input.mode.c_str());
        return 2;
    }
    const bool render_frames = cfg.display.enabled || !cli.output.empty();

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
    std::unique_ptr<SimulatorClient> simulator;
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
    } else if (is_simulator) {
        simulator = std::make_unique<SimulatorClient>(cfg.input.source);
    } else {
        const std::string src = cfg.input.source.empty() ? "data/rune_test_h264.mp4" : cfg.input.source;
        // 视频时间优先使用逐帧 PTS；input.video_fps 仅用于 PTS 缺失时外推。
        double video_fps = cfg.input.video_fps;
        if (video_fps <= 0.0 || video_fps > 1000.0) video_fps = 30.0;
        source = is_camera ? make_camera_source(src, cfg.camera.capture_width, cfg.camera.capture_height,
                                                cfg.camera.capture_fps, cfg.camera.buffer_size)
                           : make_video_source(src, video_fps);
        if (!source) {
            std::fprintf(stderr, "无法打开数据源: %s\n", src.c_str());
            return 4;
        }
        std::printf("[aim] source: %s%s\n", source->name().c_str(),
            is_video ? cv::format(" (PTS timestamps, fallback fps %.2f)", video_fps).c_str() : "");
        if (is_fixed_video) {
            std::printf("[aim] camera pose: FIXED (camera.transform; no IMU updates)\n");
        }
    }

    // ---- 主循环 ----
    const auto window = is_simulator ? "Rune Simulator Debug" : "rune_aim";
    if (cfg.display.enabled) {
        cv::namedWindow(window, cv::WINDOW_NORMAL);
        cv::resizeWindow(window, 1100, 825);
    }
    bool overlays = true, sim_control = true, sim_fire = cli.sim_fire;
    double processing_ms = 0.0;
    cv::VideoWriter output_video;
    int output_frames = 0;
    double output_fps = is_virtual ? cfg.input.hz
        : is_camera ? cfg.camera.capture_fps : cfg.input.video_fps;
    if (!std::isfinite(output_fps) || output_fps <= 0.0 || output_fps > 1000.0)
        output_fps = 30.0;

    bool rune_inited = false;
    Timestamp rune_stamp{};
    Timestamp rune_corrected_stamp{};
    Timestamp last_now{};
    double dt = 1.0 / cfg.input.hz;
    int frame_id = 0, init_count = 0, aim_count = 0;
    int fire_count = 0, shot_count = 0, hit_count = 0, observed_hit_count = 0;
    double hit_error_sum_px = 0.0, hit_error_max_px = 0.0;
    bool paused = false;
    bool tracking_corrected = false;
    bool last_fire = false;
    // 实时显示帧率（按实际完成绘制的帧数统计，适用于 video/camera/virtual）。
    double display_fps = 0.0;
    int fps_frames = 0;
    auto fps_stamp = std::chrono::steady_clock::now();
    RuneFireControl::Command last_cmd{};  // 主循环计算，可视化只读，避免重复推进火控状态机
    struct HitEvent {
        Point3d point;
        int feature_id = -1;
    };
    std::multimap<Timestamp, HitEvent> pending_hits;
    std::vector<HitEvent> frame_hits;
    cv::Mat last_display;
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
        last_cmd = {};  // Never expose a previous frame command after loss/reinitialization.
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
            fire.reset();
            last_fire = false;
            return;
        }
        model.predict(dt, now);
        rune_stamp = now;
        const bool corrected = model.correct(icons, bullseyes);
        if (model.diverged()) {
            rune_inited = false;
            fire.reset();
            last_fire = false;
            return;
        }
        if (corrected) rune_corrected_stamp = now;
        tracking_corrected = corrected;

        // ---- 火控 + 诊断 ----
        const auto state = model.state();
        auto firing_state = state;
        if (simulator) {
            firing_state.x -= simulator->muzzle.x;
            firing_state.y -= simulator->muzzle.y;
            firing_state.z -= simulator->muzzle.z;
        }
        last_cmd = fire.update(firing_state, now);
        if (simulator && last_cmd.has_attack_point) {
            last_cmd.attack_point.x += simulator->muzzle.x;
            last_cmd.attack_point.y += simulator->muzzle.y;
            last_cmd.attack_point.z += simulator->muzzle.z;
        }
        const auto& cmd = last_cmd;
        if (cmd.found) {
            ++aim_count;
            if (cmd.fire) ++fire_count;
            if (cmd.shot_started) ++shot_count;
        }
        const auto diag_samples_before = diag.stats().samples;
        diag.push_observation(now, state.rotation_angle);
        const auto diag_after = diag.stats();
        if (diag_after.samples != diag_samples_before) {
            std::printf("[predict-error] frame %d signed=%.4f rad (%.2f deg)\n", frame_id,
                diag_after.latest_error, util::rad2deg(diag_after.latest_error));
        }
        if (cmd.found && cmd.shot_started) {
            const auto prediction_mode = state.sine_valid ? "sine"
                : state.use_prediction_speed ? "linear" : "ekf";
            std::printf("[predict] frame %d target=%d mode=%s speed=%.4f ekf=%.4f rad/s cost=%.6f"
                        " sine[v=%.4f a=%.4f w=%.4f]\n",
                frame_id, cmd.target_feature_id, prediction_mode, state.rotation_speed,
                state.filter_rotation_speed, state.prediction_cost, state.sine_v,
                state.sine_a, state.sine_omega);
            const auto hit_dt = cfg.fire.algorithmic_delay + cfg.fire.shoot_delay + cmd.fly_time;
            auto clone = state;
            clone.transition(hit_dt);
            diag.push_predict(now + std::chrono::duration_cast<Timestamp::duration>(
                                      std::chrono::duration<double>(hit_dt)),
                clone.rotation_angle);
            if (render_frames && cfg.display.hitpoint && cmd.has_attack_point) {
                pending_hits.emplace(
                    now + std::chrono::duration_cast<Timestamp::duration>(
                              std::chrono::duration<double>(hit_dt)),
                    HitEvent { cmd.attack_point, cmd.target_feature_id });
            }
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
        bool ros_image = false;
#ifdef RUNE_ENABLE_ROS2
        if (ros && !ros->ok()) break;
        ros_image = ros && ros->image_due();
#endif
        if (cfg.input.max_frames > 0 && frame_id >= cfg.input.max_frames) break;
        if (paused) {
            const auto key = cv::waitKey(10);
            if (key == 'q' || key == 27) break;
            if (key == ' ') paused = false;
            if (key == 's' && !last_display.empty())
                cv::imwrite(cv::format("shot_%04d.png", frame_id), last_display);
            continue;
        }

        // 每帧的时间戳由各分支确定：虚拟模式用挂钟，video/camera 用帧**自带的采集时刻**。
        // 后者是本次改动的要点——此前在循环顶部取 Clock::now()，而流水线当轮处理的是
        // 上一轮采集的帧，观测被贴上晚一个帧周期的时间戳，EKF 状态因此系统性滞后。
        const auto processing_start = Clock::now();
        Timestamp now {};
        Timestamp following_stamp {}; // 下一视频帧时间，用于按真实帧间隔选择命中帧
        cv::Mat frame;  // 仅可视化用；与源共享像素，不拷贝
        icons.clear();
        bullseyes.clear();
        bool last_frame = false;

        if (is_virtual) {
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
            if (render_frames || ros_image)
                frame = cv::Mat::zeros(cfg.camera.image_height, cfg.camera.image_width, CV_8UC3);
        } else if (simulator) {
            auto input = simulator->grab();
            model_pose = simulator->pose;
            cfg.camera.matrix = simulator->matrix;
            cfg.camera.distortion = {0,0,0,0,0};
            cfg.camera.image_width = simulator->width;
            cfg.camera.image_height = simulator->height;
            model.update_camera(cfg.camera.matrix, cfg.camera.distortion);
            detect_worker->submit(std::move(input));
            Frame done;
            RuneDetector::Elements elements;
            detect_worker->take(done, elements);
            now = done.stamp;
            icons = std::move(elements.icons);
            bullseyes = std::move(elements.bullseyes);
            if (render_frames || ros_image) frame = *done.image;
            if (frame_id % 30 == 0)
                std::printf("[sim-detect] frame=%d icons=%zu blades=%zu\n", frame_id, icons.size(), bullseyes.size());
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
            if (!last_frame) {
                following_stamp = next.stamp;
                detect_worker->submit(std::move(next));
            }

            icons = std::move(elements.icons);
            bullseyes = std::move(elements.bullseyes);
            now = done.stamp;
            // 浅拷贝，与 Frame 共享像素；真正的拷贝只在开了可视化时的 clone() 发生。
            if ((render_frames || ros_image) && done.valid()) frame = *done.image;
        }

        // ---- 时间步进 ----
        if (!paused && last_now != Timestamp{}) {
            // 使用帧自身采集时间，避免相机抖动/丢帧时固定 dt 导致预测偏差。
            // 极端时间戳异常时限幅，防止一次坏帧把状态推飞。
            dt = std::clamp(std::chrono::duration<double>(now - last_now).count(), 1e-4, 0.1);
        }
        last_now = now;
        if (!paused) run_frame(icons, bullseyes, now);
        if (simulator) simulator->command(sim_control && last_cmd.found, last_cmd.yaw, last_cmd.pitch,
                                           sim_fire && last_cmd.fire);
        processing_ms = std::chrono::duration<double, std::milli>(Clock::now() - processing_start).count();

        frame_hits.clear();
        // 用当前帧与下一帧的真实时间中点做边界，VFR 下仍选择最接近命中时刻的帧。
        auto hit_boundary = now;
        if (following_stamp > now) {
            hit_boundary = now + (following_stamp - now) / 2;
        } else if (is_virtual) {
            hit_boundary = now + std::chrono::duration_cast<Timestamp::duration>(
                                     std::chrono::duration<double>(0.5 / cfg.input.hz));
        }
        while (!pending_hits.empty() && pending_hits.begin()->first <= hit_boundary) {
            frame_hits.push_back(pending_hits.begin()->second);
            pending_hits.erase(pending_hits.begin());
        }

#ifdef RUNE_ENABLE_ROS2
        if (ros) {
            const auto state = rune_inited ? std::optional(model.state()) : std::nullopt;
            ros->publish(now, frame, icons, bullseyes, state ? &*state : nullptr,
                         last_cmd, diag.stats(), tracking_corrected);
        }
#endif
        // ---- 可视化（统一 draw 层）----
        if (render_frames && !frame.empty()) {
            cv::Mat display = frame.clone();
            debug::DrawOptions opt{
                .keypoints = overlays && cfg.display.keypoints,
                .aimpoint = overlays && cfg.display.aimpoint,
                .hitpoint = overlays && cfg.display.hitpoint,
                .state_text = cfg.display.state_text,
                .error_text = cfg.display.error_text,
            };
            if (opt.keypoints) debug::draw_detection(display, icons, bullseyes);
            if (rune_inited) {
                const auto& cmd = last_cmd;  // 复用主循环结果，不重复调用 fire.update
                if (opt.aimpoint && tracking_corrected && cmd.found) {
                    debug::draw_aimpoint(display, model.state(),
                        cfg.fire.algorithmic_delay + cfg.fire.shoot_delay + cmd.fly_time,
                        cfg.camera.matrix, cfg.camera.distortion, model_pose);
                }
                if (opt.state_text || opt.error_text) debug::draw_status(display, cmd, diag.stats(), opt);
            }
            // 已发出的弹丸即使后续丢失跟踪，也应在其预计命中帧保留落点回放。
            if (opt.hitpoint) {
                for (const auto& hit : frame_hits) {
                    std::optional<Point2d> observed_center;
                    if (rune_inited && hit.feature_id >= 0) {
                        for (const auto& tracked : model.addition().tracked) {
                            if (tracked.feature_id == hit.feature_id) {
                                observed_center = tracked.point;
                                break;
                            }
                        }
                    }
                    const auto draw_result = debug::draw_hitpoint(display, hit.point,
                        observed_center, cfg.camera.matrix, cfg.camera.distortion, model_pose);
                    if (draw_result.hitpoint_drawn)
                        ++hit_count;
                    if (draw_result.observation_error_px) {
                        ++observed_hit_count;
                        hit_error_sum_px += *draw_result.observation_error_px;
                        hit_error_max_px = std::max(
                            hit_error_max_px, *draw_result.observation_error_px);
                    }
                }
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
            if (cfg.display.enabled) {
                cv::putText(display, cv::format("PROC FPS: %.1f", display_fps), cv::Point(20, 115),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            }
            if (simulator && cfg.display.enabled) {
                const cv::Scalar color = sim_control && simulator->auto_aim_enabled
                    ? cv::Scalar(80,255,80) : cv::Scalar(0,180,255);
                const std::vector<std::string> lines = {
                    cv::format("CONTROL %s / SERVER %s | SIM FIRE %s / ALLOWED %s",
                        sim_control?"ON":"HOLD", simulator->auto_aim_enabled?"ON":"OFF",
                        sim_fire?"ON":"OFF", simulator->fire_allowed?"YES":"NO"),
                    cv::format("Gimbal %.1f / %.1f deg | shots %u hits %u | frame %.1f ms",
                        simulator->actual_yaw, simulator->actual_pitch, simulator->shots, simulator->hits, processing_ms),
                    cv::format("R icons %zu / blades %zu | tracking %s | %s", icons.size(), bullseyes.size(),
                        tracking_corrected?"CORRECTED":rune_inited?"PREDICT ONLY":"LOST", last_cmd.reason.c_str()),
                    "Space: control hold | F: sim fire | R: reset tracker | V: overlays",
                    "D: detections | A: aimpoint | S: screenshot | Q/Esc: quit"
                };
                const int top = std::max(130, display.rows - 155);
                cv::rectangle(display, cv::Rect(0,top,display.cols,display.rows-top), cv::Scalar(22,22,22), cv::FILLED);
                for (std::size_t i=0; i<lines.size(); ++i)
                    cv::putText(display,lines[i],cv::Point(15,top+25+27*static_cast<int>(i)),
                        cv::FONT_HERSHEY_SIMPLEX,0.58,color,1,cv::LINE_AA);
            }
            if (!cli.output.empty()) {
                if (!output_video.isOpened()) {
                    const auto fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                    if (!output_video.open(cli.output, fourcc, output_fps, display.size(), true)) {
                        std::fprintf(stderr, "无法创建输出视频: %s\n", cli.output.c_str());
                        return 5;
                    }
                    std::printf("[aim] recording: %s (%.3f FPS, %dx%d)\n",
                        cli.output.c_str(), output_fps, display.cols, display.rows);
                }
                output_video.write(display);
                ++output_frames;
            }
            if (cfg.display.enabled) {
                last_display = display;
                cv::imshow(window, display);
                const auto key = cv::waitKey(1);
                if (key == 'q' || key == 27) break;
                if (key == ' ') {
                    if (simulator) sim_control = !sim_control;
                    else paused = !paused;
                }
                if (key == 'v') overlays = !overlays;
                if (key == 'd') cfg.display.keypoints = !cfg.display.keypoints;
                if (key == 'a') cfg.display.aimpoint = !cfg.display.aimpoint;
                if (simulator && key == 'f') sim_fire = !sim_fire;
                if (simulator && key == 'r') {
                    rune_inited = false;
                    fire.reset(); last_cmd = {}; last_fire = false;
                    pending_hits.clear();
                }
                // GTK builds may return -1 when this property is unsupported.
                if (cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) == 0) break;
                if (key == 's') cv::imwrite(cv::format("shot_%04d.png", frame_id), display);
            }
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
    std::printf("  shot_events : %d\n", shot_count);
    std::printf("  hit_markers : %d\n", hit_count);
    std::printf("  hit observed: %d", observed_hit_count);
    if (observed_hit_count > 0) {
        std::printf(" (mean=%.2f px, max=%.2f px)",
            hit_error_sum_px / static_cast<double>(observed_hit_count), hit_error_max_px);
    }
    std::printf("\n");
    if (!cli.output.empty())
        std::printf("  output_video: %s (%d frames @ %.3f FPS)\n",
            cli.output.c_str(), output_frames, output_fps);
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

catch (const std::exception& error) {
    std::fprintf(stderr, "[aim] %s\n", error.what());
    return 1;
}
