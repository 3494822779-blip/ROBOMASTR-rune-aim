// 外参标定后的实弹闭环测试：显示目标点与弹道瞄准点，点击实际弹孔后迭代
// 修正 fire.offset_yaw / fire.offset_pitch。程序只写独立结果文件，不修改输入配置。

#include "core/config_loader.hpp"
#include "core/frame_source.hpp"
#include "fire/trajectory.hpp"

#include <eigen3/Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr double kRadToDeg = 180.0 / std::numbers::pi;

struct Options {
    std::string config = "config/camera_sentry.yaml";
    std::string source;
    std::string output = "/tmp/ballistic_calibration.yaml";
    std::string log = "/tmp/ballistic_calibration.csv";
    double distance_m = 7.0;
    double gain = 0.7;
    bool self_test = false;
};

struct Sample {
    int index = 0;
    cv::Point2f target;
    cv::Point2f impact;
    double error_yaw = 0.0;
    double error_pitch = 0.0;
    double offset_yaw_before = 0.0;
    double offset_pitch_before = 0.0;
    double offset_yaw_after = 0.0;
    double offset_pitch_after = 0.0;
};

[[noreturn]] void usage(const char* program, int status) {
    std::fprintf(status == 0 ? stdout : stderr,
        "Usage: %s [-c config.yaml] [--distance M] [--source DEVICE|PIPELINE]\n"
        "          [--gain 0.7] [--output result.yaml] [--log rounds.csv] [--self-test]\n"
        "Mouse: right=set target, left=record actual impact\n"
        "Keys : SPACE=freeze, U=undo, S=save, [/] distance -/+, -/+ speed, R=center, Q=quit\n",
        program);
    std::exit(status);
}

auto parse_double(const char* text, const char* name, const char* program) -> double {
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value)) {
        std::fprintf(stderr, "Invalid %s: %s\n", name, text);
        usage(program, 2);
    }
    return value;
}

auto parse_args(int argc, char** argv) -> Options {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next = [&](const char* name) -> const char* {
            if (++i >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", name);
                usage(argv[0], 2);
            }
            return argv[i];
        };
        if (arg == "-c" || arg == "--config") options.config = next("--config");
        else if (arg == "--source") options.source = next("--source");
        else if (arg == "--distance")
            options.distance_m = parse_double(next("--distance"), "distance", argv[0]);
        else if (arg == "--gain") options.gain = parse_double(next("--gain"), "gain", argv[0]);
        else if (arg == "--output") options.output = next("--output");
        else if (arg == "--log") options.log = next("--log");
        else if (arg == "--self-test") options.self_test = true;
        else if (arg == "-h" || arg == "--help") usage(argv[0], 0);
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            usage(argv[0], 2);
        }
    }
    if (options.distance_m <= 0.1 || options.distance_m > 100.0 ||
        options.gain <= 0.0 || options.gain > 1.0) {
        std::fprintf(stderr, "distance must be in (0.1, 100] m and gain in (0, 1]\n");
        usage(argv[0], 2);
    }
    return options;
}

auto camera_matrix(const rmcs::cfg::CameraConfig& camera) -> cv::Mat {
    return cv::Mat(3, 3, CV_64F, const_cast<double*>(camera.matrix.data())).clone();
}

auto distortion(const rmcs::cfg::CameraConfig& camera) -> cv::Mat {
    return cv::Mat(1, 5, CV_64F, const_cast<double*>(camera.distortion.data())).clone();
}

// 将原始畸变图像中的像素投到距相机 optical forward 为 distance_m 的平面，
// 再用已标定的 camera->Odom 外参变换到弹道坐标系。
auto pixel_to_odom(const cv::Point2f& pixel, double distance_m,
    const rmcs::cfg::CameraConfig& camera) -> std::optional<Eigen::Vector3d> {
    std::vector<cv::Point2f> input { pixel }, normalized;
    cv::undistortPoints(input, normalized, camera_matrix(camera), distortion(camera));
    if (normalized.empty() || !std::isfinite(normalized[0].x) || !std::isfinite(normalized[0].y))
        return std::nullopt;

    // OpenCV (right, down, forward) -> project ROS convention (forward, left, up).
    const Eigen::Vector3d point_camera {
        distance_m, -normalized[0].x * distance_m, -normalized[0].y * distance_m
    };
    auto orientation = camera.orientation.make<Eigen::Quaterniond>();
    if (orientation.norm() < 1e-9) return std::nullopt;
    orientation.normalize();
    return camera.translation.make<Eigen::Vector3d>() + orientation * point_camera;
}

auto direction_angles(const Eigen::Vector3d& point) -> std::pair<double, double> {
    const double horizontal = std::hypot(point.x(), point.y());
    return { std::atan2(point.y(), point.x()), -std::atan2(point.z(), horizontal) };
}

auto wrap(double angle) -> double {
    return std::remainder(angle, 2.0 * std::numbers::pi);
}

// 将 Odom 中从枪口出发的一条 yaw/pitch 射线投回原始相机图像。
auto direction_to_pixel(double yaw, double pitch, double distance_m,
    const rmcs::cfg::CameraConfig& camera) -> std::optional<cv::Point2f> {
    // 使用靶距而不是无穷远方向，使 camera.translation 带来的近距离视差可见。
    const Eigen::Vector3d point_odom = distance_m * Eigen::Vector3d {
        std::cos(pitch) * std::cos(yaw),
        std::cos(pitch) * std::sin(yaw),
        -std::sin(pitch)
    };
    auto orientation = camera.orientation.make<Eigen::Quaterniond>();
    if (orientation.norm() < 1e-9) return std::nullopt;
    orientation.normalize();
    const Eigen::Vector3d point_camera = orientation.conjugate() *
        (point_odom - camera.translation.make<Eigen::Vector3d>());
    if (point_camera.x() <= 1e-6) return std::nullopt;

    // ROS camera (forward, left, up) -> OpenCV (right, down, forward).
    std::vector<cv::Point3d> object_points {
        { -point_camera.y(), -point_camera.z(), point_camera.x() }
    };
    std::vector<cv::Point2d> image_points;
    cv::projectPoints(object_points, cv::Vec3d::all(0), cv::Vec3d::all(0),
        camera_matrix(camera), distortion(camera), image_points);
    if (image_points.empty() || !std::isfinite(image_points[0].x) ||
        !std::isfinite(image_points[0].y)) return std::nullopt;
    return cv::Point2f { static_cast<float>(image_points[0].x),
                         static_cast<float>(image_points[0].y) };
}

auto ballistic_solution(const cv::Point2f& target, double distance_m,
    const rmcs::cfg::AppConfig& config) -> std::optional<rmcs::TrajectorySolution::Output> {
    const auto point = pixel_to_odom(target, distance_m, config.camera);
    if (!point) return std::nullopt;
    rmcs::TrajectorySolution trajectory;
    trajectory.input.v0 = config.fire.bullet_speed;
    trajectory.input.point = rmcs::Point3d { point->x(), point->y(), point->z() };
    return trajectory.solve();
}

auto write_results(const Options& options, const rmcs::cfg::AppConfig& config,
    const std::vector<Sample>& samples) -> bool {
    std::ofstream yaml(options.output);
    if (!yaml) return false;
    yaml << std::setprecision(12)
         << "# Generated by ballistic_test; merge this scene override into the runtime config.\n"
         << "fire:\n"
         << "  bullet_speed: " << config.fire.bullet_speed << '\n'
         << "  offset_yaw: " << config.fire.offset_yaw << '\n'
         << "  offset_pitch: " << config.fire.offset_pitch << '\n'
         << "ballistic_test:\n"
         << "  distance_m: " << options.distance_m << '\n'
         << "  samples: " << samples.size() << '\n'
         << "  gain: " << options.gain << '\n';
    yaml.close();
    if (!yaml) return false;

    std::ofstream csv(options.log);
    if (!csv) return false;
    csv << "round,target_u,target_v,impact_u,impact_v,error_yaw_deg,error_pitch_deg,"
           "offset_yaw_rad,offset_pitch_rad\n" << std::setprecision(12);
    for (const auto& sample : samples) {
        csv << sample.index << ',' << sample.target.x << ',' << sample.target.y << ','
            << sample.impact.x << ',' << sample.impact.y << ','
            << sample.error_yaw * kRadToDeg << ',' << sample.error_pitch * kRadToDeg << ','
            << sample.offset_yaw_after << ',' << sample.offset_pitch_after << '\n';
    }
    return static_cast<bool>(csv);
}

void marker(cv::Mat& image, cv::Point2f point, const cv::Scalar& color, const char* label) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) return;
    cv::drawMarker(image, point, color, cv::MARKER_CROSS, 28, 2, cv::LINE_AA);
    cv::circle(image, point, 13, color, 2, cv::LINE_AA);
    cv::putText(image, label, point + cv::Point2f(17.0F, -10.0F),
        cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv::LINE_AA);
}

void line(cv::Mat& image, int row, const std::string& text,
    const cv::Scalar& color = cv::Scalar(245, 245, 245)) {
    cv::putText(image, text, cv::Point(18, 30 + row * 27), cv::FONT_HERSHEY_SIMPLEX,
        0.62, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(image, text, cv::Point(18, 30 + row * 27), cv::FONT_HERSHEY_SIMPLEX,
        0.62, color, 1, cv::LINE_AA);
}

struct UiState {
    cv::Point2f target;
    std::optional<cv::Point2f> pending_impact;
};

void mouse_callback(int event, int x, int y, int, void* userdata) {
    auto& ui = *static_cast<UiState*>(userdata);
    if (event == cv::EVENT_RBUTTONDOWN) ui.target = cv::Point2f(x, y);
    if (event == cv::EVENT_LBUTTONDOWN) ui.pending_impact = cv::Point2f(x, y);
}

}  // namespace

int main(int argc, char** argv) {
    auto options = parse_args(argc, argv);
    auto config = rmcs::cfg::load_config(options.config);
    if (!options.source.empty()) config.input.source = options.source;

    UiState ui { cv::Point2f(static_cast<float>(config.camera.matrix[2]),
                             static_cast<float>(config.camera.matrix[5])), std::nullopt };
    if (options.self_test) {
        const auto solution = ballistic_solution(ui.target, options.distance_m, config);
        if (!solution) {
            std::fprintf(stderr, "Self-test failed: no ballistic solution\n");
            return 5;
        }
        std::printf("distance=%.3fm speed=%.3fm/s yaw=%.6frad pitch=%.6frad fly=%.6fs\n",
            options.distance_m, config.fire.bullet_speed, solution->yaw, solution->pitch,
            solution->fly_time);
        return 0;
    }

    if (config.input.source.empty()) {
        std::fprintf(stderr, "Camera source is empty; set input.source or use --source\n");
        return 3;
    }
    auto source = rmcs::make_camera_source(config.input.source, config.camera.capture_width,
        config.camera.capture_height, config.camera.capture_fps, config.camera.buffer_size);
    if (!source) {
        std::fprintf(stderr, "Failed to open camera source: %s\n", config.input.source.c_str());
        return 3;
    }

    const std::string window = "ballistic_test";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::setMouseCallback(window, mouse_callback, &ui);
    std::vector<Sample> samples;
    cv::Mat frozen;
    bool freeze = false;
    bool size_warned = false;
    std::cout << "Right-click target; fire only after the gimbal reaches AIM; "
                 "left-click the new impact.\n";

    for (;;) {
        cv::Mat frame;
        if (freeze && !frozen.empty()) {
            frame = frozen.clone();
        } else {
            auto captured = source->grab();
            if (!captured.valid()) {
                std::fprintf(stderr, "Camera read failed\n");
                return 4;
            }
            frame = captured.image->clone();
            if (!size_warned && (frame.cols != config.camera.image_width ||
                                 frame.rows != config.camera.image_height)) {
                std::fprintf(stderr,
                    "WARNING: frame is %dx%d but calibration is %dx%d; pixel geometry may be wrong\n",
                    frame.cols, frame.rows, config.camera.image_width, config.camera.image_height);
                size_warned = true;
            }
        }

        if (ui.pending_impact) {
            const auto target_point = pixel_to_odom(ui.target, options.distance_m, config.camera);
            const auto impact_point = pixel_to_odom(*ui.pending_impact, options.distance_m, config.camera);
            if (target_point && impact_point) {
                const auto [target_yaw, target_pitch] = direction_angles(*target_point);
                const auto [impact_yaw, impact_pitch] = direction_angles(*impact_point);
                Sample sample;
                sample.index = static_cast<int>(samples.size()) + 1;
                sample.target = ui.target;
                sample.impact = *ui.pending_impact;
                sample.error_yaw = wrap(impact_yaw - target_yaw);
                sample.error_pitch = wrap(impact_pitch - target_pitch);
                sample.offset_yaw_before = config.fire.offset_yaw;
                sample.offset_pitch_before = config.fire.offset_pitch;
                // 修正量为 target - impact。gain<1 可抑制弹丸散布造成的过调。
                config.fire.offset_yaw += options.gain * wrap(target_yaw - impact_yaw);
                config.fire.offset_pitch += options.gain * wrap(target_pitch - impact_pitch);
                sample.offset_yaw_after = config.fire.offset_yaw;
                sample.offset_pitch_after = config.fire.offset_pitch;
                samples.push_back(sample);
                if (!write_results(options, config, samples))
                    std::fprintf(stderr, "Failed to save calibration results\n");
                std::printf("[round %d] miss yaw=%+.3fdeg pitch=%+.3fdeg -> offsets %.8f %.8f rad\n",
                    sample.index, sample.error_yaw * kRadToDeg, sample.error_pitch * kRadToDeg,
                    config.fire.offset_yaw, config.fire.offset_pitch);
            }
            ui.pending_impact.reset();
        }

        const auto solution = ballistic_solution(ui.target, options.distance_m, config);
        marker(frame, ui.target, cv::Scalar(0, 255, 0), "TARGET / PREDICTED HIT");
        if (!samples.empty())
            marker(frame, samples.back().impact, cv::Scalar(0, 0, 255), "LAST IMPACT");
        if (solution) {
            const auto aim = direction_to_pixel(solution->yaw + config.fire.offset_yaw,
                solution->pitch + config.fire.offset_pitch, options.distance_m, config.camera);
            if (aim) marker(frame, *aim, cv::Scalar(0, 220, 255), "AIM");
            line(frame, 0, cv::format("range %.2f m  speed %.2f m/s  flight %.3f s",
                options.distance_m, config.fire.bullet_speed, solution->fly_time));
        } else {
            line(frame, 0, "NO BALLISTIC SOLUTION", cv::Scalar(0, 0, 255));
        }
        line(frame, 1, cv::format("offset yaw %+.4f deg  pitch %+.4f deg  rounds %zu",
            config.fire.offset_yaw * kRadToDeg, config.fire.offset_pitch * kRadToDeg,
            samples.size()));
        line(frame, 2, freeze ? "FROZEN - left click impact, SPACE resumes"
                              : "LIVE - right target | SPACE freeze | left impact");
        line(frame, 3, "U undo | S save | [ ] range | - + speed | R center | Q quit",
            cv::Scalar(190, 220, 255));
        cv::imshow(window, frame);

        const int key = cv::waitKey(1) & 0xff;
        if (key == 'q' || key == 27) break;
        if (key == ' ') {
            freeze = !freeze;
            if (freeze) frozen = frame.clone();
            else frozen.release();
        } else if (key == 'u' && !samples.empty()) {
            config.fire.offset_yaw = samples.back().offset_yaw_before;
            config.fire.offset_pitch = samples.back().offset_pitch_before;
            samples.pop_back();
            write_results(options, config, samples);
        } else if (key == 's') {
            if (write_results(options, config, samples))
                std::cout << "Saved " << options.output << " and " << options.log << '\n';
            else
                std::fprintf(stderr, "Failed to save calibration results\n");
        } else if (key == '[') {
            options.distance_m = std::max(0.5, options.distance_m - 0.5);
        } else if (key == ']') {
            options.distance_m = std::min(100.0, options.distance_m + 0.5);
        } else if (key == '-' || key == '_') {
            config.fire.bullet_speed = std::max(0.5, config.fire.bullet_speed - 0.5);
        } else if (key == '+' || key == '=') {
            config.fire.bullet_speed = std::min(200.0, config.fire.bullet_speed + 0.5);
        } else if (key == 'r') {
            ui.target = cv::Point2f(static_cast<float>(config.camera.matrix[2]),
                                    static_cast<float>(config.camera.matrix[5]));
        }
    }

    if (!write_results(options, config, samples)) {
        std::fprintf(stderr, "Failed to save calibration results\n");
        return 6;
    }
    std::cout << "Result: " << options.output << "\nLog: " << options.log << '\n';
    return 0;
}
