#include "core/config_loader.hpp"

#include <yaml-cpp/yaml.h>

#include <cstdio>
#include <filesystem>
#include <algorithm>
#include <optional>

namespace rmcs::cfg {
namespace {

template <typename T>
auto get_or(const YAML::Node& node, const char* key, T fallback) -> T {
    if (node && node[key]) {
        try { return node[key].as<T>(); }
        catch (...) { /* 类型不符时回退默认值 */ }
    }
    return fallback;
}

template <typename T, std::size_t N>
auto get_array(const YAML::Node& node, const char* key, const std::array<T, N>& fallback)
    -> std::array<T, N> {
    if (!node || !node[key]) return fallback;
    try {
        return node[key].as<std::array<T, N>>();
    } catch (...) {
        return fallback;
    }
}

}  // namespace

namespace {
// 把一段 yaml（template 或场景文件）的全部字段写入 cfg；未写的键保持 cfg 现值。
auto apply_node(AppConfig& cfg, const YAML::Node& root) -> void {

    // ---- input ----
    const auto input = root["input"];
    cfg.input.mode       = get_or(input, "mode", cfg.input.mode);
    cfg.input.source     = get_or(input, "source", cfg.input.source);
    cfg.input.max_frames = get_or(input, "max_frames", cfg.input.max_frames);
    cfg.input.hz         = get_or(input, "hz", cfg.input.hz);
    cfg.input.video_fps  = get_or(input, "video_fps", cfg.input.video_fps);

    // ---- camera ----
    const auto camera = root["camera"];
    cfg.camera.matrix      = get_array(camera, "matrix", cfg.camera.matrix);
    cfg.camera.distortion  = get_array(camera, "distortion", cfg.camera.distortion);
    if (camera && camera["transform"]) {
        try {
            const auto t = camera["transform"].as<std::array<double, 7>>();
            cfg.camera.translation = Translation{ t[0], t[1], t[2] };
            cfg.camera.orientation = Orientation{ t[3], t[4], t[5], t[6] };
        } catch (...) { /* 保持默认外参 */ }
    }

    // ---- detect ----
    const auto detect = root["detect"];
    cfg.detect.engine               = get_or(detect, "engine", cfg.detect.engine);
    cfg.detect.score_threshold      = static_cast<float>(get_or(detect, "score_threshold", double{ cfg.detect.score_threshold }));
    cfg.detect.keypoint_threshold   = static_cast<float>(get_or(detect, "keypoint_threshold", double{ cfg.detect.keypoint_threshold }));
    cfg.detect.center_distance      = static_cast<float>(get_or(detect, "center_distance", double{ cfg.detect.center_distance }));
    cfg.detect.refine_radius        = get_or(detect, "refine_radius", cfg.detect.refine_radius);
    cfg.detect.icon_refine_radius   = get_or(detect, "icon_refine_radius", cfg.detect.icon_refine_radius);
    cfg.detect.max_refine_shift     = static_cast<float>(get_or(detect, "max_refine_shift", double{ cfg.detect.max_refine_shift }));
    cfg.detect.min_refine_gradient  = static_cast<float>(get_or(detect, "min_refine_gradient", double{ cfg.detect.min_refine_gradient }));
    cfg.detect.min_refine_support   = get_or(detect, "min_refine_support", cfg.detect.min_refine_support);

    // ---- track ----
    const auto track = root["track"];
    cfg.track.noise_x               = get_or(track, "noise_x", cfg.track.noise_x);
    cfg.track.noise_y               = get_or(track, "noise_y", cfg.track.noise_y);
    cfg.track.noise_z               = get_or(track, "noise_z", cfg.track.noise_z);
    cfg.track.noise_rotation_angle  = get_or(track, "noise_rotation_angle", cfg.track.noise_rotation_angle);
    cfg.track.noise_rotation_speed  = get_or(track, "noise_rotation_speed", cfg.track.noise_rotation_speed);
    cfg.track.noise_face_yaw        = get_or(track, "noise_face_yaw", cfg.track.noise_face_yaw);
    cfg.track.noise_observation     = get_or(track, "noise_observation", cfg.track.noise_observation);
    cfg.track.gate_threshold        = get_or(track, "gate_threshold", cfg.track.gate_threshold);
    cfg.track.init_seed_mean_error  = get_or(track, "init_seed_mean_error", cfg.track.init_seed_mean_error);
    cfg.track.init_seed_max_error   = get_or(track, "init_seed_max_error", cfg.track.init_seed_max_error);
    cfg.track.init_center_gate      = get_or(track, "init_center_gate", cfg.track.init_center_gate);
    cfg.track.init_pitch_bound      = get_or(track, "init_pitch_bound", cfg.track.init_pitch_bound);
    cfg.track.diverge_face_angle    = get_or(track, "diverge_face_angle", cfg.track.diverge_face_angle);
    cfg.track.diverge_cov_max       = get_or(track, "diverge_cov_max", cfg.track.diverge_cov_max);
    cfg.track.diverge_pos_xy_max    = get_or(track, "diverge_pos_xy_max", cfg.track.diverge_pos_xy_max);
    cfg.track.diverge_pos_z_max     = get_or(track, "diverge_pos_z_max", cfg.track.diverge_pos_z_max);
    cfg.track.diverge_speed_factor  = get_or(track, "diverge_speed_factor", cfg.track.diverge_speed_factor);

    // ---- fire ----
    const auto fire = root["fire"];
    cfg.fire.bullet_speed       = get_or(fire, "bullet_speed", cfg.fire.bullet_speed);
    cfg.fire.shoot_delay        = get_or(fire, "shoot_delay", cfg.fire.shoot_delay);
    cfg.fire.algorithmic_delay  = get_or(fire, "algorithmic_delay", cfg.fire.algorithmic_delay);
    cfg.fire.max_fly_time       = get_or(fire, "max_fly_time", cfg.fire.max_fly_time);
    cfg.fire.pitch_max          = get_or(fire, "pitch_max", cfg.fire.pitch_max);
    cfg.fire.fire_cooldown_init = get_or(fire, "fire_cooldown_init", cfg.fire.fire_cooldown_init);
    cfg.fire.fire_cooldown      = get_or(fire, "fire_cooldown", cfg.fire.fire_cooldown);
    cfg.fire.fire_window        = get_or(fire, "fire_window", cfg.fire.fire_window);
    cfg.fire.data_life          = get_or(fire, "data_life", cfg.fire.data_life);
    cfg.fire.recover_time       = get_or(fire, "recover_time", cfg.fire.recover_time);
    cfg.fire.switch_angle       = get_or(fire, "switch_angle", cfg.fire.switch_angle);
    cfg.fire.switch_confirm     = get_or(fire, "switch_confirm", cfg.fire.switch_confirm);
    cfg.fire.offset_yaw         = get_or(fire, "offset_yaw", cfg.fire.offset_yaw);
    cfg.fire.offset_pitch       = get_or(fire, "offset_pitch", cfg.fire.offset_pitch);
    cfg.fire.max_iterate        = get_or(fire, "max_iterate", cfg.fire.max_iterate);
    cfg.fire.iterate_epsilon    = get_or(fire, "iterate_epsilon", cfg.fire.iterate_epsilon);

    // ---- virtual_rune ----
    const auto v = root["virtual_rune"];
    cfg.virtual_rune.large          = get_or(v, "large", cfg.virtual_rune.large);
    cfg.virtual_rune.x              = get_or(v, "x", cfg.virtual_rune.x);
    cfg.virtual_rune.y              = get_or(v, "y", cfg.virtual_rune.y);
    cfg.virtual_rune.z              = get_or(v, "z", cfg.virtual_rune.z);
    cfg.virtual_rune.face_yaw       = get_or(v, "face_yaw", cfg.virtual_rune.face_yaw);
    cfg.virtual_rune.pixel_noise_px = get_or(v, "pixel_noise_px", cfg.virtual_rune.pixel_noise_px);
    cfg.virtual_rune.dropout_prob   = get_or(v, "dropout_prob", cfg.virtual_rune.dropout_prob);

    // ---- diag ----
    const auto diag = root["diag"];
    cfg.diag.match_tolerance_ms = get_or(diag, "match_tolerance_ms", cfg.diag.match_tolerance_ms);
    cfg.diag.max_queue     = static_cast<std::size_t>(get_or(diag, "max_queue", static_cast<long long>(cfg.diag.max_queue)));
    cfg.diag.max_history   = static_cast<std::size_t>(get_or(diag, "max_history", static_cast<long long>(cfg.diag.max_history)));

    // ---- display ----
    const auto display = root["display"];
    cfg.display.enabled     = get_or(display, "enabled", cfg.display.enabled);
    cfg.display.keypoints   = get_or(display, "keypoints", cfg.display.keypoints);
    cfg.display.aimpoint    = get_or(display, "aimpoint", cfg.display.aimpoint);
    cfg.display.state_text  = get_or(display, "state_text", cfg.display.state_text);
    cfg.display.error_text  = get_or(display, "error_text", cfg.display.error_text);
}

// 加载一个 yaml 文件并应用到 cfg；成功返回 true。
auto load_into(const std::string& path, AppConfig& cfg) -> bool {
    try {
        apply_node(cfg, YAML::LoadFile(path));
        return true;
    } catch (...) {
        return false;
    }
}
}  // namespace

AppConfig load_config(const std::string& yaml_path) {
    AppConfig cfg;

    // Resolve paths relative to the scene file first, then fall back to the
    // process working directory. This makes launching the binary from a
    // different directory deterministic.
    const std::filesystem::path scene_path { yaml_path };
    const auto base_path = scene_path.parent_path() / "template.yaml";

    // ① 公共默认：config/template.yaml（唯一参数源；场景文件不写的字段都从这里来）
    auto base_loaded = load_into(base_path.string(), cfg);
    auto loaded_base_path = base_path;
    if (!base_loaded && base_path != std::filesystem::path { "config/template.yaml" }) {
        base_loaded = load_into("config/template.yaml", cfg);
        loaded_base_path = "config/template.yaml";
    }
    if (base_loaded) {
        std::printf("[config] base : %s\n", loaded_base_path.string().c_str());
    } else {
        std::printf("[config] base : (无 config/template.yaml，用代码内默认值)\n");
    }
    // ② 场景覆盖：-c 指定的文件只写与 template 的差异
    if (load_into(yaml_path, cfg)) {
        std::printf("[config] scene: %s\n", yaml_path.c_str());
    } else {
        std::fprintf(stderr, "[config] 无法读取配置文件: %s（仅用 template 默认）\n", yaml_path.c_str());
    }

    // Basic validation: fail safe by clamping unsafe values and report them
    // loudly instead of silently running with physically invalid parameters.
    auto clamp_report = [](const char* name, auto& value, auto lo, auto hi) {
        const auto old = value;
        value = std::clamp(value, lo, hi);
        if (value != old)
            std::fprintf(stderr, "[config] %s out of range; clamped to %.6g\n", name,
                static_cast<double>(value));
    };
    clamp_report("input.hz", cfg.input.hz, 1.0, 2000.0);
    clamp_report("detect.score_threshold", cfg.detect.score_threshold, 0.0F, 1.0F);
    clamp_report("detect.keypoint_threshold", cfg.detect.keypoint_threshold, 0.0F, 1.0F);
    clamp_report("fire.bullet_speed", cfg.fire.bullet_speed, 0.1, 200.0);
    clamp_report("fire.max_fly_time", cfg.fire.max_fly_time, 0.001, 10.0);
    clamp_report("fire.recover_time", cfg.fire.recover_time, 0.001, 10.0);
    clamp_report("virtual_rune.dropout_prob", cfg.virtual_rune.dropout_prob, 0.0, 1.0);
    if (cfg.fire.max_iterate < 1) {
        std::fprintf(stderr, "[config] fire.max_iterate invalid; clamped to 1\n");
        cfg.fire.max_iterate = 1;
    }

    // ---- 打印实际生效参数 ----
    std::printf("[config] input: mode=%s source=%s max_frames=%d\n", cfg.input.mode.c_str(),
        cfg.input.source.c_str(), cfg.input.max_frames);
    std::printf("[config] camera: fx=%.1f fy=%.1f cx=%.1f cy=%.1f\n", cfg.camera.matrix[0],
        cfg.camera.matrix[4], cfg.camera.matrix[2], cfg.camera.matrix[5]);
    std::printf("[config] detect: engine=%s score=%.2f keypoint=%.2f center_dist=%.1f\n",
        cfg.detect.engine.c_str(), cfg.detect.score_threshold, cfg.detect.keypoint_threshold,
        cfg.detect.center_distance);
    std::printf("[config] fire: bullet_speed=%.2f shoot_delay=%.3f algo_delay=%.3f\n",
        cfg.fire.bullet_speed, cfg.fire.shoot_delay, cfg.fire.algorithmic_delay);
    std::printf("[config] display: %s\n", cfg.display.enabled ? "on" : "off");

    return cfg;
}

}  // namespace rmcs::cfg
