#include "core/config_loader.hpp"

#include <yaml-cpp/yaml.h>

#include <cstdio>
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

AppConfig load_config(const std::string& yaml_path) {
    AppConfig cfg;

    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path);
    } catch (...) {
        std::fprintf(stderr, "[config] 无法读取配置文件: %s（使用全部默认值）\n", yaml_path.c_str());
        return cfg;
    }

    // ---- input ----
    const auto input = root["input"];
    cfg.input.mode       = get_or(input, "mode", cfg.input.mode);
    cfg.input.source     = get_or(input, "source", cfg.input.source);
    cfg.input.max_frames = get_or(input, "max_frames", cfg.input.max_frames);
    cfg.input.hz         = get_or(input, "hz", cfg.input.hz);

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
    cfg.track.noise_observation     = get_or(track, "noise_observation", cfg.track.noise_observation);
    cfg.track.gate_threshold        = get_or(track, "gate_threshold", cfg.track.gate_threshold);
    cfg.track.init_seed_mean_error  = get_or(track, "init_seed_mean_error", cfg.track.init_seed_mean_error);
    cfg.track.init_seed_max_error   = get_or(track, "init_seed_max_error", cfg.track.init_seed_max_error);
    cfg.track.init_center_gate      = get_or(track, "init_center_gate", cfg.track.init_center_gate);
    cfg.track.init_pitch_bound      = get_or(track, "init_pitch_bound", cfg.track.init_pitch_bound);
    cfg.track.diverge_face_angle    = get_or(track, "diverge_face_angle", cfg.track.diverge_face_angle);

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

    // ---- display ----
    const auto display = root["display"];
    cfg.display.enabled     = get_or(display, "enabled", cfg.display.enabled);
    cfg.display.keypoints   = get_or(display, "keypoints", cfg.display.keypoints);
    cfg.display.blades      = get_or(display, "blades", cfg.display.blades);
    cfg.display.aimpoint    = get_or(display, "aimpoint", cfg.display.aimpoint);
    cfg.display.state_text  = get_or(display, "state_text", cfg.display.state_text);
    cfg.display.error_text  = get_or(display, "error_text", cfg.display.error_text);

    // ---- 打印实际生效参数 ----
    std::printf("[config] %s\n", yaml_path.c_str());
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
