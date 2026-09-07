#include "fire_control.hpp"

#include "fire/trajectory.hpp"
#include "core/angle.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace rmcs {

namespace {

// 把世界系点转成 Odom 系射线角（yaw/pitch），与 TrajectorySolution 的坐标系约定一致
auto direction_to_yaw_pitch(const Point3d& point) -> std::pair<double, double> {
    const auto distance = std::hypot(point.x, point.y);
    const auto yaw      = std::atan2(point.y, point.x);
    const auto pitch    = std::atan2(point.z, distance);
    return { yaw, pitch };
}

} // namespace

struct RuneFireControl::Impl {
    Config config; // 值拷贝：避免引用成员导致移动赋值被删除

    // 数据新鲜度
    std::size_t last_update_count = 0;
    double      data_age          = 0.0;

    // 切叶检测
    double last_rotation_angle = 0.0;
    int    jump_count          = 0;
    bool   first_frame         = true;

    // 状态机
    State     state        = State::LOST;
    double    cooldown     = 0.0; // 剩余冷却时间
    double    firing_time  = 0.0; // 当前开火窗口内已累计开火时长
    Timestamp last_time    = { };

    // 恢复插值（数据过期 → 平滑回符心）
    bool    recovering         = false;
    double  recover_start_yaw   = 0.0;
    double  recover_start_pitch = 0.0;
    double  recover_elapsed     = 0.0;
    Point3d recover_center      = { };

    // 上一次输出（供恢复插值起点）
    double last_yaw   = 0.0;
    double last_pitch = 0.0;

    explicit Impl(const Config& cfg) : config(cfg) { }

    // 瞄准点外推 + 弹道固定点迭代。返回 false 表示弹道无解。
    auto aim_and_ballistic(const RuneModel::State& state, double& yaw, double& pitch,
        double& fly_time, bool& has_blade, Vector3d& ff_v, Vector3d& ff_a) -> bool {

        if (!std::isfinite(config.bullet_speed) || config.bullet_speed <= 0.0 ||
            !std::isfinite(config.max_fly_time) || config.max_fly_time <= 0.0 ||
            config.max_iterate <= 0)
            return false;

        const auto center  = Point3d { state.x, state.y, state.z };
        const auto distance = std::hypot(std::hypot(center.x, center.y), center.z);
        if (distance < 1e-6) return false;

        auto t_f = std::min(distance / config.bullet_speed, config.max_fly_time);

        auto solution = TrajectorySolution { };
        for (int i = 0; i < config.max_iterate; ++i) {
            const auto dt_hit = config.algorithmic_delay + config.shoot_delay + t_f;

            auto clone = state;
            clone.transition(dt_hit);

            const auto aimpoints = clone.get_aimpoints();
            Point3d attack;
            if (aimpoints.empty()) {
                // 收敛期/视野内无未激活符叶：瞄准符心，禁射
                attack    = clone.get_direction();
                has_blade = false;
                ff_v      = Vector3d::kZero();
                ff_a      = Vector3d::kZero();
            } else {
                attack    = aimpoints.front();
                has_blade = true;
                // P2-8：透出前馈（AimPoint 携带命中时刻符叶的射线角速度/角加速度）
                ff_v      = aimpoints.front().ff_v;
                ff_a      = aimpoints.front().ff_a;
            }

            solution.input.v0    = config.bullet_speed;
            solution.input.point = attack;

            const auto result = solution.solve();
            if (!result) return false;

            const auto prev = t_f;
            t_f             = result->fly_time;
            yaw             = result->yaw;
            pitch           = result->pitch;

            if (std::abs(t_f - prev) < config.iterate_epsilon) break;
        }

        fly_time = t_f;
        yaw      = util::normalize_angle(yaw + config.offset_yaw);
        pitch    = util::normalize_angle(pitch + config.offset_pitch);
        return true;
    }

    void begin_recover(const RuneModel::State& state) {
        recovering          = true;
        recover_elapsed     = 0.0;
        recover_start_yaw   = last_yaw;
        recover_start_pitch = last_pitch;
        recover_center      = Point3d { state.x, state.y, state.z };
    }
};

RuneFireControl::RuneFireControl() noexcept : RuneFireControl(Config { }) { }

RuneFireControl::RuneFireControl(const Config& config) noexcept
    : impl_(std::make_unique<Impl>(config)), config_(config) { }

RuneFireControl::~RuneFireControl() noexcept = default;

void RuneFireControl::reset() noexcept { *impl_ = Impl(config_); }

auto RuneFireControl::update(const RuneModel::State& state, Timestamp now) -> Command {
    auto& im = *impl_;

    // Reject corrupt tracker output before it can reach ballistic math or the
    // firing state machine. A single NaN must never result in a fire command.
    if (!std::isfinite(state.x) || !std::isfinite(state.y) || !std::isfinite(state.z) ||
        !std::isfinite(state.rotation_angle) || !std::isfinite(state.rotation_speed)) {
        auto cmd = Command{};
        cmd.state = State::LOST;
        cmd.reason = "invalid tracker state";
        return cmd;
    }

    // ---- 时间步进 ----
    double dt = 0.0;
    if (im.first_frame) {
        im.first_frame = false;
        im.last_time   = now;
    } else {
        dt = std::chrono::duration<double>(now - im.last_time).count();
        dt = std::clamp(dt, 0.0, 0.5); // 防长暂停导致的跳变
        im.last_time = now;
    }

    // ---- 数据新鲜度 ----
    const bool fresh = (state.update_count != im.last_update_count);
    if (fresh) {
        im.last_update_count = state.update_count;
        im.data_age          = 0.0;
        im.recovering        = false;
    } else {
        im.data_age += dt;
    }

    // ---- 切叶检测（连续跳变确认） ----
    if (fresh) {
        const auto d_angle =
            std::abs(util::normalize_angle(state.rotation_angle - im.last_rotation_angle));
        if (d_angle > config_.switch_angle) {
            im.jump_count++;
        } else {
            im.jump_count = 0;
        }
        im.last_rotation_angle = state.rotation_angle;
    }
    if (im.jump_count >= config_.switch_confirm) {
        im.jump_count = 0;
        // 确认切叶：重建初始冷却，避免云台超调期开火
        im.state       = State::COOLING;
        im.cooldown    = config_.fire_cooldown_init;
        im.firing_time = 0.0;
    }

    // ---- 瞄准与弹道解算 ----
    double yaw = 0.0, pitch = 0.0, fly_time = 0.0;
    bool has_blade = false;
    Vector3d ff_v = Vector3d::kZero(), ff_a = Vector3d::kZero();
    const bool ballistic_ok =
        im.aim_and_ballistic(state, yaw, pitch, fly_time, has_blade, ff_v, ff_a);
    im.last_yaw   = yaw;
    im.last_pitch = pitch;

    auto cmd          = Command { };
    // A detector miss does not immediately invalidate the EKF prediction.
    // Keep the target usable within data_life; prolonged misses are handled
    // by the recovery path below.
    cmd.found         = ballistic_ok && im.data_age <= config_.data_life;
    cmd.fly_time      = fly_time;
    cmd.yaw           = yaw;
    cmd.pitch         = pitch;
    cmd.ff_v          = ff_v;
    cmd.ff_a          = ff_a;

    // ---- 数据过期：平滑回符心 ----
    if (!ballistic_ok) {
        im.state       = State::LOST;
        im.firing_time = 0.0;
        cmd.state      = State::LOST;
        cmd.reason     = "ballistic failed";
        return cmd;
    }

    if (im.data_age > config_.data_life) {
        cmd.found = false;
        if (!im.recovering) {
            im.begin_recover(state);
            im.firing_time = 0.0;
        }
        im.recover_elapsed += dt;
        const auto recover_duration = std::max(1e-6, config_.recover_time);
        const auto k = std::min(1.0, im.recover_elapsed / recover_duration);
        const auto [center_yaw, center_pitch] = direction_to_yaw_pitch(im.recover_center);
        cmd.yaw   = im.recover_start_yaw + (center_yaw - im.recover_start_yaw) * k;
        cmd.pitch = im.recover_start_pitch + (center_pitch - im.recover_start_pitch) * k;
        cmd.state = State::RECOVERING;
        cmd.reason = "data expired, recovering to rune center";
        if (k >= 1.0) {
            im.state      = State::LOST;
            im.recovering = false;
        }
        return cmd;
    }

    // ---- 开火状态机 ----
    switch (im.state) {
    case State::LOST:
        im.state    = State::COOLING;
        im.cooldown = config_.fire_cooldown_init;
        cmd.state   = State::COOLING;
        cmd.reason  = "new target, initial cooldown";
        break;

    case State::COOLING:
        im.cooldown -= dt;
        if (im.cooldown <= 0.0) {
            im.state   = State::READY;
            cmd.state  = State::READY;
            cmd.reason = "cooldown finished";
        } else {
            cmd.state  = State::COOLING;
            cmd.reason = "cooling";
        }
        break;

    case State::READY:
        cmd.state  = State::READY;
        cmd.reason = "ready";
        break;

    case State::FIRING:
        cmd.state  = State::FIRING;
        cmd.reason = "firing window";
        break;

    case State::RECOVERING:
        im.state    = State::COOLING;
        im.cooldown = config_.fire_cooldown_init;
        cmd.state   = State::COOLING;
        cmd.reason  = "recovered, re-cooling";
        break;
    }

    // ---- 开火判定 ----
    const bool pitch_ok = std::abs(pitch) <= config_.pitch_max;
    const bool time_ok  = fly_time <= config_.max_fly_time;

    if (im.state == State::READY && has_blade && pitch_ok && time_ok) {
        cmd.fire       = true;
        im.state       = State::FIRING;
        im.firing_time = dt;
        cmd.state      = State::FIRING;
        cmd.reason     = "firing window started";
    } else if (im.state == State::FIRING && has_blade && pitch_ok && time_ok) {
        cmd.fire = true;
        im.firing_time += dt;
        if (im.firing_time >= config_.fire_window) {
            im.state       = State::COOLING;
            im.cooldown    = config_.fire_cooldown;
            im.firing_time = 0.0;
            cmd.fire       = false;
            cmd.reason     = "firing window ended, cooling";
        }
    }

    return cmd;
}

} // namespace rmcs
