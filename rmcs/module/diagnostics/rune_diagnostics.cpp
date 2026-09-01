#include "rune_diagnostics.hpp"

#include "utility/math/angle.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace rmcs {

RuneDiagnostics::RuneDiagnostics() noexcept : RuneDiagnostics(Config { }) { }

RuneDiagnostics::RuneDiagnostics(const Config& config) noexcept : config_(config) { }

void RuneDiagnostics::push_predict(Timestamp hit_time, double phase) {
    predicts_.emplace_back(hit_time, phase);
    while (predicts_.size() > config_.max_queue) {
        predicts_.pop_front();
    }
}

void RuneDiagnostics::push_observation(Timestamp time, double phase) {
    last_obs_time_  = time;
    last_obs_phase_ = phase;
    has_obs_        = true;

    // ---- 配对 ----
    // 丢弃所有命中时刻早于 (实测时刻 − 容差) 的预测（不可能再匹配上）
    const auto to_ms = [](const Timestamp::duration& d) {
        return std::chrono::duration<double, std::milli>(d).count();
    };
    while (!predicts_.empty()) {
        if (to_ms(time - predicts_.front().first) > config_.match_tolerance_ms) {
            predicts_.pop_front();
            continue;
        }
        break;
    }
    if (predicts_.empty()) return;

    // 找 |t_obs − t_hit| 最小的预测
    auto best     = predicts_.begin();
    auto best_abs = 1e30;
    for (auto it = predicts_.begin(); it != predicts_.end(); ++it) {
        const auto abs_dt = std::abs(to_ms(time - it->first));
        if (abs_dt < best_abs) {
            best_abs = abs_dt;
            best     = it;
        }
    }
    if (best_abs > config_.match_tolerance_ms) return;

    const auto error = util::normalize_angle(phase - best->second);
    predicts_.erase(predicts_.begin(), std::next(best));

    latest_error_  = error;
    max_abs_error_ = std::max(max_abs_error_, std::abs(error));
    sum_abs_error_ += std::abs(error);
    samples_++;

    if (history_.size() >= config_.max_history) {
        history_.pop_front();
    }
    history_.push_back(Sample {
        .t_hit_s    = std::chrono::duration<double>(best->first.time_since_epoch()).count(),
        .t_obs_s    = std::chrono::duration<double>(time.time_since_epoch()).count(),
        .phase_pred = best->second,
        .phase_obs  = phase,
        .error_rad  = error,
    });
}

auto RuneDiagnostics::stats() const -> Stats {
    return Stats {
        .latest_error = latest_error_,
        .mean_error   = samples_ > 0 ? sum_abs_error_ / static_cast<double>(samples_) : 0.0,
        .max_error    = max_abs_error_,
        .samples      = samples_,
    };
}

void RuneDiagnostics::reset() noexcept {
    predicts_.clear();
    history_.clear();
    has_obs_       = false;
    sum_abs_error_ = 0.0;
    max_abs_error_ = 0.0;
    latest_error_  = 0.0;
    samples_       = 0;
}

auto RuneDiagnostics::dump_csv(const std::string& path) const -> std::size_t {
    std::ofstream out(path);
    if (!out) return 0;

    out << "t_hit_s,t_obs_s,phase_pred,phase_obs,error_rad\n";
    for (const auto& s : history_) {
        out << s.t_hit_s << ',' << s.t_obs_s << ',' << s.phase_pred << ',' << s.phase_obs << ','
            << s.error_rad << '\n';
    }
    return history_.size();
}

} // namespace rmcs
