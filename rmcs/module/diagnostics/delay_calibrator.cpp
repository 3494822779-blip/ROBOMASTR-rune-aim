#include "delay_calibrator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace rmcs {

namespace {

// 线性插值到等间隔时间轴，便于稳定互相关（Climber serial_delay 同款）
auto resample_uniform(const std::deque<DelayCalibrator::Sample>& buf, double step_s,
    std::vector<double>& out_t, std::vector<double>& out_vx, std::vector<double>& out_vy,
    std::vector<double>& out_yaw, std::vector<double>& out_pitch) -> bool {

    if (buf.size() < 2) return false;
    const double t0 = buf.front().t;
    const double t1 = buf.back().t;
    if (t1 - t0 < step_s * 10) return false; // 时长太短

    const int M = static_cast<int>((t1 - t0) / step_s) + 1;
    out_t.resize(M);
    out_vx.resize(M);
    out_vy.resize(M);
    out_yaw.resize(M);
    out_pitch.resize(M);

    std::size_t j = 0;
    for (int i = 0; i < M; ++i) {
        const double tg = t0 + i * step_s;
        while (j + 1 < buf.size() && buf[j + 1].t < tg) {
            ++j;
        }
        const std::size_t j1 = std::min(j + 1, buf.size() - 1);
        const double tA = buf[j].t, tB = buf[j1].t;
        const double alpha = (tB - tA) > 1e-6 ? (tg - tA) / (tB - tA) : 0.0;
        const auto lerp = [&](double a, double b) { return a + alpha * (b - a); };
        out_t[i] = tg;
        out_vx[i] = lerp(buf[j].vision_x, buf[j1].vision_x);
        out_vy[i] = lerp(buf[j].vision_y, buf[j1].vision_y);
        out_yaw[i] = lerp(buf[j].imu_yaw_deg, buf[j1].imu_yaw_deg);
        out_pitch[i] = lerp(buf[j].imu_pitch_deg, buf[j1].imu_pitch_deg);
    }
    return true;
}

} // namespace

DelayCalibrator::DelayCalibrator() noexcept : DelayCalibrator(Config { }) { }

DelayCalibrator::DelayCalibrator(const Config& config) noexcept : config_(config) { }

void DelayCalibrator::push(const Sample& sample) {
    buffer_.push_back(sample);
    while (buffer_.size() > config_.max_buffer) {
        buffer_.pop_front();
    }
}

void DelayCalibrator::reset() noexcept { buffer_.clear(); }

auto DelayCalibrator::estimate() const -> Estimate {
    // 等间隔重采样（100Hz 时间轴），减少丢帧/不均匀采样影响
    std::vector<double> t, vx, vy, yaw, pitch;
    if (!resample_uniform(buffer_, config_.resample_step_s, t, vx, vy, yaw, pitch)) {
        return { };
    }
    const int N = static_cast<int>(vx.size());
    if (N < 60) return { };

    // 零均值，抑制漂移
    const auto mean = [](const std::vector<double>& a) {
        return std::accumulate(a.begin(), a.end(), 0.0) / std::max<std::size_t>(1, a.size());
    };
    const auto span = [](const std::vector<double>& a) {
        auto [mn, mx] = std::minmax_element(a.begin(), a.end());
        return (*mx) - (*mn);
    };

    const double mvx = mean(vx), mvy = mean(vy), myaw = mean(yaw), mpitch = mean(pitch);
    for (int i = 0; i < N; ++i) {
        vx[i] -= mvx;
        vy[i] -= mvy;
        yaw[i]   = -(yaw[i] - myaw);   // 像素与 yaw 方向相反
        pitch[i] = -(pitch[i] - mpitch); // 像素 y 与 pitch 方向相反
    }

    const double span_vx = span(vx), span_vy = span(vy);
    const double span_yaw = span(yaw), span_pitch = span(pitch);

    struct AxisResult {
        double delay_ms;
        double corr;
        bool hit_edge;
    };

    const double dt = t[1] - t[0];
    const int max_shift =
        std::min(static_cast<int>(config_.max_delay_search_s / dt), N / 4);

    const auto compute_axis = [&](const std::vector<double>& a,
                                   const std::vector<double>& b) -> AxisResult {
        double best_corr = -2.0; // 支持负相关
        int best_k = 0;
        for (int k = -max_shift; k <= max_shift; ++k) {
            const int i0 = std::max(0, -k);
            const int i1 = std::min(N, N - k);
            if (i1 - i0 < N / 5) continue; // 要求足够重叠样本
            double num = 0.0, da2 = 0.0, db2 = 0.0;
            for (int i = i0; i < i1; ++i) {
                const double da = a[i];
                const double db = b[i + k];
                num += da * db;
                da2 += da * da;
                db2 += db * db;
            }
            const double denom = std::sqrt(da2 * db2) + 1e-9;
            const double corr  = num / denom;
            if (corr > best_corr) {
                best_corr = corr;
                best_k    = k;
            }
        }
        const bool hit_edge = std::abs(best_k) == max_shift;
        // 符号约定：视觉信号滞后云台信号为正延迟。
        // 推导：a(视觉) = b(云台) 的延迟版本 → a[i] ~ b[i−d]，相关峰在 k=−d，
        // 故 delay = −k·dt。实车场景中图像（曝光+检测）必然滞后于 IMU 反馈。
        return { -best_k * dt * 1000.0, best_corr, hit_edge };
    };

    Estimate est;
    if (span_vx >= config_.min_vision_span_px && span_yaw >= config_.min_yaw_span_deg) {
        const auto r = compute_axis(vx, yaw);
        if (std::abs(r.corr) > config_.min_corr) {
            est.yaw_ms   = r.delay_ms;
            est.yaw_corr = r.corr;
            est.hit_edge = est.hit_edge || r.hit_edge;
        }
    }
    if (span_vy >= config_.min_vision_span_py && span_pitch >= config_.min_pitch_span_deg) {
        const auto r = compute_axis(vy, pitch);
        if (std::abs(r.corr) > config_.min_corr) {
            est.pitch_ms   = r.delay_ms;
            est.pitch_corr = r.corr;
            est.hit_edge   = est.hit_edge || r.hit_edge;
        }
    }
    est.valid = !std::isnan(est.yaw_ms) || !std::isnan(est.pitch_ms);
    return est;
}

} // namespace rmcs
