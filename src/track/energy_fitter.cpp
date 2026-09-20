#include "energy_fitter.hpp"

#include <eigen3/Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace rmcs {

namespace {

    auto sample_weight(double t, double reference_t) -> double {
        return std::exp2((t - reference_t) / RuneEnergyFitter::kWeightHalfLifeSeconds);
    }

    // 大符规则频率固定为 1.884 rad/s。继续从短视频自由估计频率会造成明显的
    // 外推漂移；基速、振幅和相位仍由观测使用 Cauchy IRLS 鲁棒拟合。
    constexpr double kRuneAngularFrequency = 1.884;
    constexpr int    kIrisIters    = 3;                   // IRLS 迭代次数
    constexpr double kCauchyScale  = 2.5;                 // Cauchy 尺度因子（RP 同款）

} // namespace

RuneEnergyFitter::~RuneEnergyFitter() = default;

void RuneEnergyFitter::push(double t, double theta) {
    buffer_.push_back({ t, theta });

    while (!buffer_.empty() && buffer_.back().t - buffer_.front().t > kWindowSeconds)
        buffer_.pop_front();
}

void RuneEnergyFitter::reset() { buffer_.clear(); }

template <typename Pred>
auto RuneEnergyFitter::compute_weighted_cost(const std::deque<Point>& buffer, Pred&& pred_fn)
    -> double {

    const auto reference_t = buffer.back().t;
    double weighted_sum    = 0.0;
    double weight_sum      = 0.0;
    for (const auto& pt : buffer) {
        const auto weight = sample_weight(pt.t, reference_t);
        const auto diff   = pt.theta - pred_fn(pt.t);
        weighted_sum += weight * diff * diff;
        weight_sum += weight;
    }
    return weighted_sum / weight_sum;
}

auto RuneEnergyFitter::fit_linear() const -> std::optional<LinearResult> {
    if (buffer_.size() < 2) return std::nullopt;
    if (buffer_.back().t - buffer_.front().t < kMinFitSeconds) return std::nullopt;

    const auto reference_t = buffer_.back().t;

    double sum_weight = 0.0, sum_t = 0.0, sum_theta = 0.0, sum_tt = 0.0, sum_t_theta = 0.0;
    for (const auto& pt : buffer_) {
        const auto weight = sample_weight(pt.t, reference_t);
        const auto t = pt.t - reference_t;
        sum_weight += weight;
        sum_t += weight * t;
        sum_theta += weight * pt.theta;
        sum_tt += weight * t * t;
        sum_t_theta += weight * t * pt.theta;
    }

    double denom = sum_weight * sum_tt - sum_t * sum_t;
    if (std::abs(denom) < 1e-12) return std::nullopt;

    double speed = (sum_weight * sum_t_theta - sum_t * sum_theta) / denom;
    double centered_C = (sum_theta - speed * sum_t) / sum_weight;
    double C = centered_C - speed * reference_t;

    auto pred   = [&](double t) { return centered_C + speed * (t - reference_t); };
    double cost = compute_weighted_cost(buffer_, pred);

    return LinearResult { C, speed, cost };
}

auto RuneEnergyFitter::fit_sine() const -> std::optional<FitResult> {
    if (buffer_.size() < 4) return std::nullopt;
    if (buffer_.back().t - buffer_.front().t < kMinFitSeconds) return std::nullopt;

    const auto N           = buffer_.size();
    const auto reference_t = buffer_.back().t;

    struct Sample {
        double t;
        double theta;
        double time_weight;
    };
    auto samples = std::vector<Sample> { };
    samples.reserve(N);
    double weight_sum = 0.0;
    for (const auto& point : buffer_) {
        const auto t = point.t;
        const auto w = sample_weight(t, reference_t);
        samples.push_back({ t, point.theta, w });
        weight_sum += w;
    }

    // Reused by every omega evaluation. The old implementation allocated
    // several N-row Eigen matrices and ran a dynamic QR for every IRLS step.
    auto robust_weights = std::vector<double>(N, 1.0);

    // ---- 固定 ω 的鲁棒加权最小二乘（Cauchy IRLS） ----
    // 返回 (coeff, 加权 MSE)。权重 = 时间权重 × Cauchy 残差权重。
    auto eval_omega = [&](double omega) -> std::pair<Eigen::Vector4d, double> {
        std::fill(robust_weights.begin(), robust_weights.end(), 1.0);
        Eigen::Vector4d coeff = Eigen::Vector4d::Zero();
        double mse = std::numeric_limits<double>::max();

        for (int it = 0; it < kIrisIters; ++it) {
            Eigen::Matrix4d normal = Eigen::Matrix4d::Zero();
            Eigen::Vector4d rhs    = Eigen::Vector4d::Zero();
            for (std::size_t i = 0; i < N; ++i) {
                const auto& sample = samples[i];
                const auto wt      = omega * sample.t;
                const auto basis = Eigen::Vector4d {
                    1.0, sample.t - reference_t, std::cos(wt), std::sin(wt) };
                const auto weight = sample.time_weight * robust_weights[i];
                normal.selfadjointView<Eigen::Lower>().rankUpdate(basis, weight);
                rhs.noalias() += weight * sample.theta * basis;
            }
            normal.template triangularView<Eigen::StrictlyUpper>() =
                normal.transpose().template triangularView<Eigen::StrictlyUpper>();
            const auto ldlt = normal.ldlt();
            if (ldlt.info() != Eigen::Success) {
                return { Eigen::Vector4d::Zero(), std::numeric_limits<double>::max() };
            }
            const auto diagonal = ldlt.vectorD().cwiseAbs();
            if (diagonal.minCoeff() <= diagonal.maxCoeff() * 1e-12) {
                return { Eigen::Vector4d::Zero(), std::numeric_limits<double>::max() };
            }
            coeff = ldlt.solve(rhs);
            if (!coeff.allFinite())
                return { Eigen::Vector4d::Zero(), std::numeric_limits<double>::max() };

            // Cauchy 尺度：时间加权 RMS（RP 同款），下限保护
            double time_error_sum = 0.0;
            for (const auto& sample : samples) {
                const auto wt = omega * sample.t;
                const auto prediction = coeff[0] + coeff[1] * (sample.t - reference_t)
                    + coeff[2] * std::cos(wt) + coeff[3] * std::sin(wt);
                const auto residual = sample.theta - prediction;
                time_error_sum += sample.time_weight * residual * residual;
            }
            const auto rms = std::sqrt(
                std::max(1e-12, time_error_sum / static_cast<double>(N)));
            const auto scale = std::max(1e-6, kCauchyScale * rms);
            double robust_error_sum = 0.0;
            for (std::size_t i = 0; i < N; ++i) {
                const auto& sample = samples[i];
                const auto wt      = omega * sample.t;
                const auto prediction = coeff[0] + coeff[1] * (sample.t - reference_t)
                    + coeff[2] * std::cos(wt) + coeff[3] * std::sin(wt);
                const auto residual = sample.theta - prediction;
                const auto ratio    = residual / scale;
                robust_weights[i]   = 1.0 / (1.0 + ratio * ratio);
                robust_error_sum +=
                    sample.time_weight * robust_weights[i] * residual * residual;
            }
            mse = robust_error_sum / weight_sum;
        }
        return { coeff, mse };
    };

    const auto [best_coeff, best_mse] = eval_omega(kRuneAngularFrequency);
    if (!best_coeff.allFinite() || !std::isfinite(best_mse)) return std::nullopt;
    constexpr auto best_omega = kRuneAngularFrequency;

    const double C = best_coeff(0) - best_coeff(1) * reference_t;
    const double v = best_coeff(1);
    const double A = best_coeff(2);
    const double B = best_coeff(3);

    const double a   = best_omega * std::sqrt(A * A + B * B);
    const double phi = std::atan2(B, -A);

    auto pred = [&](double t) {
        return C + v * t - a / best_omega * std::cos(best_omega * t + phi);
    };
    const double cost = compute_weighted_cost(buffer_, pred);

    return FitResult { C, v, a, best_omega, phi, cost };
}

RuneEnergyFitWorker::RuneEnergyFitWorker()
    : thread_ { [this] { run(); } } { }

RuneEnergyFitWorker::~RuneEnergyFitWorker() {
    {
        const std::lock_guard lock { mutex_ };
        stop_ = true;
    }
    ready_.notify_one();
    if (thread_.joinable()) thread_.join();
}

auto RuneEnergyFitWorker::try_submit(const RuneEnergyFitter& fitter,
    std::uint64_t generation, bool calculate_sine) noexcept -> bool {
    try {
        const std::lock_guard lock { mutex_ };
        if (stop_ || busy_ || request_ || result_) return false;
        request_.emplace(Request { fitter, generation, calculate_sine });
    } catch (...) {
        return false;
    }
    ready_.notify_one();
    return true;
}

auto RuneEnergyFitWorker::poll() noexcept -> std::optional<Result> {
    const std::lock_guard lock { mutex_ };
    if (!result_) return std::nullopt;
    auto result = std::move(result_);
    result_.reset();
    return result;
}

auto RuneEnergyFitWorker::run() noexcept -> void {
    while (true) {
        auto request = std::optional<Request> { };
        {
            std::unique_lock lock { mutex_ };
            ready_.wait(lock, [this] { return stop_ || request_.has_value(); });
            if (stop_) return;
            request = std::move(request_);
            request_.reset();
            busy_ = true;
        }

        auto result = Result { };
        result.generation  = request->generation;
        try {
            result.linear = request->fitter.fit_linear();
            if (request->calculate_sine) result.sine = request->fitter.fit_sine();
        } catch (...) {
            result.linear.reset();
            result.sine.reset();
        }

        {
            const std::lock_guard lock { mutex_ };
            busy_ = false;
            result_ = std::move(result);
        }
    }
}

} // namespace rmcs
