#include "rune_energy_fitter.hpp"

#include <eigen3/Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>

namespace rmcs {

namespace {

    auto sample_weight(double t, double reference_t) -> double {
        return std::exp2((t - reference_t) / RuneEnergyFitter::kWeightHalfLifeSeconds);
    }

    // ---- P0 升级：ω 两阶段精化 + Cauchy IRLS 鲁棒加权 ----
    // 1) 粗扫 41 步后，在最优 ω 邻域做黄金分割搜索，分辨率从 0.01 提升到 ~1e-4 rad/s；
    // 2) 每个 ω 的线性参数用 IRLS 迭代求解：总权重 = 时间权重 × Cauchy 残差权重，
    //    切叶/遮挡产生的离群相位自动降权（参考 RP-26Rune LM-IRLS 的鲁棒核设计）。
    constexpr int    kSweepSteps   = 41;                  // 粗扫步数（保持）
    constexpr double kOmegaMin     = 1.80;
    constexpr double kOmegaMax     = 2.20;
    constexpr int    kIrisIters    = 3;                   // IRLS 迭代次数
    constexpr double kCauchyScale  = 2.5;                 // Cauchy 尺度因子（RP 同款）
    constexpr double kOmegaTol     = 1e-4;                // 精化收敛（rad/s）
    constexpr int    kRefineIters  = 40;                  // 黄金分割最大迭代
    constexpr double kGoldenRatio  = 0.6180339887498949;

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
        sum_weight += weight;
        sum_t += weight * pt.t;
        sum_theta += weight * pt.theta;
        sum_tt += weight * pt.t * pt.t;
        sum_t_theta += weight * pt.t * pt.theta;
    }

    double denom = sum_weight * sum_tt - sum_t * sum_t;
    if (std::abs(denom) < 1e-12) return std::nullopt;

    double speed = (sum_weight * sum_t_theta - sum_t * sum_theta) / denom;
    double C     = (sum_theta - speed * sum_t) / sum_weight;

    auto pred   = [&](double dt) { return C + speed * dt; };
    double cost = compute_weighted_cost(buffer_, pred);

    return LinearResult { C, speed, cost };
}

auto RuneEnergyFitter::fit_sine() const -> std::optional<FitResult> {
    if (buffer_.size() < 2) return std::nullopt;
    if (buffer_.back().t - buffer_.front().t < kMinFitSeconds) return std::nullopt;

    const auto N           = static_cast<Eigen::Index>(buffer_.size());
    const auto reference_t = buffer_.back().t;

    Eigen::MatrixXd X(N, 4);
    Eigen::VectorXd y(N);
    Eigen::VectorXd time_weights(N);
    Eigen::VectorXd sqrt_time_weights(N);
    double weight_sum = 0.0;
    for (Eigen::Index i = 0; i < N; ++i) {
        const auto t = buffer_[i].t;
        y(i)         = buffer_[i].theta;
        X(i, 0)      = 1.0;
        X(i, 1)      = t;
        const auto w = sample_weight(t, reference_t);
        time_weights(i)      = w;
        sqrt_time_weights(i) = std::sqrt(w);
        weight_sum += w;
    }

    // ---- 固定 ω 的鲁棒加权最小二乘（Cauchy IRLS） ----
    // 返回 (coeff, 加权 MSE)。权重 = 时间权重 × Cauchy 残差权重。
    auto eval_omega = [&](double omega) -> std::pair<Eigen::Vector4d, double> {
        for (Eigen::Index i = 0; i < N; ++i) {
            const auto wt = omega * buffer_[i].t;
            X(i, 2)       = std::cos(wt);
            X(i, 3)       = std::sin(wt);
        }

        Eigen::VectorXd w = time_weights;
        Eigen::Vector4d coeff = Eigen::Vector4d::Zero();
        double mse = std::numeric_limits<double>::max();

        for (int it = 0; it < kIrisIters; ++it) {
            const auto sqrt_w = w.array().sqrt().matrix();
            auto weighted_X   = (X.array().colwise() * sqrt_w.array()).matrix();
            const auto weighted_y = (y.array() * sqrt_w.array()).matrix();

            coeff = weighted_X.colPivHouseholderQr().solve(weighted_y);
            const auto residual = y - X * coeff;

            // Cauchy 尺度：时间加权 RMS（RP 同款），下限保护
            const auto rms = std::sqrt(std::max(
                1e-12, (time_weights.array() * residual.array().square()).sum() / N));
            const auto scale = std::max(1e-6, kCauchyScale * rms);
            const auto ratio = residual.array() / scale;
            const auto w_res = (1.0 / (1.0 + ratio * ratio)).matrix();

            w = (time_weights.array() * w_res.array()).matrix();
            mse = (w.array() * residual.array().square()).sum() / weight_sum;
        }
        return { coeff, mse };
    };

    // ---- 阶段 1：粗扫（保持 41 步） ----
    double best_mse   = std::numeric_limits<double>::max();
    double best_omega = kOmegaMin;
    Eigen::Vector4d best_coeff = Eigen::Vector4d::Zero();
    for (int s = 0; s <= kSweepSteps; ++s) {
        const auto omega = kOmegaMin + s * (kOmegaMax - kOmegaMin) / kSweepSteps;
        const auto [coeff, mse] = eval_omega(omega);
        if (mse < best_mse) {
            best_mse   = mse;
            best_omega = omega;
            best_coeff = coeff;
        }
    }

    // ---- 阶段 2：最优邻域黄金分割精化（分辨率 0.01 → ~1e-4 rad/s） ----
    const auto step = (kOmegaMax - kOmegaMin) / kSweepSteps;
    auto lo = std::max(kOmegaMin, best_omega - step);
    auto hi = std::min(kOmegaMax, best_omega + step);
    if (hi - lo > 1e-6) {
        auto a = lo, b = hi;
        auto c = b - kGoldenRatio * (b - a);
        auto d = a + kGoldenRatio * (b - a);
        auto fc = eval_omega(c).second;
        auto fd = eval_omega(d).second;
        for (int i = 0; i < kRefineIters && (b - a) > kOmegaTol; ++i) {
            if (fc < fd) {
                b  = d;
                d  = c;
                fd = fc;
                c  = b - kGoldenRatio * (b - a);
                fc = eval_omega(c).second;
            } else {
                a  = c;
                c  = d;
                fc = fd;
                d  = a + kGoldenRatio * (b - a);
                fd = eval_omega(d).second;
            }
        }
        const auto refined = 0.5 * (a + b);
        const auto [coeff, mse] = eval_omega(refined);
        if (mse < best_mse) {
            best_mse   = mse;
            best_omega = refined;
            best_coeff = coeff;
        }
    }

    const double C = best_coeff(0);
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

} // namespace rmcs
