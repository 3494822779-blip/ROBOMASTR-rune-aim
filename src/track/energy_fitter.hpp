#pragma once

#include <deque>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>

namespace rmcs {

class RuneEnergyFitter {
public:
    struct FitResult {
        double C     = 0.0;
        double v     = 0.0;
        double a     = 0.0;
        double omega = 0.0;
        double phi   = 0.0;
        double cost  = std::numeric_limits<double>::max();
    };

    struct LinearResult {
        double C     = 0.0;
        double speed = 0.0;
        double cost  = std::numeric_limits<double>::max();
    };

    void push(double t, double theta);
    void reset();

    auto fit_linear() const -> std::optional<LinearResult>;
    auto fit_sine() const -> std::optional<FitResult>;

    static constexpr double kWindowSeconds         = 6.0;
    static constexpr double kMinFitSeconds         = 1.5;
    static constexpr double kWeightHalfLifeSeconds = 3.0;

    ~RuneEnergyFitter();

private:
    struct Point {
        double t;
        double theta;
    };
    std::deque<Point> buffer_;

    template <typename Pred>
    static auto compute_weighted_cost(const std::deque<Point>& buffer, Pred&& pred_fn) -> double;
};

// Runs the expensive model comparison on one persistent background thread.
// Only one request can be in flight; callers keep using the last accepted fit.
class RuneEnergyFitWorker {
public:
    struct Result {
        std::optional<RuneEnergyFitter::LinearResult> linear;
        std::optional<RuneEnergyFitter::FitResult> sine;
        std::uint64_t generation = 0;
    };

    RuneEnergyFitWorker();
    ~RuneEnergyFitWorker();
    RuneEnergyFitWorker(const RuneEnergyFitWorker&) = delete;
    auto operator=(const RuneEnergyFitWorker&) -> RuneEnergyFitWorker& = delete;

    // Returns false when a fit or an unread result is already pending.
    auto try_submit(const RuneEnergyFitter& fitter,
        std::uint64_t generation, bool calculate_sine) noexcept -> bool;
    auto poll() noexcept -> std::optional<Result>;

private:
    struct Request {
        RuneEnergyFitter fitter;
        std::uint64_t generation = 0;
        bool calculate_sine = false;
    };

    auto run() noexcept -> void;

    std::mutex mutex_;
    std::condition_variable ready_;
    std::optional<Request> request_;
    std::optional<Result> result_;
    bool busy_ = false;
    bool stop_ = false;
    std::thread thread_;
};

} // namespace rmcs
