#include "track/energy_fitter.hpp"
#include "track/rune_model.hpp"

#include <cmath>
#include <cstdio>

namespace {

auto near(double actual, double expected, double tolerance, const char* name) -> bool {
    if (std::abs(actual - expected) <= tolerance) return true;
    std::fprintf(stderr, "%s: actual %.12f, expected %.12f, tolerance %.12f\n",
        name, actual, expected, tolerance);
    return false;
}

} // namespace

int main() {
    constexpr double omega = 1.884;
    constexpr double v = 1.12;
    constexpr double a = 0.93;
    constexpr double phi = 0.47;
    constexpr double C = -0.31;

    rmcs::RuneEnergyFitter fitter;
    for (int i = 0; i <= 80; ++i) {
        const auto t = static_cast<double>(i) * 0.05;
        const auto theta = C + v * t - a / omega * std::cos(omega * t + phi);
        fitter.push(t, theta);
    }

    const auto fit = fitter.fit_sine();
    if (!fit) {
        std::fprintf(stderr, "sine fit unexpectedly failed\n");
        return 1;
    }
    if (!near(fit->omega, omega, 1e-12, "omega")
        || !near(fit->v, v, 1e-6, "base speed")
        || !near(fit->a, a, 1e-6, "amplitude")
        || !near(fit->cost, 0.0, 1e-12, "fit cost"))
        return 1;

    auto state = rmcs::RuneModel::State { };
    state.rotation_angle = 1.2;
    state.rotation_speed = v + a * std::sin(phi);
    state.sine_valid = true;
    state.sine_v = v;
    state.sine_a = a;
    state.sine_omega = omega;
    state.sine_phase = phi;
    state.sine_speed_correction = -0.15;

    constexpr double dt = 0.37;
    const auto expected_phase = phi + omega * dt;
    const auto expected_angle = 1.2 + v * dt
        + a / omega * (std::cos(phi) - std::cos(expected_phase))
        + state.sine_speed_correction * dt;
    const auto expected_speed = v + a * std::sin(expected_phase)
        + state.sine_speed_correction;
    state.transition(dt);

    if (!near(state.rotation_angle, expected_angle, 1e-12, "transition angle")
        || !near(state.rotation_speed, expected_speed, 1e-12, "transition speed")
        || !near(state.sine_phase, expected_phase, 1e-12, "transition phase"))
        return 1;

    return 0;
}
