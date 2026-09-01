// virtual_test —— 能量机关全链路真值闭环测试（VirtualRune → RuneModel → RuneFireControl → RuneDiagnostics）
//
// 用法：
//   ./rune_virtual_test [large=0|1] [seconds=20] [hz=200] [noise_px=0] [dropout=0]
//
// 退化模拟（P2-6）：noise_px 为关键点高斯噪声 σ（像素），dropout 为每片符叶漏检/遮挡概率。
// 带退化时误差上升属正常，用于评估算法在真实噪声下的表现；diverged 自动重锚（P2-5 替代项）
// 与火控"数据过期回符心"恢复机制在 dropout 场景下可被直接观察。
//
// 输出：
//   - 每秒一行：火控状态 + 预测误差统计（mean/max/samples）
//   - 结束时：汇总 + /tmp/rune_diag.csv（配对样本）
//
// 说明：预测误差是「预测命中相位 vs 实测相位」在配对容差内的 wrap 误差，
// 与 RP-26Rune 的 PowerRuneDiagnostics 同口径。无退化时该误差应远小于 0.01 rad，
// 反映的是 EKF+拟合+外推的纯算法误差。

#include "module/diagnostics/rune_diagnostics.hpp"
#include "module/fire_control/rune_fire_control.hpp"
#include "module/tracker/model/rune.hpp"
#include "module/tracker/model/virtual_rune.hpp"
#include "utility/math/linear.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

using namespace rmcs;

int main(int argc, char** argv) {
    const bool   large   = argc > 1 ? std::atoi(argv[1]) != 0 : false;
    const double seconds = argc > 2 ? std::atof(argv[2]) : 20.0;
    const double hz      = argc > 3 ? std::atof(argv[3]) : 200.0; // 高帧率模拟：减小诊断配对的时间离散误差
    const double noise_px = argc > 4 ? std::atof(argv[4]) : 0.0;  // P2-6：像素噪声 σ
    const double dropout  = argc > 5 ? std::atof(argv[5]) : 0.0;  // P2-6：漏检/遮挡概率
    if (seconds <= 0.0 || hz <= 0.0 || noise_px < 0.0 || dropout < 0.0 || dropout > 1.0) {
        std::fprintf(stderr,
            "usage: %s [large=0|1] [seconds>0] [hz>0] [noise_px>=0] [dropout=0..1]\n",
            argv[0]);
        return 2;
    }
    const double dt      = 1.0 / hz;

    // ---- 相机（1440×1080 演示内参；虚拟符在 6.67m 处，符半径 0.7m → 约 147px） ----
    const std::array<double, 9> K { 1400.0, 0.0, 720.0, 0.0, 1400.0, 540.0, 0.0, 0.0, 1.0 };
    const std::array<double, 5> D { 0.0, 0.0, 0.0, 0.0, 0.0 };
    const auto identity = Transform::kIdentity();

    // ---- 虚拟符（真值） ----
    VirtualRuneModel::Config vcfg;
    vcfg.enable     = true;
    vcfg.large      = large;
    vcfg.camera_matrix = K;
    vcfg.distort_coeff = D;
    vcfg.pixel_noise_px = noise_px; // P2-6
    vcfg.dropout_prob   = dropout;  // P2-6
    VirtualRuneModel virtual_rune(vcfg);
    virtual_rune.update_camera(K);
    virtual_rune.update_camera(D);
    virtual_rune.update_transform(identity);

    // ---- EKF 跟踪模型 ----
    RuneModel::Config mcfg;
    mcfg.noise_observation = 2.0; // 虚拟观测无噪声，收紧观测噪声帮助收敛
    RuneModel model(mcfg);
    model.update_camera(K, D);
    model.update_transform(identity);

    // ---- 火控 ----
    RuneFireControl::Config fcfg;
    fcfg.bullet_speed = 24.5; // 与 virtual_rune 弹道匹配的演示弹速
    RuneFireControl fire_control(fcfg);

    // ---- 诊断 ----
    RuneDiagnostics::Config dcfg;
    dcfg.match_tolerance_ms = 5.0; // 模拟帧观测时刻离散化，容差取半帧（200Hz 时为 2.5ms）
    RuneDiagnostics diag(dcfg);

    const auto t0 = Clock::now();
    bool inited = false;

    int fires = 0;
    int ready_frames = 0;

    const int total_frames = static_cast<int>(seconds * hz);
    for (int i = 0; i < total_frames; ++i) {
        // 使用真实时钟推进（RuneModel 收敛门控基于 Clock::now()），sleep 保持帧率节奏
        const auto now = Clock::now();

        // 真值推进 → 观测
        virtual_rune.update(now);
        const auto icons      = virtual_rune.icons();
        const auto bullseyes  = virtual_rune.bullseyes();

        // 跟踪
        if (!inited) {
            inited = model.init(icons, bullseyes, now);
            if (!inited) continue;
        }
        model.predict(dt, now);
        (void)model.correct(icons, bullseyes);

        const auto state = model.state();

        // 实测相位（EKF 校正后的展开角）
        diag.push_observation(now, state.rotation_angle);

        // 火控
        const auto cmd = fire_control.update(state, now);
        if (cmd.fire) fires++;
        if (cmd.state == RuneFireControl::State::READY) ready_frames++;

        // 预测命中相位入诊断
        if (cmd.found && cmd.fire) {
            const auto hit_dt = fcfg.algorithmic_delay + fcfg.shoot_delay + cmd.fly_time;
            auto clone = state;
            clone.transition(hit_dt);
            diag.push_predict(
                now + std::chrono::duration_cast<Timestamp::duration>(
                          std::chrono::duration<double>(hit_dt)),
                clone.rotation_angle);
        }

        // 每秒打印一次
        if (i % static_cast<int>(hz) == 0) {
            const auto s = diag.stats();
            std::printf("t=%6.1fs | %-9s | fire=%d | conv=%d | aim(y=%.2f p=%.2f) tf=%.3fs | "
                        "err mean=%.4f max=%.4f latest=%+.4f n=%zu\n",
                i * dt, cmd.state_name(), static_cast<int>(cmd.fire),
                static_cast<int>(model.converge()), cmd.yaw, cmd.pitch, cmd.fly_time, s.mean_error,
                s.max_error, s.latest_error, s.samples);
        }

        if (model.diverged()) {
            std::printf(">>> EKF diverged at t=%.1fs, re-init\n", i * dt);
            inited = false;
        }

        // 维持模拟帧率（真实时间推进）
        const auto next = now
            + std::chrono::duration_cast<Timestamp::duration>(std::chrono::duration<double>(dt));
        std::this_thread::sleep_until(next);
    }

    // ---- 汇总 ----
    const auto s = diag.stats();
    std::printf("\n=== summary (large=%d, %.1fs @ %.0fHz) ===\n", static_cast<int>(large), seconds,
        hz);
    std::printf("  fire frames : %d / %d (%.1f%%)\n", fires, total_frames,
        100.0 * fires / total_frames);
    std::printf("  ready frames: %d / %d\n", ready_frames, total_frames);
    std::printf("  predict err : mean=%.4f rad (%.1f deg), max=%.4f rad (%.1f deg), n=%zu\n",
        s.mean_error, s.mean_error * 180.0 / 3.14159265358979, s.max_error,
        s.max_error * 180.0 / 3.14159265358979, s.samples);
    std::printf("  csv         : /tmp/rune_diag.csv (%zu samples)\n", diag.dump_csv("/tmp/rune_diag.csv"));
    return 0;
}
