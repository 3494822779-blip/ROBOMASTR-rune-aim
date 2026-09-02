// delay_calib_test —— 链路延迟标定器离线自检
//
// 合成已知延迟的信号对（云台角 + 视觉像素位移），喂给 DelayCalibrator，
// 验证互相关恢复的延迟与真值一致。实车接入方式见文件底部注释。
//
// 用法：
//   ./rune_delay_calib_test [delay_ms=120] [noise_px=2] [seconds=30]

#include "diag/delay_calibrator.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <random>

using namespace rmcs;

int main(int argc, char** argv) {
    const double delay_ms = argc > 1 ? std::atof(argv[1]) : 120.0; // 注入的链路延迟
    const double noise_px = argc > 2 ? std::atof(argv[2]) : 2.0;   // 视觉像素噪声 σ
    const double seconds  = argc > 3 ? std::atof(argv[3]) : 30.0;

    constexpr double kFps       = 30.0;   // 模拟 30fps 采集
    constexpr double kPxPerDeg  = 25.0;   // 像素/度换算（模拟）
    constexpr double kYawAmp1   = 15.0;   // 云台慢速摆动幅度 1
    constexpr double kYawFreq1  = 0.10;   // Hz
    constexpr double kYawAmp2   = 5.0;    // 幅度 2
    constexpr double kYawFreq2  = 0.37;   // Hz

    std::mt19937 gen { 7 };
    std::normal_distribution<double> nd(0.0, noise_px);

    const auto yaw_signal = [](double t) {
        return kYawAmp1 * std::sin(2.0 * std::numbers::pi * kYawFreq1 * t)
            + kYawAmp2 * std::sin(2.0 * std::numbers::pi * kYawFreq2 * t);
    };

    DelayCalibrator calibrator;

    const int frames = static_cast<int>(seconds * kFps);
    for (int i = 0; i < frames; ++i) {
        const double t       = i / kFps;
        const double t_prev  = (i - 1) / kFps;
        const double yaw_now = yaw_signal(t);

        // 视觉信号 = 延迟后的云台角 × 像素/度（方向相反）+ 噪声
        const double vision_x = -kPxPerDeg * yaw_signal(t - delay_ms / 1000.0) + nd(gen);

        DelayCalibrator::Sample s;
        s.t             = t;
        s.vision_x      = vision_x;
        s.vision_y      = 0.0; // 只标定 yaw 轴
        s.imu_yaw_deg   = yaw_now;
        s.imu_pitch_deg = 0.0;
        calibrator.push(s);

        if (i > 0 && i % static_cast<int>(kFps) == 0) {
            const auto est = calibrator.estimate();
            if (est.valid) {
                std::printf("t=%4.0fs | est yaw delay=%7.1f ms (r=%.3f) | "
                            "pitch=%s | true=%4.0f ms\n",
                    t, est.yaw_ms, est.yaw_corr,
                    std::isnan(est.pitch_ms) ? "n/a" : "n/a", delay_ms);
            }
        }
    }

    const auto est = calibrator.estimate();
    std::printf("\n=== summary ===\n");
    std::printf("  injected delay : %.0f ms\n", delay_ms);
    if (!est.valid) {
        std::printf("  FAIL: no valid estimate (signal span too small? window too short?)\n");
        return 1;
    }
    std::printf("  estimated yaw  : %.1f ms (corr=%.3f)\n", est.yaw_ms, est.yaw_corr);
    if (est.hit_edge) std::printf("  WARNING: hit search edge, window too short\n");

    const double err = std::abs(est.yaw_ms - delay_ms);
    const bool ok = err < 20.0 && est.yaw_corr > 0.5;
    std::printf("  error          : %.1f ms -> %s\n", err, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ================= 实车接入说明 =================
// 1. 相机采集每帧：运行检测（能量机关/装甲板均可），取最大目标中心
// 2. 记录 检测结束时刻 t（统一零点）、目标中心相对图像中心的像素位移 (vision_x, vision_y)
// 3. 同一时刻读云台 IMU 姿态（yaw/pitch 度）
// 4. 每帧 push({t, vision_x, vision_y, yaw_deg, pitch_deg})
// 5. 手动摆动云台（让目标在画面中移动、跨度足够），观察 estimate() 输出
// 6. 稳定后的 yaw_ms/pitch_ms 即链路总延迟（曝光→检测→解算→下发→执行→IMU），
//    填入 RuneFireControl::Config.algorithmic_delay（建议取两轴均值，并减去预期弹道解算耗时）
