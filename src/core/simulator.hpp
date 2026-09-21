#pragma once
#include "core/frame_source.hpp"
#include "core/types.hpp"
#include <array>
#include <cstdint>
#include <string>

namespace rmcs {
// Synchronous, frame-associated request/reply protocol. No command can be relabelled
// with a newer frame, and all network operations have a bounded timeout.
class SimulatorClient {
public:
    explicit SimulatorClient(const std::string& address);
    ~SimulatorClient();
    SimulatorClient(const SimulatorClient&) = delete;
    auto operator=(const SimulatorClient&) -> SimulatorClient& = delete;
    auto grab() -> Frame;
    void command(bool valid, double yaw_rad, double pitch_rad, bool fire);
    std::array<double, 9> matrix{};
    Transform pose{};
    Translation muzzle{}; // Relative to the first muzzle position, in ROS world axes.
    int width = 0, height = 0;
    double actual_yaw = 0, actual_pitch = 0;
    bool auto_aim_enabled = false, fire_allowed = false;
    unsigned shots = 0, hits = 0;
private:
    void read(void* data, std::size_t size);
    void send_json(const std::string& data);
    int socket_ = -1;
    std::uint64_t sequence_ = 0, first_capture_ = 0;
    Timestamp first_stamp_{};
    Translation origin_{};
    double chassis_yaw_ = 0, bore_pitch_ = 0;
    bool awaiting_reply_ = false;
};
}
