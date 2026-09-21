#include "core/simulator.hpp"
#include "core/conversion.hpp"
#include <yaml-cpp/yaml.h>
#include <opencv2/imgcodecs.hpp>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace rmcs {
namespace {
[[noreturn]] void fail(const char* text) { throw std::runtime_error(text); }
double number(const YAML::Node& n, const char* key) {
    const double v = n[key].as<double>();
    if (!std::isfinite(v)) fail("non-finite simulator metadata");
    return v;
}
Translation position(const YAML::Node& n) {
    if (!n.IsSequence() || n.size() != 3) fail("missing simulator position");
    Translation t{n[0].as<double>(), n[1].as<double>(), n[2].as<double>()};
    if (!std::isfinite(t.x) || !std::isfinite(t.y) || !std::isfinite(t.z)) fail("invalid position");
    return t;
}
}
SimulatorClient::SimulatorClient(const std::string& address) {
    const char* token = std::getenv("DAEDALUS_BRIDGE_TOKEN");
    if (!token || std::strlen(token) < 16) fail("set DAEDALUS_BRIDGE_TOKEN (at least 16 characters)");
    const auto split = address.find(':');
    const auto host = address.substr(0, split);
    const auto port = split == std::string::npos ? "7447" : address.substr(split + 1);
    addrinfo hints{}, *addresses = nullptr;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_INET;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &addresses)) fail("cannot resolve simulator host");
    for (auto* a = addresses; a; a = a->ai_next) {
        socket_ = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (socket_ < 0) continue;
        timeval timeout{3, 0};
        setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(socket_, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        if (connect(socket_, a->ai_addr, a->ai_addrlen) == 0) break;
        close(socket_); socket_ = -1;
    }
    freeaddrinfo(addresses);
    if (socket_ < 0) fail("cannot connect to simulator");
    try {
        YAML::Emitter quoted; quoted << YAML::DoubleQuoted << token;
        send_json(std::string("{\"version\":1,\"token\":") + quoted.c_str() + "}");
    } catch (...) { close(socket_); socket_ = -1; throw; }
}
SimulatorClient::~SimulatorClient() { if (socket_ >= 0) close(socket_); }
void SimulatorClient::read(void* data, std::size_t size) {
    auto* p = static_cast<char*>(data);
    while (size) {
        const auto n = recv(socket_, p, size, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) fail("simulator disconnected or receive timed out");
        p += n; size -= n;
    }
}
void SimulatorClient::send_json(const std::string& data) {
    const auto length = htonl(static_cast<std::uint32_t>(data.size()));
    std::string packet(reinterpret_cast<const char*>(&length), 4); packet += data;
    std::size_t sent = 0;
    while (sent < packet.size()) {
        const auto n = send(socket_, packet.data() + sent, packet.size() - sent, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) fail("simulator send failed");
        sent += n;
    }
}
auto SimulatorClient::grab() -> Frame {
    if (awaiting_reply_) fail("simulator frame must be answered before receiving another");
    std::uint32_t sizes[2]; read(sizes, sizeof(sizes));
    const auto meta_size = ntohl(sizes[0]), jpeg_size = ntohl(sizes[1]);
    if (!meta_size || meta_size > 4096 || !jpeg_size || jpeg_size > 16*1024*1024) fail("invalid simulator packet length");
    std::string raw(meta_size, '\0'); read(raw.data(), raw.size());
    std::vector<unsigned char> jpeg(jpeg_size); read(jpeg.data(), jpeg.size());
    const auto received = Clock::now();
    const auto meta = YAML::Load(raw);
    const auto seq = meta["frame_seq"].as<std::uint64_t>();
    if (meta["version"].as<int>() != 1 || seq <= sequence_) fail("invalid simulator frame sequence/version");
    sequence_ = seq;
    width = meta["width"].as<int>(); height = meta["height"].as<int>();
    if (width <= 0 || height <= 0 || width > 8192 || height > 8192 || std::int64_t(width)*height > 16777216) fail("invalid image size");
    matrix = {number(meta,"fx"),0,number(meta,"cx"),0,number(meta,"fy"),number(meta,"cy"),0,0,1};
    if (matrix[0] <= 0 || matrix[4] <= 0) fail("invalid focal length");
    chassis_yaw_ = number(meta,"chassis_yaw_deg"); bore_pitch_ = number(meta,"bore_pitch_deg");
    const auto camera = position(meta["camera_position"]);
    const auto gun = position(meta["muzzle_position"]);
    const auto capture = meta["capture_unix_ns"].as<std::uint64_t>();
    if (!first_capture_) { first_capture_ = capture; first_stamp_ = received; origin_ = gun; }
    if (capture < first_capture_ || capture-first_capture_ > 86400000000000ULL) fail("invalid capture clock");
    muzzle = Translation{gun.x-origin_.x, gun.y-origin_.y, gun.z-origin_.z};
    const auto q = meta["camera_orientation_xyzw"];
    if (!q.IsSequence() || q.size()!=4) fail("invalid camera quaternion");
    Eigen::Quaterniond optical(q[3].as<double>(),q[0].as<double>(),q[1].as<double>(),q[2].as<double>());
    if (!optical.coeffs().allFinite() || std::abs(optical.norm()-1)>0.01) fail("invalid camera orientation");
    // Metadata maps optical axes into ROS world; the tracker expects FLU camera axes.
    const Eigen::Quaterniond flu(optical.normalized().toRotationMatrix()*util::kCoordTransformMatrix.transpose());
    pose = Transform{Translation{camera.x-origin_.x,camera.y-origin_.y,camera.z-origin_.z}, Orientation{flu}};
    auto image = std::make_shared<cv::Mat>(cv::imdecode(jpeg,cv::IMREAD_COLOR));
    if (image->empty() || image->cols!=width || image->rows!=height) fail("invalid simulator JPEG");
    actual_yaw = number(meta,"yaw_deg"); actual_pitch = number(meta,"pitch_deg");
    auto_aim_enabled = meta["auto_aim_enabled"].as<bool>();
    fire_allowed = meta["sim_fire_allowed"].as<bool>();
    shots = meta["simulated_shots"].as<unsigned>(); hits = meta["simulated_hits"].as<unsigned>();
    awaiting_reply_ = true;
    if (sequence_%60 == 0 || capture == first_capture_)
        std::printf("[sim] seq=%llu image=%dx%d yaw=%.2f pitch=%.2f auto=%d fire_allowed=%d shots=%u hits=%u\n",
            static_cast<unsigned long long>(sequence_),width,height,number(meta,"yaw_deg"),number(meta,"pitch_deg"),
            meta["auto_aim_enabled"].as<bool>(),meta["sim_fire_allowed"].as<bool>(),
            meta["simulated_shots"].as<unsigned>(),meta["simulated_hits"].as<unsigned>());
    return {image, first_stamp_ + std::chrono::nanoseconds(capture-first_capture_)};
}
void SimulatorClient::command(bool valid, double yaw_rad, double pitch_rad, bool fire) {
    const double yaw = std::remainder(util::rad2deg(yaw_rad)-chassis_yaw_,360.0);
    const double pitch = -util::rad2deg(pitch_rad)-bore_pitch_;
    valid = valid && std::isfinite(yaw) && std::isfinite(pitch) && std::abs(pitch)<=90;
    char packet[512];
    std::snprintf(packet,sizeof(packet),"{\"version\":1,\"frame_seq\":%llu,\"source_frame_seq\":%llu,\"target_valid\":%s,\"yaw_deg\":%.8f,\"pitch_deg\":%.8f,\"fire\":%s}",
        static_cast<unsigned long long>(sequence_),static_cast<unsigned long long>(sequence_),
        valid?"true":"false",valid?yaw:0,valid?pitch:0,valid&&fire?"true":"false");
    send_json(packet); awaiting_reply_ = false;
}
}
