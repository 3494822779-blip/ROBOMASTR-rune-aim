#include "ros/telemetry.hpp"
#include "debug/draw.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <opencv2/imgcodecs.hpp>

namespace rmcs {
struct RosTelemetry::Impl {
    cfg::AppConfig config;
    rclcpp::Context::SharedPtr context = std::make_shared<rclcpp::Context>();
    rclcpp::Node::SharedPtr node;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr image;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr status;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr aim;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers;
    Timestamp steady_origin = Clock::now();
    int64_t epoch_origin = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    Timestamp last_image{};
    std::chrono::duration<double> image_period;
    explicit Impl(const cfg::AppConfig& c, double fps) : config(c), image_period(1.0 / fps) {
        context->init(0, nullptr);
        rclcpp::install_signal_handlers();
        node = std::make_shared<rclcpp::Node>("rune_aim", rclcpp::NodeOptions().context(context));
        const auto qos = rclcpp::SensorDataQoS().keep_last(1);
        image = node->create_publisher<sensor_msgs::msg::CompressedImage>("/rune/image/compressed", qos);
        status = node->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/rune/status", qos);
        aim = node->create_publisher<geometry_msgs::msg::Vector3Stamped>("/rune/aim", qos);
        markers = node->create_publisher<visualization_msgs::msg::MarkerArray>("/rune/markers", qos);
    }
};
RosTelemetry::RosTelemetry(const cfg::AppConfig& c, double fps) : impl_(std::make_unique<Impl>(c, fps)) {}
RosTelemetry::~RosTelemetry() { impl_->context->shutdown("rune_aim exiting"); }
bool RosTelemetry::ok() const { return impl_->context->is_valid(); }
bool RosTelemetry::image_due() const {
    return impl_->image->get_subscription_count() > 0 &&
        Clock::now() - impl_->last_image >= impl_->image_period;
}
void RosTelemetry::publish(Timestamp time, const cv::Mat& frame,
    const std::vector<RuneIcon>& icons, const std::vector<RuneBullseye>& bullseyes,
    const RuneModel::State* state, const RuneFireControl::Command& command,
    const RuneDiagnostics::Stats& stats, bool corrected) {
    auto& p = *impl_;
    // Preserve acquisition/synthetic video time, mapped once from steady to Unix time.
    std_msgs::msg::Header header;
    header.stamp = rclcpp::Time(p.epoch_origin +
        std::chrono::duration_cast<std::chrono::nanoseconds>(time - p.steady_origin).count());
    header.frame_id = "odom";
    diagnostic_msgs::msg::DiagnosticArray status;
    status.header = header;
    diagnostic_msgs::msg::DiagnosticStatus s;
    s.name = "rune_aim"; s.hardware_id = p.config.input.mode;
    s.level = state ? s.OK : s.WARN;
    s.message = command.state_name();
    auto add = [&](const char* key, auto value) {
        diagnostic_msgs::msg::KeyValue kv;
        kv.key = key; kv.value = std::to_string(value); s.values.push_back(kv);
    };
    add("tracking", state != nullptr); add("corrected", corrected);
    add("found", command.found); add("fire", command.fire);
    add("icons", icons.size()); add("bullseyes", bullseyes.size());
    add("yaw_rad", command.yaw); add("pitch_rad", command.pitch);
    add("fly_time_s", command.fly_time);
    add("mean_error_rad", stats.mean_error); add("max_error_rad", stats.max_error);
    add("error_samples", stats.samples);
    if (state) { add("phase_rad", state->rotation_angle); add("speed_rad_s", state->get_rotation_speed()); }
    diagnostic_msgs::msg::KeyValue reason; reason.key = "reason"; reason.value = command.reason;
    s.values.push_back(reason); status.status.push_back(s); p.status->publish(status);
    // Valid solutions only: gaps in plots represent lost targets, not zero-angle commands.
    if (command.found) {
        geometry_msgs::msg::Vector3Stamped aim; aim.header = header;
        aim.vector.x = command.yaw; aim.vector.y = command.pitch; aim.vector.z = command.fly_time;
        p.aim->publish(aim);
    }
    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker clear;
    clear.header = header; clear.action = clear.DELETEALL; markers.markers.push_back(clear);
    auto sphere = [&](double x, double y, double z, int id, float r, float g, float b, double size) {
        visualization_msgs::msg::Marker m;
        m.header = header; m.ns = "rune"; m.id = id; m.type = m.SPHERE;
        m.action = m.ADD; m.pose.orientation.w = 1;
        m.pose.position.x = x; m.pose.position.y = y; m.pose.position.z = z;
        m.scale.x = m.scale.y = m.scale.z = size;
        m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1;
        m.lifetime = rclcpp::Duration::from_seconds(0.5);
        markers.markers.push_back(m);
    };
    if (state) {
        sphere(state->x, state->y, state->z, 0, 1, 0, 1, 0.12);
        int id = 1;
        for (const auto& point : state->get_aimpoints())
            sphere(point.x, point.y, point.z, id++, 0, 1, 0, 0.16);
        if (command.found) {
            auto future = *state;
            future.transition(p.config.fire.algorithmic_delay + p.config.fire.shoot_delay + command.fly_time);
            for (const auto& point : future.get_aimpoints()) {
                sphere(point.x, point.y, point.z, 10, 1, 1, 0, 0.20); break;
            }
        }
    }
    p.markers->publish(markers);
    if (!frame.empty() && image_due()) {
        p.last_image = Clock::now();
        cv::Mat display = frame.clone();
        debug::draw_detection(display, icons, bullseyes);
        debug::draw_status(display, command, stats, debug::DrawOptions{});
        sensor_msgs::msg::CompressedImage image;
        image.header = header; image.header.frame_id = "camera_optical";
        image.format = "bgr8; jpeg compressed bgr8";
        if (cv::imencode(".jpg", display, image.data, {cv::IMWRITE_JPEG_QUALITY, 75}))
            p.image->publish(image);
    }
}
}
