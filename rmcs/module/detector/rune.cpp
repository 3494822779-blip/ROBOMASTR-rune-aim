#include "rune.hpp"
#include "module/gpu/rune_gpu.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

namespace rmcs {
namespace {
constexpr int kInputWidth = 640;
constexpr int kInputHeight = 480;
constexpr int kClasses = 3;
constexpr int kPoints = 5;
constexpr int kChannels = 18;
constexpr int kCandidates = 6300;

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
    }
};

struct Candidate {
    int class_id;
    float score;
    std::array<cv::Point2f, kPoints> points;
    std::array<float, kPoints> point_scores;
    float quality;
    cv::Point2f nms_center;
};
}

struct RuneDetector::Impl {
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream = nullptr;
    void* input_device = nullptr;
    void* output_device = nullptr;
    std::vector<float> output_host;
    std::vector<Candidate> candidates;
    std::vector<Candidate> selected;
    std::vector<gpu::Point> gpu_points;
    std::vector<gpu::RefineResult> refined;
    const char* input_name = nullptr;
    const char* output_name = nullptr;
    gpu::RuneGpuPipeline gpu_pipeline;

    ~Impl() {
        if (stream) cudaStreamDestroy(stream);
        if (input_device) cudaFree(input_device);
        if (output_device) cudaFree(output_device);
    }
};

RuneDetector::RuneDetector() noexcept : impl_(std::make_unique<Impl>()) {}
RuneDetector::~RuneDetector() noexcept = default;
RuneDetector::RuneDetector(RuneDetector&&) noexcept = default;
auto RuneDetector::operator=(RuneDetector&&) noexcept -> RuneDetector& = default;

auto RuneDetector::initialize() noexcept -> bool {
    if (impl_->stream) { cudaStreamDestroy(impl_->stream); impl_->stream = nullptr; }
    if (impl_->input_device) { cudaFree(impl_->input_device); impl_->input_device = nullptr; }
    if (impl_->output_device) { cudaFree(impl_->output_device); impl_->output_device = nullptr; }
    impl_->context.reset();
    impl_->engine.reset();
    impl_->runtime.reset();
    std::ifstream file(config.engine_path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const auto size = file.tellg();
    file.seekg(0);
    std::vector<char> data(static_cast<std::size_t>(size));
    if (!file.read(data.data(), size)) return false;

    impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
    if (!impl_->runtime) return false;
    impl_->engine.reset(impl_->runtime->deserializeCudaEngine(data.data(), data.size()));
    if (!impl_->engine) return false;
    impl_->context.reset(impl_->engine->createExecutionContext());
    if (!impl_->context) return false;

    impl_->input_name = impl_->engine->getIOTensorName(0);
    impl_->output_name = impl_->engine->getIOTensorName(1);
    if (!impl_->input_name || !impl_->output_name) return false;
    const auto input_dims = impl_->engine->getTensorShape(impl_->input_name);
    const auto output_dims = impl_->engine->getTensorShape(impl_->output_name);
    if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3 ||
        input_dims.d[2] != kInputHeight || input_dims.d[3] != kInputWidth ||
        output_dims.nbDims != 3 || output_dims.d[0] != 1 ||
        output_dims.d[1] != kChannels || output_dims.d[2] != kCandidates) return false;

    impl_->output_host.resize(kChannels * kCandidates);
    if (cudaMalloc(&impl_->input_device, 3 * kInputHeight * kInputWidth * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&impl_->output_device, impl_->output_host.size() * sizeof(float)) != cudaSuccess ||
        cudaStreamCreate(&impl_->stream) != cudaSuccess) return false;
    impl_->candidates.reserve(kCandidates);
    impl_->selected.reserve(32);
    impl_->gpu_points.reserve(32 * kPoints);
    impl_->refined.reserve(32);
    return impl_->context->setTensorAddress(impl_->input_name, impl_->input_device) &&
           impl_->context->setTensorAddress(impl_->output_name, impl_->output_device);
}

auto RuneDetector::detect(const cv::Mat& image) noexcept -> Elements {
    if (image.empty() || !impl_->context) return {};

    const float scale = std::min(kInputWidth / static_cast<float>(image.cols),
                                 kInputHeight / static_cast<float>(image.rows));
    const int resized_width = static_cast<int>(std::round(image.cols * scale));
    const int resized_height = static_cast<int>(std::round(image.rows * scale));
    const int pad_x = (kInputWidth - resized_width) / 2;
    const int pad_y = (kInputHeight - resized_height) / 2;
    if (image.type() != CV_8UC3 ||
        !impl_->gpu_pipeline.upload_bgr(image.ptr<unsigned char>(), image.cols, image.rows,
                                        image.step, impl_->stream) ||
        !impl_->gpu_pipeline.preprocess(static_cast<float*>(impl_->input_device),
                                        kInputWidth, kInputHeight, scale, pad_x, pad_y,
                                        impl_->stream)) return {};

    if (!impl_->context->enqueueV3(impl_->stream)) return {};
    if (cudaMemcpyAsync(impl_->output_host.data(), impl_->output_device,
            impl_->output_host.size() * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream)
        != cudaSuccess) return {};
    if (cudaStreamSynchronize(impl_->stream) != cudaSuccess) return {};

    const float* output_data = impl_->output_host.data();
    auto& candidates = impl_->candidates;
    candidates.clear();
    for (int column = 0; column < kCandidates; ++column) {
        int class_id = 0;
        float score = output_data[column];
        for (int c = 1; c < kClasses; ++c) {
            const float value = output_data[c * kCandidates + column];
            if (value > score) { score = value; class_id = c; }
        }
        if (score < config.score_threshold) continue;
        Candidate item{class_id, score, {}, {}, 0.0F, {}};
        int valid_points = 0;
        float valid_score_sum = 0.0F;
        cv::Point2f valid_center{};
        bool in_bounds = true;
        for (int p = 0; p < kPoints; ++p) {
            const int base = kClasses + p * 3;
            const float x = (output_data[(base + 0) * kCandidates + column] - pad_x) / scale;
            const float y = (output_data[(base + 1) * kCandidates + column] - pad_y) / scale;
            const float point_score = output_data[(base + 2) * kCandidates + column];
            item.points[p] = {x, y};
            item.point_scores[p] = point_score;
            if (point_score >= config.keypoint_threshold) {
                ++valid_points;
                valid_score_sum += point_score;
                valid_center += item.points[p];
            }
            in_bounds &= x >= 0 && x < image.cols && y >= 0 && y < image.rows;
        }
        if (valid_points >= 3 && in_bounds) {
            item.quality = item.score * (valid_score_sum / valid_points);
            item.nms_center = valid_center * (1.0F / valid_points);
            candidates.push_back(item);
        }
    }

    std::stable_sort(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.quality > b.quality; });
    auto& selected = impl_->selected;
    selected.clear();
    for (const auto& item : candidates) {
        const auto item_center = item.nms_center;
        const bool duplicate = std::any_of(selected.begin(), selected.end(), [&](const Candidate& kept) {
            const auto delta = item_center - kept.nms_center;
            return delta.dot(delta) < config.center_distance * config.center_distance;
        });
        if (!duplicate) selected.push_back(item);
    }

    Elements result;
    result.icons.reserve(selected.size());
    result.bullseyes.reserve(selected.size());
    auto& gpu_points = impl_->gpu_points;
    auto& refined = impl_->refined;
    gpu_points.resize(selected.size() * kPoints);
    refined.resize(selected.size());
    for (std::size_t i = 0; i < selected.size(); ++i)
        for (int p = 0; p < kPoints; ++p)
            gpu_points[i * kPoints + p] = {selected[i].points[p].x, selected[i].points[p].y};
    const bool refinement_launched = impl_->gpu_pipeline.refine(
        gpu_points.data(), static_cast<int>(selected.size()), refined.data(),
        config.refine_radius, config.icon_refine_radius, config.max_refine_shift,
        config.min_refine_gradient, impl_->stream);
    const bool refinement_ready = refinement_launched &&
                                  cudaStreamSynchronize(impl_->stream) == cudaSuccess &&
                                  cudaGetLastError() == cudaSuccess;
    for (std::size_t i = 0; i < selected.size(); ++i) {
        const auto& item = selected[i];
        // Prefer OpenCV local refinement. When local image evidence is
        // insufficient, retain the neural detector's original semantic points.
        auto points = item.points;
        if (refinement_ready && refined[i].valid)
            for (int p = 0; p < kPoints; ++p)
                points[p] = {refined[i].points[p].x, refined[i].points[p].y};
        // Tracker contract: icon R; corners top,left,bottom,right; activation.
        // P1-4：保留 3 类激活信息（0=未激活, 1=小符激活, 2=大符激活），不再合并为 bool。
        const auto activation =
            item.class_id == 0 ? RuneBullseye::Activation::Inactive
            : item.class_id == 1 ? RuneBullseye::Activation::SmallActive
                                 : RuneBullseye::Activation::BigActive;
        result.icons.push_back({Point2d{points[2]}, item.point_scores[2]});
        result.bullseyes.push_back({
            .center = Point2d{(points[0] + points[1] + points[3] + points[4]) * 0.25F},
            .corners = {{Point2d{points[0]}, Point2d{points[1]},
                         Point2d{points[4]}, Point2d{points[3]}}},
            .active = item.class_id != 0,
            .score = item.score,
            .activation = activation,
        });
    }
    return result;
}
}
