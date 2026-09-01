#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>

namespace rmcs::gpu {

struct Point { float x, y; };
struct RefineResult { Point points[5]; int valid; };

class RuneGpuPipeline {
public:
    RuneGpuPipeline() = default;
    ~RuneGpuPipeline();
    RuneGpuPipeline(const RuneGpuPipeline&) = delete;
    auto operator=(const RuneGpuPipeline&) -> RuneGpuPipeline& = delete;

    auto upload_bgr(const unsigned char* host, int width, int height,
                    std::size_t host_pitch, cudaStream_t stream) -> bool;
    auto preprocess(float* tensor, int output_width, int output_height,
                    float scale, int pad_x, int pad_y, cudaStream_t stream) -> bool;
    // `target_count` targets are supplied, with exactly five points per target.
    auto refine(const Point* points, int target_count, RefineResult* results,
                int blade_radius, int icon_radius, float max_shift,
                float min_gradient, cudaStream_t stream) -> bool;

private:
    unsigned char* image_ = nullptr;
    std::size_t pitch_ = 0;
    int width_ = 0;
    int height_ = 0;
    Point* seeds_ = nullptr;
    RefineResult* results_ = nullptr;
    int capacity_ = 0;
};

}
