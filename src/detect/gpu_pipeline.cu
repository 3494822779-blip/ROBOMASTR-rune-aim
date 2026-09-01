#include "gpu_pipeline.hpp"

#include <cmath>

namespace rmcs::gpu {
namespace {
__device__ float gray_at(const unsigned char* image, std::size_t pitch, int width, int height,
                         int x, int y) {
    x = max(0, min(width - 1, x)); y = max(0, min(height - 1, y));
    const auto* p = image + y * pitch + 3 * x;
    return 0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2];
}

__global__ void preprocess_kernel(const unsigned char* image, std::size_t pitch,
                                  int width, int height, float* output,
                                  int out_width, int out_height, float scale,
                                  int pad_x, int pad_y) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_width || y >= out_height) return;
    float r = 114.0f, g = 114.0f, b = 114.0f;
    const float source_x = (x - pad_x + 0.5f) / scale - 0.5f;
    const float source_y = (y - pad_y + 0.5f) / scale - 0.5f;
    if (source_x >= 0 && source_y >= 0 && source_x < width - 1 && source_y < height - 1) {
        const int x0 = static_cast<int>(floorf(source_x));
        const int y0 = static_cast<int>(floorf(source_y));
        const float ax = source_x - x0, ay = source_y - y0;
        const auto* p00 = image + y0 * pitch + 3 * x0;
        const auto* p01 = p00 + 3;
        const auto* p10 = image + (y0 + 1) * pitch + 3 * x0;
        const auto* p11 = p10 + 3;
        auto sample = [&](int c) {
            return (1-ax)*(1-ay)*p00[c] + ax*(1-ay)*p01[c] +
                   (1-ax)*ay*p10[c] + ax*ay*p11[c];
        };
        b = sample(0); g = sample(1); r = sample(2);
    }
    const int index = y * out_width + x;
    const int plane = out_width * out_height;
    output[index] = r / 255.0f;
    output[plane + index] = g / 255.0f;
    output[2 * plane + index] = b / 255.0f;
}

__device__ bool refine_blade(const unsigned char* image, std::size_t pitch, int width, int height,
                             Point seed, Point center, int radius, float threshold, Point& out) {
    float nx = seed.x - center.x, ny = seed.y - center.y;
    const float length = sqrtf(nx*nx + ny*ny);
    if (length < 2) return false;
    nx /= length; ny /= length;
    float best = 0; int best_offset = 0;
    const int half_width = min(5, max(2, radius / 3));
    for (int offset = -radius; offset <= radius; ++offset) {
        float response = 0; int support = 0;
        for (int lateral = -half_width; lateral <= half_width; ++lateral) {
            const float px = seed.x + nx*offset - ny*lateral;
            const float py = seed.y + ny*offset + nx*lateral;
            const int x = __float2int_rn(px), y = __float2int_rn(py);
            if (x < 1 || y < 1 || x >= width-1 || y >= height-1) continue;
            const float gx = gray_at(image,pitch,width,height,x+1,y)-gray_at(image,pitch,width,height,x-1,y);
            const float gy = gray_at(image,pitch,width,height,x,y+1)-gray_at(image,pitch,width,height,x,y-1);
            response += sqrtf(gx*gx+gy*gy); ++support;
        }
        if (support && response/support > best) { best=response/support; best_offset=offset; }
    }
    if (best < threshold || abs(best_offset) > radius) return false;
    out = {seed.x + nx*best_offset, seed.y + ny*best_offset};
    return true;
}

__global__ void refine_kernel(const unsigned char* image, std::size_t pitch, int width, int height,
                              const Point* seeds, RefineResult* results, int count,
                              int blade_radius, int icon_radius, float max_shift, float threshold) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const Point* input = seeds + i*5;
    RefineResult value{};
    const Point center{(input[0].x+input[1].x+input[3].x+input[4].x)*.25f,
                       (input[0].y+input[1].y+input[3].y+input[4].y)*.25f};
    bool valid = true;
    const int ids[4] = {0,1,3,4};
    for (int n=0;n<4;++n) {
        if (!refine_blade(image,pitch,width,height,input[ids[n]],center,
                          blade_radius,threshold,value.points[ids[n]])) {
            valid = false;
            value.points[ids[n]] = input[ids[n]];
        }
    }
    double sw=0,sx=0,sy=0; int support=0;
    const Point icon=input[2];
    for(int dy=-icon_radius;dy<=icon_radius;++dy) for(int dx=-icon_radius;dx<=icon_radius;++dx) {
        const int x=__float2int_rn(icon.x)+dx, y=__float2int_rn(icon.y)+dy;
        if(x<1||y<1||x>=width-1||y>=height-1) continue;
        const float gx=gray_at(image,pitch,width,height,x+1,y)-gray_at(image,pitch,width,height,x-1,y);
        const float gy=gray_at(image,pitch,width,height,x,y+1)-gray_at(image,pitch,width,height,x,y-1);
        const float mag=fabsf(gx)+fabsf(gy); if(mag<2*threshold) continue;
        const float spatial=expf(-(dx*dx+dy*dy)/(0.7f*icon_radius*icon_radius));
        const float w=mag*spatial; sw+=w; sx+=w*x; sy+=w*y; ++support;
    }
    if(support<6||sw<=0) { valid=false; value.points[2]=input[2]; }
    else value.points[2]={(float)(sx/sw),(float)(sy/sw)};
    for(int p=0;p<5;++p) { const float dx=value.points[p].x-input[p].x,dy=value.points[p].y-input[p].y;
        if(dx*dx+dy*dy>max_shift*max_shift) valid=false; }
    value.valid=valid; results[i]=value;
}
}

RuneGpuPipeline::~RuneGpuPipeline(){ cudaFree(image_); cudaFree(seeds_); cudaFree(results_); }
auto RuneGpuPipeline::upload_bgr(const unsigned char* host,int width,int height,std::size_t host_pitch,cudaStream_t stream)->bool{
    if(width!=width_||height!=height_){ cudaFree(image_); image_=nullptr; if(cudaMallocPitch(&image_,&pitch_,width*3,height)!=cudaSuccess)return false; width_=width;height_=height; }
    return cudaMemcpy2DAsync(image_,pitch_,host,host_pitch,width*3,height,cudaMemcpyHostToDevice,stream)==cudaSuccess;
}
auto RuneGpuPipeline::preprocess(float* tensor,int ow,int oh,float scale,int px,int py,cudaStream_t stream)->bool{
    preprocess_kernel<<<dim3((ow+15)/16,(oh+15)/16),dim3(16,16),0,stream>>>(image_,pitch_,width_,height_,tensor,ow,oh,scale,px,py);
    return cudaGetLastError()==cudaSuccess;
}
auto RuneGpuPipeline::refine(const Point* points,int target_count,RefineResult* host_results,int br,int ir,float shift,float grad,cudaStream_t stream)->bool{
    if(target_count<=0)return true;
    if(!points||!host_results||!image_)return false;
    if(target_count>capacity_){
        Point* new_seeds=nullptr;
        RefineResult* new_results=nullptr;
        if(cudaMalloc(&new_seeds,target_count*5*sizeof(Point))!=cudaSuccess)return false;
        if(cudaMalloc(&new_results,target_count*sizeof(RefineResult))!=cudaSuccess){
            cudaFree(new_seeds);
            return false;
        }
        cudaFree(seeds_);
        cudaFree(results_);
        seeds_=new_seeds;
        results_=new_results;
        capacity_=target_count;
    }
    if(cudaMemcpyAsync(seeds_,points,target_count*5*sizeof(Point),cudaMemcpyHostToDevice,stream)!=cudaSuccess)return false;
    refine_kernel<<<(target_count+31)/32,32,0,stream>>>(image_,pitch_,width_,height_,seeds_,results_,target_count,br,ir,shift,grad);
    return cudaGetLastError()==cudaSuccess&&cudaMemcpyAsync(host_results,results_,target_count*sizeof(RefineResult),cudaMemcpyDeviceToHost,stream)==cudaSuccess;
}
}
