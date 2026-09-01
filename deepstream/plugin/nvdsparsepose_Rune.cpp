#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {
constexpr int kClasses = 3;
constexpr int kKeypoints = 5;
constexpr int kChannels = kClasses + kKeypoints * 3;
constexpr float kKeypointThreshold = 0.8F;
constexpr float kBoxPadding = 0.15F;
constexpr float kNmsDistance = 30.0F;

float clamp(float value, float low, float high) {
  return std::max(low, std::min(value, high));
}

void nmsCenter(const NvDsInferInstanceMaskInfo &object, float &x, float &y) {
  x = 0.0F;
  y = 0.0F;
  int valid = 0;
  for (int point = 0; point < kKeypoints; ++point) {
    if (object.mask[point * 3 + 2] < kKeypointThreshold) continue;
    x += object.mask[point * 3 + 0];
    y += object.mask[point * 3 + 1];
    ++valid;
  }
  if (valid > 0) { x /= valid; y /= valid; }
}

void releaseMask(NvDsInferInstanceMaskInfo &object) {
  delete[] object.mask;
  object.mask = nullptr;
}
}  // namespace

extern "C" bool NvDsInferParseRunePose(
    const std::vector<NvDsInferLayerInfo> &outputLayersInfo,
    const NvDsInferNetworkInfo &networkInfo,
    const NvDsInferParseDetectionParams &detectionParams,
    std::vector<NvDsInferInstanceMaskInfo> &objectList) {
  if (outputLayersInfo.empty()) {
    std::cerr << "Rune parser: output layer is missing\n";
    return false;
  }

  const NvDsInferLayerInfo &layer = outputLayersInfo[0];
  const NvDsInferDims &dims = layer.inferDims;
  int candidates = 0;
  if (dims.numDims == 2 && dims.d[0] == kChannels) {
    candidates = dims.d[1];
  } else if (dims.numDims == 3 && dims.d[0] == 1 && dims.d[1] == kChannels) {
    candidates = dims.d[2];
  } else {
    std::cerr << "Rune parser: expected output [18,N] or [1,18,N], got [";
    for (unsigned int i = 0; i < dims.numDims; ++i) {
      std::cerr << (i ? "," : "") << dims.d[i];
    }
    std::cerr << "]\n";
    return false;
  }

  if (!layer.buffer || candidates <= 0) {
    std::cerr << "Rune parser: output buffer is empty\n";
    return false;
  }

  const auto *output = static_cast<const float *>(layer.buffer);
  std::vector<NvDsInferInstanceMaskInfo> proposals;
  proposals.reserve(128);

  for (int n = 0; n < candidates; ++n) {
    int class_id = 0;
    float confidence = output[n];
    for (int c = 1; c < kClasses; ++c) {
      const float score = output[c * candidates + n];
      if (score > confidence) {
        confidence = score;
        class_id = c;
      }
    }
    const float threshold = class_id < static_cast<int>(detectionParams.perClassPreclusterThreshold.size())
                                ? detectionParams.perClassPreclusterThreshold[class_id]
                                : 0.25F;
    if (confidence < threshold) continue;

    float *keypoints = new float[kKeypoints * 3];
    float min_x = static_cast<float>(networkInfo.width);
    float min_y = static_cast<float>(networkInfo.height);
    float max_x = 0.0F;
    float max_y = 0.0F;
    int valid = 0;
    float keypoint_score_sum = 0.0F;
    for (int p = 0; p < kKeypoints; ++p) {
      const int base = kClasses + p * 3;
      const float x = clamp(output[(base + 0) * candidates + n], 0.0F, networkInfo.width);
      const float y = clamp(output[(base + 1) * candidates + n], 0.0F, networkInfo.height);
      const float score = output[(base + 2) * candidates + n];
      keypoints[p * 3 + 0] = x;
      keypoints[p * 3 + 1] = y;
      keypoints[p * 3 + 2] = score;
      if (score >= kKeypointThreshold) {
        min_x = std::min(min_x, x); min_y = std::min(min_y, y);
        max_x = std::max(max_x, x); max_y = std::max(max_y, y);
        ++valid;
        keypoint_score_sum += score;
      }
    }
    if (valid < 3 || max_x <= min_x || max_y <= min_y) {
      delete[] keypoints;
      continue;
    }

    const float pad_x = (max_x - min_x) * kBoxPadding;
    const float pad_y = (max_y - min_y) * kBoxPadding;
    NvDsInferInstanceMaskInfo object{};
    object.left = clamp(min_x - pad_x, 0.0F, networkInfo.width);
    object.top = clamp(min_y - pad_y, 0.0F, networkInfo.height);
    const float right = clamp(max_x + pad_x, 0.0F, networkInfo.width);
    const float bottom = clamp(max_y + pad_y, 0.0F, networkInfo.height);
    object.width = right - object.left;
    object.height = bottom - object.top;
    object.detectionConfidence = confidence * (keypoint_score_sum / valid);
    object.classId = class_id;
    object.mask = keypoints;
    object.mask_width = networkInfo.width;
    object.mask_height = networkInfo.height;
    object.mask_size = sizeof(float) * kKeypoints * 3;
    proposals.push_back(object);
  }

  std::stable_sort(proposals.begin(), proposals.end(), [](const auto &a, const auto &b) {
    return a.detectionConfidence > b.detectionConfidence;
  });
  objectList.clear();
  for (auto &proposal : proposals) {
    bool keep = true;
    for (const auto &selected : objectList) {
      float px, py, sx, sy;
      nmsCenter(proposal, px, py);
      nmsCenter(selected, sx, sy);
      const float dx = px - sx;
      const float dy = py - sy;
      if (dx * dx + dy * dy < kNmsDistance * kNmsDistance) {
        keep = false;
        break;
      }
    }
    if (keep) objectList.push_back(proposal);
    else releaseMask(proposal);
  }
  return true;
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseRunePose);
