#pragma once
// gimbal_extrinsic —— 陀螺仪-相机外参标定
//
// 问题：相机和IMU都装在云台上，装配时的相对姿态未知。此工具通过多组观测
// 求解 IMU→相机 的固定旋转矩阵。
//
// 原理：给定 N≥2 组(image_R_target, imu_R_base) 变换对，求 X 使得
//   image_R_target · X = X · imu_R_base  (对所有观测 i)
// 即 AX = XB 形式，用 SVD 求解。

#include "core/types.hpp"

#include <optional>
#include <vector>

namespace rmcs {

struct ExtrinsicSample {
    // 某一时刻：(1) 相机看到符的相对姿态 (从 PnP 或其他方式得到)
    Eigen::Matrix3d image_R_target;
    // (2) IMU 在云台坐标系中的姿态 (从陀螺仪读出或回读 camera.transform 的 orientation)
    Eigen::Matrix3d imu_R_base;
};

// 从至少 2 个样本求解 camera→imu 的旋转矩阵
auto solve_gimbal_extrinsic(const std::vector<ExtrinsicSample>& samples)
    -> std::optional<Eigen::Matrix3d>;

}  // namespace rmcs
