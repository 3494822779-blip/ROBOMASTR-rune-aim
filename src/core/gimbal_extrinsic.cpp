#include "core/gimbal_extrinsic.hpp"

#include <eigen3/Eigen/SVD>
#include <algorithm>
#include <cmath>

namespace rmcs {

auto solve_gimbal_extrinsic(const std::vector<ExtrinsicSample>& samples)
    -> std::optional<Eigen::Matrix3d> {
    if (samples.size() < 2) return std::nullopt;

    // Construct the 12×12 constraint matrix from AX=XB ⟹ vec(A)·X·vec(B^T)=0
    // For each sample i: A_i·X = X·B_i
    // Vectorize: (B_i^T ⊗ A_i - I)·vec(X) = 0
    Eigen::MatrixXd M(9 * static_cast<int>(samples.size()), 9);
    M.setZero();

    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto& A = samples[i].image_R_target;
        const auto& B = samples[i].imu_R_base;

        // B^T ⊗ A (Kronecker product) - I₉
        Eigen::Matrix<double, 9, 9> constraint;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        int global_row = row * 3 + r;
                        int global_col = col * 3 + c;
                        constraint(global_row, global_col) =
                            B(col, row) * A(r, c) - (global_row == global_col ? 1.0 : 0.0);
                    }
                }
            }
        }

        M.block<9, 9>(static_cast<int>(i) * 9, 0) = constraint;
    }

    // SVD to find null space: M·v ≈ 0, v is the right singular vector for smallest singular value
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullV);
    const auto& V = svd.matrixV();
    const auto& singular_values = svd.singularValues();

    // Find the smallest singular value index
    int min_idx = 0;
    double min_sv = singular_values(0);
    for (int i = 1; i < singular_values.rows(); ++i) {
        if (singular_values(i) < min_sv) {
            min_sv = singular_values(i);
            min_idx = i;
        }
    }

    // Extract the corresponding column (right singular vector)
    Eigen::VectorXd X_vec = V.col(min_idx);

    // Reshape back to 3×3 matrix
    Eigen::Matrix3d X_raw;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            X_raw(i, j) = X_vec(i * 3 + j);
        }
    }

    // Enforce orthogonality via Procrustes: closest orthogonal matrix to X_raw
    // Use SVD: X_raw = U·Σ·V^T, then X = U·V^T
    Eigen::JacobiSVD<Eigen::Matrix3d> procrustes(X_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d X = procrustes.matrixU() * procrustes.matrixV().transpose();

    // Ensure det=1 (proper rotation, not reflection)
    if (X.determinant() < 0.0) {
        X.col(2) *= -1.0;
    }

    return X;
}

}  // namespace rmcs
