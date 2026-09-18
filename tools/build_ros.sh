#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source /opt/ros/humble/setup.bash
cmake -S . -B build-ros -DENABLE_ROS2=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CXX:-g++-14}" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build-ros -j"${BUILD_JOBS:-2}"
