#!/usr/bin/env bash
set -euo pipefail

# P2-7：库已剥离 rclcpp（纯 C++），不再需要 source ROS2 环境。

# 自动定位 nvcc（CUDA 通常不在 PATH 中）
if [ -z "${CUDACXX:-}" ]; then
  for cand in /usr/local/cuda/bin/nvcc /usr/local/cuda-12/bin/nvcc /usr/local/cuda-12.6/bin/nvcc; do
    if [ -x "$cand" ]; then
      export CUDACXX="$cand"
      break
    fi
  done
fi

build_type="${BUILD_TYPE:-Release}"
build_dir="${BUILD_DIR:-build}"
jobs="${JOBS:-$(nproc)}"
cc="${CC:-gcc-14}"
cxx="${CXX:-g++-14}"

command -v cmake >/dev/null || { echo "error: cmake not found" >&2; exit 1; }
command -v "$cc" >/dev/null || { echo "error: C compiler '$cc' not found; set CC" >&2; exit 1; }
command -v "$cxx" >/dev/null || { echo "error: C++ compiler '$cxx' not found; set CXX" >&2; exit 1; }
if [ -z "${CUDACXX:-}" ]; then
  echo "error: nvcc not found; set CUDACXX to its absolute path" >&2
  exit 1
fi

CC="$cc" CXX="$cxx" cmake -S . -B "$build_dir" \
  -DCMAKE_BUILD_TYPE="$build_type" -DBUILD_TESTING=ON
cmake --build "$build_dir" -j"$jobs"
ctest --test-dir "$build_dir" --output-on-failure
mkdir -p lib
cp -f "$build_dir/librune_full.a" lib/librune_full.a
