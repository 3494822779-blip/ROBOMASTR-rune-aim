#!/usr/bin/env bash
# 构建脚本：不再依赖 ROS2（rclcpp 已剥离）
set -eo pipefail
export PATH=/usr/local/cuda/bin:$PATH
set -u
mkdir -p lib
cc_bin="${CC:-}"
cxx_bin="${CXX:-}"
if [[ -z "$cc_bin" ]]; then
  if command -v gcc-14 >/dev/null 2>&1; then cc_bin=gcc-14; else cc_bin=gcc; fi
fi
if [[ -z "$cxx_bin" ]]; then
  if command -v g++-14 >/dev/null 2>&1; then cxx_bin=g++-14; else cxx_bin=g++; fi
fi
jobs="${BUILD_JOBS:-$(nproc)}"
cmake -S . -B build -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}" \
  -DCMAKE_C_COMPILER="$cc_bin" -DCMAKE_CXX_COMPILER="$cxx_bin"
cmake --build build -j"$jobs"
cp -f build/librune_core.a lib/librune_core.a
echo "构建完成：build/rune_aim  build/rune_bench  build/delay_calib  lib/librune_core.a"
