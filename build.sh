#!/usr/bin/env bash
# 构建脚本：不再依赖 ROS2（rclcpp 已剥离）
set -eo pipefail
export PATH=/usr/local/cuda/bin:$PATH
set -u
mkdir -p lib
CC=gcc-14 CXX=g++-14 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
cp -f build/librune_core.a lib/librune_core.a
echo "构建完成：build/rune_aim  build/rune_bench  build/delay_calib  lib/librune_core.a"
