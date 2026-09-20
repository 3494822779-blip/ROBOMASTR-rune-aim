cmake_minimum_required(VERSION 3.22)
project(rune_aim LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (Debug, Release, RelWithDebInfo)")
option(ENABLE_ROS2 "Build optional ROS 2 telemetry (--ros)" OFF)
option(ENABLE_SANITIZERS "Enable ASan/UBSan for host C++ code" OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d ximgproc videoio highgui)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
# yaml-cpp：系统 config 有时不可用，手动定义 imported target 最稳
find_library(YAML_CPP_LIB NAMES yaml-cpp REQUIRED)
if(NOT TARGET yaml-cpp::yaml-cpp)
  add_library(yaml-cpp::yaml-cpp UNKNOWN IMPORTED)
  set_target_properties(yaml-cpp::yaml-cpp PROPERTIES
    IMPORTED_LOCATION "${YAML_CPP_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "/usr/include")
endif()

# detector 与 rune_model 各有一个 detector/rune 语义，归档成员重名会串：
# 用 OBJECT_OUTPUT_NAME 显式区分（模块级拆分后仍保留，防御未来同名目标）
set_source_files_properties(src/detect/detector.cpp PROPERTIES OBJECT_OUTPUT_NAME rune_detector)
set_source_files_properties(src/track/rune_model.cpp PROPERTIES OBJECT_OUTPUT_NAME rune_model)

add_library(rune_core STATIC
  # ---- 检测 ----
  src/detect/detector.cpp
  src/detect/detect_worker.cpp
  src/detect/gpu_pipeline.cu
  # ---- 跟踪 ----
  src/track/rune_model.cpp
  src/track/energy_fitter.cpp
  src/track/virtual_rune.cpp
  # ---- 火控 ----
  src/fire/fire_control.cpp
  src/fire/trajectory.cpp
  # ---- 诊断 ----
  src/diag/diagnostics.cpp
  src/diag/delay_calibrator.cpp
  # ---- 可视化调试 ----
  src/debug/draw.cpp
  # ---- 核心 ----
  src/core/camera.cpp
  src/core/frame_source.cpp
  src/core/reprojection.cpp
  src/core/rune_pnp.cpp
  src/core/config_loader.cpp
)
target_include_directories(rune_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(rune_core PUBLIC ${OpenCV_LIBS} Eigen3::Eigen CUDA::cudart nvinfer yaml-cpp::yaml-cpp Threads::Threads)
target_compile_options(rune_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--Wno-deprecated-gpu-targets>
)
if(ENABLE_SANITIZERS)
  target_compile_options(rune_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined;-fno-omit-frame-pointer>)
  # rune_core is static, so sanitizer runtimes must be linked by its executables.
  target_link_options(rune_core INTERFACE -fsanitize=address,undefined)
endif()
set_target_properties(rune_core PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

# ---- 统一命令行入口（video / virtual / camera 三模式） ----
add_executable(rune_aim tools/rune_aim.cpp)
target_link_libraries(rune_aim rune_core)

# ---- 检测性能基准 ----
add_executable(rune_bench tools/rune_bench.cpp)
target_link_libraries(rune_bench PRIVATE rune_core)

# ---- 链路延迟标定 ----
add_executable(delay_calib tools/delay_calib.cpp)
target_link_libraries(delay_calib PRIVATE rune_core)

# ---- 相机棋盘格内参标定 ----
add_executable(camera_calib tools/camera_calib.cpp)
target_link_libraries(camera_calib PRIVATE ${OpenCV_LIBS})

# ---- 外参标定后的实弹落点 / 弹道偏置迭代测试 ----
add_executable(ballistic_test tools/ballistic_test.cpp)
target_link_libraries(ballistic_test PRIVATE rune_core)

if(BUILD_TESTING)
  add_executable(rune_prediction_test tests/prediction_test.cpp)
  target_link_libraries(rune_prediction_test PRIVATE rune_core)
  add_test(NAME rune_prediction_test COMMAND rune_prediction_test)
endif()

if(ENABLE_ROS2)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(diagnostic_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(visualization_msgs REQUIRED)
  target_sources(rune_aim PRIVATE src/ros/telemetry.cpp)
  target_compile_definitions(rune_aim PRIVATE RUNE_ENABLE_ROS2)
  ament_target_dependencies(rune_aim rclcpp sensor_msgs diagnostic_msgs geometry_msgs visualization_msgs)
endif()
