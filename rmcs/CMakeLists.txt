cmake_minimum_required(VERSION 3.22)
project(rune_full LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d ximgproc videoio highgui)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# 修复：detector 与 tracker 各有一个 rune.cpp，归档成员重名会导致链接器取错目标文件。
# 用 OBJECT_OUTPUT_NAME 显式区分。
set_source_files_properties(module/detector/rune.cpp PROPERTIES OBJECT_OUTPUT_NAME rune_detector)
set_source_files_properties(module/tracker/model/rune.cpp PROPERTIES OBJECT_OUTPUT_NAME rune_model)

add_library(rune_full STATIC
  # ---- 检测（神经 TensorRT） ----
  module/detector/rune.cpp
  module/gpu/rune_gpu.cu

  # ---- 跟踪 / 拟合 / 虚拟符 ----
  module/tracker/model/rune.cpp
  module/tracker/model/rune_energy_fitter.cpp
  module/tracker/model/virtual_rune.cpp

  # ---- 弹道（吸纳自 rmcs_auto_aim_v2） ----
  module/fire_control/trajectory_solution.cpp

  # ---- 火控状态机（新，RP RuneDecisionModule 设计） ----
  module/fire_control/rune_fire_control.cpp

  # ---- 预测误差诊断（新，RP PowerRuneDiagnostics 设计） ----
  module/diagnostics/rune_diagnostics.cpp

  # ---- 链路延迟互相关标定（新，吸纳 Climber serial_delay 核心算法） ----
  module/diagnostics/delay_calibrator.cpp

  # ---- PnP / 重投影 / 角点优化（此前漏编译，RuneModel::init 依赖） ----
  utility/math/camera.cpp
  utility/math/reprojection.cpp
  utility/math/corners_optimizor.cpp
  utility/math/solve_pnp/pnp_solution.cpp
)
target_include_directories(rune_full PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rune_full PUBLIC ${OpenCV_LIBS} Eigen3::Eigen CUDA::cudart nvinfer)
target_compile_options(rune_full PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--Wno-deprecated-gpu-targets>
)
set_target_properties(rune_full PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

# ---- 检测器性能测试（原有） ----
add_executable(rune_video_test test/rune_video_test.cpp)
target_link_libraries(rune_video_test PRIVATE rune_full)

# ---- 全链路真值闭环测试（虚拟符） ----
add_executable(rune_virtual_test app/virtual_test.cpp)
target_link_libraries(rune_virtual_test PRIVATE rune_full)
if(BUILD_TESTING)
  add_test(NAME rune_virtual_small
    COMMAND rune_virtual_test 0 3 100 0 0)
  add_test(NAME rune_virtual_large
    COMMAND rune_virtual_test 1 7 100 0 0)
  set_tests_properties(rune_virtual_small rune_virtual_large PROPERTIES TIMEOUT 15)
endif()

# ---- 视频回放链路测试（检测→跟踪→火控） ----
add_executable(rune_video_aim app/video_test.cpp)
target_link_libraries(rune_video_aim PRIVATE rune_full)

# ---- 链路延迟标定器离线自检（吸纳 Climber serial_delay 算法） ----
add_executable(rune_delay_calib_test app/delay_calib_test.cpp)
target_link_libraries(rune_delay_calib_test PRIVATE rune_full)
if(BUILD_TESTING)
  add_test(NAME rune_delay_calib COMMAND rune_delay_calib_test 120 2 30)
  set_tests_properties(rune_delay_calib PROPERTIES TIMEOUT 60)
endif()
