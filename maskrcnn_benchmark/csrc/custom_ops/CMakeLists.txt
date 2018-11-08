cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(maskrcnn_benchmark_customops)

find_package(Torch REQUIRED)

add_library(maskrcnn_benchmark_customops SHARED custom_ops.cpp ../cpu/nms_cpu.cpp ../cpu/ROIAlign_cpu.cpp)

target_compile_features(maskrcnn_benchmark_customops PUBLIC cxx_range_for)
include_directories(BEFORE ../)
target_link_libraries(maskrcnn_benchmark_customops "${TORCH_LIBRARIES}")
target_link_libraries(maskrcnn_benchmark_customops "opencv_core" "opencv_highgui")
target_compile_definitions(maskrcnn_benchmark_customops PRIVATE maskrcnn_benchmark_customops_EXPORTS)
