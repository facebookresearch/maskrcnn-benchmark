cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(maskrcnn_benchmark_cppdemo)

find_package(Torch REQUIRED NO_CMAKE_FIND_ROOT_PATH)

add_executable(traced_model_demo traced_model.cpp)

target_compile_features(traced_model_demo PUBLIC cxx_range_for)

target_link_libraries(traced_model_demo "${TORCH_LIBRARIES}")
target_link_libraries(traced_model_demo "dl")

find_package (OpenCV REQUIRED NO_CMAKE_FIND_ROOT_PATH)
target_link_libraries(traced_model_demo "opencv_core" "opencv_highgui" "opencv_imgcodecs" "opencv_imgproc")

