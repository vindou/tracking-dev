# trk_flow: the inference process of the flow module (Jetson: TensorRT and CUDA from JetPack).
#   cmake -S . -B build -DTRACKER_DIR=$HOME/tracker && make -C build -j6
# The tracker library is used through its C interface (trk_flowio), built here from the tracker sources.
cmake_minimum_required(VERSION 3.16)
project(trk_flow LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(TRACKER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tracker" CACHE PATH "tracker source directory")
add_subdirectory(${TRACKER_DIR} tracker_build EXCLUDE_FROM_ALL)

find_path(TRT_INCLUDE NvInfer.h PATHS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu REQUIRED)
find_library(TRT_LIB nvinfer REQUIRED)
find_path(CUDA_INCLUDE cuda_runtime.h PATHS /usr/local/cuda/include REQUIRED)
find_library(CUDA_LIB cudart PATHS /usr/local/cuda/lib64 REQUIRED)

add_executable(trk_flow trk_flow.cpp)
target_compile_options(trk_flow PRIVATE -Wall -Wextra)
target_include_directories(trk_flow PRIVATE ${TRACKER_DIR}/src)
target_include_directories(trk_flow SYSTEM PRIVATE ${TRT_INCLUDE} ${CUDA_INCLUDE})
target_link_libraries(trk_flow PRIVATE trk_flowio ${TRT_LIB} ${CUDA_LIB})
