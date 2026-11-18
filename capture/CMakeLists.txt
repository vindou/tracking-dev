cmake_minimum_required(VERSION 3.16)
project(d435_capture LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

option(CAPTURE_TSAN "Build with ThreadSanitizer" OFF)
option(CAPTURE_ASAN "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(CAPTURE_WERROR "Treat warnings as errors" ON)

add_compile_options(-Wall -Wextra)
if(CAPTURE_WERROR)
  add_compile_options(-Werror)
endif()
if(CAPTURE_TSAN)
  add_compile_options(-fsanitize=thread -g -O1)
  add_link_options(-fsanitize=thread)
endif()

if(CAPTURE_ASAN)
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer -g -O1)
  add_link_options(-fsanitize=address,undefined)
endif()
if(CAPTURE_TSAN AND CAPTURE_ASAN)
  message(FATAL_ERROR "CAPTURE_TSAN and CAPTURE_ASAN cannot be combined")
endif()

find_package(Threads REQUIRED)

# Camera-independent core: builds and is unit-tested on Linux and macOS.
add_library(capture_core STATIC
  src/event_signal.cpp
  src/histogram.cpp
  src/matcher.cpp
  src/pairing_engine.cpp
  src/shm_sink.cpp)
target_include_directories(capture_core PUBLIC src)
target_link_libraries(capture_core PUBLIC Threads::Threads)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(capture_core PUBLIC rt)
endif()

add_executable(capture_tests
  tests/test_main.cpp
  tests/test_ring.cpp
  tests/test_matcher.cpp
  tests/test_shm.cpp
  tests/test_shm_v2.cpp
  tests/test_fps.cpp)
target_link_libraries(capture_tests PRIVATE capture_core)

add_executable(d435_pair_reader tools/pair_reader.cpp)
target_link_libraries(d435_pair_reader PRIVATE capture_core)

# Capture program: needs librealsense2 (on the Jetson: cmake -DCMAKE_PREFIX_PATH=$HOME/.local ..)
find_package(realsense2 QUIET)
if(realsense2_FOUND)
  add_executable(d435_capture
    src/main.cpp
    src/camera.cpp
    src/config.cpp
    src/status.cpp)
  target_link_libraries(d435_capture PRIVATE capture_core realsense2::realsense2)
  # rpath so that the binary finds librealsense2 in ~/.local/lib without LD_LIBRARY_PATH
  set_target_properties(d435_capture PROPERTIES BUILD_RPATH "${realsense2_DIR}/../.." INSTALL_RPATH "${realsense2_DIR}/../..")
  message(STATUS "librealsense2 ${realsense2_VERSION}: building d435_capture")
else()
  message(STATUS "librealsense2 not found: building core library, tests and reader only")
endif()
