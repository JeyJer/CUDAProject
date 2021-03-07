cmake_minimum_required(VERSION 3.16)
project(projet_cuda CUDA CXX)

set(CMAKE_CUDA_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fpermissive")

include_directories(
        lib
        lib/common
        lib/gpu
        lib/cpu
        /usr/include
        ${OpenCV_INCLUDE_DIRS})

find_package (PkgConfig REQUIRED)
find_package (OpenCV REQUIRED)

add_executable(img_processor ImageProcessor.cu)

target_link_libraries(img_processor ${OpenCV_LIBS})

set_target_properties(
        img_processor
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(gpu_img_processor
        gpu_img_processor.cu
        lib/gpu/gpu_img_transform.cuh
        lib/gpu/gpu_img_transform.cu
        lib/common/utilities.hpp
        lib/common/utilities.cpp
        lib/common/menu_lib.hpp
        lib/common/menu_lib.cpp
        )

target_link_libraries(gpu_img_processor ${OpenCV_LIBS})

set_target_properties(
        gpu_img_processor
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(std_img_processor
        cpu_img_processor.cpp
        lib/common/utilities.hpp
        lib/common/utilities.cpp
        lib/common/menu_lib.hpp
        lib/common/menu_lib.cpp
        )

target_link_libraries(std_img_processor ${OpenCV_LIBS})