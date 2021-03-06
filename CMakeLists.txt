cmake_minimum_required(VERSION 3.16)
project(projet_cuda CUDA CXX)

set(CMAKE_CUDA_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fpermissive")

include_directories(
        lib
        /usr/include
        ${OpenCV_INCLUDE_DIRS})

find_package (PkgConfig REQUIRED)
find_package (OpenCV REQUIRED)

add_executable(cuda_img_processor ImageProcessor.cu)

target_link_libraries(cuda_img_processor ${OpenCV_LIBS})

set_target_properties(
        cuda_img_processor
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(std_img_processor
        cpu_img_processor.cpp
        lib/convolution_transform.hpp
        lib/convolution_transform.cpp)

target_link_libraries(std_img_processor ${OpenCV_LIBS})