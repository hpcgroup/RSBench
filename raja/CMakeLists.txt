cmake_minimum_required(VERSION 3.21)

project(RSBench VERSION 1.0 LANGUAGES CXX CUDA)

find_package(RAJA REQUIRED)
find_package(Umpire REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -forward-unknown-to-host-compiler -arch=sm_70 -extended-lambda -restrict -keep ${CUDA_EXTRA_FLAGS}")

set(SOURCE main.cpp io.cpp simulation.cpp init.cpp utils.cpp material.cpp)

add_executable(RSBench ${SOURCE})
target_link_libraries(RSBench RAJA umpire)

set_source_files_properties(${SOURCE} PROPERTIES
									  LANGUAGE CUDA)

set_target_properties(RSBench PROPERTIES
							CMAKE_CXX_STANDARD 17
							CUDA_SEPERABLE_COMPILATION ON
							CUDA_STANDARD 17
							CUDA_STANDARD_REQUIRED ON)
