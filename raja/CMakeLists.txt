cmake_minimum_required(VERSION 3.21)

project(RSBench VERSION 1.0 LANGUAGES CXX CUDA)

set(CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_ARCHITECTURES 70)

find_package(RAJA REQUIRED)
find_package(umpire REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -gencode arch=compute_70,code=sm_70 -forward-unknown-to-host-compiler -extended-lambda -restrict -keep ${CUDA_EXTRA_FLAGS}")

set(SOURCE main.cpp io.cpp simulation.cpp init.cpp utils.cpp material.cpp)

add_executable(RSBench ${SOURCE})

set_source_files_properties(${SOURCE} PROPERTIES
									  LANGUAGE CUDA)

set_target_properties(RSBench PROPERTIES
							CXX_STANDARD 17
							CUDA_SEPERABLE_COMPILATION ON
							CUDA_STANDARD 17
							CUDA_STANDARD_REQUIRED ON
							CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(RSBench RAJA umpire)
