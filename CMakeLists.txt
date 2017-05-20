cmake_minimum_required(VERSION 3.7)
project(PrimalDualSetCover)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp)
find_package(Eigen3 REQUIRED)

include_directories(EIGEN3_INCLUDE_DIR)

add_executable(PrimalDualSetCover ${SOURCE_FILES})
target_link_libraries (PrimalDualSetCover Eigen3::Eigen)