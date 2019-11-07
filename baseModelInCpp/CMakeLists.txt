cmake_minimum_required(VERSION 3.15)
project(baseModelInCpp)

set(CMAKE_CXX_STANDARD 11)
set(Torch_DIR /Users/peterzheng/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
add_executable(baseModelInCpp main.cpp)
target_link_libraries(baseModelInCpp ${TORCH_LIBRARIES})