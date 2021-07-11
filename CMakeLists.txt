cmake_minimum_required(VERSION 3.17)
project(test)

set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_FLAGS "-fsanitize=address")

find_package(BLAS REQUIRED)

add_subdirectory(extern/xtl)
add_subdirectory(extern/xtensor)
add_subdirectory(extern/xtensor-blas)

add_subdirectory(lib)

add_executable(deep_learning main.cpp)
target_link_libraries(deep_learning pdl)