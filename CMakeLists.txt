project(Texture_Synthesis)
find_package(OpenCV REQUIRED)

cmake_minimum_required(VERSION 2.8)

# Switching compiler to g++ on macos
if(NOT CMAKE_COMPILER_IS_GNUCXX)
  message("-- Setting compiler to g++")
  set(CMAKE_CXX_COMPILER "g++")
endif()

# Additional included path
include_directories(inc/)

file(GLOB allsrc "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(CommonLib ${allsrc})

add_executable(TestNeighborhood src/testneighborhood.cpp)
target_link_libraries(TestNeighborhood ${OpenCV_LIBS} CommonLib)

add_executable(Texture_Synthesis Texture_synthesis.cpp)
target_link_libraries(Texture_Synthesis ${OpenCV_LIBS} CommonLib)
