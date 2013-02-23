project(Texture_Synthesis)
find_package(OpenCV REQUIRED)

# Switching compiler to g++ on macos
if(NOT CMAKE_COMPILER_IS_GNUCXX)
  message("-- Setting compiler to g++")
  set(CMAKE_CXX_COMPILER "g++")
endif()

# Additional included path
include_directories(inc/)

add_library(Solver src/alglibinternal.cpp src/alglibmisc.cpp src/ap.cpp src/linalg.cpp src/solvers.cpp)
add_executable(Test src/solve_opt.cpp)
target_link_libraries(Test Solver)

add_library(Common src/common.cpp)
add_executable(TestNeighborhood src/testneighborhood.cpp)
target_link_libraries(TestNeighborhood ${OpenCV_LIBS} Common)

add_executable(Texture_Synthesis Texture_synthesis.cpp)
target_link_libraries(Texture_Synthesis ${OpenCV_LIBS})
