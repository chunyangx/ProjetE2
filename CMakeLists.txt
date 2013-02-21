project(Texture_Synthesis)
find_package(OpenCV REQUIRED)
add_executable(Texture_Synthesis Texture_synthesis.cpp)
target_link_libraries(Texture_Synthesis ${OpenCV_LIBS})
