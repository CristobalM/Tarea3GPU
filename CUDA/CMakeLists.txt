cmake_minimum_required(VERSION 3.10)
project(Tarea3GPU)

set(CMAKE_CXX_STANDARD 11)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.0/)
find_package(CUDA QUIET REQUIRED)
find_package(OpenGL)
find_package(GLUT)

cuda_add_library(cudalib SHARED
        GameOfLife.cu GameOfLife.h
        defs.h)

add_executable(Tarea3GPU main.cpp SequentialGameOfLife.cpp SequentialGameOfLife.h)
add_executable(VisualizacionConsola ConsoleVisualization.cpp )
target_link_libraries(Tarea3GPU cudalib ${CUDA_LIBRARIES})
target_link_libraries(VisualizacionConsola cudalib ${CUDA_LIBRARIES})
#string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_50,code=sm_50")
#target_include_directories(Tarea3GPU_OpenCL PRIVATE OpenCL)
