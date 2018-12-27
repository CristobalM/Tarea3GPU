cmake_minimum_required(VERSION 3.12)
project(Tarea3GPUOpenCL)

set(CMAKE_CXX_STANDARD 11)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.0/)
find_package(OpenCL REQUIRED)
find_package(OpenGL)
find_package(GLUT)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

add_executable(Tarea3GPUOpenCL HighLife.cpp)
target_link_libraries(Tarea3GPUOpenCL OpenCL::OpenCL ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
#string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_50,code=sm_50")
