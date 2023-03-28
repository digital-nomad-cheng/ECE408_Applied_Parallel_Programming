
project(ECE408)
cmake_minimum_required(VERSION 3.18)

find_package(CUDA REQUIRED)

if (CUDA_FOUND)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wall;")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")    
    
    # adding debuggin to CUDA NVCC flags. For Nvidia's Nsight tools.
    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")
    include_directories(${CMAKE_SOURCE_DIR})
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

file( GLOB wb_src lib/*.cpp lib/*.c )

include_directories("lib")

CUDA_ADD_EXECUTABLE(MP0 MP0/template.cu ${wb_src})
CUDA_ADD_EXECUTABLE(MP1 MP1/template.cu ${wb_src})
CUDA_ADD_EXECUTABLE(MP2 MP2/template.cu ${wb_src})
CUDA_ADD_EXECUTABLE(MP3 MP3/template.cu ${wb_src})
CUDA_ADD_EXECUTABLE(MP4 MP4/template.cu ${wb_src})
CUDA_ADD_EXECUTABLE(MP5.1 MP5.1/template.cu ${wb_src})
CUDA_ADD_EXECUTABLE(MP5.2_ks MP5.2/template_ks.cu ${wb_src})
CUDA_ADD_EXECUTABLE(MP5.2_bk MP5.2/template_bk.cu ${wb_src})

# nvcc MP0/template.cu ${wb_src} -Ilib -o template

