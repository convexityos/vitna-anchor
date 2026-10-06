cmake_minimum_required(VERSION 3.15)
project(vitna_anchor C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# The CUDA path. -DVITNA_CUDA=ON compiles src/model_cuda.cu with nvcc into a
# library both executables link, so --device cuda can run the model on a GPU.
# Off, the default, the build is the CPU engine alone, as before.
option(VITNA_CUDA "Build the CUDA path, for --device cuda" OFF)
if(VITNA_CUDA)
    # For this machine's GPU, unless architectures are named, as in
    # -DCMAKE_CUDA_ARCHITECTURES=86.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
        if(CMAKE_VERSION VERSION_LESS 3.24)
            message(FATAL_ERROR "Name the GPU architectures, as in -DCMAKE_CUDA_ARCHITECTURES=86: finding this machine's own needs CMake 3.24 or later")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    # The CUDA runtime links statically, and on Windows its library is built
    # against the static C runtime, so this build uses that runtime throughout
    # rather than mixing two.
    if(MSVC)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

include_directories(include)

file(GLOB SOURCES "src/*.c")
set(LIB_SOURCES ${SOURCES})
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*/main\\.c$")

add_executable(vitna-anchor ${SOURCES})
add_executable(vitna-anchor-tests ${LIB_SOURCES} tests/unit_tests.c)
if(VITNA_CUDA)
    add_library(vitna-cuda STATIC src/model_cuda.cu)
endif()

enable_testing()
add_test(NAME unit COMMAND vitna-anchor-tests)

foreach(target vitna-anchor vitna-anchor-tests)
    if(WIN32)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS _WINSOCK_DEPRECATED_NO_WARNINGS)
        target_link_libraries(${target} PRIVATE ws2_32)
    else()
        find_package(Threads REQUIRED)
        target_link_libraries(${target} PRIVATE Threads::Threads m)
    endif()

    if(MSVC)
        target_compile_options(${target} PRIVATE /W3 /O2)
        # A UTF-8 program, for paths with any letters in them: the manifest says how.
        target_sources(${target} PRIVATE windows/utf8.manifest)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -O3)
    endif()

    if(VITNA_CUDA)
        target_compile_definitions(${target} PRIVATE VITNA_CUDA=1)
        target_link_libraries(${target} PRIVATE vitna-cuda)
    endif()
endforeach()
