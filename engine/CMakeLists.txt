cmake_minimum_required(VERSION 3.15)
project(vitna_anchor C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB SOURCES "src/*.c")

add_executable(vitna-anchor ${SOURCES})

if(WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_WINSOCK_DEPRECATED_NO_WARNINGS)
    target_link_libraries(vitna-anchor PRIVATE ws2_32)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(vitna-anchor PRIVATE Threads::Threads m)
endif()

if(MSVC)
    target_compile_options(vitna-anchor PRIVATE /W3 /O2)
else()
    target_compile_options(vitna-anchor PRIVATE -Wall -Wextra -O3)
endif()
