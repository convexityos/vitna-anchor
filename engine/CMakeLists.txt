cmake_minimum_required(VERSION 3.15)
project(vitna_anchor C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB SOURCES "src/*.c")
set(LIB_SOURCES ${SOURCES})
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*/main\\.c$")

add_executable(vitna-anchor ${SOURCES})
add_executable(vitna-anchor-tests ${LIB_SOURCES} tests/unit_tests.c)

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
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -O3)
    endif()
endforeach()
