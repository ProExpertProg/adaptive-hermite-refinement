cmake_minimum_required(VERSION 3.21)
project(adaptive_hermite_refinement)

set(CMAKE_CXX_STANDARD 20)

# ==========================
# Library dependencies
# TODO: find locally first if available
# ==========================
include(FetchContent)
FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

FetchContent_Declare(
        mdspan
        GIT_REPOSITORY https://github.com/kokkos/mdspan.git
        GIT_TAG 9ceface91483775a6c74d06ebf717bbb2768452f
)
FetchContent_MakeAvailable(mdspan)

FetchContent_Declare(
        cnpy
        GIT_REPOSITORY https://github.com/rogersce/cnpy.git
        GIT_TAG 4e8810b1a8637695171ed346ce68f6984e585ef4
)
FetchContent_MakeAvailable(cnpy)

target_include_directories(cnpy PUBLIC ${CNPY_SOURCE_DIR})

# ==========================
# Project code
# ==========================
enable_testing()
add_subdirectory(fftw-cpp)
target_link_libraries(fftw-cpp INTERFACE mdspan)

option(ENABLE_AVX512 "Enable AVX512 instructions (e.g. might want to turn off for the sake of Valgrind)" OFF)
if (ENABLE_AVX512)
    add_compile_options("-mavx512f" "-march=native")
endif ()

option(ENABLE_CILK "Enable Cilk parallelism" ON)
if (ENABLE_CILK)
    add_compile_options("-fopencilk")
    add_link_options("-fopencilk")
    add_compile_definitions(CILK_ENABLED)
endif ()

# TODO: cleanup (currently just to test formatting)
add_executable(adaptive_hermite_refinement lib/test-triangle.cpp)
target_include_directories(adaptive_hermite_refinement PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(adaptive_hermite_refinement argparse mdspan)

add_library(src-lib OBJECT lib/Naive.cpp lib/Naive.hpp lib/HermiteRunner.cpp lib/HermiteRunner.hpp)
target_include_directories(src-lib PUBLIC lib/)
target_link_libraries(src-lib mdspan fftw-cpp cnpy)

add_subdirectory(bench)

add_executable(naive lib/driver.cpp)
target_link_libraries(naive argparse src-lib)

option(AHR_TEST "Enable AHR tests" ON)
if (AHR_TEST)
    enable_testing()
    add_subdirectory(test)
endif ()