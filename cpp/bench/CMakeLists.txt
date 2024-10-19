include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../lib)

add_executable(bench-fft fftw/bench-fft.cpp)
target_link_libraries(bench-fft argparse fftw-cpp)

add_executable(bench-fft-multi fftw/bench-fft-multi.cpp)
target_link_libraries(bench-fft-multi argparse fftw-cpp)

add_executable(bench-fft-fusion fftw/bench-fft-fusion.cpp)
target_link_libraries(bench-fft-fusion argparse fftw-cpp)

option(BENCHMARK_ENABLE_TESTING "Enable testing for Google Benchmark." OFF)

include(FetchContent)
FetchContent_Declare(
        google-benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG 3fd1e6a7aee12e6878d7e039f947ea81140b4f5a
)
FetchContent_MakeAvailable(google-benchmark)

add_executable(bench-naive naive.cpp)
target_link_libraries(bench-naive src-lib benchmark::benchmark_main)

if (ENABLE_CILK)
    add_executable(bench-fftw-cilk fftw-cilk.cpp)
    target_link_libraries(bench-fftw-cilk fftw-cpp benchmark::benchmark fftw3_threads)

    add_executable(bench-fftw-pthread fftw-cilk.cpp)
    target_compile_definitions(bench-fftw-pthread PUBLIC FFTW_PTHREADS)
    target_link_libraries(bench-fftw-pthread fftw-cpp benchmark::benchmark fftw3_threads)
endif ()