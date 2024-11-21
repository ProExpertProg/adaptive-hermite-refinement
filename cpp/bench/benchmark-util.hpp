#pragma once
#include <benchmark/benchmark.h>

/// Add `min` as a statistic to the benchmark, useful for serial execution.
/// Not reported when only running 1 repetition.
#define BENCHMARK_WMIN(...)                                                                        \
  BENCHMARK(__VA_ARGS__)->ComputeStatistics("min", [](const std::vector<double> &v) {              \
    return *std::min_element(v.begin(), v.end());                                                  \
  })
