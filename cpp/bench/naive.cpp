#include "Naive.hpp"

#include <benchmark/benchmark.h>
#include <iostream>

using namespace ahr;
static void BM_Naive(benchmark::State &state) {
  std::ostringstream oss;

  Dim const M = state.range(0);
  Dim const X = state.range(1);
  Dim const N = state.range(2);

  Naive naive{M, X, X};
  naive.init("gauss");

  // Lower CFL -> lower dt -> fewer repeats
  CFLFrac = 0.05;

  // Run once, to get the timestep right, and avoid repeats during benchmarking
  naive.run(1, 0);

  // Benchmarking
  for (auto _ : state) {
    naive.run(N, 0);
  }
}

BENCHMARK(BM_Naive)
    ->ArgsProduct({{2, 4, 10, 45}, {128, 256, 512}, {5}})
    ->ArgsProduct({{2, 4}, {1024, 2048}, {5}})
    ->ArgsProduct({{2}, {4096}, {5}})
    ->Unit(benchmark::kMillisecond);
