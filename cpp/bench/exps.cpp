#include "Exponentials.hpp"

#include "benchmark-util.hpp"
#include <iostream>

using namespace ahr;
using namespace ahr::exp;
template <typename Exp> static void BM_ExpKXKY(benchmark::State &state) {
  Dim const X = state.range(0);
  Dim const Y = state.range(1);

  Grid grid{1, X, Y};
  Real dt = 1.0;
  HyperCoefficients hyper = HyperCoefficients::calculate(dt, grid);

  Exp exp{grid};
  exp.update(hyper, dt);

  for (auto _ : state) {
    grid.for_each_kxky([&](Dim kx, Dim ky) { benchmark::DoNotOptimize(exp(kx, ky)); });
  }
}

template <typename Exp> static void BM_ExpM(benchmark::State &state) {
  Dim const M = state.range(0);
  Dim const X = state.range(1);
  Dim const Y = state.range(2);

  Grid grid{M, X, Y};
  Real dt = 1.0;
  HyperCoefficients hyper = HyperCoefficients::calculate(dt, grid);

  Exp exp{grid};
  exp.update(hyper, dt);

  for (auto _ : state) {
    for (Dim m = 0; m < M; ++m) {
      grid.for_each_kxky([&](Dim kx, Dim ky) { benchmark::DoNotOptimize(exp(m)); });
    }
  }
}

BENCHMARK_WMIN(BM_ExpKXKY<Eta>)
    ->ArgsProduct({{2048, 4096, 8192}, {2048, 4096, 8192}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK_WMIN(BM_ExpKXKY<Nu>)
    ->ArgsProduct({{2048, 4096, 8192}, {2048, 4096, 8192}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK_WMIN(BM_ExpKXKY<NuG>)
    ->ArgsProduct({{2048, 4096, 8192}, {2048, 4096, 8192}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK_WMIN(BM_ExpM<GM>)
    ->ArgsProduct({{2, 4, 10}, {2048, 4096}, {2048, 4096}})
    ->Unit(benchmark::kMillisecond);
