#include "CachedExponentials.hpp"
#include "Exponentials.hpp"

#include "benchmark-util.hpp"
#include <iostream>

using namespace ahr;
using namespace ahr::exp;
template <space_like Exp> static void BM_ExpKXKY(benchmark::State &state) {
  Dim const M = state.range(0);
  Dim const X = state.range(1);
  Dim const Y = state.range(2);

  Grid grid{M, X, Y};
  Real dt = 1.0;
  HyperCoefficients hyper = HyperCoefficients::calculate(dt, grid);
  Exp exp{grid};

  for (auto _ : state) {
    dt *= 1.2;
    exp.update(hyper, dt);
    for (int m = 0; m < M; ++m) {
      FOREACH_KXKY(grid, { benchmark::DoNotOptimize(exp(kx, ky)); });
    }
  }
}

template <moment_like Exp> static void BM_ExpM(benchmark::State &state) {
  Dim const M = state.range(0);
  Dim const X = state.range(1);
  Dim const Y = state.range(2);

  Grid grid{M, X, Y};
  Real dt = 1.0;
  HyperCoefficients hyper = HyperCoefficients::calculate(dt, grid);

  Exp exp{grid};

  for (auto _ : state) {
    dt *= 1.2;
    exp.update(hyper, dt);
    for (Dim m = 0; m < M; ++m) {
      FOREACH_KXKY(grid, { benchmark::DoNotOptimize(exp(m)); });
    }
  }
}

BENCHMARK_WMIN(BM_ExpKXKY<Eta>)
    ->ArgsProduct({{2, 4, 16}, {2048, 4096}, {2048, 4096}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK_WMIN(BM_ExpKXKY<Nu>)
    ->ArgsProduct({{2, 4, 16}, {2048, 4096}, {2048, 4096}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK_WMIN(BM_ExpKXKY<NuG>)
    ->ArgsProduct({{2, 4, 16}, {2048, 4096}, {2048, 4096}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK_WMIN(BM_ExpM<GM>)
    ->ArgsProduct({{4, 8, 16}, {2048, 4096}, {2048, 4096}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_WMIN(BM_ExpKXKY<CachedKXKY<Eta>>)
    ->ArgsProduct({{2, 4, 16}, {2048, 4096}, {2048, 4096}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK_WMIN(BM_ExpKXKY<CachedKXKY<Nu>>)
    ->ArgsProduct({{2, 4, 16}, {2048, 4096}, {2048, 4096}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK_WMIN(BM_ExpKXKY<CachedKXKY<NuG>>)
    ->ArgsProduct({{2, 4, 16}, {2048, 4096}, {2048, 4096}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK_WMIN(BM_ExpM<CachedM<GM>>)
    ->ArgsProduct({{4, 8, 16}, {2048, 4096}, {2048, 4096}})
    ->Unit(benchmark::kMillisecond);
