#include "Exponentials.hpp"

#include "benchmark-util.hpp"
#include <iostream>

using namespace ahr;
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

BENCHMARK_WMIN(BM_ExpKXKY<exp::Eta>)
    ->ArgsProduct({{2048, 4096, 8192}, {2048, 4096, 8192}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK_WMIN(BM_ExpKXKY<exp::Nu>)
    ->ArgsProduct({{2048, 4096, 8192}, {2048, 4096, 8192}})
    ->Unit(benchmark::kMillisecond);
BENCHMARK_WMIN(BM_ExpKXKY<exp::NuG>)
    ->ArgsProduct({{2048, 4096, 8192}, {2048, 4096, 8192}})
    ->Unit(benchmark::kMillisecond);
