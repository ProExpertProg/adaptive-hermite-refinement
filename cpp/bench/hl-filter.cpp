#include "Filter.hpp"

#include <benchmark/benchmark.h>

static constexpr auto N_WARMUP_ITERS = 5;

using namespace ahr;
template <class Filter> static void BM_HouLiFilter(benchmark::State &state) {
  Dim const X = state.range(0);
  Dim const Y = state.range(1);

  Grid grid{1, X, Y};
  Filter filter{grid};

  auto buf = grid.cBufXY();

  // Warm-up
  for (int i = 0; i < N_WARMUP_ITERS; i++) {
    filter(buf);
  }

  for (auto _ : state) {
    filter(buf);
  }
}

BENCHMARK(BM_HouLiFilter<HouLiFilter>)
    ->ArgsProduct({{2048, 4096, 8192}, {2048, 4096, 8192}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_HouLiFilter<HouLiFilterCached>)
    ->ArgsProduct({{2048, 4096, 8192}, {2048, 4096, 8192}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_HouLiFilter<HouLiFilterCached1D>)
    ->ArgsProduct({{2048, 4096, 8192}, {2048, 4096, 8192}})
    ->Unit(benchmark::kMillisecond);
