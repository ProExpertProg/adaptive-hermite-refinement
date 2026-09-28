#include "PrepareDerivatives.hpp"

#include <benchmark/benchmark.h>

static constexpr auto N_WARMUP_ITERS = 5;

using namespace ahr;
template <class Prepare> static void BM_Prepare(benchmark::State &state) {
  Dim const X = state.range(0);
  Dim const Y = state.range(1);

  Grid grid{1, X, Y};
  Prepare prepare{grid};

  auto buf = grid.cBufXY();
  auto buf2 = grid.cBufXY();
  auto buf3 = grid.cBufXY();

  // Warm-up
  for (int i = 0; i < N_WARMUP_ITERS; i++) {
    prepare(buf, {buf2, buf3});
  }

  for (auto _ : state) {
    prepare(buf, {buf2, buf3});
  }
}

BENCHMARK(BM_Prepare<PrepareDerivatives>)
    ->ArgsProduct({{2048, 4096, 8192}, {2048, 4096, 8192}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Prepare<PrepareDerivativesVector>)
    ->ArgsProduct({{2048, 4096, 8192}, {2048, 4096, 8192}})
    ->Unit(benchmark::kMillisecond);
