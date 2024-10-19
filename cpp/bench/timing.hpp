#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <numeric>

#ifdef CILK_ENABLED
#include <cilk/cilkscale.h>
#else
using wsp_t = int;
int wsp_getworkspan() { return 0; }
static inline void wsp_dump(wsp_t wsp, const char *tag) { return; }
#endif

using fms = std::chrono::duration<float, std::milli>;
constexpr unsigned N_TRIALS = 10;

std::pair<std::array<fms, 3>, std::array<wsp_t, N_TRIALS>> benchmark(std::ostream &out,
                                                                     auto beforeEach, auto run) {
  using namespace std::chrono_literals;
  std::array<fms, N_TRIALS> diff{};
  std::array<wsp_t, N_TRIALS> diff_wsp{};

  for (unsigned i = 0; i < N_TRIALS; i++) {
    beforeEach();
    auto start_wsp = wsp_getworkspan();
    auto start = std::chrono::high_resolution_clock::now();
    run();
    auto end = std::chrono::high_resolution_clock::now();
    auto end_wsp = wsp_getworkspan();
    diff[i] = end - start;
    out << diff[i].count() << "ms" << std::endl;

    diff_wsp[i] = end_wsp - start_wsp;
  }

  std::sort(diff.begin(), diff.end());
  int half = diff.size() / 2;
  fms median;
  if constexpr (diff.size() % 2 == 1) {
    median = diff[half];
  } else {
    median = (diff[half - 1] + diff[half]) / 2;
  }

  auto minimum = diff[0];
  auto average = std::accumulate(diff.begin(), diff.end(), 0.0ms) / N_TRIALS;

  out << "Minimum: " << minimum.count() << "ms" << std::endl;
  out << "Median: " << median.count() << "ms" << std::endl;
  out << "Average: " << average.count() << "ms" << std::endl;

  return {{minimum, median, average}, diff_wsp};
}