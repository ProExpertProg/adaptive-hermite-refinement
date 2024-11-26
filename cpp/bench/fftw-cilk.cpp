#include "fftw-cpp/fftw-cpp.h"

#include "benchmark-util.hpp"
#include <cilk/cilk.h>
#include <cilk/cilkscale.h>
#include <cstring>
#include <iostream>
#include <thread>

static void copy(fftw::buffer const &in, fftw::buffer &out) {
  std::memcpy(out.data(), in.data(), in.size() * sizeof(fftw::buffer::element_type));
}

static void BM_copy_1D(benchmark::State &state) {
  size_t const X = state.range(0);

  fftw::buffer in{X};
  fftw::buffer out{X};

  // warmup
  copy(in, out);

  for (auto _ : state) {
    copy(in, out);
  }
}

static void BM_FFTW_1D(benchmark::State &state) {
  size_t const X = state.range(0);

  fftw::buffer in{X};
  fftw::buffer out{X};

  auto const plan = fftw::plan<>::dft(in, out, fftw::FORWARD, fftw::MEASURE);

  // warmup
  plan();

  for (auto _ : state) {
    plan();
  }
}

static void BM_copy_2D(benchmark::State &state) {
  size_t const X = state.range(0);
  size_t const Y = state.range(1);

  fftw::mdbuffer<2> in{X, Y};
  fftw::mdbuffer<2> out{X, Y};

  // warmup
  copy(in.container(), out.container());

  for (auto _ : state) {
    copy(in.container(), out.container());
  }
}

static void BM_FFTW_2D(benchmark::State &state) {
  size_t const X = state.range(0);
  size_t const Y = state.range(1);

  fftw::mdbuffer<2> in{X, Y};
  fftw::mdbuffer<2> out{X, Y};

  auto const plan = fftw::plan<2>::dft(in, out, fftw::FORWARD, fftw::MEASURE);

  // warmup
  plan();

  for (auto _ : state) {
    plan();
  }
}

auto constexpr K = 1024;
auto constexpr M = K * K;

BENCHMARK_WMIN(BM_copy_1D)
    ->RangeMultiplier(2)
    ->Range(16 * M, 64 * M)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_WMIN(BM_FFTW_1D)
    ->RangeMultiplier(2)
    ->Range(16 * M, 64 * M)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_WMIN(BM_copy_2D)
    ->ArgsProduct({{4 * K, 8 * K, 16 * K}, {4 * K, 8 * K, 16 * K}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_WMIN(BM_FFTW_2D)
    ->ArgsProduct({{4 * K, 8 * K, 16 * K}, {4 * K, 8 * K, 16 * K}})
    ->Unit(benchmark::kMillisecond);

// TODO pthreads, openmp, cilk -> compare all
#ifndef FFTW_PTHREADS
void parallel_for(void *(*work)(char *), char *jobdata, size_t elsize, int njobs, void *data) {
  // std::cout << "parallel_for: " << njobs << std::endl;
  cilk_for(int i = 0; i < njobs; ++i) { work(jobdata + i * elsize); }
}
#endif

int main(int argc, char **argv) {
  fftw_init_threads();
  int njobs = int(std::thread::hardware_concurrency()) / 2;

#ifdef FFTW_PTHREADS
#define cilk_scope_
#else
#define cilk_scope_ cilk_scope
  fftw_threads_set_callback(parallel_for, nullptr);
#endif

  if (auto const n = std::getenv("BENCH_NJOBS"); n) { njobs = std::stoi(n); }

  std::cout << "Using " << njobs << " jobs for FFTW" << std::endl;
  fftw_plan_with_nthreads(njobs);

  // make sure cilk isn't initialized and deinitialized for each benchmark
  cilk_scope_ {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
  }
}
