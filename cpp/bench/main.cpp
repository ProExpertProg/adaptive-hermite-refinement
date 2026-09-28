#include <benchmark/benchmark.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include "cilk.hpp"

int main(int argc, char **argv) {
  // No logging in benchmarks (unless overridden via environment variable)
  spdlog::set_level(spdlog::level::off);
  spdlog::cfg::load_env_levels();

  // Invoke a cilk_scope to avoid Cilk startup/shutdown in benchmarks
  cilk_scope {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
  }
}