#include <benchmark/benchmark.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

int main(int argc, char **argv) {
  // No logging in benchmarks (unless overridden via environment variable)
  spdlog::set_level(spdlog::level::off);
  spdlog::cfg::load_env_levels();

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}