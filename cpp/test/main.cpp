#include <gtest/gtest.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

int main(int argc, char *argv[]) {
  // Tests should not output info (unless overridden via environment variable)
  spdlog::set_level(spdlog::level::warn);
  spdlog::cfg::load_env_levels();

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}