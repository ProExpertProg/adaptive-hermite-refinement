#include "typedefs.hpp"

#include "CliParams.hpp"
#include "HermiteRunner.hpp"
#include "Naive.hpp"
#include <iostream>
#include <spdlog/cfg/env.h>

// TODO
using namespace ahr;

int main(int argc, const char *argv[]) {
  // Load log levels from SPDLOG_LEVEL env variable
  spdlog::cfg::load_env_levels();

  CliParams cliParams{"naive"};

  try {
    cliParams.parse(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  auto const p = cliParams.get();
  Naive naive{p.M, p.X, p.X};
  HermiteRunner &runner = naive;

  runner.init("OT01");
  runner.run(p.N, p.saveInterval);
}
