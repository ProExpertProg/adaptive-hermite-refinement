#include "fftw-cpp/fftw-cpp.h"

#include "cilk.hpp"
#include "timing.hpp"

#include <argparse/argparse.hpp>
#include <iostream>
#include <numbers>

int main(int argc, const char *argv[]) {
  argparse::ArgumentParser arguments("ahr");

  arguments.add_argument("X").help("Size of FFT domain").scan<'i', int>().default_value(128);

  arguments.add_argument("M")
      .help("Number of moments")
      .scan<'i', unsigned long>()
      .default_value(100ul);

  arguments.add_argument("C")
      .help("Number of transforms computed with one call")
      .scan<'i', int>()
      .default_value(1);

  try {
    arguments.parse_args(argc, argv);
    if (arguments.get<unsigned long>("M") % arguments.get<int>("C")) {
      throw std::invalid_argument("C should divide M");
    }
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << arguments;
    return 1;
  }

  const auto X = arguments.get<int>("X"), C = arguments.get<int>("C");
  const auto M = arguments.get<unsigned long>("M"), MC = M / C;
  const auto X2 = X * X;

  std::cout << "Running " << M << " transforms of size " << X << "^2 in batches of " << C
            << std::endl;

  fftw::buffer in{X2 * M}, out{X2 * M};
  std::vector<fftw_plan> plans{MC};

  int rank = 2;
  int n[] = {X, X};
  int howmany = C;
  int idist = n[0] * n[1], odist = n[0] * n[1];

  for (int m = 0; m < MC; ++m) {
    auto in_m = nullptr;
    for (int c = 0; c < C; ++c) {
      for (int i = 0; i < X2; ++i) {
        int x2 = i % X2, x1 = i / X2;
        in[(m * C + c) * X2 + i] = {std::cos(2.0 * std::numbers::pi * (x1 * x2) / (X2)),
                                    std::sin(2.0 * std::numbers::pi * (x1 * x2) / (X2))};
      }
    }
    plans[m] =
        fftw_plan_many_dft(rank, n, howmany, in.unwrap() + m * C * X2, n, 1, X2,
                           out.unwrap() + m * C * X2, n, 1, X2, fftw::BACKWARD, fftw::MEASURE);
  }

  std::cout << "Planning complete!" << std::endl;

  auto [_, wsp] = benchmark(
      std::cout, [&]() {},
      [&]() {
        cilk_for(int m = 0; m < MC; m++) { fftw_execute(plans[m]); }
      });

  for (unsigned i = 0; i < N_TRIALS; i++) {
    wsp_dump(wsp[i], "fft");
  }

  for (auto plan : plans) {
    fftw_destroy_plan(plan);
  }
}