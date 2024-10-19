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

  try {
    arguments.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << arguments;
    return 1;
  }

  const auto X = arguments.get<int>("X");
  const auto M = arguments.get<unsigned long>("M");

  std::cout << "Running " << M << " transforms of size " << X << "x" << X << std::endl;

  std::vector<fftw::mdbuffer<2u>> in{}, out{};
  std::vector<fftw::plan<2u>> plans{M};
  in.reserve(M);
  out.reserve(M);
  for (int m = 0; m < M; ++m) {
    in.emplace_back(X, X);
    out.emplace_back(X, X);
    for (int x1 = 0; x1 < in[m].extent(0); ++x1) {
      for (int x2 = 0; x2 < in[m].extent(1); ++x2) {
        in[m](x1, x2) = {std::cos(2.0 * std::numbers::pi * (x1 * x2) / (X * X)),
                         std::sin(2.0 * std::numbers::pi * (x1 * x2) / (X * X))};
      }
    }

    plans[m] = fftw::plan<2u>::dft(in[m], out[m], fftw::BACKWARD, fftw::MEASURE);
  }

  std::cout << "Planning complete!" << std::endl;

  auto [_, wsp] = benchmark(
      std::cout, [&]() {},
      [&]() {
        cilk_for(int m = 0; m < M; m++) { plans[m](); }
      });

  for (unsigned i = 0; i < N_TRIALS; i++) {
    wsp_dump(wsp[i], "fft");
  }
}