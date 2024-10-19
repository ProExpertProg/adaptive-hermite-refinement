#include "fftw-cpp/fftw-cpp.h"

#include "timing.hpp"

#include <argparse/argparse.hpp>
#include <iostream>
#include <numbers>

int main(int argc, const char *argv[]) {
  namespace stdex = std::experimental;
  argparse::ArgumentParser arguments("ahr");

  arguments.add_argument("X")
      .help("Size of FFT domain (columns)")
      .scan<'i', int>()
      .default_value(128);
  arguments.add_argument("Y").help("Size of FFT domain (rows)").scan<'i', int>().default_value(128);
  arguments.add_argument("-c", "--check").help("Check correctness").flag();

  try {
    arguments.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << arguments;
    return 1;
  }

  const auto X = arguments.get<int>("X");
  const auto Y = arguments.get<int>("Y");
  const auto check = arguments.get<bool>("--check");

  std::cout << "Running transforms of size " << X << "x" << X << std::endl;

  std::vector<fftw::mdbuffer<2u>> in{}, out{}, mid{};
  in.reserve(6);
  out.reserve(6);
  mid.reserve(6);

  auto make_buf = [&]() { return fftw::mdbuffer<2u>{Y, X}; };
  std::generate_n(std::back_inserter(in), 6, make_buf);
  std::generate_n(std::back_inserter(out), 6, make_buf);
  std::generate_n(std::back_inserter(mid), 6, make_buf);

  fftw::plan<2u> plan_2d = fftw::plan<2u>::dft(in[0], out[0], fftw::FORWARD, fftw::MEASURE);

  auto sliceX = [&](auto &buf, size_t x) {
    return stdex::submdspan(buf.to_mdspan(), stdex::full_extent, x);
  };
  auto sliceY = [&](auto &buf, size_t y) {
    return stdex::submdspan(buf.to_mdspan(), y, stdex::full_extent);
  };

  fftw::plan<1u> plan_1d_x =
      fftw::plan<1u>::dft(sliceY(in[1], 0), sliceY(mid[1], 0), fftw::FORWARD, fftw::MEASURE);
  fftw::plan<1u> plan_1d_y =
      fftw::plan<1u>::dft(sliceX(mid[1], 0), sliceX(out[1], 0), fftw::FORWARD, fftw::MEASURE);

  fftw::plan<2u> plan_1d_many_x =
      fftw::plan<2u>::dft_many(in[3], mid[3], {true, false}, fftw::FORWARD, fftw::MEASURE);
  fftw::plan<2u> plan_1d_many_y =
      fftw::plan<2u>::dft_many(mid[3], out[3], {false, true}, fftw::FORWARD, fftw::MEASURE);

  assert(in[0].extent(0) == Y);
  assert(in[0].extent(1) == X);
  for (int y = 0; y < Y; ++y) {
    for (int x = 0; x < X; ++x) {
      for (auto &buf : in) {
        buf(y, x) = {std::cos(2.0 * std::numbers::pi * (y * x) / (X * Y)),
                     std::sin(2.0 * std::numbers::pi * (y * x) / (X * Y))};
      }
    }
  }

  std::cout << "Planning complete!" << std::endl;
  std::cout << "2D transform:" << std::endl;
  auto [_, wsp] = benchmark(
      std::cout, [&]() {},
      [&]() {
        plan_2d();

        // normalize
        for (int y = 0; y < Y; y++) {
          for (int x = 0; x < X; x++) {
            out[0](y, x) /= X * Y;
          }
        }
      });

  // Do equivalent 1D FFTs
  std::cout << "1D transform:" << std::endl;
  auto [_2, wsp2] = benchmark(
      std::cout, [&]() {},
      [&]() {
        for (int y = 0; y < Y; y++) {
          plan_1d_x(sliceY(in[1], y), sliceY(mid[1], y));
        }

        for (int x = 0; x < X; x++) {
          plan_1d_y(sliceX(mid[1], x), sliceX(out[1], x));
        }

        // normalize
        for (int y = 0; y < Y; y++) {
          for (int x = 0; x < X; x++) {
            out[1](y, x) /= X * Y;
          }
        }
      });

  // Do equivalent 1D FFTs with fusion
  std::cout << "1D transform (fusion):" << std::endl;
  auto [_3, wsp3] = benchmark(
      std::cout, [&]() {},
      [&]() {
        for (int y = 0; y < Y; y++) {
          plan_1d_x(sliceY(in[2], y), sliceY(mid[2], y));
          for (int x = 0; x < X; x++) {
            mid[2](y, x) /= X * Y;
          }
        }
        for (int x = 0; x < X; x++) {
          plan_1d_y(sliceX(mid[2], x), sliceX(out[2], x));
        }
      });

  // Do equivalent 1D FFTs with many for y only
  std::cout << "1D transform (many, y):" << std::endl;
  auto [_4, wsp4] = benchmark(
      std::cout, [&]() {},
      [&]() {
        for (int y = 0; y < Y; y++) {
          plan_1d_x(sliceY(in[3], y), sliceY(mid[3], y));
        }

        plan_1d_many_y(mid[3], out[3]);

        // normalize
        for (int y = 0; y < Y; y++) {
          for (int x = 0; x < X; x++) {
            out[3](y, x) /= X * Y;
          }
        }
      });

  // Do equivalent 1D FFTs with many for y only, fusion
  std::cout << "1D transform (many, y, fusion):" << std::endl;
  auto [_5, wsp5] = benchmark(
      std::cout, [&]() {},
      [&]() {
        for (int y = 0; y < Y; y++) {
          plan_1d_x(sliceY(in[4], y), sliceY(mid[4], y));
          for (int x = 0; x < X; x++) {
            mid[4](y, x) /= X * Y;
          }
        }
        plan_1d_many_y(mid[4], out[4]);
      });

  // Do equivalent 1D FFTs with many for both
  std::cout << "1D transform (many):" << std::endl;
  auto [_6, wsp6] = benchmark(
      std::cout, [&]() {},
      [&]() {
        plan_1d_many_x(in[5], mid[5]);
        plan_1d_many_y(mid[5], out[5]);

        // normalize
        for (int y = 0; y < Y; y++) {
          for (int x = 0; x < X; x++) {
            out[5](y, x) /= X * Y;
          }
        }
      });

  if (check) {
    for (int x = 0; x < X; x++) {
      for (int y = 0; y < Y; y++) {
        for (size_t i = 0; i < 6; i++) {
          if (std::abs(out[0](y, x) - out[i](y, x)) > 1e-10) {
            std::cerr << "Buffer " << i << ": mismatch at (" << x << ", " << y
                      << "): " << out[0](y, x) << " vs " << out[i](y, x) << std::endl;
            return 1;
          }
        }
      }
    }
  }
}