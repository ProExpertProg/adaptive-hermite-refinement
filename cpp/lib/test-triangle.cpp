#include "typedefs.hpp"

#include "cilk.hpp"
#include <algorithm>
#include <argparse/argparse.hpp>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <experimental/mdspan>
#include <iostream>
#include <numbers>
#include <span>
#include <tuple>

using namespace ahr;
using Moments = std::vector<std::vector<std::vector<Complex>>>;

constexpr Real INITIAL_DT = 1e-4;
constexpr Complex INJECT = 3.2i;
constexpr Complex INIT_MOMENT{1e-8, 1e-8};

Real getDT(const Moments &tmpMoments) {
  Real maxMoment = 0;
  for (auto &row : tmpMoments) {
    for (auto &point : row) {
      maxMoment = std::max(maxMoment, std::norm(point[0]));
    }
  }

  Real dt = 1 / std::sqrt(maxMoment);
  return std::min(dt, 1e-2);
}

std::complex<Real> getIk(Dim k) { return Real(k) * std::numbers::inv_pi_v<Real> * 1i / 2.0; }

Complex integrateMoment(std::span<Complex, 3> gm, Real dt, Complex ikx, Complex iky, Dim m, Dim M) {
  Real lower = std::sqrt((Real(m) + 1.f) / 2.f), upper = std::sqrt(Real(m) / 2.f),
       current = std::pow(Real(m) / Real(M), 6.f);
  return -((ikx + iky) * (gm[0] * lower + gm[2] * upper) + gm[1] * current) * dt + gm[1];
}

auto calculateMomentsNaive(Dim nr, Dim M, Dim Kx, Dim Ky)
    -> std::tuple<Moments, std::vector<Real>> {

  std::vector<Real> dts;
  dts.reserve(nr);

  Real dt = INITIAL_DT;
  // Ky by Kx by M+1
  Moments moments{Ky, {Kx, std::vector(M + 1, INIT_MOMENT)}};
  auto tmpMoments = moments;

  for (Dim t = 0; t < nr; ++t) {
    for (Dim ky = 0; ky < Ky; ++ky) {
      for (Dim kx = 0; kx < Kx; ++kx) {
        Complex iky = getIk(ky + 1);
        Complex ikx = getIk(kx + 1);

        auto &momentsForPoint = moments[ky][kx];
        std::array<Complex, 3> tmp{INJECT, momentsForPoint[0], momentsForPoint[1]};
        tmpMoments[ky][kx][0] = integrateMoment(tmp, dt, ikx, iky, 0, M);

        for (Dim m = 1; m < M; ++m) {
          tmpMoments[ky][kx][m] = integrateMoment(
              std::span(momentsForPoint).subspan(m - 1).first<3>(), dt, ikx, iky, m, M);
        }
      }
    }

    // find next dt
    dts.push_back(dt);
    dt = getDT(tmpMoments);

    // should be cheap, simplifies code a lot
    std::swap(moments, tmpMoments);
  }

  return {moments, dts};
}

auto calculateMomentsTriangle(Dim nr, Dim M, Dim Kx, Dim Ky)
    -> std::tuple<Moments, std::vector<Real>> {
  using namespace std::complex_literals;

  std::vector<Real> dts;
  dts.reserve(nr);

  Real dt = INITIAL_DT;
  // Ky by Kx by M+1
  Moments moments{Ky, {Kx, std::vector(M + 1, Complex(0))}};
  auto tmpMoments = moments;

  for (Dim t = 0; t < nr; ++t) {
    for (Dim ky = 0; ky < Ky; ++ky) {
      for (Dim kx = 0; kx < Kx; ++kx) {
        Complex iky = getIk(ky + 1);
        Complex ikx = getIk(kx + 1);

        auto &momentsForPoint = moments[ky][kx];
        std::array<Complex, 3> tmp{INJECT, momentsForPoint[0], momentsForPoint[1]};
        tmpMoments[ky][kx][0] = integrateMoment(tmp, dt, ikx, iky, 0, M);

        Dim nComputedMoments = std::min(M + t, nr) - t;
        for (Dim m = 1; m < nComputedMoments; ++m) {
          tmpMoments[ky][kx][m] = integrateMoment(
              std::span(momentsForPoint).subspan(m - 1).first<3>(), dt, ikx, iky, m, M);
        }

        // copy over rest of moments, need to keep boundary values
        // TODO not necessary everywhere, just copy over 1 (others inside
        // moments should stay untouched
        //  - although there is tmpMoments?)?
        // TODO save 2x the values because boundary, utilize tmpMoments properly
        Dim nTotalMoments = std::max(nr, M);
        assert(nTotalMoments > nComputedMoments);
        std::copy_n(moments[ky][kx].begin() + nComputedMoments, nTotalMoments - nComputedMoments,
                    tmpMoments[ky][kx].begin());
      }
    }

    // find next dt
    dts.push_back(dt);
    dt = getDT(tmpMoments);
    std::swap(moments, tmpMoments);
  }

  // Now we can invert the loops
  for (Dim ky = 0; ky < Ky; ++ky) {
    for (Dim kx = 0; kx < Kx; ++kx) {
      for (Dim t = 0; t < nr; ++t) {

        Complex iky = getIk(ky + 1);
        Complex ikx = getIk(kx + 1);

        // moments is holding all the values that we might need from the
        // appropriate time in history, so just use them.
        // TODO ^ this is not true, as two values from the boundary are needed
        // (not just one)
        //  will need to store half of the boundary separately - maybe we can
        //  use the fact we have two moments arrays anyway?
        auto &momentsForPoint = moments[ky][kx];

        // copy the ones needed in the future
        Dim nTotalTriangleMoments = std::max(nr, M);
        //                assert(nTotalTriangleMoments > nComputedMoments);
        //                std::copy_n(moments[ky][kx].begin() +
        //                nComputedMoments, nTotalMoments - nComputedMoments,
        //                            tmpMoments[ky][kx].begin());
        //                for (Dim m = 1; m < M; ++m) {
        //                    tmpMoments[ky][kx][m] =
        //                    integrateMoment(std::span(momentsForPoint).subspan(m
        //                    - 1).first<3>(), dt,
        //                                                            ikx, iky,
        //                                                            m, M);
        //                }
      }
    }
    std::swap(moments, tmpMoments);
  }

  return {moments, dts};
}

int main(int argc, const char *argv[]) {
  std::cout << "Hello!" << std::endl;

  argparse::ArgumentParser arguments("ahr");

  arguments.add_argument("K")
      .help("Size of Ky, Kx for domain")
      .scan<'i', Dim>()
      .default_value(Dim{256});

  arguments.add_argument("M").help("Number of moments").scan<'i', Dim>().default_value(Dim{100});

  arguments.add_argument("nr").help("Number of timesteps").scan<'i', Dim>().default_value(Dim{20});

  try {
    arguments.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << arguments;
    return 1;
  }

  auto K = arguments.get<Dim>("K"), M = arguments.get<Dim>("M"), nr = arguments.get<Dim>("nr");

  auto start = std::chrono::high_resolution_clock::now();
  auto [moments, dts] = calculateMomentsNaive(nr, M, K, K);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> diff = end - start;

  std::cout << "Time:" << diff.count() << "ms" << std::endl;

  std::cout << "Moments:\n";
  for (auto moment : moments[0][0]) {
    std::cout << moment << " ";
  }
  std::cout << std::endl;
  std::cout << "DTs:\n";
  for (auto dt : dts) {
    std::cout << dt << " ";
  }
  std::cout << std::endl;
}