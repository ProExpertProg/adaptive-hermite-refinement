#pragma once

#include "Naive.hpp"
#include "constants.hpp"
#include "grid.hpp"
#include <experimental/mdarray>
#include <utility>

namespace ahr {
namespace stdex = std::experimental;

// Note that these differ from julia because they're 0-indexed.
Real xx(Dim x, Dim X) { return lx * (Real(x) - Real(X) / 2.0) / Real(X); }

Real yy(Dim y, Dim Y) { return ly * (Real(y) - Real(Y) / 2.0) / Real(Y); }

auto equilibriumGauss(Grid const &g) {
  auto aParEq = g.rBufXY(), phiEq = g.rBufXY();

  g.for_each_xy([&](Dim x, Dim y) {
    using std::numbers::pi;
    aParEq(x, y) = a0 * std::exp(-sq(yy(y, g.Y) * 2 * pi * 2 / ly)) *
                   std::exp(-sq(xx(x, g.X) * 2 * pi * 2 / lx));

    phiEq(x, y) = 0;
  });
  return std::make_pair(std::move(aParEq), std::move(phiEq));
}

auto equilibriumOT01(Grid const &g) {
  auto aParEq = g.rBufXY(), phiEq = g.rBufXY();

  g.for_each_xy([&](Dim x, Dim y) {
    using std::numbers::pi;
    aParEq(x, y) = std::cos(4 * pi * xx(x, g.X) / lx) + 2 * std::cos(2 * pi * yy(y, g.Y) / ly);
    phiEq(x, y) = -2 * (std::cos(2 * pi * xx(x, g.X) / lx) + std::cos(2 * pi * yy(y, g.Y) / ly));
  });

  return std::make_pair(std::move(aParEq), std::move(phiEq));
}

auto equilibrium(std::string_view name, Grid const &g) {
  if (name == "gauss") {
    return equilibriumGauss(g);
  } else if (name == "OT01") {
    return equilibriumOT01(g);
  } else {
    throw std::invalid_argument("name must be one of gauss or OT01");
  }
}
} // namespace ahr
