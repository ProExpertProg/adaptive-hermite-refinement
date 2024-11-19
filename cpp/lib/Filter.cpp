//
// Created by Luka on 11/2/2024.
//

#include "Filter.hpp"
namespace ahr {
void HouLiFilter::operator()(Grid::View::C_XY view) const {
  grid.for_each_kxky([&](Dim kx, Dim ky) {
    view(kx, ky) *= std::exp(-36.0 * std::pow(grid.kx_(kx) / grid.KX, 36.0)) *
                    std::exp(-36.0 * std::pow(grid.ky_(ky) / grid.KY, 36.0));
  });
}

HouLiFilterCached::HouLiFilterCached(Grid const &grid)
    : HouLiFilter(grid), factors(std::array{grid.KX, grid.KY}) {
  grid.for_each_kxky([&](Dim kx, Dim ky) {
    factors(kx, ky) = std::exp(-36.0 * std::pow(grid.kx_(kx) / grid.KX, 36.0)) *
                      std::exp(-36.0 * std::pow(grid.ky_(ky) / grid.KY, 36.0));
  });
}

void HouLiFilterCached::operator()(Grid::View::C_XY view) const {
  grid.for_each_kxky([&](Dim kx, Dim ky) { view(kx, ky) *= factors(kx, ky); });
}

HouLiFilterCached1D::HouLiFilterCached1D(Grid const &grid)
    : HouLiFilter(grid), factors_x(grid.KX), factors_y(grid.KY) {
  for (Dim kx = 0; kx < grid.KX; ++kx) {
    factors_x.at(kx) = std::exp(-36.0 * std::pow(grid.kx_(kx) / grid.KX, 36.0));
  }
  for (Dim ky = 0; ky < grid.KY; ++ky) {
    factors_y.at(ky) = std::exp(-36.0 * std::pow(grid.ky_(ky) / grid.KY, 36.0));
  }
}

void HouLiFilterCached1D::operator()(Grid::View::C_XY view) const {
  grid.for_each_kxky([&](Dim kx, Dim ky) {
    // Extra multiplication at runtime for lower memory cost
    view(kx, ky) *= factors_x[kx] * factors_y[ky];
  });
}
} // namespace ahr