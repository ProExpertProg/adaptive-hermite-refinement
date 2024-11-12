//
// Created by Luka on 11/2/2024.
//

#include "Filter.hpp"

#include <eve/module/core.hpp>
#include <immintrin.h>

namespace ahr {
void HouLiFilter::operator()(Grid::View::C_XY view) {
  grid.for_each_kxky([&](Dim kx, Dim ky) {
    view(kx, ky) *=
        exp(-36.0 * pow(kx_(kx) / grid.KX, 36.0)) * exp(-36.0 * pow(ky_(ky) / grid.KY, 36.0));
  });
}

HouLiFilterCached::HouLiFilterCached(Grid const &grid)
    : HouLiFilter(grid), factors(std::array{grid.KX, grid.KY}) {
  grid.for_each_kxky([&](Dim kx, Dim ky) {
    factors(kx, ky) =
        exp(-36.0 * pow(kx_(kx) / grid.KX, 36.0)) * exp(-36.0 * pow(ky_(ky) / grid.KY, 36.0));
  });
}

void HouLiFilterCached::operator()(Grid::View::C_XY view) {
  grid.for_each_kxky([&](Dim kx, Dim ky) { view(kx, ky) *= factors(kx, ky); });
}

HouLiFilterCached1D::HouLiFilterCached1D(Grid const &grid)
    : HouLiFilter(grid), factors_x(grid.KX), factors_y(grid.KY) {
  for (Dim kx = 0; kx < grid.KX; ++kx) {
    factors_x.at(kx) = exp(-36.0 * pow(kx_(kx) / grid.KX, 36.0));
  }
  for (Dim ky = 0; ky < grid.KY; ++ky) {
    factors_y.at(ky) = exp(-36.0 * pow(ky_(ky) / grid.KY, 36.0));
  }
}

void HouLiFilterCached1D::operator()(Grid::View::C_XY view) {
  grid.for_each_kxky([&](Dim kx, Dim ky) {
    // Extra multiplication at runtime for lower memory cost
    view(kx, ky) *= factors_x[kx] * factors_y[ky];
  });
}

HouLiFilterCached1DVector::HouLiFilterCached1DVector(Grid const &grid) : HouLiFilterCached1D(grid) {
  assert(grid.KX > R_WIDTH);
}

void HouLiFilterCached1DVector::operator()(Grid::View::C_XY view) {
  // This method applies the HouLi filter to view using vector instructions.
  // An array of contiguous complex numbers is simply treated as a real array
  // with double the length.
  // The code is vectorized along the kx (continuous) dimension.
  // We load R_WIDTH factors_x and expand them into two registers, duplicating each element.
  // That way, each two consecutive real numbers (real and imaginary parts)
  // get multiplied with the same factor.
  // Finally, we read the complex numbers, multiply with kx- and ky-factors, and write it back.
  // Duplication demo:
  // vfx_full = {kx, kx+1, kx+2, kx+3, kx+4, kx+5, kx+6, kx+7}
  // into
  // lower_fx = {kx, kx, kx+1, kx+1, kx+2, kx+2, kx+3, kx+3}
  // upper_fx = {kx+4, kx+4, kx+5, kx+5, kx+6, kx+6, kx+7, kx+7}
  //
  // TODO multiple fy at once to better reuse fx

  static_assert(view.stride(0) == 1); // contiguous in kx

  // permutation indices
  static_assert(C_WIDTH == 4); // idx are hardcoded
  VIdx const lower_idx = {0, 0, 1, 1, 2, 2, 3, 3};
  VIdx const upper_idx = {4, 4, 5, 5, 6, 6, 7, 7};

  for (int ky = 0; ky < grid.KY; ++ky) {
    // avoid std::vector dereference inside loop:
    // broadcast fy value into vector
    VReal vfy{factors_y[ky]};
    // prepare iteration address for fx
    Real *fx_addr = factors_x.data();

    int kx = 0;
    // Make sure the last element in the 2nd vector isn't past the end
    // Process 1 vector of factors at a time (2 vectors of complex)
    for (; kx <= grid.KX - R_WIDTH; kx += R_WIDTH, fx_addr += R_WIDTH) {
      // get address for two vectors we're writing to
      auto *view_addr = (Real *)&view(kx, ky);
      auto *upper_view_addr = view_addr + R_WIDTH;
      VReal input_lower{view_addr};
      VReal input_upper{upper_view_addr};

      // Load factors
      VReal vfx_full{fx_addr};

      // Permute lower factors, multiply lower input
      VReal lower_fx = _mm512_permutex2var_pd(vfx_full, lower_idx, vfx_full);
      eve::store(input_lower * lower_fx * vfy, view_addr);

      // Permute upper factors, multiply upper input
      VReal upper_fx = _mm512_permutex2var_pd(vfx_full, upper_idx, vfx_full);
      eve::store(input_upper * upper_fx * vfy, upper_view_addr);
    }

    // tail
    for (; kx < grid.KX; ++kx) {
      view(kx, ky) *= factors_x[kx] * factors_y[ky];
    }
  }
}
} // namespace ahr