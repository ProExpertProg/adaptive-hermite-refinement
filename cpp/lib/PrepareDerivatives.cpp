#include "PrepareDerivatives.hpp"

#include <eve/module/core.hpp>

namespace ahr {

void PrepareDerivatives::operator()(View::C_XY const &in, DxDy<View::C_XY> out) const {
  grid.for_each_kxky([&](Dim kx, Dim ky) {
    Complex in_norm = 1i * in(kx, ky) * XYNorm;
    out.DX(kx, ky) = kx_(kx) * in_norm;
    out.DY(kx, ky) = ky_(ky) * in_norm;
  });
}

void PrepareDerivativesVector::operator()(View::C_XY const &in, DxDy<View::C_XY> out) const {
  eve::logical<VIdx> const even_mask{[](int idx, int) { return idx % 2 == 0; }};
  VReal const kx_v_init{[](int idx, int) { return Real(idx / 2); }};
  for (Dim ky = 0; ky < grid.KY; ky += KY_TILE) {
    // broadcast ky values
    using TileReal = std::array<VReal, KY_TILE>;
    TileReal ky_v;
    for (Dim tile_ky = 0; tile_ky < KY_TILE; ++tile_ky) {
      ky_v[tile_ky] = ky_(ky + tile_ky);
    }

    // Initialize kx values
    VReal kx_v = kx_v_init;

    Dim kx = 0;
    for (; kx <= grid.KX - C_WIDTH; kx += C_WIDTH, kx_v += C_WIDTH) {
      auto input = [&](int i) { return (Real *)&in(kx, ky + i); };
      auto out_dx = [&](int i) { return (Real *)&out.DX(kx, ky + i); };
      auto out_dy = [&](int i) { return (Real *)&out.DY(kx, ky + i); };

      for (int i = 0; i < KY_TILE; ++i) {
        auto in_v = VReal{input(i)};
        // (a + bi) * i -> (-b + ai)
        // swap real and imaginary parts
        auto swapped = eve::swap_adjacent(in_v);
        // selectively negate
        auto mul_with_i = eve::minus[even_mask](swapped);
        // normalize
        auto in_norm = mul_with_i * XYNorm;

        eve::store(kx_v * in_norm, out_dx(i));
        eve::store(ky_v[i] * in_norm, out_dy(i));
      }
    }

    // tail
    for (; kx < grid.KX; ++kx) {
      for (int i = 0; i < KY_TILE; ++i) {
        Complex in_norm = 1i * in(kx, ky + i) * XYNorm;
        out.DX(kx, ky + i) = kx_(kx) * in_norm;
        out.DY(kx, ky + i) = ky_(ky + i) * in_norm;
      }
    }
  }
}
} // namespace ahr