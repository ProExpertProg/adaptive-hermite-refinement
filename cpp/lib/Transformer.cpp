//
// Created by Luka on 10/23/2024.
//

#include "Transformer.hpp"

namespace ahr {
void Transformer::init() {
  auto r = grid.rBufXY();
  auto c = grid.cBufXY();
  fftFwd = fftw::plan_r2c<2u>::dft(r.to_mdspan(), c.to_mdspan(), fftw::ESTIMATE);
  fftBwd = fftw::plan_c2r<2u>::dft(c.to_mdspan(), r.to_mdspan(), fftw::ESTIMATE);
}

void Transformer::fft(Grid::View::R_XY in, Grid::View::C_XY out) const { fftFwd(in, out); }

void Transformer::bfft(Grid::View::C_XY in, Grid::View::R_XY out) const { fftBwd(in, out); }

void Transformer::normalize(Grid::View::C_XY view, Grid::View::C_XY out) const {
  FOREACH_KXKY (grid, { out(kx, ky) = view(kx, ky) * XYNorm; })
    ;
}

void Transformer::normalize(Grid::View::R_XY view, Grid::View::R_XY out) const {
  FOREACH_XY (grid, { out(x, y) = view(x, y) * XYNorm; })
    ;
}

} // namespace ahr
