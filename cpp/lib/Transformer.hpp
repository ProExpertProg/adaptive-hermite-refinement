#pragma once

#include "grid.hpp"

namespace ahr {

class Transformer {

public:
  explicit Transformer(Grid const &grid) : grid(grid) {}

  /// Plan FFTs, etc.
  void init();

  /// Forward FFT
  void fft(Grid::View::R_XY in, Grid::View::C_XY out) const;

  /// Backwards FFT (unnormalized)
  void bfft(Grid::View::C_XY in, Grid::View::R_XY out) const;

  /// Normalize a complex buffer (can be in-place)
  void normalize(Grid::View::C_XY view, Grid::View::C_XY out) const;

  /// Normalize a real buffer (can be in-place)
  void normalize(Grid::View::R_XY view, Grid::View::R_XY out) const;

private:
  fftw::plan_r2c<2u> fftFwd{};
  fftw::plan_c2r<2u> fftBwd{};

  Grid const &grid;

  /// Normalization factor for FFT
  Real XYNorm{1.0 / double(grid.X) / double(grid.Y)};
};

} // namespace ahr