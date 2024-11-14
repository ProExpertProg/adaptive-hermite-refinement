#pragma once

#include "constants.hpp"
#include "grid.hpp"

namespace ahr {

class PrepareDerivatives {
  using View = Grid::View;
  template <class T> using DxDy = Grid::DxDy<T>;

public:
  explicit PrepareDerivatives(Grid const &grid) : grid(grid) {}

  void operator()(View::C_XY const &in, DxDy<View::C_XY> out) const;

private:
  Grid const &grid;

  /// Normalization factor for FFT
  const Real XYNorm{1.0 / Real(grid.X) / Real(grid.Y)};

  // TODO extract to common utility
  [[nodiscard]] Real ky_(Dim ky) const {
    return (ky <= (grid.KY / 2) ? Real(ky) : Real(ky) - Real(grid.KY)) * Real(lx) / Real(ly);
  }
  [[nodiscard]] Real kx_(Dim kx) const { return Real(kx); }
};
} // namespace ahr
