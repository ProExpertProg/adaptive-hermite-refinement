#pragma once

#include "constants.hpp"
#include "grid.hpp"
#include <eve/wide.hpp>

namespace ahr {

class PrepareDerivatives {
protected:
  using View = Grid::View;
  template <class T> using DxDy = Grid::DxDy<T>;

public:
  explicit PrepareDerivatives(Grid const &grid) : grid(grid) {}

  void operator()(View::C_XY const &in, DxDy<View::C_XY> out) const;

protected:
  Grid const &grid;

  /// Normalization factor for FFT
  const Real XYNorm{1.0 / Real(grid.X) / Real(grid.Y)};

  // TODO extract to common utility
  [[nodiscard]] Real ky_(Dim ky) const {
    return (ky <= (grid.KY / 2) ? Real(ky) : Real(ky) - Real(grid.KY)) * Real(lx) / Real(ly);
  }
  [[nodiscard]] Real kx_(Dim kx) const { return Real(kx); }
};

class PrepareDerivativesVector : protected PrepareDerivatives {
public:
  explicit PrepareDerivativesVector(Grid const &grid) : PrepareDerivatives(grid) {}

  void operator()(View::C_XY const &in, DxDy<View::C_XY> out) const;

protected:
  using VReal = eve::wide<Real>;

  // TODO vec_utils
  using VIdx = eve::wide<long long>;
  static auto constexpr R_WIDTH = VReal::size();
  static auto constexpr C_WIDTH = VReal::size() / 2;

  static constexpr auto KY_TILE = 4;
};

} // namespace ahr
