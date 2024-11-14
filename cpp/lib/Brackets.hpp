#pragma once
#include "constants.hpp"
#include "grid.hpp"

namespace ahr {

class Transformer;
class HouLiFilter;

class Brackets {
public:
  Brackets(Grid const &grid, Transformer const &tf, HouLiFilter const &hlFilter)
      : grid(grid), tf(tf), hlFilter(hlFilter) {}

private:
  Grid const &grid;
  Transformer const &tf;
  HouLiFilter const &hlFilter; // TODO vectorized

  using View = Grid::View;
  using Buf = Grid::Buf;
  template <class T> using DxDy = Grid::DxDy<T>;

public:
  /// Compute real δx and δy derivatives of complex op, store in output
  void derivatives(View::C_XY const &op, DxDy<View::R_XY> output) const;

  /// Compute the bracket of two complex fields using their derivatives
  [[nodiscard]] Buf::C_XY halfBracket(DxDy<View::R_XY> op1, DxDy<View::R_XY> op2) const;

  /// Compute the bracket of two complex fields using their values
  [[nodiscard]] Buf::C_XY fullBracket(View::C_XY op1, View::C_XY op2) const;

private:
  /// Prepares the δx and δy of viewPH in phase space, as well as over-normalizes
  /// (after inverse FFT, values will be properly normalized)
  void prepareDXY_PH(View::C_XY const &view_K, View::C_XY const &viewDX_K,
                     View::C_XY const &viewDY_K) const;

  /// Computes bracket [op1, op2], expects normalized values
  void bracket(DxDy<View::R_XY> const &op1, DxDy<View::R_XY> const &op2,
               View::R_XY const &output) const;

  /// Normalization factor for FFT
  const Real XYNorm{1.0 / Real(grid.X) / Real(grid.Y)};

  // TODO extract to common utility
  [[nodiscard]] Real ky_(Dim ky) const {
    return (ky <= (grid.KY / 2) ? Real(ky) : Real(ky) - Real(grid.KY)) * Real(lx) / Real(ly);
  }
  [[nodiscard]] Real kx_(Dim kx) const { return Real(kx); }
};
} // namespace ahr
