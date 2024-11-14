#pragma once
#include "PrepareDerivatives.hpp"
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
  PrepareDerivatives prepareDXY{grid};

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
  /// Computes bracket [op1, op2], expects normalized values
  void bracket(DxDy<View::R_XY> const &op1, DxDy<View::R_XY> const &op2,
               View::R_XY const &output) const;
};
} // namespace ahr
