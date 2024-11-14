#include "Brackets.hpp"
#include "Transformer.hpp"
#include "Filter.hpp"

namespace ahr {

void Brackets::prepareDXY_PH(View::C_XY const &view_K, View::C_XY const &viewDX_K,
                             View::C_XY const &viewDY_K) const {
  grid.for_each_kxky([&](Dim kx, Dim ky) {
    viewDX_K(kx, ky) = kx_(kx) * 1i * view_K(kx, ky) * XYNorm;
    viewDY_K(kx, ky) = ky_(ky) * 1i * view_K(kx, ky) * XYNorm;
  });
}

void Brackets::bracket(DxDy<View::R_XY> const &op1, DxDy<View::R_XY> const &op2,
                       View::R_XY const &output) const {
  grid.for_each_xy([&](Dim x, Dim y) {
    output(x, y) = op1.DX(x, y) * op2.DY(x, y) - op1.DY(x, y) * op2.DX(x, y);
  });
}

void Brackets::derivatives(View::C_XY const &op, DxDy<View::R_XY> output) const {
  DxDy<Buf::C_XY> Der_K{grid.KX, grid.KY};
  prepareDXY_PH(op, Der_K.DX, Der_K.DY);
  tf.bfft(Der_K.DX, output.DX);
  tf.bfft(Der_K.DY, output.DY);
}

Brackets::Buf::C_XY Brackets::halfBracket(DxDy<View::R_XY> derOp1, DxDy<View::R_XY> derOp2) const {
  Buf::R_XY br = grid.rBufXY();
  Buf::C_XY br_K = grid.cBufXY();
  bracket(derOp1, derOp2, br);
  tf.fft(br, br_K);
  hlFilter(br_K);
  br_K(0, 0) = 0;
  return br_K;
}

[[nodiscard]] Brackets::Buf::C_XY Brackets::fullBracket(View::C_XY op1, View::C_XY op2) const {
  auto derOp1 = grid.dBufXY(), derOp2 = grid.dBufXY();
  derivatives(op1, derOp1);
  derivatives(op2, derOp2);

  return halfBracket(derOp1, derOp2);
}

} // namespace ahr