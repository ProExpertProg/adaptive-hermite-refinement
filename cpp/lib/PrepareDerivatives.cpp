#include "PrepareDerivatives.hpp"

namespace ahr {

void PrepareDerivatives::operator()(View::C_XY const &in, DxDy<View::C_XY> out) const {
  grid.for_each_kxky([&](Dim kx, Dim ky) {
    out.DX(kx, ky) = kx_(kx) * 1i * in(kx, ky) * XYNorm;
    out.DY(kx, ky) = ky_(ky) * 1i * in(kx, ky) * XYNorm;
  });
}

} // namespace ahr