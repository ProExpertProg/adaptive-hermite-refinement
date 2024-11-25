#include "PrepareDerivatives.hpp"

#include "debug.hpp"
#include "util.hpp"

#include <gtest/gtest.h>

namespace ahr {

template <typename TestedPrepare> class TestPrepareDerivatives : public ::testing::Test {
protected:
  Grid grid{5, 32, 32};
  PrepareDerivatives prepare{grid};
  TestedPrepare prepare_t{grid};
};

using Types = ::testing::Types<PrepareDerivativesVector>;
TYPED_TEST_SUITE(TestPrepareDerivatives, Types);

TYPED_TEST(TestPrepareDerivatives, Prepare) {
  auto const KX = this->grid.KX, KY = this->grid.KY;
  auto buf = this->grid.cBufXY(), buf_t = this->grid.cBufXY();
  Grid::DxDy<Grid::Buf::C_XY> bufD{KX, KY};
  Grid::DxDy<Grid::Buf::C_XY> bufD_t{KX, KY};

  // initialize buffers
  FOREACH_KXKY(this->grid, {
    buf(kx, ky) = buf_t(kx, ky) =
        1024.0 * Complex{std::sin(2 * pi * kx / KX), std::cos(2 * pi * ky / KY)};
  });

  this->prepare(buf, bufD);
  this->prepare_t(buf_t, bufD_t);

  ASSERT_THAT(bufD_t.DX.to_mdspan(), MdspanElementsAllClose(bufD.DX.to_mdspan(), 1e-16, 1e-15));
  ASSERT_THAT(bufD_t.DY.to_mdspan(), MdspanElementsAllClose(bufD.DY.to_mdspan(), 1e-16, 1e-15));
}

} // namespace ahr
