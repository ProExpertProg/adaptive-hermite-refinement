#include "grid.hpp"

#include <gmock/gmock.h>

namespace ahr {
TEST(Grid, BufferAlloc) {
  Grid grid{5, 16, 24};
  ASSERT_EQ(grid.KX, 9);
  ASSERT_EQ(grid.KY, 24);

  auto cxy = grid.cBufXY();
  EXPECT_EQ(cxy.extents(), (Grid::Buf::C_XY::extents_type{9, 24}));

  auto rxy = grid.rBufXY();
  EXPECT_EQ(rxy.extents(), (Grid::Buf::R_XY::extents_type{16, 24}));

  auto cmxy = grid.cBufMXY();
  EXPECT_EQ(cmxy.extents(), (Grid::Buf::C_MXY::extents_type{9, 24, 5}));

  auto rmxy = grid.rBufMXY();
  EXPECT_EQ(rmxy.extents(), (Grid::Buf::R_MXY::extents_type{16, 24, 5}));
}

TEST(Grid, Slicing) {
  Grid grid{5, 16, 24};
  auto cmxy = grid.cBufMXY();
  auto rmxy = grid.rBufMXY();

  auto cmxy_slice = Grid::sliceXY(cmxy, 2);
  auto rmxy_slice = Grid::sliceXY(rmxy, 2);

  static_assert(std::is_same_v<decltype(cmxy_slice), Grid::View::C_XY>);
  static_assert(std::is_same_v<decltype(rmxy_slice), Grid::View::R_XY>);

  EXPECT_EQ(cmxy_slice.extents(), grid.cBufXY().extents());
  EXPECT_EQ(rmxy_slice.extents(), grid.rBufXY().extents());

  for (Dim kx = 0; kx < grid.KX; ++kx) {
    for (Dim ky = 0; ky < grid.KY; ++ky) {
      EXPECT_EQ(&cmxy_slice(kx, ky), &cmxy(kx, ky, 2));
    }
  }

  for (Dim x = 0; x < grid.X; ++x) {
    for (Dim y = 0; y < grid.Y; ++y) {
      EXPECT_EQ(&rmxy_slice(x, y), &rmxy(x, y, 2));
    }
  }
}

TEST(Grid, ForEach) {
  Grid grid{5, 16, 24};

  struct MockLoop {
    MOCK_METHOD(void, fun, (Dim, Dim));
  } mockLoop;

  for (Dim kx = 0; kx < grid.KX; ++kx) {
    for (Dim ky = 0; ky < grid.KY; ++ky) {
      EXPECT_CALL(mockLoop, fun(kx, ky));
    }
  }

  FOREACH_KXKY(grid, { mockLoop.fun(kx, ky); });
  testing::Mock::VerifyAndClearExpectations(&mockLoop);

  for (Dim x = 0; x < grid.X; ++x) {
    for (Dim y = 0; y < grid.Y; ++y) {
      EXPECT_CALL(mockLoop, fun(x, y));
    }
  }

  grid.for_each_xy([&](Dim x, Dim y) { mockLoop.fun(x, y); });
  testing::Mock::VerifyAndClearExpectations(&mockLoop);
}

}; // namespace ahr
