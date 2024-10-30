#include "Exporter.hpp"
#include "Transformer.hpp"
#include "debug.hpp"
#include "util.hpp"
#include <gtest/gtest.h>

namespace ahr {

class TestExporter : public ::testing::Test {
protected:
  Grid grid{5, 16, 24};
  Transformer tf{grid};
  fs::path const tmp_dir{fs::temp_directory_path()};
  Exporter exporter{grid, tf, tmp_dir};

  TestExporter() { tf.init(); }
};

TEST_F(TestExporter, RoundTripReal) {

  auto rBuf = grid.rBufXY();
  grid.for_each_xy([&](Dim x, Dim y) { rBuf(x, y) = Real(x + y * grid.X); });

  exporter.exportTo("test.npy", rBuf);

  auto const path = tmp_dir / "test.npy";
  ASSERT_TRUE(fs::exists(path));

  // Import buffer and compare
  auto rBuf2 = exporter.importRealBuf("test.npy");
  // No math, tolerance is 0
  EXPECT_THAT(rBuf2.to_mdspan(), MdspanElementsAllClose(rBuf.to_mdspan(), 0.0));

  // Import into npy view and compare
  auto rNpy = exporter.importReal("test.npy");
  EXPECT_THAT(rNpy.view(), MdspanElementsAllClose(rBuf.to_mdspan(), 0.0));
}

TEST_F(TestExporter, RoundTripComplex) {
  auto cBuf = grid.cBufXY(), cBuf2 = grid.cBufXY();
  grid.for_each_kxky([&](Dim kx, Dim ky) {
    cBuf(kx, ky) = {Real(kx) + Real(ky * grid.KX), Real(kx) - Real(ky * grid.KX)};
  });
  tf.normalize(cBuf, cBuf2);

  // Use absolute path this time
  exporter.exportTo(tmp_dir / "test.npy", cBuf);

  auto const path = tmp_dir / "test.npy";
  ASSERT_TRUE(fs::exists(path));

  // Import buffer and compare
  auto rBuf = exporter.importReal("test.npy");
  auto rBuf2 = grid.rBufXY();
  tf.bfft(cBuf2, rBuf2);

  // No math, tolerance is 0
  EXPECT_THAT(rBuf.view(), MdspanElementsAllClose(rBuf2.to_mdspan(), 0.0));
}

} // namespace ahr