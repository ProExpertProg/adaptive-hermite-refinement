#include "Transformer.hpp"

#include "util.hpp"
#include <gtest/gtest.h>

namespace ahr {

TEST(Transformer, Forward) {
  Grid grid{5, 16, 24}; // Example grid dimensions
  Transformer tf{grid};
  tf.init();

  auto r = grid.rBufXY();
  auto c = grid.cBufXY();

  // Initialize the input grid with a delta function
  FOREACH_XY(grid, { r(x, y) = (x == 0 && y == 0) ? 1.0 : 0.0; });

  // Perform FFT
  tf.fft(r, c);

  // Check the output (constant)
  FOREACH_KXKY(grid, {
    EXPECT_NEAR(c(kx, ky).real(), 1.0, 1e-6);
    EXPECT_NEAR(c(kx, ky).imag(), 0.0, 1e-6);
  });
}

TEST(Transformer, Backward) {
  Grid grid{5, 16, 24};
  Transformer tf{grid};
  tf.init();

  auto r = grid.rBufXY();
  auto c = grid.cBufXY();

  // Initialize the input with a constant
  FOREACH_KXKY(grid, { c(kx, ky) = 1.0; });

  tf.bfft(c, r);

  // Check the output (delta function)
  FOREACH_XY(grid, {
    if (x == 0 && y == 0) {
      EXPECT_NEAR(r(x, y), grid.X * grid.Y, 1e-6);
    } else {
      EXPECT_NEAR(r(x, y), 0.0, 1e-6);
    }
  });
}

TEST(Transformer, RoundTrip) {
  Grid grid{5, 16, 24};
  Transformer tf{grid};
  tf.init();

  auto r = grid.rBufXY(), r2 = grid.rBufXY();
  auto c = grid.cBufXY(), c2 = grid.cBufXY();

  FOREACH_XY(grid, {
    using namespace std;
    using namespace std::numbers;
    r(x, y) = cos(4 * pi * x / grid.X) + 2 * cos(2 * pi * y / grid.Y);
  });

  tf.fft(r, c);
  tf.normalize(c, c2);
  tf.bfft(c2, r2);

  EXPECT_THAT(r2.to_mdspan(), MdspanElementsAllClose(r.to_mdspan(), 1e-10, 1e-10));
}

} // namespace ahr