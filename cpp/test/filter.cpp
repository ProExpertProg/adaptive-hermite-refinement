#include "Filter.hpp"
#include "grid.hpp"

#include "debug.hpp"
#include "util.hpp"

#include <gtest/gtest.h>

namespace ahr {

template <typename TestedFilter> class TestFilter : public ::testing::Test {
protected:
  Grid grid{5, 32, 32};
  HouLiFilter filter{grid};
  TestedFilter filter_t{grid};
};

using Types = ::testing::Types<HouLiFilterCached, HouLiFilterCached1D, HouLiFilterCached1DVector>;
TYPED_TEST_SUITE(TestFilter, Types);

TYPED_TEST(TestFilter, Filter) {
  auto buf = this->grid.cBufXY();
  auto buf_t = this->grid.cBufXY();

  // initialize buffers
  this->grid.for_each_kxky([&](Dim kx, Dim ky) {
    buf(kx, ky) = {std::sin(2 * pi * kx), std::cos(2 * pi * ky)};
    buf_t(kx, ky) = {std::sin(2 * pi * kx), std::cos(2 * pi * ky)};
  });

  this->filter(buf);
  this->filter_t(buf_t);

  ASSERT_THAT(buf_t.to_mdspan(), MdspanElementsAllClose(buf.to_mdspan(), 1e-16, 1e-15));
}

} // namespace ahr
