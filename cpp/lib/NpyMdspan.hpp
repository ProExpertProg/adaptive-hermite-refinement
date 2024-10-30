#pragma once

#include "grid.hpp"
#include <cnpy.h>
#include <filesystem>

namespace ahr {
namespace fs = std::filesystem;

/// Owning holder of a npy array with convenience to see it as an mdspan.
/// We can use this to avoid needlessly copying into an mdarray.
class NpyMdspan {
  cnpy::NpyArray array_;

public:
  explicit NpyMdspan(cnpy::NpyArray array) : array_(std::move(array)) {}

  // Layout-right
  using ViewXY = stdex::mdspan<Real, stdex::dextents<size_t, 2u>>;

  // TODO(luka) const view
  Grid::View::R_XY view() {
    // Reverse the dimensions
    auto shape = array_.shape;
    std::reverse(shape.begin(), shape.end());
    std::span<size_t, 2> const extents{shape.data(), 2};

    // Return a layout_left view
    return Grid::View::R_XY{array_.data<Real>(), extents};
  }

  [[nodiscard]] bool valid() const { return array_.word_size == sizeof(Real); }
};

} // namespace ahr