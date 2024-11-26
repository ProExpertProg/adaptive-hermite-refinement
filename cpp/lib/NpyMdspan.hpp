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
  ViewXY view() {
    std::span<size_t, 2> const extents{array_.shape.data(), 2};
    return ViewXY{array_.data<Real>(), extents};
  }

  [[nodiscard]] bool valid() const { return array_.word_size == sizeof(Real); }
};

} // namespace ahr