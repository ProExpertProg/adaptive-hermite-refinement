#include "Exporter.hpp"
#include "Transformer.hpp"
#include <cnpy.h>
#include <filesystem>

namespace ahr {
namespace fs = std::filesystem;
void Exporter::exportTo(fs::path const &filename, Grid::View::C_XY cView) {
  // fft overwrites the input, so we need to copy it to a temporary buffer
  auto tempK = grid.cBufXY();
  auto temp = grid.rBufXY();

  // Copy and also normalize
  tf.normalize(cView, tempK);

  // Backwards FFT
  tf.bfft(tempK, temp);

  // Write the real buffer to file
  exportTo(filename, temp);
}

void Exporter::exportTo(fs::path const &filename, Grid::View::R_XY rView) {
  // TODO Copy the data to a layout-right buffer
  //  stdex::mdarray<Real, stdex::dextents<size_t, 2u>> rArray{rView.extents()};
  //  grid.for_each_xy([&](Dim x, Dim y) { rArray(x, y) = rView(x, y); });
  //  cnpy::npy_save(prefix_dir / filename, rArray.data(), {grid.X, grid.Y}, "w");

  // Dimensions are switched because we use layout_left
  auto const path = filename.is_absolute() ? filename : prefix_dir / filename;
  cnpy::npy_save(path, rView.data_handle(), {grid.Y, grid.X}, "w");
}

NpyMdspan
Exporter::importReal(const fs::path &filename) { // NOLINT(*-convert-member-functions-to-static)
  auto const path = filename.is_absolute() ? filename : prefix_dir / filename;
  return NpyMdspan{cnpy::npy_load(path)};
}

void Exporter::importReal(const fs::path &filename, Grid::View::R_XY rView) {
  auto npy = importReal(filename);
  if (!npy.valid()) { throw std::runtime_error("Invalid npy file"); }
  if (npy.view().extents() != rView.extents()) { throw std::runtime_error("Incompatible extents"); }

  // Copy the data
  grid.for_each_xy([&](Dim x, Dim y) { rView(x, y) = npy.view()(x, y); });
}

Grid::Buf::R_XY Exporter::importRealBuf(const fs::path &filename) {
  auto rBuf = grid.rBufXY();
  importReal(filename, rBuf.to_mdspan());
  return rBuf;
}

} // namespace ahr