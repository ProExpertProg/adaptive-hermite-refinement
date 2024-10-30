#pragma once

#include "NpyMdspan.hpp"
#include "grid.hpp"
#include <filesystem>
#include <optional>

namespace ahr {
namespace fs = std::filesystem;
class Transformer;

/// This class is responsible for exporting and importing .npy buffers.
/// It can automatically transform complex buffers to real before exporting.
/// Any relative path is interpreted as relative to the prefix_dir.
class Exporter {
  Grid const &grid;
  Transformer const &tf;
  fs::path prefix_dir;

public:
  Exporter(Grid const &grid, Transformer const &transformer,
           std::optional<fs::path> prefix_dir = std::nullopt)
      : grid(grid), tf(transformer) {
    if (prefix_dir) {
      this->prefix_dir = fs::canonical(*prefix_dir);
    } else if (auto prefix_dir_env = std::getenv("EXPORT_PREFIX_DIR"); prefix_dir_env) {
      this->prefix_dir = fs::canonical(prefix_dir_env);
    } else {
      this->prefix_dir = fs::current_path();
    }

    fs::create_directories(this->prefix_dir);
  }

  /// Export the complex view to file by transforming it to real first.
  void exportTo(fs::path const &filename, Grid::View::C_XY cView);

  /// Export the real buffer to file.
  void exportTo(fs::path const &filename, Grid::View::R_XY rView);

  /// Import a real buffer from file, and write it to the given view.
  void importReal(const fs::path &filename, Grid::View::R_XY rView);

  /// Import a real buffer from file, and return it as an owning NpyMdspan.
  [[nodiscard]] NpyMdspan importReal(fs::path const &filename);

  /// Import a real buffer from file, and return it as a real buffer.
  [[nodiscard]] Grid::Buf::R_XY importRealBuf(fs::path const &filename);

};
} // namespace ahr
