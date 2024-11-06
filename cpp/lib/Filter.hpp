#pragma once
#include "constants.hpp"
#include "grid.hpp"

namespace ahr {

class HouLiFilter {
public:
  explicit HouLiFilter(Grid const &grid) : grid(grid) {}

  void operator()(Grid::View::C_XY view);

protected:
  Grid const &grid;

  // TODO extract to common utility
  [[nodiscard]] Real ky_(Dim ky) const {
    return (ky <= (grid.KY / 2) ? Real(ky) : Real(ky) - Real(grid.KY)) * Real(lx) / Real(ly);
  }
  [[nodiscard]] Real kx_(Dim kx) const { return Real(kx); }
};

class HouLiFilterCached : HouLiFilter {
public:
  explicit HouLiFilterCached(Grid const &grid);
  void operator()(Grid::View::C_XY view);

private:
  /// Pre-calculated factors for the Hou-Li filter.
  /// Note that this is a real buffer with dimensions (KX,KY)
  Grid::Buf::R_XY factors;
};

class HouLiFilterCached1D : protected HouLiFilter {
public:
  explicit HouLiFilterCached1D(Grid const &grid);
  void operator()(Grid::View::C_XY view);

protected:
  std::vector<Real> factors_x, factors_y;
};
} // namespace ahr