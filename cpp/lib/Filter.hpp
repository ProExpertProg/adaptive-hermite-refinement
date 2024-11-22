#pragma once
#include "constants.hpp"
#include "grid.hpp"

namespace ahr {

class HouLiFilter {
public:
  explicit HouLiFilter(Grid const &grid) : grid(grid) {}

  void operator()(Grid::View::C_XY view) const;

protected:
  Grid const &grid;
};

class HouLiFilterCached : HouLiFilter {
public:
  explicit HouLiFilterCached(Grid const &grid);
  void operator()(Grid::View::C_XY view) const;

private:
  /// Pre-calculated factors for the Hou-Li filter.
  /// Note that this is a real buffer with dimensions (KX,KY)
  Grid::Buf::R_XY factors;
};

class HouLiFilterCached1D : protected HouLiFilter {
public:
  explicit HouLiFilterCached1D(Grid const &grid);
  void operator()(Grid::View::C_XY view) const;

protected:
  std::vector<Real> factors_x, factors_y;
};
} // namespace ahr