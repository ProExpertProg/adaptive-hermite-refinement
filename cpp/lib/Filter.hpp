#pragma once
#include "constants.hpp"
#include "grid.hpp"
#include <eve/wide.hpp>

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

class HouLiFilterCached1DVector : HouLiFilterCached1D {
public:
  explicit HouLiFilterCached1DVector(Grid const &grid);
  void operator()(Grid::View::C_XY view) const;

private:
  using VIdx = eve::wide<long long>;
  using VReal = eve::wide<Real>;
  static auto constexpr R_WIDTH = VReal::size();
  static auto constexpr C_WIDTH = VReal::size() / 2;
};
} // namespace ahr