#pragma once
#include "constants.hpp"
#include "grid.hpp"
#include <eve/wide.hpp>

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

class HouLiFilterCached1DVector : HouLiFilterCached1D {
public:
  explicit HouLiFilterCached1DVector(Grid const &grid);
  void operator()(Grid::View::C_XY view);

private:
  using VIdx = eve::wide<long long>;
  using VReal = eve::wide<Real>;
  static auto constexpr R_WIDTH = VReal::size();
  static auto constexpr C_WIDTH = VReal::size() / 2;

  VReal duplicateLower(VReal src); ///< duplicate lower half of src
  VReal duplicateUpper(VReal src); ///< duplicate upper half of src
};
} // namespace ahr