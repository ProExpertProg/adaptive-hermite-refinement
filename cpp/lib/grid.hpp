#pragma once

#include "cilk.hpp"
#include "constants.hpp"
#include "typedefs.hpp"

#include <fftw-cpp/fftw-cpp.h>

#define FOREACH_XY(grid, ...)                                                                      \
  do {                                                                                             \
    cilk_for (Dim y = 0; y < (grid).Y; ++y) {                                                      \
      for (Dim x = 0; x < (grid).X; ++x) {                                                         \
        __VA_ARGS__;                                                                               \
      }                                                                                            \
    }                                                                                              \
  } while (0)

#define FOREACH_KXKY(grid, ...)                                                                    \
  do {                                                                                             \
    cilk_for (Dim ky = 0; ky < (grid).KY; ++ky) {                                                  \
      for (Dim kx = 0; kx < (grid).KX; ++kx) {                                                     \
        __VA_ARGS__;                                                                               \
      }                                                                                            \
    }                                                                                              \
  } while (0)

namespace ahr {
namespace stdex = std::experimental;

struct Grid {
  Dim const M, X, Y;
  Dim const KX{X / 2 + 1}, KY{Y};

  Grid(Dim M, Dim X, Dim Y) : M(M), X(X), Y(Y) {}

private:
  template <size_t D, bool IsReal>
  using buf_left = fftw::basic_mdbuffer<Real, stdex::dextents<std::size_t, D>, Complex,
                                        stdex::layout_left, IsReal>;

public:
  struct Buf {
    using C_XY = buf_left<2u, false>;
    using R_XY = buf_left<2u, true>;
    using C_MXY = buf_left<3u, false>;
    using R_MXY = buf_left<3u, true>;
  };

  struct View {
    using C_XY = stdex::mdspan<Complex, stdex::dextents<Dim, 2u>, stdex::layout_left>;
    using R_XY = stdex::mdspan<Real, stdex::dextents<Dim, 2u>, stdex::layout_left>;
  };

private:
  struct {
    Buf::C_XY::extents_type cXY;
    Buf::R_XY::extents_type rXY;
    Buf::C_MXY::extents_type cMXY;
    Buf::R_MXY::extents_type rMXY;
  } extents{
      stdex::extents{KX, KY},
      std::array{X, Y},
      std::array{M, KX, KY},
      std::array{M, X, Y},
  };

  struct {
    Buf::C_XY::mapping_type cXY;
    Buf::R_XY::mapping_type rXY;
    Buf::C_MXY::mapping_type cMXY;
    Buf::R_MXY::mapping_type rMXY;
  } mapping{
      Buf::C_XY::extents_type{KX, KY},
      Buf::R_XY::extents_type{X, Y},
      Buf::C_MXY::extents_type{KX, KY, M},
      Buf::R_MXY::extents_type{X, Y, M},
  };

public:
  /// @defgroup Buffer allocators
  /// @{
  [[nodiscard]] Buf::C_XY cBufXY() const { return {mapping.cXY}; }
  [[nodiscard]] Buf::R_XY rBufXY() const { return {mapping.rXY}; }
  [[nodiscard]] Buf::C_MXY cBufMXY() const { return {mapping.cMXY}; }
  [[nodiscard]] Buf::R_MXY rBufMXY() const { return {mapping.rMXY}; }
  /// @}

  /// Named pair for holding both dx and dy derivatives
  template <class T> struct DxDy {
    T DX, DY;

    template <typename... Args>
      requires std::constructible_from<T, Args...>
    explicit DxDy(Args... args) : DX{args...}, DY{args...} {}

    DxDy(T dx, T dy) : DX(std::move(dx)), DY(std::move(dy)) {}

    template <class U>
      requires std::convertible_to<T, U>
    operator DxDy<U>() { // NOLINT(google-explicit-constructor)
      return {U(DX), U(DY)};
    }
  };

  // TODO test?
  [[nodiscard]] DxDy<Buf::R_XY> dBufXY() const { return {rBufXY(), rBufXY()}; }
  [[nodiscard]] DxDy<Buf::R_MXY> dBufMXY() const { return {rBufMXY(), rBufMXY()}; }

  static View::C_XY sliceXY(Buf::C_MXY &moments, Dim m) {
    return stdex::submdspan(moments.to_mdspan(), stdex::full_extent, stdex::full_extent, m);
  }

  static View::R_XY sliceXY(Buf::R_MXY &moments, Dim m) {
    return stdex::submdspan(moments.to_mdspan(), stdex::full_extent, stdex::full_extent, m);
  }

  static DxDy<View::R_XY> sliceXY(DxDy<Buf::R_MXY> &moments, Dim m) {
    return {sliceXY(moments.DX, m), sliceXY(moments.DY, m)};
  }

  // TODO move iteration to a separate class

  /// Iterate in real space
  void for_each_xy(std::invocable<Dim, Dim> auto fun) const { FOREACH_XY((*this), fun(x, y)); }

  /// Iterate in phase space
  void for_each_kxky(std::invocable<Dim, Dim> auto fun) const {
    FOREACH_KXKY((*this), fun(kx, ky));
  }

  [[nodiscard]] Real ky_(Dim ky) const {
    return (ky <= (KY / 2) ? Real(ky) : Real(ky) - Real(KY)) * Real(lx) / Real(ly);
  }
  [[nodiscard]] Real kx_(Dim kx) const { return Real(kx); }

  [[nodiscard]] Real kPerp2(Dim kx, Dim ky) const {
    auto dkx = kx_(kx), dky = ky_(ky);
    return dkx * dkx + dky * dky;
  }

  [[nodiscard]] Real kPerp(Dim kx, Dim ky) const { return std::sqrt(kPerp2(kx, ky)); }
};

// Options for various buffer types:
// - Grid::Buf::XY_K
// - Grid::Buf::C_XY
// - Grid::Buf::CXY
// - Grid::Buf::KXY
// - Grid::Buf::KXKY
// - Grid::Buf::MKXKY
// - Grid::CBuf::MXY
// - Grid::CBuf::XY
// - Grid::Buf::XY
// - Grid::BufXY
// - Grid::BufMXY
// - Grid::CBufMXY

} // namespace ahr
