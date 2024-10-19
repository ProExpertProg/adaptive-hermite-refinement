#pragma once

#include "HermiteRunner.hpp"
#include "constants.hpp"
#include "nonlinears.hpp"

#include <fftw-cpp/fftw-cpp.h>
#include <iomanip>
#include <type_traits>

#define _ln1(x) #x
#define _ln2(x) _ln1(x)
#define debug(name, var) print(__FILE__ ":" _ln2(__LINE__) " " name, var)
#define debug2(var) debug(#var, (var))
#define debugXY(var)                                                                               \
  debug2((var).DX);                                                                                \
  debug2((var).DY)

namespace ahr {
namespace stdex = std::experimental;

class Naive : public ahr::HermiteRunner {
public:
  /**
   * Construct a new simulation using the Naive approach.
   * @param out The stream to which we're logging.
   * @param M The number of moments. Note that moments are 0-indexed, meaning that the highest
   * moment is M-1.
   * @param X The size of the X domain.
   * @param Y The size of the Y domain.
   */
  Naive(std::ostream &out, Dim M, Dim X, Dim Y);

  void init(std::string_view equilibriumName) override;

  void run(Dim N, Dim saveInterval) override;

  mdarray<Real, dextents<Dim, 2u>> getFinalAPar() override;

private:
  template <size_t D, bool IsReal>
  using buf_left = fftw::basic_mdbuffer<Real, stdex::dextents<std::size_t, D>, Complex,
                                        stdex::layout_left, IsReal>;

public:
  using Buf3D_K = buf_left<3u, false>;
  using Buf2D_K = buf_left<2u, false>;
  using Buf3D = buf_left<3u, true>;
  using Buf2D = buf_left<2u, true>;

  using CViewXY = stdex::mdspan<Complex, stdex::dextents<Dim, 2u>, stdex::layout_left>;
  using ViewXY = stdex::mdspan<Real, stdex::dextents<Dim, 2u>, stdex::layout_left>;

private:
  Dim const M, X, Y, KX{X / 2 + 1}, KY{Y};
  Real dt{-1};        ///< timestep
  Real elapsedT{0.0}; ///< total time elapsed

  void hlFilter(CViewXY &complexArray);
  void fft(ViewXY in, CViewXY out); ///< FFT with Hou-Li Filter

  fftw::plan_r2c<2u> fft_base{};
  fftw::plan_c2r<2u> fftInv{};
  Real bPerpMax{0};

  static constexpr Dim N_E = 0;
  static constexpr Dim A_PAR = 1;
  static constexpr Dim G_MIN = 2;
  const Dim LAST = M - 1; ///< This is equivalent to ngtot in Viriato

  template <class Buffer> struct DxDy {
    using buffer_t = Buffer;
    Buffer DX, DY;

    template <typename... Args>
      requires std::constructible_from<Buffer, Args...>
    explicit DxDy(Args &&...args)
        : DX{std::forward<Args>(args)...}, DY{std::forward<Args>(args)...} {}

    DxDy(Buffer dx, Buffer dy) : DX(dx), DY(dy) {}

    template <class U>
      requires std::convertible_to<Buffer, U>
    operator DxDy<U>() { // NOLINT(google-explicit-constructor)
      return {U(DX), U(DY)};
    }
  };

  /// \defgroup Buffers for all the physical quantities used.
  /// Names ending in K mean the values are in phase space.
  /// @{

  /// g_m: moment values for moments $m \in [0,M-1]$.
  /// The following values are special moments:
  /// - m=0: n_e (charge density)
  /// - m=1: A∥ (or Apar, parallel velocity)
  /// TODO maybe instead of these enormous amounts of memory, we could reuse (parallelism might
  /// suffer)
  Buf3D_K moments_K{KX, KY, M}, momentsNew_K{KX, KY, M};

  /// A|| equilibrium value, used in corrector step
  Buf2D_K aParEq_K{KX, KY};

  /// Φ: the electrostatic potential.
  Buf2D_K phi_K{KX, KY}, phi_K_New{KX, KY};

  /// sq(∇⊥) A∥, also parallel electron velocity
  Buf2D_K ueKPar_K{KX, KY}, ueKPar_K_New{KX, KY};
  /// @}

  void for_each_xy(std::invocable<Dim, Dim> auto fun) const {
    for (Dim y = 0; y < Y; ++y) {
      for (Dim x = 0; x < X; ++x) {
        fun(x, y);
      }
    }
  }

  /// Iterate in phase space, will later be changed to account for phase space dims
  void for_each_kxky(std::invocable<Dim, Dim> auto fun) const {
    for (Dim ky = 0; ky < KY; ++ky) {
      for (Dim kx = 0; kx < KX; ++kx) {
        fun(kx, ky);
      }
    }
  }

  void for_each_mxy(std::invocable<Dim, Dim, Dim> auto fun) {
    for (Dim m = 0; m < M; ++m) {
      for (Dim y = 0; y < Y; ++y) {
        for (Dim x = 0; x < X; ++x) {
          fun(m, x, y);
        }
      }
    }
  }

  /// Returns a 2D mdspan of values in the XY space for a specified m.
  template <class Buf>
    requires std::same_as<std::decay_t<Buf>, Buf3D> or std::same_as<std::decay_t<Buf>, Buf3D_K>
  static auto sliceXY(Buf &moments, Dim m) {
    return stdex::submdspan(moments.to_mdspan(), stdex::full_extent, stdex::full_extent, m);
  }

  /// Returns a DxDy of 2D mdspans for a specified m.
  template <class Buf>
    requires std::same_as<Buf, Buf3D> // or std::same_as<Buf, Buf3D_K> // TODO likely no need for
                                      // DxDy in K space?
  static auto sliceXY(DxDy<Buf> &moments, Dim m) {
    return DxDy{sliceXY(moments.DX, m), sliceXY(moments.DY, m)};
  }

  /// Prepares the δx and δy of viewPH in phase space, as well as over-normalizes
  /// (after inverse FFT, values will be properly normalized)
  void prepareDXY_PH(CViewXY view_K, CViewXY viewDX_K, CViewXY viewDY_K) {
    for_each_kxky([&](Dim kx, Dim ky) {
      viewDX_K(kx, ky) = kx_(kx) * 1i * view_K(kx, ky) / double(X) / double(Y);
      viewDY_K(kx, ky) = ky_(ky) * 1i * view_K(kx, ky) / double(X) / double(Y);
    });
  }

  /// computes bracket [view, other], expects normalized values
  void bracket(const ViewXY &dxOp1, const ViewXY &dyOp1, const ViewXY &dxOp2, const ViewXY &dyOp2,
               const ViewXY &output) {
    for_each_xy([&](Dim x, Dim y) {
      output(x, y) = dxOp1(x, y) * dyOp2(x, y) - dyOp1(x, y) * dxOp2(x, y);
    });
  }

  /// bracket overload for DxDy params
  void bracket(const DxDy<ViewXY> &op1, const DxDy<ViewXY> &op2, const ViewXY &output) {
    bracket(op1.DX, op1.DY, op2.DX, op2.DY, output);
  }

  /// Bracket that only takes inputs and allocates temporaries and output
  [[nodiscard]] Buf2D_K fullBracket(CViewXY op1, CViewXY op2);

  /// Compute derivatives in real space and store them in output
  void derivatives(const CViewXY &value, DxDy<ViewXY> output);

  /// Bracket that takes in derivatives that were already computed
  [[nodiscard]] Buf2D_K halfBracket(DxDy<ViewXY> op1, DxDy<ViewXY> op2);

  // =================
  // Math helpers
  // TODO other file/class
  // =================

  [[nodiscard]] Real ky_(Dim ky) const {
    return (ky <= (KY / 2) ? Real(ky) : Real(ky) - Real(KY)) * Real(lx) / Real(ly);
  };
  [[nodiscard]] Real kx_(Dim kx) const { return Real(kx); };

  [[nodiscard]] Real kPerp2(Dim kx, Dim ky) const {
    auto dkx = kx_(kx), dky = ky_(ky);
    return dkx * dkx + dky * dky;
  }

  [[nodiscard]] Real kPerp(Dim kx, Dim ky) const { return std::sqrt(kPerp2(kx, ky)); }

  [[nodiscard]] Real exp_nu(Dim kx, Dim ky, Real nu2, Real dt) const {
    return std::exp(-(nu * kPerp2(kx, ky) + nu2 * std::pow(kPerp2(kx, ky), hyper_order)) * dt);
  }

  [[nodiscard]] Real exp_gm(Dim m, Real hyper_nuei, Real dt) const {
    return exp(-(Real(m) * nu_ei + std::pow(m, 2 * hyper_morder) * hyper_nuei) * dt);
  }

  [[nodiscard]] Real exp_eta(Dim kx, Dim ky, Real res2, Real dt) const {
    return std::exp(-(res * kPerp2(kx, ky) + res2 * std::pow(kPerp2(kx, ky), hyper_order)) * dt /
                    (1.0 + kPerp2(kx, ky) * de * de));
  }

  /// getTimestep calculates flows and magnetic fields to determine a dt.
  /// It also updates bPerpMax in the process.
  [[nodiscard]] Real getTimestep(DxDy<ViewXY> dPhi, DxDy<ViewXY> dNE, DxDy<ViewXY> dAPar) {
    // compute flows
    DxDy<ViewXY> ve, b;
    Real vyMax{0}, vxMax{0}, bxMax{0}, byMax{0};
    bPerpMax = 0;

    // Note that this is minus in Viriato, but we don't care because we're taking the absolute value
    // anyway.
    ve.DX = dPhi.DY;
    ve.DY = dPhi.DX;
    b.DX = dAPar.DY;
    b.DY = dAPar.DX;

    for_each_xy([&](Dim x, Dim y) {
      bxMax = std::max(bxMax, std::abs(b.DX(x, y)));
      byMax = std::max(byMax, std::abs(b.DY(x, y)));
      bPerpMax = std::max(bPerpMax, std::sqrt(b.DX(x, y) * b.DX(x, y) + b.DY(x, y) * b.DY(x, y)));
      vxMax = std::max(vxMax, std::abs(ve.DX(x, y)));
      vyMax = std::max(vyMax, std::abs(ve.DY(x, y)));
      if (rhoI >= smallRhoI) {
        vxMax = std::max(vxMax, rhoS * rhoS * std::abs(dNE.DX(x, y)));
        vyMax = std::max(vyMax, rhoS * rhoS * std::abs(dNE.DY(x, y)));
      }
    });

    Real kperpDum2 = std::pow(ky_(KY / 2), 2) + std::pow(Real(KX), 2);
    Real omegaKaw;
    if (rhoI < smallRhoI) {
      omegaKaw = std::sqrt(1.0 + kperpDum2 * (3.0 / 4.0 * rhoI * rhoI + rhoS * rhoS)) *
                 ky_(KY / 2) * bPerpMax / (1.0 + kperpDum2 * de * de);
    } else {
      omegaKaw =
          std::sqrt(kperpDum2 *
                    (rhoS * rhoS - rhoI * rhoI / (Gamma0(0.5 * kperpDum2 * rhoI * rhoI) - 1.0))) *
          ky_(KY / 2 + 1) * bPerpMax / std::sqrt(1.0 + kperpDum2 * de * de);
    }

    Real dx = lx / Real(X), dy = ly / Real(Y);

    Real CFLFlow;
    if (M > 2) {
      CFLFlow = std::min({dx / vxMax, dy / vyMax, 2.0 / omegaKaw,
                          std::min(dx / bxMax, dy / byMax) / (rhoS / de) / std::sqrt(LAST)});
    } else {
      CFLFlow = std::min({dx / vxMax, dy / vyMax, 2.0 / omegaKaw, dx / bxMax, dy / byMax});
    }

    // DEBUG
    out << "vxmax: " << vxMax << " vymax: " << vyMax << std::endl;
    out << "vxmax: " << vxMax << " vymax: " << vyMax << std::endl;
    out << "bxmax: " << bxMax << " bymax: " << byMax << std::endl;
    out << "bperp_max: " << bPerpMax << " omegakaw: " << omegaKaw << std::endl;
    out << "CFLFlow: " << CFLFlow << std::endl;
    out << "calculated dt: " << CFLFlow * CFLFrac << std::endl;

    return CFLFrac * CFLFlow;
  }

  // TODO this is a terrible hack
  template <typename View>
    requires(not std::same_as<View, Buf2D>) and (not std::same_as<View, Buf2D_K>)

  void print(std::string_view name, View view) const {
    out << name << ":\n";
    for (int x = 0; x < view.extent(0); ++x) {
      for (int y = 0; y < view.extent(1); ++y) {
        out << std::setprecision(16) << view(x, y) << " ";
      }
      out << std::endl;
    }
    out << "===========================================" << std::endl;
  }

  template <typename Buf>
    requires std::same_as<Buf, Buf2D> or std::same_as<Buf, Buf2D_K>
  [[maybe_unused]] void print(std::string_view name, Buf &view) const {
    print(name, view.to_mdspan());
  }

public:
  struct Energies {
    Real magnetic{0.0}, kinetic{0.0};

    Real total() const { return magnetic + kinetic; }
  };

  Energies calculateEnergies() const;

  Real elapsedTime() const { return elapsedT; }

  // Returns a const CViewXY
  auto getMoment_K(Dim m) const { return sliceXY(moments_K, m); }

  Buf2D getMoment(Dim m) const;

private:
  Real updateTimestep(Real dt, Real tempDt, bool noInc, Real relative_error) const;

public:
  // TODO(luka) separate exporting utility
  void exportToNpy(std::string path, ViewXY view) const;

  // Will also normalize and inverseFFT
  void exportToNpy(std::string path, CViewXY view) const;

private:
  // If view = viewOut, then we're normalizing in place.
  void normalize(Naive::ViewXY view, Naive::ViewXY viewOut) const;

  void exportTimestep(Dim t);
};
}; // namespace ahr